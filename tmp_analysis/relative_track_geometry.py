"""Pairwise geometry in original-frame units, separate from pretrained fusion.

This module creates observable motion/shape features. It does not fit a gate,
replace bbox_transform inputs, assign IDs, estimate camera motion or claim that
constant velocity is correct. An optional camera transform must be supplied
externally in original pixel coordinates for the stated reference->query time.
"""
import numpy as np


def boxes_xyxy(value):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim == 1:
        value = value[None]
    if value.ndim != 2 or value.shape[1] != 4 or not np.isfinite(value).all():
        raise ValueError('Expected finite N x 4 xyxy boxes')
    if np.any(value[:, 2:] <= value[:, :2]):
        raise ValueError('Expected positive box width and height')
    return value


def warp_boxes(boxes, homography):
    boxes = boxes_xyxy(boxes)
    h = np.asarray(homography, dtype=np.float64)
    if h.shape != (3, 3) or not np.isfinite(h).all():
        raise ValueError('Expected finite 3 x 3 externally estimated homography')
    scale = np.max(np.abs(h))
    if scale == 0:
        raise ValueError('Camera homography must have full rank')
    h = h/scale
    if np.linalg.matrix_rank(h) != 3:
        raise ValueError('Camera homography must have full rank')
    x1, y1, x2, y2 = boxes.T
    corners = np.stack((np.stack((x1,y1),1), np.stack((x2,y1),1),
                        np.stack((x2,y2),1), np.stack((x1,y2),1)),1)
    homogeneous = np.concatenate((corners, np.ones((*corners.shape[:2],1))),2) @ h.T
    den = homogeneous[:,:,2]
    if np.any(np.abs(den) <= 1e-9) or np.any(np.min(den,1)*np.max(den,1) <= 0):
        raise ValueError('Camera warp crosses the projective horizon')
    xy = homogeneous[:,:,:2] / homogeneous[:,:,2:]
    return boxes_xyxy(np.concatenate((xy.min(1),xy.max(1)),1))


def warp_aligned(boxes, homographies, reference_times):
    h = np.asarray(homographies, dtype=np.float64)
    if h.shape == (3,3):
        if len(reference_times) and np.any(reference_times != reference_times[0]):
            raise ValueError('A shared camera warp requires the same reference frame')
        return warp_boxes(boxes,h)
    if h.shape != (len(boxes),3,3):
        raise ValueError('Supply one reference->query camera warp per memory')
    return np.concatenate([warp_boxes(boxes[j],h[j]) for j in range(len(boxes))],0) if len(boxes) else boxes.copy()


def pair_geometry(query_boxes, memory_boxes, query_frame_index,
                  memory_frame_indices, previous_memory_boxes=None,
                  previous_frame_indices=None, memory_to_query_homography=None,
                  previous_to_memory_homographies=None):
    """Return Nquery x Nmemory motion/shape cues with explicit missing history.

    Observed displacement is relative to camera-warped memory. Velocity uses
    previous->memory original-frame spacing. If an ego-motion warp is used,
    historical boxes also need previous->memory warps before velocity can be
    called compensated. Missing previous state disables CV prediction per
    memory; it never invents a zero velocity observation.
    """
    q, m = boxes_xyxy(query_boxes), boxes_xyxy(memory_boxes)
    mt = np.asarray(memory_frame_indices, dtype=np.float64)
    if mt.shape != (len(m),) or not np.isfinite(mt).all():
        raise ValueError('One original frame index per memory box is required')
    qt = float(query_frame_index)
    if not np.isfinite(qt) or np.any(qt <= mt):
        raise ValueError('Query must follow every reference in original-frame units')
    dt = qt - mt
    have_camera = memory_to_query_homography is not None
    mw = warp_aligned(m, memory_to_query_homography, mt) if have_camera else m.copy()
    qc, mc = (q[:,:2]+q[:,2:])/2, (mw[:,:2]+mw[:,2:])/2
    qw, mwh = q[:,2:]-q[:,:2], mw[:,2:]-mw[:,:2]
    scale = np.sqrt(qw[:,None,:] * mwh[None,:,:])
    displacement = (qc[:,None,:]-mc[None,:,:])/scale
    log_size_change = np.log(qw[:,None,:]/mwh[None,:,:])
    velocity = np.zeros((len(m),2), dtype=np.float64)
    valid = np.zeros(len(m), dtype=bool)
    have_previous = previous_memory_boxes is not None or previous_frame_indices is not None
    if have_previous:
        if previous_memory_boxes is None or previous_frame_indices is None:
            raise ValueError('Historical boxes and frame indices must be supplied together')
        previous = np.asarray(previous_memory_boxes, dtype=np.float64)
        pt = np.asarray(previous_frame_indices, dtype=np.float64)
        if previous.shape != m.shape or pt.shape != mt.shape:
            raise ValueError('History must align with memory boxes')
        valid = np.isfinite(previous).all(1) & np.isfinite(pt)
        if np.any(valid & (pt >= mt)):
            raise ValueError('Historical frame index must precede reference')
        if np.any(valid):
            boxes_xyxy(previous[valid])
        if have_camera and np.any(valid) and previous_to_memory_homographies is None:
            raise ValueError('Camera-compensated velocity requires aligned historical camera warps')
        aligned_previous = previous.copy()
        if previous_to_memory_homographies is not None:
            hist_h = np.asarray(previous_to_memory_homographies,dtype=np.float64)
            if hist_h.shape != (len(m),3,3):
                raise ValueError('One historical camera warp per memory is required')
            for j in np.flatnonzero(valid):
                aligned_previous[j] = warp_boxes(previous[j],hist_h[j])[0]
        last_c = (m[:,:2]+m[:,2:])/2
        prev_c = (aligned_previous[:,:2]+aligned_previous[:,2:])/2
        velocity[valid] = (last_c[valid]-prev_c[valid])/(mt[valid]-pt[valid])[:,None]
    # Prediction starts in the memory coordinate system; map the predicted
    # center+box to query coordinates with the same supplied camera warp.
    predicted = m.copy()
    shift = velocity * dt[:,None]
    predicted[:,:2] += shift
    predicted[:,2:] += shift
    if have_camera:
        predicted = warp_aligned(predicted,memory_to_query_homography,mt)
    predicted_c = (predicted[:,:2]+predicted[:,2:])/2
    innovation = (qc[:,None,:]-predicted_c[None,:,:])/scale
    lo = np.maximum(q[:,None,:2],predicted[None,:,:2])
    hi = np.minimum(q[:,None,2:],predicted[None,:,2:])
    area = np.prod(np.maximum(hi-lo,0.),axis=-1)
    iou = area/(np.prod(qw,1)[:,None]+np.prod(predicted[:,2:]-predicted[:,:2],1)[None,:]-area)
    innovation[:,~valid,:] = np.nan
    return dict(frame_gap=dt, normalized_displacement=displacement,
                log_size_change=log_size_change, normalized_CV_innovation=innovation,
                predicted_box_iou=iou, velocity_history_available=valid,
                prediction_type=np.where(valid,'constant_velocity','static_reference'),
                time_unit='original_frames', camera_warp_supplied=have_camera)


def self_test():
    q=np.array([[30,20,40,30],[50,40,70,60]],float)
    m=np.array([[10,10,20,20],[20,20,40,40]],float)
    p=m-np.array([5,0,5,0])
    a=pair_geometry(q,m,40,[10,10],p,[0,0])
    b=pair_geometry(q+[1000,2000,1000,2000],m+[1000,2000,1000,2000],40,[10,10],p+[1000,2000,1000,2000],[0,0])
    for key in ('normalized_displacement','log_size_change','normalized_CV_innovation','predicted_box_iou'):
        np.testing.assert_allclose(a[key],b[key],atol=1e-12)
    c=pair_geometry(q*[3,2,3,2],m*[3,2,3,2],40,[10,10],p*[3,2,3,2],[0,0])
    for key in ('normalized_displacement','log_size_change','normalized_CV_innovation','predicted_box_iou'):
        np.testing.assert_allclose(a[key],c[key],atol=1e-12)
    cv=pair_geometry([40,0,50,10],[10,0,20,10],40,[10],[[0,0,10,10]],[0])
    np.testing.assert_allclose(cv['normalized_CV_innovation'],0.,atol=1e-12)
    missing=pair_geometry(q,m,40,[10,10])
    assert not missing['velocity_history_available'].any() and np.isnan(missing['normalized_CV_innovation']).all()
    camera=np.array([[1,0,20],[0,1,0],[0,0,1]],float)
    compensated=pair_geometry([30,10,40,20],[10,10,20,20],40,[10],memory_to_query_homography=camera)
    np.testing.assert_allclose(compensated['normalized_displacement'],0.,atol=1e-12)
    np.testing.assert_allclose(warp_boxes(m,camera),warp_boxes(m,-camera),atol=1e-12)
    np.testing.assert_allclose(warp_boxes(m,camera),warp_boxes(m,camera*1e-12),atol=1e-12)
    try:
        warp_boxes(m,[[1,0,0],[0,1,0],[1,1,0]])
    except ValueError:
        pass
    else:
        raise AssertionError('Accepted rank-deficient camera homography')
    try:
        pair_geometry(q,m,40,[10,20],memory_to_query_homography=camera)
    except ValueError:
        pass
    else:
        raise AssertionError('Accepted shared warp across different reference frames')
    aligned=pair_geometry(q,m,40,[10,20],memory_to_query_homography=np.stack((camera,camera)))
    assert aligned['camera_warp_supplied'] is True
    try:
        pair_geometry([30,10,40,20],[10,10,20,20],40,[10],[[0,10,10,20]],[0],camera)
    except ValueError:
        pass
    else:
        raise AssertionError('Accepted uncompensated historical velocity')
    return dict(status='PASS_RELATIVE_GEOMETRY_SYNTHETIC',checks=11,
                real_cache_opened=False,trained=False,method_efficacy_established=False)


if __name__ == '__main__':
    import json
    print(json.dumps(self_test()))
