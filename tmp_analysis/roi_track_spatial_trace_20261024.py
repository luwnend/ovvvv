"""Read-only runtime hooks for the active track head, never the shared extractor."""
import argparse
import hashlib
import json
from pathlib import Path

import torch


class ROITrackSpatialTrace:
    def __init__(self, head):
        if head.num_convs < 1 or head.num_fcs < 1:
            raise ValueError('Expected spatial conv branch followed by FC branch')
        self.head = head
        self.events = None
        self.handles = [head.register_forward_pre_hook(self._input),
                        head.convs[-1].register_forward_hook(self._spatial),
                        head.register_forward_hook(self._output)]

    def begin(self):
        if self.events is not None:
            raise RuntimeError('Previous frame was not consumed')
        self.events = {'input': [], 'spatial': [], 'output': []}

    def _record(self, name, tensor):
        if self.events is None:
            raise RuntimeError('Track head ran outside an explicit frame')
        self.events[name].append(tensor.detach().clone())

    def _input(self, module, arguments):
        if len(arguments) != 1:
            raise RuntimeError('Unexpected track-head call signature')
        self._record('input', arguments[0])

    def _spatial(self, module, arguments, output):
        self._record('spatial', output)

    def _output(self, module, arguments, output):
        self._record('output', output)

    def take(self, keep_mask):
        events, self.events = self.events, None
        if events is None or any(len(events[k]) != 1 for k in events):
            raise RuntimeError('Pilot requires exactly one actual track-head call per frame')
        before, spatial, raw = (events[k][0] for k in ('input', 'spatial', 'output'))
        if before.ndim != 4 or before.shape != spatial.shape or len(raw) != len(before):
            raise RuntimeError('Spatial/track row correspondence changed')
        keep = torch.as_tensor(keep_mask, device=raw.device, dtype=torch.bool)
        if keep.shape != (len(raw),):
            raise RuntimeError('Tracker keep-mask must index detection rows')
        # Replay only the FC branch, so the track-head hooks cannot recurse.
        with torch.no_grad():
            replay = spatial.reshape(len(spatial), -1)
            for fc in self.head.fcs:
                replay = self.head.relu(fc(replay))
            replay = self.head.fc_embed(replay)
        if not torch.equal(raw, replay):
            raise RuntimeError('FC replay is not exactly the actual raw head output')
        rows = torch.nonzero(keep, as_tuple=False).flatten()
        return dict(roi_input=before[keep], conv_spatial=spatial[keep],
                    raw_output=raw[keep], pretracker_det_row=rows,
                    full_raw_output=raw, native_FC_replay_exact=True)

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.events = None


def synthetic_round(number):
    torch.manual_seed(20261024 + number)
    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.num_convs, self.num_fcs = 1, 1
            self.convs = torch.nn.ModuleList([torch.nn.Conv2d(3, 3, 1)])
            self.fcs = torch.nn.ModuleList([torch.nn.Linear(12, 7)])
            self.relu = torch.nn.ReLU(inplace=True)
            self.fc_embed = torch.nn.Linear(7, 5)
        def forward(self, x):
            x = self.convs[0](x).reshape(len(x), -1)
            return self.fc_embed(self.relu(self.fcs[0](x)))
    head = Head().eval()
    x = torch.randn(4, 3, 2, 2)
    with torch.no_grad():
        expected = head(x).clone()
        trace = ROITrackSpatialTrace(head)
        trace.begin(); actual = head(x); data = trace.take([True, False, True, False])
        assert torch.equal(actual, expected)
        assert data['pretracker_det_row'].tolist() == [0, 2]
        assert torch.equal(data['raw_output'], expected[[0, 2]])
        assert torch.equal(data['roi_input'], x[[0, 2]])
        # Shared-extractor calls are irrelevant: this traces the track head.
        trace.begin(); head(x); head(x)
        try:
            trace.take([True] * 4)
        except RuntimeError:
            rejected = True
        else:
            raise AssertionError('Multiple track calls silently accepted')
        trace.close()
        assert torch.equal(head(x), expected)
    return dict(status='PASS_ROI_TRACK_TRACE_SYNTHETIC', round=number,
                checks=['native_FC_replay_and_retained_detection_row_alignment',
                        'multiple_head_calls_rejected_and_hooks_do_not_change_output'],
                module_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                actual_GPU_or_real_cache_used=False, training=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--round', type=int, choices=(1, 2, 3), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    value = synthetic_round(args.round)
    with args.output.open('x') as stream:
        json.dump(value, stream, indent=2); stream.write('\n')
    print(json.dumps(value))
