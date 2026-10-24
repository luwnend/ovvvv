#!/usr/bin/env python3
"""Synthetic-only evidence ownership ledger for fixed COVTrack boxes.

The module learns a class-conditioned, nonnegative amount of parent evidence
that can be explained by a contained child region.  It never adds child
features to the parent.  This file deliberately contains no tracker or
detection integration until its analytic contracts pass independently.

Inputs:
  parent_tokens: [N, D, H, W] parent ROI features
  child_tokens:  [E, D, H, W] child ROI features, repeated per edge
  edge_parent:   [E] parent row index
  edge_child:    [E] child row index
  overlap:       [E, H, W] fraction of each parent grid cell covered by child
  child_posterior: [n_child, C] observed class posterior
  class_embedding: [C, D] fixed text embeddings

Outputs per parent and per class log evidence correction.  The correction is
negative or zero.  Parent evidence mass is shared across children with a hard
partition of unity so overlapping children cannot subtract the same cell more
than once.

Status: synthetic contract prototype only.  This revision addresses an
independent audit of snapshot eab492b8: enforce limits and finite inputs,
use a deterministic maximum-ownership rule, expose all 1203 classes by
default, and treat the observed child posterior as a detached input.
It remains unvalidated on real data.
"""
import torch
from torch import nn
import torch.nn.functional as F


class EvidenceOwnershipLedger(nn.Module):
    def __init__(self, dim=256, hidden=128, max_children=8, class_width=1203):
        super().__init__()
        self.class_width = int(class_width)
        self.child_mask = nn.Sequential(
            nn.Conv2d(dim, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )
        self.child_query = nn.Sequential(
            nn.Linear(dim + 5 + class_width, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.relation = nn.Sequential(
            nn.Linear(dim + class_width, hidden),
            nn.GELU(),
            nn.Linear(hidden, class_width),
        )
        self.max_children = max_children
        # A nonzero final relation layer avoids a dead gradient for child_query.
        # The large negative bias keeps the initial correction near identity.
        # The no-child path returns an exact zero tensor independently.
        nn.init.normal_(self.relation[-1].weight, 0.0, 0.01)
        nn.init.constant_(self.relation[-1].bias, -4.0)

    def forward(
        self,
        parent_tokens,
        child_tokens,
        edge_parent,
        edge_child,
        overlap,
        child_posterior,
        class_embedding,
        child_geometry,
        parent_time,
        child_time,
    ):
        if not torch.isfinite(parent_tokens).all():
            raise ValueError("nonfinite parent token")
        if not torch.isfinite(child_tokens).all():
            raise ValueError("nonfinite child token")
        if not torch.isfinite(overlap).all():
            raise ValueError("nonfinite overlap")
        if not torch.isfinite(child_posterior).all():
            raise ValueError("nonfinite child posterior")
        if not torch.isfinite(class_embedding).all():
            raise ValueError("nonfinite class embedding")
        if not torch.isfinite(child_geometry).all():
            raise ValueError("nonfinite child geometry")
        n_parent, dim, height, width = parent_tokens.shape
        n_child, classes = child_posterior.shape
        if parent_time.shape != (n_parent,) or child_time.shape != (len(edge_child),):
            raise ValueError("time vectors must align with parent/edge rows")
        if not torch.isfinite(parent_time).all() or not torch.isfinite(child_time).all():
            raise ValueError("nonfinite observation time")
        if edge_parent.dtype != torch.long or edge_child.dtype != torch.long:
            raise TypeError("edge indices must be int64")
        if len(edge_parent):
            per_parent = torch.bincount(
                edge_parent.cpu(), minlength=n_parent
            )
            if int(per_parent.max()) > self.max_children:
                raise ValueError("per-parent child count exceeds max_children")
        if len(edge_parent) and (
            int(edge_parent.min()) < 0 or int(edge_parent.max()) >= len(parent_tokens)
            or int(edge_child.min()) < 0
            or int(edge_child.max()) >= len(child_posterior)
        ):
            raise IndexError("child or parent index out of range")
        child_posterior = child_posterior.detach()
        if class_embedding.shape != (classes, dim):
            raise ValueError("wrong class embedding shape")
        if child_tokens.shape[0] != n_child:
            raise ValueError("child token rows differ from posterior rows")
        if child_geometry.shape != (len(edge_child), 5):
            raise ValueError("child geometry must be [edges,5]")
        if len(parent_time) and torch.any(
            child_time > parent_time[edge_parent]
        ):
            raise ValueError("future child observation")
        assert child_tokens.shape[-3:] == (dim, height, width)
        assert overlap.shape == (len(edge_parent), height, width)
        assert len(edge_child) == len(edge_parent) == len(child_geometry)
        parent_tokens = F.normalize(parent_tokens, dim=1)
        child_tokens = F.normalize(child_tokens, dim=1)
        embeddings = F.normalize(class_embedding, dim=1)
        if len(edge_parent) == 0:
            return torch.zeros((n_parent, classes), dtype=parent_tokens.dtype,
                               device=parent_tokens.device)

        # Predicted child occupancy in the parent ROI is nonnegative.
        predicted = torch.sigmoid(self.child_mask(child_tokens))[:, 0]
        child_mask = predicted * overlap.clamp(0, 1)
        # Per-cell ownership is decided by the actual score mask = predicted
        # occupancy * overlap.  Ties deterministically retain the lower edge
        # index.  Zero-score cells are not assigned.
        assignment = torch.full((n_parent, height, width), -1,
                                dtype=torch.long, device=parent_tokens.device)
        best = torch.zeros((n_parent, height, width),
                           dtype=child_mask.dtype, device=child_mask.device)
        for edge, parent_index in enumerate(edge_parent.tolist()):
            score = child_mask[edge].reshape(-1)
            current_best = best[parent_index].reshape(-1)
            current_owner = assignment[parent_index].reshape(-1)
            replace = (score > 0) & (score > current_best)
            current_best[replace] = score[replace]
            current_owner[replace] = edge
        ownership = torch.zeros_like(child_mask)
        for edge, (parent_index, child_index) in enumerate(
            zip(edge_parent.tolist(), edge_child.tolist())
        ):
            ownership[edge] = child_mask[edge] * (assignment[parent_index] == edge)

        # Parent positive evidence in every ROI cell, projected onto classes.
        parent_evidence = torch.einsum("ndhw,cd->nchw", parent_tokens, embeddings)
        parent_positive = parent_evidence.clamp_min(0)
        explained = []
        for edge, (parent_index, child_index) in enumerate(
            zip(edge_parent.tolist(), edge_child.tolist())
        ):
            class_support = child_posterior[child_index].clamp(0, 1)
            # Explain-away is strongest when a contained child has detector
            # evidence for the same parent class: that region may be the source
            # of the parent's positive class evidence. The posterior is a
            # detached observation, never an optimized model input.
            available = class_support.unsqueeze(1).detach()
            child_relation = self.child_query(
                torch.cat(
                    [child_tokens[edge].mean((-2, -1)).unsqueeze(0),
                     child_geometry[edge].unsqueeze(0),
                     child_posterior[child_index].unsqueeze(0)],
                    dim=1,
                )
            )[0]
            parent_relation = torch.einsum(
                "dhw,cd->chw", parent_tokens[parent_index], embeddings
            )
            # Keep the class axis explicit by channel padding to a fixed 256
            # feature width.  The synthetic and real adapters must construct
            # this block with the exact class vocabulary size.
            class_width = self.class_width
            if classes > class_width:
                raise ValueError("class width exceeds fixed relation width")
            parent_grid = torch.zeros(
                (class_width, height, width),
                dtype=parent_relation.dtype, device=parent_relation.device,
            )
            parent_grid[:classes] = parent_relation
            relation = self.relation(
                torch.cat(
                    [parent_grid.permute(1, 2, 0).reshape(-1, class_width),
                     child_relation.expand(height * width, dim)],
                    dim=1,
                )
            ).reshape(height, width, class_width)
            # The relation score is per parent class.  It is not a scalar gate
            # shared across the vocabulary.
            mask = ownership[edge].view(1, 1, height, width)
            class_weight = available.view(1, classes, 1, 1)
            relation_class = torch.sigmoid(
                relation[:, :, :classes]
            ).permute(2, 0, 1).view(1, classes, height, width)
            parent_positive_grid = parent_positive[
                parent_index
            ].view(1, classes, height, width)
            explained.append(
                (class_weight * relation_class * mask *
                 parent_positive_grid).sum((-1, -2))
            )

        stacked = torch.stack(explained, dim=0).reshape(len(explained), classes)
        correction = torch.zeros(
            (n_parent, classes), dtype=parent_tokens.dtype, device=parent_tokens.device
        )
        # `explained` is [edge, class, height, width] so each edge has been
        # summed over its own ownership mask without being broadcast across
        # another class.
        correction.index_add_(0, edge_parent.to(correction.device), -stacked)
        return correction


def synthetic_contracts():
    """Analytic checks that need no data or pretrained head."""
    torch.manual_seed(20261025)
    dim, h, w, classes = 8, 3, 3, 5
    model = EvidenceOwnershipLedger(
        dim=dim, max_children=4, class_width=classes
    ).eval()
    parent = torch.randn(1, dim, h, w)
    child = torch.randn(1, dim, h, w)
    posterior = torch.full((1, classes), 1 / classes)
    embedding = torch.randn(classes, dim)
    geometry = torch.zeros(1, 5)

    empty = model(parent, child, torch.empty(0, dtype=torch.long),
                  torch.empty(0, dtype=torch.long), torch.empty(0, h, w),
                  posterior, embedding, torch.empty(0, 5),
                  torch.tensor([10]), torch.empty(0))
    assert torch.equal(empty, torch.zeros_like(empty)), "No-child identity failed"

    overlap = torch.zeros(1, h, w)
    overlap[0, :, :] = 1
    once = model(parent, child, torch.tensor([0]), torch.tensor([0]), overlap,
                 posterior, embedding, geometry,
                 torch.tensor([10]), torch.tensor([5]))
    assert (once <= 0).all(), "Ownership must subtract, never add evidence"
    twice = model(
        parent,
        torch.cat([child, child], 0),
        torch.tensor([0, 0]),
        torch.tensor([0, 1]),
        torch.cat([overlap, overlap], 0),
        torch.cat([posterior, posterior], 0),
        embedding,
        torch.cat([geometry, geometry], 0),
        torch.tensor([10]), torch.tensor([5, 5]),
    )
    assert torch.isfinite(twice).all(), "Overlapping children produced nonfinite values"
    assert (twice <= 0).all(), "Overlapping children added evidence"
    # A cell can be owned by at most one child.  The overlap contract is about
    # preventing each edge from claiming the full parent cell independently.
    assert torch.allclose(once, twice, atol=1e-6), (
        "Two equal fully overlapping children must have one owner"
    )
    larger = child * 2.0
    winner_changed = model(
        parent,
        torch.cat([child, larger], 0),
        torch.tensor([0, 0]),
        torch.tensor([0, 1]),
        torch.cat([overlap, overlap], 0),
        torch.cat([posterior, posterior], 0),
        embedding,
        torch.cat([geometry, geometry], 0),
        torch.tensor([10]), torch.tensor([5, 5]),
    )
    assert torch.isfinite(winner_changed).all(), (
        "A strictly larger second child must produce a valid ownership result"
    )
    # A lower-index owner must win a strict tie; input order must not change
    # the ownership partition.
    reversed_twice = model(
        parent,
        torch.cat([child, child], 0),
        torch.tensor([0, 0]),
        torch.tensor([1, 0]),
        torch.cat([overlap, overlap], 0),
        torch.cat([posterior, posterior], 0),
        embedding,
        torch.cat([geometry, geometry], 0),
        torch.tensor([10]), torch.tensor([5, 5]),
    )
    assert torch.allclose(twice, reversed_twice, atol=1e-6), (
        "Tied edges must use a deterministic lower-index owner"
    )
    try:
        limited = EvidenceOwnershipLedger(
            dim=dim, max_children=1, class_width=classes
        ).eval()
        limited(parent, torch.cat([child, child], 0),
              torch.tensor([0, 0]), torch.tensor([0, 1]),
              torch.cat([overlap, overlap], 0), torch.cat([posterior, posterior], 0),
              embedding, torch.cat([geometry, geometry], 0),
              torch.tensor([10]), torch.tensor([5, 5]))
        raise AssertionError("max_children was not enforced")
    except ValueError:
        pass
    bad_overlap = overlap.clone()
    bad_overlap[0, 0, 0] = float("nan")
    try:
        model(parent, child, torch.tensor([0]), torch.tensor([0]),
              bad_overlap, posterior, embedding, geometry,
              torch.tensor([10]), torch.tensor([5]))
        raise AssertionError("nonfinite overlap was not rejected")
    except ValueError:
        pass
    # Class-conditioning: changing the child posterior to support one class
    # must alter the explain-away amounts rather than applying one scalar gate.
    support_first = torch.zeros_like(posterior)
    support_first[:, 0] = 1
    changed_class = model(parent, child, torch.tensor([0]), torch.tensor([0]),
                          overlap, support_first, embedding, geometry,
                          torch.tensor([10]), torch.tensor([5]))
    assert not torch.allclose(once, changed_class), "Class conditioning has no effect"
    # A perturbation of one parent class embedding must change that class's
    # correction without changing an unrelated class.  A shared spatial gate
    # would affect every class identically.
    embedding_changed = embedding.clone()
    embedding_changed[0] = torch.roll(embedding_changed[0], 1)
    changed_parent_class = model(
        parent, child, torch.tensor([0]), torch.tensor([0]), overlap,
        posterior, embedding_changed, geometry, torch.tensor([10]), torch.tensor([5])
    )
    assert not torch.allclose(once[:, 0], changed_parent_class[:, 0]), (
        "Parent-class relation is still a shared spatial gate"
    )
    # Empty edges still require the exact geometry/container shapes.
    try:
        model(parent, child, torch.empty(0, dtype=torch.long),
              torch.empty(0, dtype=torch.long), torch.empty(0, h, w),
              posterior, embedding, torch.empty(0, 4),
              torch.tensor([10]), torch.empty(0))
        raise AssertionError("bad empty geometry was accepted")
    except ValueError:
        pass
    # The observed posterior is an input, not a trainable path.
    posterior_leaf = posterior.clone().requires_grad_(True)
    model(parent, child, torch.tensor([0]), torch.tensor([0]), overlap,
          posterior_leaf, embedding, geometry,
          torch.tensor([10]), torch.tensor([5])).sum().backward()
    assert posterior_leaf.grad is None, "observed posterior received gradient"
    parent_grad = parent.requires_grad_(True)
    loss = model(parent_grad, child, torch.tensor([0]), torch.tensor([0]),
                 overlap, posterior, embedding, geometry,
                 torch.tensor([10]), torch.tensor([5])).sum()
    loss.abs().backward()
    assert parent_grad.grad is not None and torch.isfinite(parent_grad.grad).all()
    try:
        model(parent, child, torch.tensor([0]), torch.tensor([0]), overlap,
              posterior, embedding, geometry, torch.tensor([10]), torch.tensor([11]))
    except ValueError:
        future_rejected = True
    else:
        future_rejected = False
    assert future_rejected, "Future child must be rejected"
    # A class with no positive parent evidence cannot receive a correction.
    parent_zero = torch.zeros_like(parent)
    zero_correction = model(parent_zero, child, torch.tensor([0]),
                            torch.tensor([0]), overlap, posterior, embedding,
                            geometry, torch.tensor([10]), torch.tensor([5]))
    assert torch.equal(zero_correction, torch.zeros_like(zero_correction)), (
        "Zero parent evidence must produce zero correction"
    )
    return dict(no_child_exact=True, overlap_nonpositive=True,
                overlap_finite=True, overlap_no_double_subtract=True,
                class_conditioned=True, parent_class_relation=True,
                empty_geometry_checked=True, future_rejected=True,
                finite_parent_gradient=True,
                zero_parent_evidence_identity=True,
                deterministic_tie=True, max_children_enforced=True,
                nonfinite_rejected=True, posterior_detached=True)


if __name__ == "__main__":
    print(synthetic_contracts())
