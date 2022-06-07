#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Anchor.
"""

from __future__ import annotations

from one.core import console

__all__ = [
    "check_anchor_order",
]


# MARK: - Functional

def check_anchor_order(m):
    """Check anchor order against stride order for YOLO Detect() module m,
    and correct if necessary.
    """
    a  = m.anchor_grid.prod(-1).view(-1)  # anchor area
    da = a[-1] - a[0]                # delta a
    ds = m.stride[-1] - m.stride[0]  # delta s
    if da.sign() != ds.sign():  # same order
        console.log("Reversing anchor order")
        m.anchors[:]     = m.anchors.flip(0)
        m.anchor_grid[:] = m.anchor_grid.flip(0)


'''
def check_anchors(dataset, model, thr: float = 4.0, image_size: Int3T = 640):
    """Check anchor fit to data, recompute if necessary."""
    console.log("\nAnalyzing anchors... ", end="")
    m      = model.module.model[-1] if hasattr(model, "module") else model.model[-1]   # Detect()
    shapes = image_size * dataset.shapes / dataset.shapes.max(1, keep_dims=True)
    scale  = np.random.uniform(0.9, 1.1, size=(shapes.shape[0], 1))  # Augment scale
    wh     = torch.tensor(
        np.concatenate(
            [ll[:, 3:5] * s for s, ll in zip(shapes * scale, dataset.target)]
        )
    ).float()  # wh

    def metric(k):  # compute metric
        r    = wh[:, None] / k[None]
        x    = torch.min(r, 1.0 / r).min(2)[0]        # ratio metric
        best = x.max(1)[0]                            # best_x
        aat  = (x > 1.0 / thr).float().sum(1).mean()  # anchors above threshold
        bpr  = (best > 1.0 / thr).float().mean()      # best possible recall
        return bpr, aat

    bpr, aat = metric(m.anchor_grid.clone().cpu().view(-1, 2))
    console.log("anchors/target = %.2f, Best Possible Recall (BPR) = %.4f" %
                (aat, bpr), end="")
    if bpr < 0.98:  # threshold to recompute
        console.log(". Attempting to generate improved anchors, please wait...")
        na          = m.anchor_grid.numel() // 2  # number of anchors
        new_anchors = kmean_anchors(dataset, n=na, image_size=image_size, thr=thr, gen=1000, verbose=False)
        new_bpr = metric(new_anchors.reshape(-1, 2))[0]
        if new_bpr > bpr:  # replace anchors
            new_anchors      = torch.tensor(new_anchors, device=m.anchors.device).type_as(m.anchors)
            m.anchor_grid[:] = new_anchors.clone().view_as(m.anchor_grid)  # For inference
            # loss
            m.anchors[:] = (new_anchors.clone().view_as(m.anchors) /
                            m.stride.to(m.anchors.device).view(-1, 1, 1))
            check_anchor_order(m)
            console.log("New anchors saved to model. Update model *.yaml to "
                        "use these anchors in the future.")
        else:
            console.log("Original anchors better than new anchors. Proceeding "
                        "with original anchors.")
    console.log("")  # newline
'''
