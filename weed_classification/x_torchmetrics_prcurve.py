#! /usr/bin/env python

"""
script to test torchmetrics to generate a pr curve
"""

import torch
import torchmetrics

from torchmetrics import PrecisionRecallCurve

pred = torch.tensor([0, 1, 2, 3])
target = torch.tensor([0, 1, 1, 0])
pr_curve = PrecisionRecallCurve(pos_label=1)
prec, reca, thresh = pr_curve(pred, target)

print(prec)

print(reca)

print(thresh)

pred = torch.tensor([[0.75, 0.05, 0.05, 0.05, 0.05],
                     [0.05, 0.75, 0.05, 0.05, 0.05],
                     [0.05, 0.05, 0.75, 0.05, 0.05],
                     [0.05, 0.05, 0.05, 0.75, 0.05]])
target = torch.tensor([0, 1, 3, 2])
pr_curve = PrecisionRecallCurve(num_classes=5)
p, r, t = pr_curve(pred, target)
p
r
t

