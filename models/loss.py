from typing import Any
from piq import LPIPS
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure, MultiScaleStructuralSimilarityIndexMeasure
import torch
import torch.nn as nn


class LPIPSLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.lpips = LPIPS(reduction='none')
        self.lpips.requires_grad_(False)

    def forward(self, pred, target, weight=1):
        return (self.lpips(F.interpolate((pred + 1) / 2, size=224, mode='bilinear'),
                           F.interpolate((target.clip(-1, 1) + 1) / 2, size=224, mode='bilinear')) * weight)


def L2Loss(pred, target):
    return (pred - target).flatten(1).norm(dim=-1, p=2).mean()

def L2Loss_ph(pred, target):
    return ((((pred - target).flatten(1)**2).sum(-1) + (0.00054*32) ** 2)**0.5 - 0.00054*32)

def L1Loss(pred, target):
    return (pred - target).flatten(1).norm(dim=-1, p=1).mean()


class SSIMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ssim = StructuralSimilarityIndexMeasure()
        if torch.cuda.is_available():
            self.ssim = self.ssim.cuda()

    def forward(self, pred, target):
        return (1 - self.ssim(F.interpolate((pred + 1) / 2, size=224, mode='bilinear'),
                              F.interpolate((target.clip(-1, 1) + 1) / 2, size=224, mode='bilinear')).mean()) / 2


class MSSSIMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.msssim = MultiScaleStructuralSimilarityIndexMeasure()
        if torch.cuda.is_available():
            self.msssim = self.msssim.cuda()

    def forward(self, pred, target):
        return (1 - self.msssim(F.interpolate((pred + 1) / 2, size=224, mode='bilinear'),
                                F.interpolate((target.clip(-1, 1) + 1) / 2, size=224, mode='bilinear')).mean()) * 0.84 + \
            0.16 * (pred - target).flatten(1).norm(dim=-1, p=1).mean()


def get_distance_loss(loss_type, compile=False):
    if loss_type == "l1":
        metric = L1Loss
    elif loss_type == "l2":
        metric = L2Loss
    elif loss_type == "l2_ph":
        metric = L2Loss_ph
    elif loss_type == "ssim":
        metric = SSIMLoss()
    elif loss_type == "msssim":
        metric = MSSSIMLoss()
    elif loss_type == "lpips":
        metric = LPIPSLoss()
    if compile:
        return torch.compile(metric)
    return metric
