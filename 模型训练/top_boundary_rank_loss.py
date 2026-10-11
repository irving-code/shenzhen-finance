import math

import torch
import torch.nn.functional as F


def compute_top_boundary_loss(pred, target, dates, top_flags, boundary_valid, tau):
    if pred.ndim != 1 or target.ndim != 1 or dates.ndim != 1:
        raise ValueError("预测、标签和日期必须是一维数据")
    if top_flags.ndim != 1 or boundary_valid.ndim != 1:
        raise ValueError("Top 标记和边界资格必须是一维数据")
    if not (pred.numel() == target.numel() == dates.numel() == top_flags.numel() == boundary_valid.numel()):
        raise ValueError("边界排序输入数量不一致")
    if pred.numel() == 0 or pred.numel() > 512:
        raise ValueError("同日边界排序批次大小异常")
    if not torch.isfinite(pred).all() or not torch.isfinite(target).all():
        raise ValueError("边界排序输入包含非有限值")
    if not torch.all(dates == dates[0]):
        raise ValueError("边界排序只接受同一交易日的批次")
    if top_flags.dtype != torch.bool or boundary_valid.dtype != torch.bool:
        raise ValueError("Top 标记和边界资格必须为布尔类型")
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("tau 必须是有限正数")

    top_indices = torch.where(top_flags & boundary_valid)[0]
    rest_indices = torch.where(boundary_valid & ~top_flags)[0]
    pair_count = int(top_indices.numel() * rest_indices.numel())
    if pair_count == 0:
        loss = pred.sum() * 0.0
    else:
        prediction_delta = (
            pred[top_indices].unsqueeze(1) - pred[rest_indices].unsqueeze(0)
        ).reshape(-1)
        loss = (tau * F.softplus(-prediction_delta / tau)).mean()
    return {
        "loss_top_boundary": loss,
        "top_pair_count": pair_count,
        "top_count": int((top_flags & boundary_valid).sum().item()),
        "rest_count": int((boundary_valid & ~top_flags).sum().item()),
    }
