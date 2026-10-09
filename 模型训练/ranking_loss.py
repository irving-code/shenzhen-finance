import math

import torch
import torch.nn.functional as F


def compute_joint_loss(pred, target, dates, rank_weight, beta=0.02, tau=0.02):
    if pred.ndim != 1 or target.ndim != 1 or dates.ndim != 1:
        raise ValueError("pred、target 和 dates 必须是一维数据")
    if pred.numel() == 0 or pred.numel() != target.numel() or pred.numel() != dates.numel():
        raise ValueError("预测、标签和日期数量必须相同且非空")
    if pred.numel() > 512:
        raise ValueError("同日训练批次不能超过 512 条记录")
    if not torch.isfinite(pred).all() or not torch.isfinite(target).all():
        raise ValueError("预测值和标签必须全部有限")
    if not torch.all(dates == dates[0]):
        raise ValueError("排序损失只接受同一交易日的批次")
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta 必须是有限正数")
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("tau 必须是有限正数")
    if not math.isfinite(rank_weight) or rank_weight < 0:
        raise ValueError("rank_weight 必须是有限非负数")

    loss_return = F.smooth_l1_loss(pred, target, beta=beta, reduction="mean")
    if rank_weight == 0:
        loss_rank = pred.sum() * 0.0
        valid_pair_count = 0
    else:
        pair_indices = torch.triu_indices(
            pred.numel(), pred.numel(), offset=1, device=pred.device
        )
        left, right = pair_indices[0], pair_indices[1]
        target_delta = target[left] - target[right]
        valid = target_delta != 0
        valid_pair_count = int(valid.sum().item())
        if valid_pair_count:
            direction = target_delta[valid].sign()
            prediction_delta = pred[left[valid]] - pred[right[valid]]
            margins = direction * prediction_delta
            loss_rank = (tau * F.softplus(-margins / tau)).mean()
        else:
            loss_rank = pred.sum() * 0.0

    weighted_rank_loss = rank_weight * loss_rank
    loss_total = loss_return + weighted_rank_loss
    return {
        "loss_total": loss_total,
        "loss_return": loss_return,
        "loss_rank": loss_rank,
        "weighted_rank_loss": weighted_rank_loss,
        "valid_pair_count": valid_pair_count,
    }
