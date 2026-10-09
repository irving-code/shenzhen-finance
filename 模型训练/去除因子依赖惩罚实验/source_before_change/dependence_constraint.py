import numpy as np
import torch

from cross_section_experiment import project_path, read_json


def source_feature_indices(config):
    base_metadata = read_json(project_path(config, "base_metadata"))
    feature_names = base_metadata["feature_columns"] + config["features"]["extra_columns"]
    groups = []
    seen = set()
    for names in config["dependence"]["source_groups"]:
        indices = [feature_names.index(name) for name in names]
        if len(indices) != len(set(indices)) or seen.intersection(indices):
            raise ValueError("因子来源组存在重复通道")
        groups.append(indices)
        seen.update(indices)
    if len(groups) != 7 or any(index < 0 or index >= 37 for group in groups for index in group):
        raise ValueError("因子来源组与 37 通道输入不一致")
    return groups


def compute_neutral_values(config, reader, frame, chunk_size=8192):
    if chunk_size <= 0:
        raise ValueError("中性值计算批次必须为正数")
    positions = frame.index[frame["training_eligible"].eq(1)].to_numpy(dtype=np.int64)
    if len(positions) != config["data"]["training_eligible"]:
        raise ValueError("中性值计算的训练资格数量不一致")
    groups = source_feature_indices(config)
    indices = np.asarray(sorted({index for group in groups for index in group}), dtype=np.int64)
    values = np.empty((len(positions), len(indices)), dtype=np.float32)
    for start in range(0, len(positions), chunk_size):
        stop = min(start + chunk_size, len(positions))
        batch = reader.read_batch(positions[start:stop])
        values[start:stop] = batch["x"][:, -1, indices]
    if not np.isfinite(values).all():
        raise ValueError("训练期因子中性值样本包含非有限值")
    medians = np.median(values, axis=0).astype(np.float32)
    neutral = np.zeros(37, dtype=np.float32)
    neutral[indices] = medians
    if not np.isfinite(neutral).all():
        raise ValueError("因子中性值包含非有限值")
    return neutral


def replace_source_group(inputs, group_indices, neutral_values):
    if inputs.ndim != 3 or inputs.shape[1:] != (20, 37):
        raise ValueError("依赖约束输入必须具有 (batch,20,37) 形状")
    indices = torch.as_tensor(group_indices, dtype=torch.long, device=inputs.device)
    neutral = torch.as_tensor(neutral_values, dtype=inputs.dtype, device=inputs.device)
    if neutral.shape != (37,) or not torch.isfinite(neutral).all():
        raise ValueError("因子中性值必须是 37 个有限数值")
    if indices.numel() == 0 or torch.any(indices < 0) or torch.any(indices >= 37):
        raise ValueError("因子来源组索引无效")
    replaced = inputs.clone()
    replaced[:, :, indices] = neutral[indices].view(1, 1, -1)
    return replaced


def dependence_penalty(predictions, replaced_predictions, limit, weight, denominator_floor):
    if predictions.ndim != 1 or replaced_predictions.shape != predictions.shape or predictions.numel() < 2:
        raise ValueError("依赖程度计算需要数量相同且至少两条的一维预测")
    if not torch.isfinite(predictions).all() or not torch.isfinite(replaced_predictions).all():
        raise ValueError("依赖程度预测包含非有限值")
    if not 0.0 < limit < 1.0 or weight < 0.0 or denominator_floor <= 0.0:
        raise ValueError("依赖程度参数超出有效范围")
    original_std = predictions.std(unbiased=False)
    change_std = (predictions - replaced_predictions).std(unbiased=False)
    ratio = change_std / original_std.clamp_min(denominator_floor)
    penalty = weight * torch.relu(ratio - limit).square()
    if not torch.isfinite(ratio) or not torch.isfinite(penalty):
        raise ValueError("依赖程度或惩罚项非有限")
    return ratio, penalty
