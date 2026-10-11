import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cross_section_experiment import (
    file_sha256, index_path, load_config, project_path, read_json,
    source_hashes, temporary_path, validate_completed_input, write_json,
)
from cross_section_input import CrossSectionInputReader, SameDateBatchSampler
from evaluate_cross_section_rank import model_for_group, run_path
from ranking_loss import compute_joint_loss
from top_boundary_rank_loss import compute_top_boundary_loss
from train_cross_section_rank import checked_batch, set_determinism


def input_hash(input_record):
    return hashlib.sha256(
        json.dumps(input_record, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def training_frame(config):
    frame = pd.read_parquet(index_path(config, "train_fit"), columns=[
        "ts_code", "target_date", "panel_row_id", "training_eligible",
        "y_ret_1d", "flag_limit_up",
    ])
    eligible = frame.loc[frame["training_eligible"].eq(1)]
    if len(eligible) != config["data"]["training_eligible"]:
        raise ValueError("训练资格数量与配置不一致")
    if eligible.duplicated(["ts_code", "target_date"]).any():
        raise ValueError("训练索引存在重复股票日期键")
    if not np.isfinite(eligible["y_ret_1d"].to_numpy(dtype=np.float64)).all():
        raise ValueError("训练标签包含非有限值")
    if not np.isfinite(eligible["flag_limit_up"].to_numpy(dtype=np.float64)).all():
        raise ValueError("训练涨停字段包含非有限值")
    frame["boundary_valid"] = False
    frame["top_flag"] = False
    for _, group in eligible.groupby("target_date", sort=True):
        valid = group.loc[group["flag_limit_up"].eq(0) & group["y_ret_1d"].notna()]
        if valid.empty:
            continue
        top_count = max(len(valid) // 10, 1)
        ordered = valid.sort_values("y_ret_1d", ascending=False, kind="mergesort")
        top_index = ordered.index[:top_count]
        frame.loc[valid.index, "boundary_valid"] = True
        frame.loc[top_index, "top_flag"] = True
    boundary_count = int((frame["boundary_valid"] & frame["training_eligible"].eq(1)).sum())
    top_count = int((frame["top_flag"] & frame["training_eligible"].eq(1)).sum())
    if boundary_count == 0 or top_count == 0:
        raise ValueError("训练 Top 组标签为空")
    return frame


def save_checkpoint(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_path(path)
    torch.save(state, temporary)
    os.replace(temporary, path)


def train(config, config_hash, input_record, run_dir, top_weight, max_epochs):
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    data_hash = input_hash(input_record)
    snapshot_path = run_dir / "配置快照.json"
    snapshot = {
        "config_sha256": config_hash,
        "input_sha256": data_hash,
        "top_boundary_weight": top_weight,
        "global_rank_weight": config["ranking"]["weight"],
        "seed": config["seed"],
        "feature_mode": config["feature_mode"],
        "source_sha256": {
            **source_hashes(config["protocol"]["schema_version"]),
            "top_boundary_rank_loss.py": file_sha256(Path(__file__).with_name("top_boundary_rank_loss.py")),
            "train_top_boundary_rank.py": file_sha256(Path(__file__)),
        },
    }
    if snapshot_path.exists():
        existing = read_json(snapshot_path)
        if existing != snapshot:
            raise ValueError("训练目录已有不一致的配置快照")
    else:
        write_json(snapshot_path, snapshot)
    frame = training_frame(config)
    device = torch.device("cuda")
    seed = config["seed"]
    group = config["model_id"]
    rank_weight = config["ranking"]["weight"]
    set_determinism(seed)
    model = model_for_group(config, group, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    reader = CrossSectionInputReader(
        project_path(config, "base_metadata"),
        project_path(config, "input_dir") / "输入元数据.json",
        index_path(config, "train_fit"), config["feature_mode"],
    )
    directory = run_path(run_dir, group, rank_weight, seed)
    completed_epochs_path = run_dir / "调度状态.json"
    completed_epochs = 0
    if completed_epochs_path.exists():
        status = read_json(completed_epochs_path)
        if status["top_boundary_weight"] != top_weight:
            raise ValueError("已有训练状态的 Top 边界权重不一致")
        completed_epochs = int(status["completed_epochs"])
    for epoch in range(completed_epochs + 1, max_epochs + 1):
        model.train()
        start = time.monotonic()
        totals = {
            "sample_count": 0, "batch_count": 0, "valid_pair_count": 0,
            "top_pair_count": 0, "top_row_count": 0, "rest_row_count": 0,
            "return_sum": 0.0, "rank_sum": 0.0, "top_sum": 0.0,
            "weighted_rank_sum": 0.0, "weighted_top_sum": 0.0,
            "total_sum": 0.0, "grad_norm_sum": 0.0,
        }
        sampler_hash = bytes(32)
        sampler = SameDateBatchSampler(frame, seed, epoch, config["training"]["batch_size"])
        for batch_id, positions in enumerate(sampler):
            batch = checked_batch(reader, frame, positions)
            source = frame.iloc[positions]
            inputs = torch.from_numpy(batch["x"]).to(device=device, dtype=torch.float32)
            targets = torch.from_numpy(batch["y"]).to(device=device, dtype=torch.float32)
            dates = torch.from_numpy(batch["target_date"]).to(device=device, dtype=torch.int64)
            top_flags = torch.from_numpy(source["top_flag"].to_numpy(dtype=bool)).to(device=device)
            boundary_valid = torch.from_numpy(source["boundary_valid"].to_numpy(dtype=bool)).to(device=device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(inputs)
            global_losses = compute_joint_loss(
                predictions, targets, dates, rank_weight,
                config["ranking"]["beta"], config["ranking"]["tau"],
            )
            top_losses = compute_top_boundary_loss(
                predictions, targets, dates, top_flags, boundary_valid,
                config["ranking"]["tau"],
            )
            weighted_top = top_weight * top_losses["loss_top_boundary"]
            loss_total = global_losses["loss_total"] + weighted_top
            loss_total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config["training"]["gradient_clip"],
            )
            if not torch.isfinite(grad_norm):
                raise ValueError("梯度范数非有限")
            optimizer.step()
            with torch.no_grad():
                if not torch.isfinite(torch.nn.utils.parameters_to_vector(model.parameters())).all():
                    raise ValueError("更新后的模型参数非有限")
            size = len(positions)
            totals["sample_count"] += size
            totals["batch_count"] += 1
            totals["valid_pair_count"] += global_losses["valid_pair_count"]
            totals["top_pair_count"] += top_losses["top_pair_count"]
            totals["top_row_count"] += top_losses["top_count"]
            totals["rest_row_count"] += top_losses["rest_count"]
            totals["return_sum"] += float(global_losses["loss_return"].detach()) * size
            totals["rank_sum"] += float(global_losses["loss_rank"].detach())
            totals["top_sum"] += float(top_losses["loss_top_boundary"].detach())
            totals["weighted_rank_sum"] += float(global_losses["weighted_rank_loss"].detach())
            totals["weighted_top_sum"] += float(weighted_top.detach())
            totals["total_sum"] += float(loss_total.detach())
            totals["grad_norm_sum"] += float(grad_norm.detach())
            sampler_hash = hashlib.sha256(
                sampler_hash + positions.tobytes() + batch["target_date"].tobytes()
                + "|".join(batch["ts_code"]).encode("utf-8")
            ).digest()
        if totals["sample_count"] != config["data"]["training_eligible"]:
            raise ValueError("训练周期没有完整覆盖训练资格记录")
        if totals["batch_count"] != config["sampling"]["expected_batches_per_epoch"]:
            raise ValueError("训练批次数量与配置不一致")
        if totals["top_pair_count"] == 0:
            raise ValueError("训练周期没有有效 Top/普通边界配对")
        elapsed = time.monotonic() - start
        record = {
            "epoch": epoch, "top_boundary_weight": top_weight,
            "global_rank_weight": rank_weight, "seed": seed,
            "sample_count": totals["sample_count"], "batch_count": totals["batch_count"],
            "valid_pair_count": totals["valid_pair_count"],
            "top_pair_count": totals["top_pair_count"],
            "top_row_count": totals["top_row_count"], "rest_row_count": totals["rest_row_count"],
            "loss_return": totals["return_sum"] / totals["sample_count"],
            "loss_global_rank": totals["rank_sum"] / totals["batch_count"],
            "loss_top_boundary": totals["top_sum"] / totals["batch_count"],
            "weighted_global_rank": totals["weighted_rank_sum"] / totals["batch_count"],
            "weighted_top_boundary": totals["weighted_top_sum"] / totals["batch_count"],
            "loss_total": totals["total_sum"] / totals["batch_count"],
            "grad_norm": totals["grad_norm_sum"] / totals["batch_count"],
            "elapsed_seconds": elapsed,
            "sampling_sha256": sampler_hash.hex(),
            "parameter_count": sum(item.numel() for item in model.parameters()),
        }
        write_json(directory / f"epoch_{epoch:02d}_training.json", record)
        save_checkpoint(directory / f"checkpoint_epoch_{epoch:02d}.pt", {
            "config_sha256": config_hash, "input_sha256": data_hash,
            "group": group, "rank_weight": rank_weight, "top_boundary_weight": top_weight,
            "seed": seed, "epoch": epoch, "input_dim": 37,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        })
        write_json(completed_epochs_path, {
            "top_boundary_weight": top_weight, "global_rank_weight": rank_weight,
            "seed": seed, "completed_epochs": epoch, "max_epochs": max_epochs,
            "stopped": epoch == max_epochs,
        })
        print(json.dumps(record, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--top-weight", type=float, default=0.3)
    parser.add_argument("--max-epochs", type=int, default=8)
    args = parser.parse_args()
    if args.top_weight < 0 or not np.isfinite(args.top_weight):
        raise ValueError("Top 边界权重必须为有限非负数")
    if args.max_epochs < 1:
        raise ValueError("训练周期必须为正整数")
    config, config_hash = load_config(args.config)
    if not torch.cuda.is_available():
        raise ValueError("本机 CUDA 设备不可用")
    if config["dependence"]["training_penalty_enabled"] is True:
        raise ValueError("本实验要求关闭因子依赖训练惩罚")
    input_record = validate_completed_input(config)
    train(config, config_hash, input_record, args.run_dir, args.top_weight, args.max_epochs)


if __name__ == "__main__":
    main()
