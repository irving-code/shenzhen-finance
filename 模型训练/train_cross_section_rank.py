import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import arch
import loguru
import numpy as np
import pandas as pd
import pyarrow
import scipy
import torch

from cross_section_experiment import (
    file_sha256, index_path, load_config, project_path, read_json,
    source_hashes, temporary_path, training_dependence_enabled,
    validate_completed_input, write_json,
)
from cross_section_input import CrossSectionInputReader, SameDateBatchSampler
from dependence_constraint import dependence_penalty, replace_source_group, source_feature_indices
from evaluate_cross_section_rank import evaluate_validation, model_for_group, run_path
from ranking_loss import compute_joint_loss


def set_determinism(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def environment_record():
    if not torch.cuda.is_available():
        raise ValueError("CUDA 设备不可用")
    device = torch.cuda.get_device_properties(0)
    return {
        "python": sys.version.split()[0], "python_executable": str(Path(sys.executable).resolve()),
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "numpy": np.__version__, "pandas": pd.__version__, "pyarrow": pyarrow.__version__,
        "scipy": scipy.__version__, "loguru": loguru.__version__, "arch": arch.__version__,
        "device": device.name, "device_memory_bytes": device.total_memory,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def save_checkpoint(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_path(path)
    torch.save(state, temporary)
    os.replace(temporary, path)


def restore_random(state):
    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])
    torch.set_rng_state(state["torch_cpu_random_state"].cpu())
    torch.cuda.set_rng_state_all([item.cpu() for item in state["torch_cuda_random_state"]])


def capture_state(config_hash, input_hash, group, rank_weight, seed, epoch,
                  next_batch, total_steps, sampler_hash, totals, model, optimizer):
    return {
        "config_sha256": config_hash, "input_sha256": input_hash,
        "group": group, "rank_weight": rank_weight, "seed": seed,
        "input_dim": model.input_projection[0].in_features,
        "epoch": epoch, "next_batch": next_batch, "total_steps": total_steps,
        "sampler_version": "same_date_v1", "sampler_sha256": sampler_hash,
        "totals": totals, "parameter_count": sum(item.numel() for item in model.parameters()),
        "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
        "python_random_state": random.getstate(), "numpy_random_state": np.random.get_state(),
        "torch_cpu_random_state": torch.get_rng_state(),
        "torch_cuda_random_state": torch.cuda.get_rng_state_all(),
    }


def load_state(path, config_hash, input_hash, group, rank_weight, seed, epoch,
               model, optimizer, device):
    state = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "config_sha256": config_hash, "input_sha256": input_hash,
        "group": group, "rank_weight": rank_weight, "seed": seed,
        "epoch": epoch, "input_dim": model.input_projection[0].in_features,
        "sampler_version": "same_date_v1",
    }
    for key, value in expected.items():
        if state[key] != value:
            raise ValueError(f"检查点 {key} 与当前运行不一致")
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    for optimizer_state in optimizer.state.values():
        for key, value in optimizer_state.items():
            if isinstance(value, torch.Tensor):
                optimizer_state[key] = value.to(device)
    restore_random(state)
    return state


def training_index(config):
    frame = pd.read_parquet(index_path(config, "train_fit"), columns=[
        "ts_code", "target_date", "panel_row_id", "training_eligible", "y_ret_1d",
    ])
    eligible = frame.loc[frame["training_eligible"].eq(1)]
    if len(eligible) != config["data"]["training_eligible"]:
        raise ValueError("训练位置表数量与计划不一致")
    if eligible.duplicated(["ts_code", "target_date"]).any():
        raise ValueError("训练位置表存在重复股票日期键")
    if not np.isfinite(eligible["y_ret_1d"].to_numpy(dtype=np.float64)).all():
        raise ValueError("训练位置表包含非有限标签")
    return frame


def checked_batch(reader, frame, positions):
    batch = reader.read_batch(positions)
    source = frame.iloc[positions]
    if not source["training_eligible"].eq(1).all():
        raise ValueError("训练批次包含非训练资格记录")
    if not np.array_equal(source["ts_code"].astype(str).to_numpy(), batch["ts_code"]):
        raise ValueError("训练批次股票身份不一致")
    if not np.array_equal(source["target_date"].to_numpy(dtype=np.int64), batch["target_date"]):
        raise ValueError("训练批次日期不一致")
    if not np.array_equal(source["panel_row_id"].to_numpy(dtype=np.int64), batch["panel_row_id"]):
        raise ValueError("训练批次面板身份不一致")
    if len(np.unique(batch["ts_code"])) != len(batch["ts_code"]):
        raise ValueError("同日训练批次存在重复股票")
    if not np.isfinite(batch["y"]).all():
        raise ValueError("训练批次存在非有限标签")
    return batch


def train_seed_epoch(config, config_hash, input_hash, run_dir, group,
                     rank_weight, seed, epoch, frame, device, resume, neutral_values):
    set_determinism(seed)
    directory = run_path(run_dir, group, rank_weight, seed)
    directory.mkdir(parents=True, exist_ok=True)
    model = model_for_group(config, group, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    recovery_path = directory / "checkpoint_recovery.pt"
    epoch_path = directory / f"checkpoint_epoch_{epoch:02d}.pt"
    previous_path = directory / f"checkpoint_epoch_{epoch - 1:02d}.pt"
    next_batch = total_steps = 0
    sampler_hash = bytes(32)
    totals = {
        "sample_count": 0, "batch_count": 0, "valid_pair_count": 0,
        "return_sum": 0.0, "rank_pair_sum": 0.0,
        "weighted_rank_sum": 0.0, "total_sum": 0.0,
        "grad_norm_sum": 0.0, "prediction_std_sum": 0.0,
        "dependence_ratio_sum": 0.0, "dependence_penalty_sum": 0.0,
        "dependence_group_counts": [0] * 7,
        "dependence_group_ratio_sums": [0.0] * 7,
        "dependence_group_exceed_counts": [0] * 7,
        "elapsed_seconds": 0.0, "device_memory_peak_bytes": 0,
    }
    if recovery_path.exists() and epoch > 1 and previous_path.exists():
        recovery = torch.load(recovery_path, map_location="cpu", weights_only=False)
        if recovery["epoch"] == epoch - 1:
            if recovery["config_sha256"] != config_hash or recovery["input_sha256"] != input_hash:
                raise ValueError("已完成周期的恢复检查点身份不一致")
            recovery_path.unlink()
    if epoch > 1 and not recovery_path.exists():
        prior = load_state(previous_path, config_hash, input_hash, group, rank_weight,
                           seed, epoch - 1, model, optimizer, device)
        total_steps = prior["total_steps"]
    if recovery_path.exists():
        if not resume:
            raise ValueError("已有周期恢复检查点，必须显式提供 --resume")
        state = load_state(recovery_path, config_hash, input_hash, group, rank_weight,
                           seed, epoch, model, optimizer, device)
        next_batch = state["next_batch"]
        total_steps = state["total_steps"]
        sampler_hash = bytes.fromhex(state["sampler_sha256"])
        totals = state["totals"]
    mode = config["feature_mode"]
    reader = CrossSectionInputReader(
        project_path(config, "base_metadata"),
        project_path(config, "input_dir") / "输入元数据.json",
        index_path(config, "train_fit"), mode,
    )
    sampler = SameDateBatchSampler(frame, seed, epoch, config["training"]["batch_size"])
    dependence_enabled = training_dependence_enabled(config, group)
    dependence_groups = source_feature_indices(config) if dependence_enabled else None
    start = time.monotonic()
    carried_seconds = totals["elapsed_seconds"]
    torch.cuda.reset_peak_memory_stats(device)
    model.train()
    for batch_id, positions in enumerate(sampler):
        if batch_id < next_batch:
            continue
        batch = checked_batch(reader, frame, positions)
        inputs = torch.from_numpy(batch["x"]).to(device)
        targets = torch.from_numpy(batch["y"]).to(device)
        dates = torch.from_numpy(batch["target_date"]).to(device)
        optimizer.zero_grad(set_to_none=True)
        cpu_rng_before = torch.get_rng_state()
        cuda_rng_before = torch.cuda.get_rng_state_all()
        predictions = model(inputs)
        losses = compute_joint_loss(
            predictions, targets, dates, rank_weight,
            config["ranking"]["beta"], config["ranking"]["tau"],
        )
        loss_total = losses["loss_total"]
        dependence_ratio = None
        dependence_term = None
        dependence_group_index = None
        if dependence_enabled:
            dependence_group_index = (batch_id + seed) % len(dependence_groups)
            replaced_inputs = replace_source_group(
                inputs, dependence_groups[dependence_group_index], neutral_values,
            )
            cpu_rng_after = torch.get_rng_state()
            cuda_rng_after = torch.cuda.get_rng_state_all()
            torch.set_rng_state(cpu_rng_before)
            torch.cuda.set_rng_state_all(cuda_rng_before)
            replaced_predictions = model(replaced_inputs)
            torch.set_rng_state(cpu_rng_after)
            torch.cuda.set_rng_state_all(cuda_rng_after)
            dependence_ratio, dependence_term = dependence_penalty(
                predictions, replaced_predictions,
                config["dependence"]["limit"],
                config["dependence"]["penalty_weight"],
                config["dependence"]["denominator_floor"],
            )
            loss_total = loss_total + dependence_term
        losses["loss_total"] = loss_total
        loss_total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config["training"]["gradient_clip"]
        )
        if not torch.isfinite(grad_norm):
            raise ValueError("梯度范数非有限")
        optimizer.step()
        with torch.no_grad():
            if not torch.isfinite(torch.nn.utils.parameters_to_vector(model.parameters())).all():
                raise ValueError("更新后的模型参数非有限")
        size = len(positions)
        pairs = losses["valid_pair_count"]
        totals["sample_count"] += size
        totals["batch_count"] += 1
        totals["valid_pair_count"] += pairs
        totals["return_sum"] += float(losses["loss_return"].detach()) * size
        totals["rank_pair_sum"] += float(losses["loss_rank"].detach()) * pairs
        totals["weighted_rank_sum"] += float(losses["weighted_rank_loss"].detach())
        totals["total_sum"] += float(losses["loss_total"].detach())
        totals["grad_norm_sum"] += float(grad_norm.detach())
        totals["prediction_std_sum"] += float(predictions.detach().std(unbiased=False))
        if dependence_ratio is not None:
            totals["dependence_ratio_sum"] += float(dependence_ratio.detach())
            totals["dependence_penalty_sum"] += float(dependence_term.detach())
            totals["dependence_group_counts"][dependence_group_index] += 1
            totals["dependence_group_ratio_sums"][dependence_group_index] += float(dependence_ratio.detach())
            totals["dependence_group_exceed_counts"][dependence_group_index] += int(float(dependence_ratio.detach()) > config["dependence"]["limit"])
        sampler_hash = hashlib.sha256(
            sampler_hash + positions.tobytes() + batch["target_date"].tobytes()
            + "|".join(batch["ts_code"]).encode("utf-8")
        ).digest()
        total_steps += 1
        if (batch_id + 1) % config["training"]["checkpoint_interval_batches"] == 0:
            elapsed = time.monotonic() - start
            totals["elapsed_seconds"] = carried_seconds + elapsed
            totals["device_memory_peak_bytes"] = max(totals["device_memory_peak_bytes"], torch.cuda.max_memory_allocated(device))
            save_checkpoint(recovery_path, capture_state(
                config_hash, input_hash, group, rank_weight, seed, epoch,
                batch_id + 1, total_steps, sampler_hash.hex(), totals, model, optimizer,
            ))
            completed_batches = batch_id + 1 - next_batch
            seconds_per_batch = elapsed / completed_batches
            progress = {
                "epoch": epoch, "batch": batch_id + 1,
                "total_batches": config["sampling"]["expected_batches_per_epoch"],
                "elapsed_seconds": totals["elapsed_seconds"],
                "seconds_per_batch": seconds_per_batch,
                "estimated_epoch_remaining_seconds": seconds_per_batch * (config["sampling"]["expected_batches_per_epoch"] - batch_id - 1),
                "loss_total": float(loss_total.detach()),
            }
            write_json(Path(run_dir) / "训练进度.json", progress)
            print(json.dumps(progress, ensure_ascii=False), flush=True)
    if totals["sample_count"] != config["data"]["training_eligible"]:
        raise ValueError("训练周期没有完整覆盖全部合格记录")
    if totals["batch_count"] != config["sampling"]["expected_batches_per_epoch"]:
        raise ValueError("训练周期批次数量不一致")
    totals["elapsed_seconds"] = carried_seconds + time.monotonic() - start
    totals["device_memory_peak_bytes"] = max(totals["device_memory_peak_bytes"], torch.cuda.max_memory_allocated(device))
    write_json(directory / f"epoch_{epoch:02d}_training.json", {
        "group": group, "rank_weight": rank_weight, "seed": seed, "epoch": epoch,
        "elapsed_seconds": totals["elapsed_seconds"], "sampling_sha256": sampler_hash.hex(),
        "device_memory_peak_bytes": totals["device_memory_peak_bytes"],
        "sample_count": totals["sample_count"], "batch_count": totals["batch_count"],
        "valid_pair_count": totals["valid_pair_count"],
        "loss_return": totals["return_sum"] / totals["sample_count"],
        "loss_rank": totals["rank_pair_sum"] / totals["valid_pair_count"] if totals["valid_pair_count"] else 0.0,
        "weighted_rank_step_mean": totals["weighted_rank_sum"] / totals["batch_count"],
        "loss_total_step_mean": totals["total_sum"] / totals["batch_count"],
        "grad_norm_step_mean": totals["grad_norm_sum"] / totals["batch_count"],
        "prediction_std_step_mean": totals["prediction_std_sum"] / totals["batch_count"],
        "dependence_ratio_step_mean": (
            totals["dependence_ratio_sum"] / totals["batch_count"] if dependence_enabled else None
        ),
        "dependence_penalty_step_mean": totals["dependence_penalty_sum"] / totals["batch_count"],
        "dependence_group_counts": totals["dependence_group_counts"],
        "dependence_group_mean_ratios": [
            total / count if count else None
            for total, count in zip(totals["dependence_group_ratio_sums"], totals["dependence_group_counts"], strict=True)
        ],
        "dependence_group_exceed_fractions": [
            total / count if count else None
            for total, count in zip(totals["dependence_group_exceed_counts"], totals["dependence_group_counts"], strict=True)
        ],
        "parameter_count": sum(item.numel() for item in model.parameters()),
        "total_steps": total_steps,
    })
    save_checkpoint(epoch_path, capture_state(
        config_hash, input_hash, group, rank_weight, seed, epoch,
        totals["batch_count"], total_steps, sampler_hash.hex(), totals, model, optimizer,
    ))
    if recovery_path.exists():
        recovery_path.unlink()
    return model


def evaluate_saved_epoch(config, config_hash, input_hash, run_dir, group,
                         rank_weight, seed, epoch, device):
    directory = run_path(run_dir, group, rank_weight, seed)
    checkpoint = directory / f"checkpoint_epoch_{epoch:02d}.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state["config_sha256"] != config_hash or state["input_sha256"] != input_hash:
        raise ValueError("验证检查点与当前配置或输入不一致")
    model = model_for_group(config, group, device)
    model.load_state_dict(state["model_state_dict"])
    return evaluate_validation(config, run_dir, group, rank_weight, seed, epoch, model, device)


def run_training(config, config_hash, input_record, run_dir, resume):
    if config["protocol"]["execution_status"] != "authorized_to_run":
        raise ValueError("正式训练仍处于暂停状态")
    input_hash = hashlib.sha256(json.dumps(input_record, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    snapshot = read_json(Path(run_dir) / "配置快照.json")
    if snapshot["config_sha256"] != config_hash or snapshot["input_sha256"] != input_hash:
        raise ValueError("实验运行目录与配置或输入身份不一致")
    if snapshot["source_sha256"] != source_hashes(config["protocol"]["schema_version"]):
        raise ValueError("训练程序摘要与预检记录不一致")
    if read_json(Path(run_dir) / "verification" / "预检结果.json")["status"] != "passed":
        raise ValueError("真实数据预检尚未通过")
    neutral_record = read_json(Path(run_dir) / "dependence_neutral_values.json")
    if neutral_record["config_sha256"] != config_hash or neutral_record["input_sha256"] != input_hash:
        raise ValueError("依赖约束中性值与当前配置或输入身份不一致")
    neutral_values = torch.tensor(neutral_record["values"], dtype=torch.float32, device="cuda")
    if any(Path(run_dir).joinpath("runs").glob("*")) and not resume:
        raise ValueError("运行目录已有正式训练记录，必须显式提供 --resume")
    frame = training_index(config)
    device = torch.device("cuda")
    state_path = Path(run_dir) / "调度状态.json"
    state = read_json(state_path) if state_path.exists() else {
        "model_id": config["model_id"], "rank_weight": config["ranking"]["weight"],
        "seed": config["seed"], "completed_epochs": 0, "best_monitor": None,
        "stale_epochs": 0, "stopped": False,
    }
    expected_state = {
        "model_id": config["model_id"], "rank_weight": config["ranking"]["weight"],
        "seed": config["seed"],
    }
    if any(state[key] != value for key, value in expected_state.items()):
        raise ValueError("训练状态与单模型配置不一致")
    if state["stopped"]:
        return
    group = config["model_id"]
    rank_weight = config["ranking"]["weight"]
    seed = config["seed"]
    alpha = config["selection"]["alpha"]
    for epoch in range(state["completed_epochs"] + 1, config["training"]["max_epochs"] + 1):
        metrics_path = run_path(run_dir, group, rank_weight, seed) / f"epoch_{epoch:02d}" / "validation_metrics.json"
        checkpoint_path = run_path(run_dir, group, rank_weight, seed) / f"checkpoint_epoch_{epoch:02d}.pt"
        runtime_path = metrics_path.parent / "validation_runtime.json"
        if metrics_path.exists() and runtime_path.exists() and checkpoint_path.exists():
            metrics = read_json(metrics_path)
        elif checkpoint_path.exists():
            metrics = evaluate_saved_epoch(
                config, config_hash, input_hash, run_dir, group, rank_weight, seed, epoch, device,
            )
        else:
            model = train_seed_epoch(
                config, config_hash, input_hash, run_dir, group, rank_weight, seed,
                epoch, frame, device, resume, neutral_values,
            )
            metrics = evaluate_validation(
                config, run_dir, group, rank_weight, seed, epoch, model, device,
            )
        alpha_metrics = metrics[str(alpha)]
        monitor = (
            float(alpha_metrics["score"]["final_score"])
            if alpha_metrics.get("quality_pass") is True and alpha_metrics.get("score") is not None
            else None
        )
        early_start = config["training"]["early_stopping_count_start_epoch"]
        if monitor is not None and (
            state["best_monitor"] is None
            or monitor > state["best_monitor"] + config["training"]["early_stopping_min_improvement"]
        ):
            state["best_monitor"] = monitor
            state["stale_epochs"] = 0
        elif epoch >= early_start:
            state["stale_epochs"] += 1
        state["completed_epochs"] = epoch
        state["stopped"] = epoch == config["training"]["max_epochs"] or (
            epoch >= config["training"]["min_epochs"]
            and state["stale_epochs"] >= config["training"]["early_stopping_patience"]
        )
        state["stop_reason"] = (
            "max_epochs" if epoch == config["training"]["max_epochs"]
            else "early_stopping" if state["stopped"]
            else None
        )
        write_json(state_path, state)
        print(json.dumps({
            "model_id": group, "epoch": epoch, "validation_score": monitor,
            "best_validation_score": state["best_monitor"],
            "stale_epochs": state["stale_epochs"], "stopped": state["stopped"],
        }, ensure_ascii=False), flush=True)
        if state["stopped"]:
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--check-environment", action="store_true")
    parser.add_argument("--run-dir")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config, config_hash = load_config(args.config)
    set_determinism(config["seed"])
    environment = environment_record()
    if args.check_environment:
        print(json.dumps(environment, ensure_ascii=False, indent=2))
        return
    if args.device != "cuda" or not args.run_dir:
        raise ValueError("正式训练需要 --run-dir 与 CUDA 设备")
    run_dir = Path(args.run_dir).resolve()
    input_record = validate_completed_input(config)
    run_training(config, config_hash, input_record, run_dir, args.resume)


if __name__ == "__main__":
    main()
