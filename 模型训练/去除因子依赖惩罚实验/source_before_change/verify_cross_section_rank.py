import argparse
import hashlib
import importlib.util
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cross_section_experiment import (
    file_sha256, index_path, load_config, project_path, read_json,
    source_hashes, validate_completed_input, write_json,
)
from cross_section_input import CrossSectionInputReader, SameDateBatchSampler
from dependence_constraint import (
    compute_neutral_values, dependence_penalty, replace_source_group, source_feature_indices,
)
from evaluate_cross_section_rank import model_for_group, smooth_predictions, strict_score
from ranking_loss import compute_joint_loss
from train_cross_section_rank import (
    capture_state, environment_record, load_state, save_checkpoint, set_determinism,
    training_index,
)


def reader_for(config, split, mode):
    return CrossSectionInputReader(
        project_path(config, "base_metadata"),
        project_path(config, "input_dir") / "输入元数据.json",
        index_path(config, split), mode,
    )


def verify_dates(config):
    dates = np.load(project_path(config, "factor_dates"))
    if dates.ndim != 1 or not np.all(dates[1:] > dates[:-1]):
        raise ValueError("交易日期表不是严格递增的一维数组")
    record = {}
    for split, field, lower, upper, excluded, target_date, realization_date in (
        ("train_fit", "training_eligible", None, config["splits"]["training_label_end"],
         config["data"]["training_boundary_excluded"], 20221230, 20230103),
        ("validation", "selection_eligible", config["splits"]["selection_label_start"],
         config["splits"]["selection_label_end"],
         config["data"]["selection_boundary_excluded"], 20231229, 20240102),
    ):
        frame = pd.read_parquet(index_path(config, split), columns=[
            "ts_code", "target_date", "prediction_eligible", field, "label_realization_date", "y_ret_1d",
        ])
        eligible = frame.loc[frame[field].eq(1)]
        realization = eligible["label_realization_date"].to_numpy(dtype=np.int64)
        if upper is not None and np.any(realization > upper):
            raise ValueError(f"{split} 标签实现日期越过区间终点")
        if lower is not None and np.any(realization < lower):
            raise ValueError(f"{split} 标签实现日期早于区间起点")
        if not np.isfinite(eligible["y_ret_1d"].to_numpy(dtype=np.float64)).all():
            raise ValueError(f"{split} 有效标签包含非有限值")
        positions = np.searchsorted(dates, eligible["target_date"].to_numpy(dtype=np.int64))
        if np.any(positions + 1 >= len(dates)) or not np.array_equal(dates[positions + 1], realization):
            raise ValueError(f"{split} 下一交易日映射不一致")
        removed = frame.loc[frame["prediction_eligible"].eq(1) & frame[field].eq(0)]
        if len(removed) != excluded:
            raise ValueError(f"{split} 标签边界排除数量不一致")
        if not removed["target_date"].eq(target_date).all() or not removed["label_realization_date"].eq(realization_date).all():
            raise ValueError(f"{split} 标签边界排除日期不一致")
        record[split] = {
            "eligible_rows": len(eligible), "excluded_rows": len(removed),
            "excluded_target_date": target_date, "excluded_realization_date": realization_date,
        }
    return record


def verify_sampling(config, frame):
    expected_positions = frame.index[frame["training_eligible"].eq(1)].to_numpy(dtype=np.int64)
    record = {}
    for seed in (config["seed"],):
        hashes = []
        for epoch in (1, 2):
            visited = np.zeros(len(frame), dtype=np.uint8)
            digest = bytes(32)
            batch_count = 0
            for positions in SameDateBatchSampler(frame, seed, epoch, 512):
                if len(positions) > 512 or len(positions) == 0:
                    raise ValueError("训练采样批次规模异常")
                if frame.iloc[positions]["target_date"].nunique() != 1:
                    raise ValueError("训练采样批次跨越交易日期")
                if np.any(visited[positions]):
                    raise ValueError("训练采样重复访问原文件行号")
                visited[positions] = 1
                digest = hashlib.sha256(digest + positions.tobytes()).digest()
                batch_count += 1
            if not np.array_equal(np.flatnonzero(visited), expected_positions):
                raise ValueError("训练采样未完整覆盖合格原文件行号")
            hashes.append(digest.hex())
            record[f"seed_{seed}_epoch_{epoch}"] = {
                "rows": int(visited.sum()), "batch_count": batch_count,
                "sampler_sha256": digest.hex(),
            }
        if hashes[0] == hashes[1]:
            raise ValueError("相邻周期的采样顺序完全相同")
    return record


def verify_all_windows(config):
    result = {}
    for split in config["data"]["rows"]:
        base = reader_for(config, split, "base27")
        extended = reader_for(config, split, "extended37")
        count = 0
        for old, new in zip(
            base.iter_inference_batches(4096), extended.iter_inference_batches(4096), strict=True
        ):
            if not np.array_equal(old["x"], new["x"][:, :, :27]):
                raise ValueError(f"{split} 原有 27 通道与扩展输入不一致")
            if not np.isfinite(new["x"]).all():
                raise ValueError(f"{split} 扩展输入出现非有限值")
            if split == "competition_test" and ("y" in old or "y" in new):
                raise ValueError("官方测试输入包含标签")
            if not np.array_equal(old["ts_code"], new["ts_code"]):
                raise ValueError(f"{split} 两组输入股票身份不一致")
            count += len(new["x"])
        if count != config["data"]["prediction_eligible"][split]:
            raise ValueError(f"{split} 合格窗口扫描数量不一致")
        result[split] = {"windows": count, "base_channels": 27, "extended_channels": 37}
    return result


def verify_real_gpu_batches(config, frame, verification_dir, neutral_values):
    eligible = frame.loc[frame["training_eligible"].eq(1)]
    large_date = int(eligible.groupby("target_date").size().loc[lambda counts: counts.ge(512)].index[0])
    positions = eligible.index[eligible["target_date"].eq(large_date)].to_numpy(dtype=np.int64)[:512]
    if len(positions) != 512:
        raise ValueError("真实训练日期不足 512 条记录")
    result = {}
    for group in (config["model_id"],):
        set_determinism(config["seed"])
        mode = config["feature_mode"]
        rank_weight = config["ranking"]["weight"]
        batch = reader_for(config, "train_fit", mode).read_batch(positions)
        model = model_for_group(config, group, torch.device("cuda"))
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )
        initial = [parameter.detach().clone() for parameter in model.parameters()]
        features = torch.from_numpy(batch["x"]).to("cuda")
        target = torch.from_numpy(batch["y"]).to("cuda")
        dates = torch.from_numpy(batch["target_date"]).to("cuda")
        optimizer.zero_grad(set_to_none=True)
        cpu_rng_before = torch.get_rng_state()
        cuda_rng_before = torch.cuda.get_rng_state_all()
        pred = model(features)
        loss = compute_joint_loss(pred, target, dates, rank_weight)
        loss_total = loss["loss_total"]
        if group == config["dependence"]["enabled_model"]:
            source_groups = source_feature_indices(config)
            source_index = (config["seed"] + 1) % len(source_groups)
            replaced_features = replace_source_group(
                features, source_groups[source_index], neutral_values,
            )
            cpu_rng_after = torch.get_rng_state()
            cuda_rng_after = torch.cuda.get_rng_state_all()
            torch.set_rng_state(cpu_rng_before)
            torch.cuda.set_rng_state_all(cuda_rng_before)
            replaced_pred = model(replaced_features)
            torch.set_rng_state(cpu_rng_after)
            torch.cuda.set_rng_state_all(cuda_rng_after)
            ratio, dependence_term = dependence_penalty(
                pred, replaced_pred, config["dependence"]["limit"],
                config["dependence"]["penalty_weight"],
                config["dependence"]["denominator_floor"],
            )
            loss_total = loss_total + dependence_term
            if not torch.equal(
                replaced_features[:, :, source_groups[source_index]],
                torch.as_tensor(neutral_values, device="cuda")[source_groups[source_index]].view(1, 1, -1).expand(
                    len(positions), 20, -1,
                ),
            ):
                raise ValueError("因子替换没有覆盖全部时间位置")
            result["dependence"] = {
                "source_group_index": source_index,
                "source_features": source_groups[source_index],
                "ratio": float(ratio.detach()),
                "penalty": float(dependence_term.detach()),
            }
        loss["loss_total"] = loss_total
        loss_total.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(norm):
            raise ValueError("真实批次梯度非有限")
        optimizer.step()
        torch.cuda.synchronize()
        if not any(not torch.equal(before, after) for before, after in zip(initial, model.parameters())):
            raise ValueError("真实批次没有更新模型参数")
        result[group] = {
            "rows": len(positions), "shape": list(batch["x"].shape),
            "loss_total": float(loss["loss_total"].detach()),
            "loss_return": float(loss["loss_return"].detach()),
            "loss_rank": float(loss["loss_rank"].detach()),
            "valid_pair_count": loss["valid_pair_count"],
            "gradient_norm": float(norm.detach()),
        }
        if group == config["model_id"]:
            label_values = batch["y"]
            unequal = np.flatnonzero(label_values != label_values[0])
            if len(unequal) == 0:
                raise ValueError("真实批次中没有可检验的非并列股票对")
            selected_pair = np.array([0, int(unequal[0])], dtype=np.int64)
            pair_prediction = pred.detach()[selected_pair].clone().requires_grad_(True)
            pair_target = target[selected_pair]
            pair_dates = dates[selected_pair]
            pair_loss = compute_joint_loss(
                pair_prediction, pair_target, pair_dates, rank_weight=1.0,
            )["loss_rank"]
            pair_gradient = torch.autograd.grad(pair_loss, pair_prediction)[0]
            direction = torch.sign(pair_target[0] - pair_target[1])
            if not (direction * (pair_gradient[0] - pair_gradient[1]) < 0):
                raise ValueError("真实股票对的排序梯度方向异常")
            result["rank_direction"] = {
                "ts_codes": batch["ts_code"][selected_pair].tolist(),
                "target_date": int(large_date),
                "target_difference": float((pair_target[0] - pair_target[1]).detach()),
                "gradient_difference": float((pair_gradient[0] - pair_gradient[1]).detach()),
            }
        del features, target, dates, pred, loss, model, optimizer
        torch.cuda.empty_cache()
    inference_reader = reader_for(config, "validation", "extended37")
    valid = pd.read_parquet(index_path(config, "validation"), columns=["prediction_eligible"])
    inference_positions = valid.index[valid["prediction_eligible"].eq(1)].to_numpy(dtype=np.int64)[:4096]
    inference = inference_reader.read_batch(inference_positions)
    set_determinism(42)
    model = model_for_group(config, config["model_id"], torch.device("cuda"))
    model.eval()
    with torch.inference_mode():
        prediction = model(torch.from_numpy(inference["x"]).to("cuda"))
    torch.cuda.synchronize()
    if prediction.shape != (4096,) or not torch.isfinite(prediction).all():
        raise ValueError("真实 4096 条推理批次异常")
    result["inference"] = {
        "rows": len(inference_positions), "shape": list(inference["x"].shape),
        "device_memory_peak_bytes": torch.cuda.max_memory_allocated(),
    }
    write_json(verification_dir / "真实批次核验.json", result)
    return result


def verify_checkpoint_recovery(config, config_sha256, input_sha256, frame, verification_dir, neutral_values):
    set_determinism(42)
    reader = reader_for(config, "train_fit", config["feature_mode"])
    sampler = iter(SameDateBatchSampler(frame, config["seed"], 1, 512))
    first_positions = next(sampler)
    first = reader.read_batch(first_positions)
    second = reader.read_batch(next(sampler))
    source_groups = source_feature_indices(config)
    device = torch.device("cuda")
    model = model_for_group(config, config["model_id"], device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.0001)

    def update(current_model, current_optimizer, batch, batch_id):
        current_model.train()
        current_optimizer.zero_grad(set_to_none=True)
        inputs = torch.from_numpy(batch["x"]).to(device)
        target = torch.from_numpy(batch["y"]).to(device)
        dates = torch.from_numpy(batch["target_date"]).to(device)
        cpu_before = torch.get_rng_state()
        cuda_before = torch.cuda.get_rng_state_all()
        prediction = current_model(inputs)
        cpu_after = torch.get_rng_state()
        cuda_after = torch.cuda.get_rng_state_all()
        replaced = replace_source_group(inputs, source_groups[(batch_id + config["seed"]) % 7], neutral_values)
        torch.set_rng_state(cpu_before)
        torch.cuda.set_rng_state_all(cuda_before)
        changed = current_model(replaced)
        torch.set_rng_state(cpu_after)
        torch.cuda.set_rng_state_all(cuda_after)
        loss = compute_joint_loss(prediction, target, dates, config["ranking"]["weight"])["loss_total"]
        _, penalty = dependence_penalty(
            prediction, changed, config["dependence"]["limit"],
            config["dependence"]["penalty_weight"], config["dependence"]["denominator_floor"],
        )
        loss = loss + penalty
        loss.backward()
        torch.nn.utils.clip_grad_norm_(current_model.parameters(), 1.0)
        current_optimizer.step()
        return prediction.detach().clone()

    update(model, optimizer, first, 0)
    path = verification_dir / "恢复核验.pt"
    save_checkpoint(path, capture_state(
        config_sha256, input_sha256, config["model_id"], config["ranking"]["weight"], config["seed"], 1, 1, 1,
        hashlib.sha256(first_positions.tobytes()).hexdigest(),
        {"sample_count": len(first["x"])}, model, optimizer,
    ))
    continuous_prediction = update(model, optimizer, second, 1)
    continuous_parameters = [parameter.detach().clone() for parameter in model.parameters()]
    set_determinism(42)
    recovered = model_for_group(config, config["model_id"], device)
    recovered_optimizer = torch.optim.AdamW(recovered.parameters(), lr=0.0003, weight_decay=0.0001)
    load_state(path, config_sha256, input_sha256, config["model_id"], config["ranking"]["weight"], config["seed"], 1,
               recovered, recovered_optimizer, device)
    restored_prediction = update(recovered, recovered_optimizer, second, 1)
    rtol = config["verification"]["rtol"]
    atol = config["verification"]["atol"]
    if not torch.allclose(continuous_prediction, restored_prediction, rtol=rtol, atol=atol):
        raise ValueError("恢复后的真实批次预测不一致")
    if any(not torch.allclose(a, b, rtol=rtol, atol=atol) for a, b in zip(continuous_parameters, recovered.parameters())):
        raise ValueError("恢复后的模型参数不一致")
    result = {
        "first_batch_rows": len(first["x"]), "next_batch_rows": len(second["x"]),
        "first_batch_date": int(first["target_date"][0]),
        "next_batch_date": int(second["target_date"][0]),
        "checkpoint_sha256": file_sha256(path),
        "prediction_max_absolute_difference": float(torch.max(torch.abs(
            continuous_prediction - restored_prediction
        )).cpu()),
    }
    write_json(verification_dir / "恢复核验.json", result)
    return result


def verify_official_score(config, verification_dir):
    index = pd.read_parquet(index_path(config, "validation"), columns=[
        "ts_code", "target_date", "prediction_eligible", "y_ret_1d", "flag_limit_up",
    ])
    eligible = index.loc[index["prediction_eligible"].eq(1)]
    dates = eligible["target_date"].drop_duplicates().iloc[:3].to_numpy(dtype=np.int64)
    positions = eligible.index[eligible["target_date"].isin(dates)].to_numpy(dtype=np.int64)
    reader = reader_for(config, "validation", "extended37")
    set_determinism(42)
    model = model_for_group(config, config["model_id"], torch.device("cuda"))
    model.eval()
    values = []
    with torch.inference_mode():
        for chunk in np.array_split(positions, math.ceil(len(positions) / 4096)):
            batch = reader.read_batch(chunk)
            values.append(model(torch.from_numpy(batch["x"]).to("cuda")).cpu().numpy())
    prediction = np.concatenate(values).astype(np.float64)
    observed = index.iloc[positions].reset_index(drop=True)
    frame = pd.DataFrame({
        "ts_code": observed["ts_code"].astype(str),
        "trade_date": observed["target_date"].to_numpy(dtype=np.int64),
        "pred": prediction,
        "y_ret_1d": observed["y_ret_1d"].to_numpy(dtype=np.float64),
        "flag_limit_up": observed["flag_limit_up"].to_numpy(dtype=np.float64),
        "prediction_eligible": np.ones(len(observed), dtype=np.int8),
    }).sort_values(["ts_code", "trade_date"], kind="mergesort").reset_index(drop=True)
    local, daily = strict_score(frame, "pred")
    smoothed = smooth_predictions(frame, 1.0)
    if not np.array_equal(smoothed["smooth_pred"].to_numpy(), smoothed["pred"].to_numpy()):
        raise ValueError("alpha=1.0 平滑结果与原始预测不一致")
    csv_dir = verification_dir / "official_csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    frame[["ts_code", "trade_date", "pred"]].to_csv(csv_dir / "submission.csv", index=False)
    frame[["ts_code", "trade_date", "y_ret_1d"]].to_csv(csv_dir / "测试集_Y.csv", index=False)
    frame[["ts_code", "trade_date", "flag_limit_up"]].to_csv(csv_dir / "测试集_X.csv", index=False)
    official_path = Path(config["project_root"]) / "赛题五" / "evaluate.py"
    spec = importlib.util.spec_from_file_location("official_competition_evaluate", official_path)
    if spec is None or spec.loader is None:
        raise ValueError("官方评分模块无法加载")
    official_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official_module)
    official = official_module.evaluate(str(csv_dir / "submission.csv"), str(csv_dir))
    comparisons = {
        "ic_mean": local["rank_ic"]["mean"],
        "ic_std": local["rank_ic"]["std"],
        "icir": local["rank_ic"]["icir"],
        "ic_positive_ratio": local["rank_ic"]["positive_ratio"],
        "annual_excess": local["top_decile"]["annual_excess"],
        "top1_annual_ret": local["top_decile"]["top1_annual_return"],
        "mean_turnover": local["turnover"]["mean_turnover"],
        "final_score": local["final_score"],
    }
    difference = {key: abs(float(official[key]) - float(value)) for key, value in comparisons.items()}
    for key, value in comparisons.items():
        if not np.isclose(official[key], value, rtol=config["verification"]["score_rtol"],
                          atol=config["verification"]["score_atol"]):
            raise ValueError(f"官方评分 {key} 与统一评价入口不一致")
    result = {"dates": dates.tolist(), "rows": len(frame), "official": official,
              "local": comparisons, "absolute_difference": difference,
              "daily_metric_rows": len(daily)}
    write_json(verification_dir / "官方评分核验.json", result)
    return result


def run_preflight(config, config_sha256, run_dir):
    started = time.monotonic()
    run_dir = Path(run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError("预检运行目录已有文件")
    set_determinism(42)
    environment = environment_record()
    if environment["device"] != config["execution"]["device"]:
        raise ValueError("当前显卡与本机训练配置不一致")
    input_record = validate_completed_input(config)
    input_sha256 = hashlib.sha256(
        json.dumps(input_record, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    run_dir.mkdir(parents=True, exist_ok=True)
    verification_dir = run_dir / "verification"
    verification_dir.mkdir()
    write_json(run_dir / "配置快照.json", {
        "config_sha256": config_sha256, "input_sha256": input_sha256,
        "config": config, "source_sha256": source_hashes(),
    })
    write_json(run_dir / "环境记录.json", environment)
    write_json(run_dir / "输入接入记录.json", input_record)
    print(json.dumps({"preflight": "input_verified", "elapsed_seconds": time.monotonic() - started}), flush=True)
    dates = verify_dates(config)
    write_json(verification_dir / "日期资格核验.json", dates)
    frame = training_index(config)
    neutral_reader = reader_for(config, "train_fit", "extended37")
    neutral_values = compute_neutral_values(config, neutral_reader, frame)
    write_json(run_dir / "dependence_neutral_values.json", {
        "config_sha256": config_sha256,
        "input_sha256": input_sha256,
        "method": config["dependence"]["replacement"],
        "source_rows": config["data"]["training_eligible"],
        "values": neutral_values.tolist(),
    })
    print(json.dumps({"preflight": "neutral_values_ready", "elapsed_seconds": time.monotonic() - started}), flush=True)
    sampling = verify_sampling(config, frame)
    write_json(verification_dir / "采样核验.json", sampling)
    print(json.dumps({"preflight": "sampling_verified", "elapsed_seconds": time.monotonic() - started}), flush=True)
    windows = verify_all_windows(config)
    write_json(verification_dir / "窗口核验.json", windows)
    print(json.dumps({"preflight": "windows_verified", "elapsed_seconds": time.monotonic() - started}), flush=True)
    gpu = verify_real_gpu_batches(config, frame, verification_dir, neutral_values)
    recovery = verify_checkpoint_recovery(
        config, config_sha256, input_sha256, frame, verification_dir, neutral_values,
    )
    official = verify_official_score(config, verification_dir)
    write_json(verification_dir / "预检结果.json", {
        "status": "passed", "config_sha256": config_sha256,
        "input_sha256": input_sha256, "dates": dates, "sampling": sampling,
        "windows": windows, "gpu": gpu, "recovery": recovery,
        "official_score": official,
        "elapsed_seconds": time.monotonic() - started,
    })
    print(json.dumps({"status": "passed", "run_dir": str(run_dir)}, ensure_ascii=False))


def write_final_manifest(run_dir, result):
    files = []
    for path in run_dir.rglob("*"):
        if path.is_file() and path.name not in ("最终核验.json", "执行状态.json") and not path.name.endswith(".writing"):
            files.append({"path": path.relative_to(run_dir).as_posix(),
                          "bytes": path.stat().st_size, "sha256": file_sha256(path)})
    write_json(run_dir / "最终核验.json", {**result, "files": files})


def run_final_audit(config, config_sha256, run_dir):
    run_dir = Path(run_dir)
    snapshot = read_json(run_dir / "配置快照.json")
    if snapshot["config_sha256"] != config_sha256:
        raise ValueError("最终核验配置摘要不一致")
    current_input = validate_completed_input(config)
    input_sha256 = hashlib.sha256(json.dumps(current_input, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    if snapshot["input_sha256"] != input_sha256 or snapshot["source_sha256"] != source_hashes():
        raise ValueError("最终核验输入或程序摘要不一致")
    status = read_json(run_dir / "调度状态.json")
    group, seed = config["model_id"], config["seed"]
    weight, alpha = config["ranking"]["weight"], config["selection"]["alpha"]
    if not status["stopped"] or status["model_id"] != group or status["seed"] != seed:
        raise ValueError("单模型正式训练尚未结束或身份不一致")
    epochs = status["completed_epochs"]
    if not config["training"]["min_epochs"] <= epochs <= config["training"]["max_epochs"]:
        raise ValueError("训练周期数量异常")
    directory = run_dir / "runs" / group / f"lambda_{weight:g}" / f"seed_{seed}"
    for epoch in range(1, epochs + 1):
        checkpoint = directory / f"checkpoint_epoch_{epoch:02d}.pt"
        training = directory / f"epoch_{epoch:02d}_training.json"
        metrics = directory / f"epoch_{epoch:02d}" / "validation_metrics.json"
        if not checkpoint.exists() or not training.exists() or not metrics.exists():
            raise ValueError("正式训练检查点、训练记录或验证指标缺失")
        record = read_json(training)
        if record["sample_count"] != config["data"]["training_eligible"]:
            raise ValueError("正式训练周期覆盖数量不一致")
        if record["batch_count"] != config["sampling"]["expected_batches_per_epoch"]:
            raise ValueError("正式训练批次数量不一致")
        if sum(record["dependence_group_counts"]) != record["batch_count"]:
            raise ValueError("因子依赖约束没有覆盖全部训练批次")
    selected = read_json(run_dir / "选定配置.json")
    choice = selected["final_candidate"]
    if selected["status"] == "no_eligible_configuration":
        checks = read_json(run_dir / "选择核验记录.json")
        if choice is not None or {item["epoch"] for item in checks} != set(range(1, epochs + 1)):
            raise ValueError("模型选择失败记录没有覆盖全部训练周期")
        for item in checks:
            epoch_dir = directory / f"epoch_{item['epoch']:02d}"
            quality = read_json(epoch_dir / "validation_metrics.json")[str(alpha)]
            if item["status"] == "prediction_quality_failed":
                failed = quality["quality_pass"] is not True or quality["score"] is None
            elif item["status"] == "dependence_failed":
                failed = read_json(epoch_dir / "dependence_metrics.json")[str(seed)]["passed"] is False
            elif item["status"] == "stability_failed":
                failed = read_json(epoch_dir / "stability_metrics.json")["seeds"][str(seed)]["alphas"][str(alpha)]["passed"] is False
            else:
                raise ValueError("模型选择失败记录包含未知检查状态")
            if not failed:
                raise ValueError("模型淘汰原因与原始检查记录不一致")
        if (run_dir / "evaluation" / "本地留出集指标.json").exists() or (run_dir / "competition_test").exists():
            raise ValueError("缺少合格模型时出现留出集或测试集产物")
        if not (run_dir / "模型结果评估报告.md").exists():
            raise ValueError("模型选择失败评估报告缺失")
        write_final_manifest(run_dir, {
            "status": "passed", "model_selection_status": selected["status"],
            "completed_runs": 1, "completed_epochs": epochs,
            "holdout_evaluated": False, "competition_test_predicted": False,
        })
        return
    if selected["status"] != "selected" or choice is None:
        raise ValueError("缺少通过核验的模型选择记录")
    if (choice["group"], choice["seed"], choice["rank_weight"], choice["alpha"]) != (group, seed, weight, alpha):
        raise ValueError("模型选择身份与固定配置不一致")
    if not 1 <= choice["epoch"] <= epochs:
        raise ValueError("模型选择周期不在已完成训练范围内")
    epoch_dir = directory / f"epoch_{choice['epoch']:02d}"
    quality = read_json(epoch_dir / "validation_metrics.json")[str(alpha)]
    if quality["quality_pass"] is not True:
        raise ValueError("所选周期未通过预测质量检查")
    dependence = read_json(epoch_dir / "dependence_metrics.json")[str(seed)]
    if dependence["passed"] is not True or dependence["epoch"] != choice["epoch"]:
        raise ValueError("所选周期未通过因子依赖检查")
    stability = read_json(epoch_dir / "stability_metrics.json")
    if (
        stability["group"] != group or stability["rank_weight"] != weight
        or stability["epoch"] != choice["epoch"]
        or stability["config_sha256"] != config_sha256
        or stability["input_sha256"] != snapshot["input_sha256"]
        or set(stability["seeds"]) != {str(seed)}
    ):
        raise ValueError("最终选择的双批量稳定性身份不一致")
    if stability["seeds"][str(seed)]["alphas"][str(alpha)]["passed"] is not True:
        raise ValueError("所选周期未通过双批量稳定性检查")
    holdout = read_json(run_dir / "evaluation" / "本地留出集指标.json")
    if (holdout["model_id"], holdout["seed"], holdout["epoch"], holdout["alpha"]) != (group, seed, choice["epoch"], alpha):
        raise ValueError("本地留出集评价身份不一致")
    coverage = read_json(run_dir / "competition_test" / "覆盖审计.json")
    if (coverage["eligible_predictions"] != config["data"]["prediction_eligible"]["competition_test"]
            or coverage["uncovered_rows"] != config["data"]["competition_test_uncovered"]
            or coverage["complete_submission"] is not False):
        raise ValueError("官方测试覆盖数量不一致")
    prediction = pd.read_parquet(run_dir / "competition_test" / "合格记录预测.parquet", columns=[
        "ts_code", "trade_date", "selected_pred",
    ])
    expected = pd.read_parquet(index_path(config, "competition_test"), columns=[
        "ts_code", "target_date", "prediction_eligible",
    ])
    expected = expected.loc[expected["prediction_eligible"].eq(1), ["ts_code", "target_date"]]
    expected = expected.rename(columns={"target_date": "trade_date"})
    if len(prediction) != len(expected) or prediction.duplicated(["ts_code", "trade_date"]).any():
        raise ValueError("官方测试合格记录预测键数量不一致")
    merged = prediction[["ts_code", "trade_date"]].merge(
        expected, on=["ts_code", "trade_date"], how="outer", indicator=True,
        validate="one_to_one",
    )
    if not merged["_merge"].eq("both").all() or not np.isfinite(prediction["selected_pred"]).all():
        raise ValueError("官方测试预测键或数值与共同索引不一致")
    expected_outputs = ["周期指标.csv"]
    if holdout["metrics"][str(alpha)]["score"] is not None:
        expected_outputs.append("月度指标.csv")
    for filename in expected_outputs:
        if not (run_dir / "evaluation" / filename).exists():
            raise ValueError(f"最终评价产物缺失: {filename}")
    write_final_manifest(run_dir, {
        "status": "passed", "completed_candidates": 1,
        "completed_runs": 1, "completed_epochs": epochs,
        "coverage": coverage,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--operation", required=True, choices=("preflight", "final-audit"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise ValueError("核验需要可用的 CUDA 设备")
    config, config_sha256 = load_config(args.config)
    if args.operation == "preflight":
        run_preflight(config, config_sha256, Path(args.run_dir).resolve())
    else:
        run_final_audit(config, config_sha256, Path(args.run_dir).resolve())


if __name__ == "__main__":
    main()
