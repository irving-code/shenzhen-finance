import argparse
from functools import cmp_to_key
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch

from cross_section_experiment import (
    file_sha256, index_path, load_config, project_path, read_json, validate_completed_input, write_json,
)
from cross_section_input import CrossSectionInputReader
from dependence_constraint import dependence_penalty, replace_source_group, source_feature_indices
from evaluate_first_lstm_result import evaluate_frame
from lstm_transformer_small import LSTMTransformerRegressor
from run_second_score_experiment import smooth_predictions


def run_path(run_dir, group, rank_weight, seed):
    return Path(run_dir) / "runs" / group / f"lambda_{rank_weight:g}" / f"seed_{seed}"


def model_for_group(config, group, device):
    if group != config["model_id"] or config["feature_mode"] != "extended37":
        raise ValueError("模型标识或输入模式与单模型配置不一致")
    input_dim = 37
    return LSTMTransformerRegressor(input_dim=input_dim, **config["model"]).to(device)


def predict_split(config, model, split, device, batch_size=None):
    mode = "base27" if model.input_projection[0].in_features == 27 else "extended37"
    reader = CrossSectionInputReader(
        project_path(config, "base_metadata"),
        project_path(config, "input_dir") / "输入元数据.json",
        index_path(config, split), mode,
    )
    model.eval()
    chunks = []
    with torch.inference_mode():
        inference_batch_size = batch_size or config["training"]["inference_batch_size"]
        for batch in reader.iter_inference_batches(inference_batch_size):
            features = torch.from_numpy(batch["x"]).to(device=device, dtype=torch.float32)
            predictions = model(features).cpu().numpy().astype(np.float64)
            if not np.isfinite(predictions).all():
                raise ValueError(f"{split} 模型预测存在非有限值")
            chunks.append(pd.DataFrame({
                "ts_code": batch["ts_code"],
                "trade_date": batch["target_date"],
                "pred": predictions,
                "prediction_eligible": batch["prediction_eligible"],
            }))
    result = pd.concat(chunks, ignore_index=True)
    expected = config["data"]["prediction_eligible"][split]
    if len(result) != expected or result.duplicated(["ts_code", "trade_date"]).any():
        raise ValueError(f"{split} 预测数量或股票日期键不符合交付清单")
    return result


def attach_evaluation(config, prediction, split):
    if split == "competition_test":
        raise ValueError("官方测试集没有可用标签")
    index = pd.read_parquet(index_path(config, split), columns=[
        "ts_code", "target_date", "prediction_eligible", "selection_eligible",
        "y_ret_1d", "flag_limit_up",
    ])
    eligible = index.loc[index["prediction_eligible"].eq(1)].reset_index(drop=True)
    if len(prediction) != len(eligible):
        raise ValueError("评价预测数量与共同索引资格数量不一致")
    if not np.array_equal(prediction["ts_code"].astype(str).to_numpy(), eligible["ts_code"].astype(str).to_numpy()):
        raise ValueError("评价预测股票身份与共同索引不一致")
    if not np.array_equal(prediction["trade_date"].to_numpy(dtype=np.int64), eligible["target_date"].to_numpy(dtype=np.int64)):
        raise ValueError("评价预测日期与共同索引不一致")
    result = prediction.copy()
    result["y_ret_1d"] = eligible["y_ret_1d"].to_numpy(dtype=np.float64)
    result["flag_limit_up"] = eligible["flag_limit_up"].to_numpy(dtype=np.float64)
    result["selection_eligible"] = eligible["selection_eligible"].to_numpy(dtype=np.int8)
    if result["flag_limit_up"].isna().any():
        raise ValueError("评价涨停状态含缺失值")
    labels = result["y_ret_1d"].to_numpy(dtype=np.float64)
    if not np.isfinite(labels[~np.isnan(labels)]).all():
        raise ValueError("评价标签含非有限值")
    return result


def daily_metrics(frame, signal_column):
    rows = []
    previous = None
    for date, group in frame.groupby("trade_date", sort=True):
        valid_ic = group.loc[group["y_ret_1d"].notna()]
        ic = float("nan")
        if len(valid_ic) >= 30:
            if valid_ic[signal_column].nunique() < 2 or valid_ic["y_ret_1d"].nunique() < 2:
                raise ValueError(f"{date} 的 Spearman 相关系数无法定义")
            ic = float(spearmanr(valid_ic[signal_column], valid_ic["y_ret_1d"]).statistic)
            if not math.isfinite(ic):
                raise ValueError(f"{date} 的 Rank IC 非有限")
        valid_return = group.loc[group["flag_limit_up"].eq(0) & group["y_ret_1d"].notna()]
        top_return = market_return = excess = float("nan")
        top_count = 0
        if len(valid_return) >= 100:
            ordered = valid_return.sort_values(signal_column, ascending=False)
            top_count = max(len(ordered) // 10, 1)
            top_return = float(ordered["y_ret_1d"].iloc[:top_count].mean())
            market_return = float(ordered["y_ret_1d"].mean())
            excess = top_return - market_return
        valid_turnover = group.loc[group["flag_limit_up"].eq(0)]
        turnover = float("nan")
        if len(valid_turnover) >= 100:
            ordered = valid_turnover.sort_values(signal_column, ascending=False)
            count = max(len(ordered) // 10, 1)
            current = set(ordered["ts_code"].iloc[:count])
            if previous is not None:
                turnover = 1.0 - len(current & previous) / len(current | previous)
            previous = current
        else:
            previous = None
        rows.append({
            "trade_date": int(date), "rank_ic": ic, "top_return": top_return,
            "market_return": market_return, "excess_return": excess,
            "turnover": turnover, "sample_count": len(group),
            "ic_count": len(valid_ic), "return_count": len(valid_return),
            "turnover_count": len(valid_turnover), "top_count": top_count,
        })
    return pd.DataFrame(rows)


def strict_score(frame, signal_column):
    if frame.empty or frame.duplicated(["ts_code", "trade_date"]).any():
        raise ValueError("评价记录为空或存在重复股票日期键")
    if not np.isfinite(frame[signal_column].to_numpy(dtype=np.float64)).all():
        raise ValueError("评价预测包含非有限值")
    daily = daily_metrics(frame, signal_column)
    if daily["rank_ic"].notna().sum() < 2:
        raise ValueError("有效 Rank IC 日期不足两个")
    score = evaluate_frame(frame, signal_column)
    checks = (
        (float(daily["rank_ic"].mean()), score["rank_ic"]["mean"]),
        (float(daily["excess_return"].mean() * 252), score["top_decile"]["annual_excess"]),
        (float(daily["turnover"].mean()), score["turnover"]["mean_turnover"]),
    )
    if any(not np.isclose(a, b, rtol=1e-10, atol=1e-12) for a, b in checks):
        raise ValueError("逐日指标与统一评价入口不一致")
    return score, daily


def prediction_quality(config, frame, expected_dates):
    rows = []
    resolution_dates = 0
    constant_dates = 0
    warning_dates = 0
    for date, group in frame.groupby("trade_date", sort=True):
        labeled = group.loc[group["y_ret_1d"].notna()]
        if len(labeled) < config["quality"]["minimum_ic_rows"]:
            raise ValueError(f"{date} 的质量核验标签数量不足")
        prediction = labeled["pred"].to_numpy(dtype=np.float64)
        label = labeled["y_ret_1d"].to_numpy(dtype=np.float64)
        prediction_std = float(prediction.std(ddof=0))
        label_std = float(label.std(ddof=0))
        if label_std == 0.0:
            raise ValueError(f"{date} 的真实标签为常数")
        unique_count = int(np.unique(labeled["pred"].to_numpy(dtype=np.float32)).size)
        tie_ratio = 1.0 - unique_count / len(labeled)
        span = float(prediction.max() - prediction.min())
        magnitude = float(np.abs(prediction).max())
        ulp = abs(float(np.spacing(np.float32(magnitude))))
        resolution_limited = span <= config["quality"]["resolution_ulp_multiplier"] * ulp
        constant_prediction = unique_count < 2
        std_ratio = prediction_std / label_std
        warning = (
            std_ratio < config["quality"]["std_ratio_warning_below"]
            or tie_ratio > config["quality"]["tie_ratio_warning_above"]
        )
        constant_dates += int(constant_prediction)
        resolution_dates += int(resolution_limited)
        warning_dates += int(warning)
        rows.append({
            "trade_date": int(date), "rows": len(labeled),
            "pred_std": prediction_std, "label_std": label_std,
            "std_ratio": std_ratio, "unique_count": unique_count,
            "tie_ratio": tie_ratio, "prediction_span": span,
            "output_magnitude": magnitude, "ulp": ulp,
            "resolution_limited": resolution_limited,
            "constant_prediction": constant_prediction, "warning": warning,
        })
    if len(rows) != expected_dates:
        raise ValueError("预测质量日期数量与配置不一致")
    fraction = resolution_dates / len(rows)
    passed = (
        constant_dates == 0
        and fraction < config["quality"]["resolution_invalid_date_fraction_at_least"]
    )
    return {
        "status": "passed" if passed else "invalid",
        "passed": passed, "dates": len(rows),
        "constant_prediction_dates": constant_dates,
        "resolution_limited_dates": resolution_dates,
        "resolution_limited_fraction": fraction,
        "warning_dates": warning_dates, "daily": rows,
    }


def evaluate_prediction(config, prediction, split, alphas, output_dir, selection_only=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction.to_parquet(output_dir / f"{split}_raw_predictions.parquet", index=False, compression="zstd")
    evaluation = attach_evaluation(config, prediction, split)
    quality_frame = evaluation.loc[evaluation["selection_eligible"].eq(1)].copy() if selection_only else evaluation
    quality_dates = config["data"]["selection_dates"] if selection_only else int(quality_frame["trade_date"].nunique())
    quality = prediction_quality(config, quality_frame, quality_dates)
    write_json(output_dir / f"{split}_raw_prediction_quality.json", quality)
    results = {}
    for alpha in alphas:
        if not quality["passed"]:
            results[str(alpha)] = {"score": None, "quality_pass": False, "quality": quality}
            continue
        smoothed = smooth_predictions(evaluation, alpha)
        selected = smoothed.loc[smoothed["selection_eligible"].eq(1)].copy() if selection_only else smoothed
        if selection_only and len(selected) != config["data"]["selection_eligible"]:
            raise ValueError("验证选择资格数量不一致")
        undefined_dates = [
            int(date) for date, daily in selected.groupby("trade_date", sort=True)
            if len(daily.loc[daily["y_ret_1d"].notna()]) >= config["quality"]["minimum_ic_rows"]
            and daily.loc[daily["y_ret_1d"].notna(), "smooth_pred"].nunique() < 2
        ]
        if undefined_dates:
            results[str(alpha)] = {
                "score": None, "quality_pass": False, "quality": quality,
                "undefined_correlation_dates": undefined_dates,
            }
            continue
        score, daily = strict_score(selected, "smooth_pred")
        daily.to_parquet(output_dir / f"{split}_alpha_{alpha:g}_daily.parquet", index=False, compression="zstd")
        target = selected["y_ret_1d"].to_numpy(dtype=np.float64)
        prediction_values = selected["smooth_pred"].to_numpy(dtype=np.float64)
        valid = np.isfinite(target)
        error = prediction_values[valid] - target[valid]
        beta = config["ranking"]["beta"]
        smooth_l1 = np.where(np.abs(error) < beta, error * error / (2 * beta), np.abs(error) - beta / 2)
        results[str(alpha)] = {
            "score": score, "quality_pass": True, "quality": quality,
            "smooth_l1": float(smooth_l1.mean()),
            "mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(np.square(error).mean())),
            "pred_mean": float(prediction_values.mean()), "pred_std": float(prediction_values.std()),
            "pred_p001": float(np.quantile(prediction_values, 0.001)),
            "pred_p999": float(np.quantile(prediction_values, 0.999)),
            "pred_tie_ratio": float(1.0 - selected["smooth_pred"].nunique() / len(selected)),
            "rows": len(selected), "dates": int(selected["trade_date"].nunique()),
        }
    write_json(output_dir / f"{split}_metrics.json", results)
    return results


def evaluate_validation(config, run_dir, group, rank_weight, seed, epoch, model, device):
    start = time.monotonic()
    output = run_path(run_dir, group, rank_weight, seed) / f"epoch_{epoch:02d}"
    prediction = predict_split(config, model, "validation", device)
    metrics = evaluate_prediction(
        config, prediction, "validation", [config["selection"]["alpha"]], output,
        selection_only=True,
    )
    write_json(output / "validation_runtime.json", {"elapsed_seconds": time.monotonic() - start})
    return metrics


def better(candidate, current, tolerance):
    if current is None:
        return True
    for key in ("final_score", "rank_ic", "annual_excess"):
        difference = candidate[key] - current[key]
        if abs(difference) > tolerance:
            return difference > 0
    for key, reverse in (("epoch", False), ("rank_weight", False), ("alpha", True)):
        if candidate[key] != current[key]:
            return candidate[key] > current[key] if reverse else candidate[key] < current[key]
    return candidate["group"] < current["group"]


def compare_candidates(left, right, tolerance):
    if better(left, right, tolerance):
        return -1
    if better(right, left, tolerance):
        return 1
    return 0


def top_portfolio(frame, signal, portfolio):
    if portfolio == "return":
        eligible = frame.loc[frame["flag_limit_up"].eq(0) & frame["y_ret_1d"].notna()]
    elif portfolio == "turnover":
        eligible = frame.loc[frame["flag_limit_up"].eq(0)]
    else:
        raise ValueError("未知稳定性持仓类型")
    if len(eligible) < 100:
        return None
    ordered = eligible.sort_values(signal, ascending=False, kind="mergesort")
    count = max(len(ordered) // 10, 1)
    return set(ordered["ts_code"].iloc[:count])


def overlap_ratio(left, right):
    if left is None or right is None or len(left) != len(right) or not left:
        return None
    return len(left.intersection(right)) / len(left)


def evaluate_stability_seed(config, run_dir, group, rank_weight, seed, epoch, device):
    model = load_checkpoint_model(config, run_dir, group, rank_weight, seed, epoch, device)
    batch_sizes = config["quality"]["stability"]["batch_sizes"]
    predictions = [
        predict_split(config, model, "validation", device, batch_size)
        for batch_size in batch_sizes
    ]
    base = attach_evaluation(config, predictions[0], "validation")
    alternate = attach_evaluation(config, predictions[1], "validation")
    if not np.array_equal(base[["ts_code", "trade_date"]].to_numpy(), alternate[["ts_code", "trade_date"]].to_numpy()):
        raise ValueError("双批量预测股票日期身份不一致")
    selection = base.loc[base["selection_eligible"].eq(1)].reset_index(drop=True)
    alternate = alternate.loc[alternate["selection_eligible"].eq(1)].reset_index(drop=True)
    if len(selection) != config["data"]["selection_eligible"]:
        raise ValueError("双批量预测选择资格数量不一致")
    raw_column_a = "pred"
    raw_column_b = "pred_alternate"
    selection[raw_column_b] = alternate["pred"].to_numpy(dtype=np.float64)
    alpha_predictions = {}
    for alpha in [config["selection"]["alpha"]]:
        smooth_left = smooth_predictions(base, alpha)
        smooth_right = smooth_predictions(alternate, alpha)
        alpha_predictions[str(alpha)] = (
            smooth_left.loc[smooth_left["selection_eligible"].eq(1), "smooth_pred"].to_numpy(dtype=np.float64),
            smooth_right.loc[smooth_right["selection_eligible"].eq(1), "smooth_pred"].to_numpy(dtype=np.float64),
        )
    alpha_results = {}
    settings = config["quality"]["stability"]
    for alpha in [config["selection"]["alpha"]]:
        left_values, right_values = alpha_predictions[str(alpha)]
        daily_results = []
        for date, group_frame in selection.groupby("trade_date", sort=True):
            positions = group_frame.index.to_numpy(dtype=np.int64)
            valid = group_frame["y_ret_1d"].notna().to_numpy()
            raw_left = group_frame[raw_column_a].to_numpy(dtype=np.float64)
            raw_right = group_frame[raw_column_b].to_numpy(dtype=np.float64)
            raw_spearman = None
            if valid.sum() >= config["quality"]["minimum_ic_rows"] and np.unique(raw_left[valid]).size >= 2 and np.unique(raw_right[valid]).size >= 2:
                raw_spearman = float(spearmanr(raw_left[valid], raw_right[valid]).statistic)
            chosen_left = left_values[positions]
            chosen_right = right_values[positions]
            selected_spearman = None
            if valid.sum() >= config["quality"]["minimum_ic_rows"] and np.unique(chosen_left[valid]).size >= 2 and np.unique(chosen_right[valid]).size >= 2:
                selected_spearman = float(spearmanr(chosen_left[valid], chosen_right[valid]).statistic)
            left_frame = group_frame.copy()
            right_frame = group_frame.copy()
            left_frame["raw_signal"] = raw_left
            right_frame["raw_signal"] = raw_right
            left_frame["selected_signal"] = chosen_left
            right_frame["selected_signal"] = chosen_right
            overlaps = {}
            for signal_kind, column_left, column_right in (
                ("raw", "raw_signal", "raw_signal"),
                ("selected", "selected_signal", "selected_signal"),
            ):
                for portfolio in settings["required_portfolios"]:
                    left_set = top_portfolio(left_frame, column_left, portfolio)
                    right_set = top_portfolio(right_frame, column_right, portfolio)
                    overlaps[f"{signal_kind}_{portfolio}"] = overlap_ratio(left_set, right_set)
            passed = (
                raw_spearman is not None
                and selected_spearman is not None
                and raw_spearman >= settings["minimum_daily_spearman"]
                and selected_spearman >= settings["minimum_daily_spearman"]
                and all(value is not None and value >= settings["minimum_top_decile_overlap"] for value in overlaps.values())
            )
            daily_results.append({
                "trade_date": int(date), "raw_spearman": raw_spearman,
                "selected_spearman": selected_spearman,
                "overlaps": overlaps, "passed": passed,
            })
        passing_dates = sum(item["passed"] for item in daily_results)
        required_dates = math.ceil(settings["minimum_passing_date_fraction"] * len(daily_results))
        alpha_results[str(alpha)] = {
            "passing_dates": passing_dates, "total_dates": len(daily_results),
            "required_passing_dates": required_dates,
            "passed": passing_dates >= required_dates,
            "daily": daily_results,
        }
    return {"seed": seed, "epoch": epoch, "batch_sizes": batch_sizes, "alphas": alpha_results}


def evaluate_stability_epoch(config, run_dir, group, rank_weight, epoch, device, output_path=None):
    first = run_path(run_dir, group, rank_weight, config["seed"]) / f"epoch_{epoch:02d}"
    results = {
        "group": group, "rank_weight": rank_weight, "epoch": epoch,
        "config_sha256": read_json(Path(run_dir) / "配置快照.json")["config_sha256"],
        "input_sha256": read_json(Path(run_dir) / "配置快照.json")["input_sha256"],
        "batch_sizes": config["quality"]["stability"]["batch_sizes"],
        "seeds": {},
    }
    seed = config["seed"]
    results["seeds"][str(seed)] = evaluate_stability_seed(
        config, run_dir, group, rank_weight, seed, epoch, device,
    )
    write_json(output_path or first / "stability_metrics.json", results)
    return results


def load_or_evaluate_stability_epoch(config, run_dir, group, rank_weight, epoch, device):
    path = run_path(run_dir, group, rank_weight, config["seed"]) / f"epoch_{epoch:02d}" / "stability_metrics.json"
    if not path.exists():
        return evaluate_stability_epoch(config, run_dir, group, rank_weight, epoch, device)
    report = read_json(path)
    snapshot = read_json(Path(run_dir) / "配置快照.json")
    if (
        report["group"] != group
        or report["rank_weight"] != rank_weight
        or report["epoch"] != epoch
        or report["config_sha256"] != snapshot["config_sha256"]
        or report["input_sha256"] != snapshot["input_sha256"]
        or report["batch_sizes"] != config["quality"]["stability"]["batch_sizes"]
        or set(report["seeds"]) != {str(config["seed"])}
    ):
        raise ValueError("已保存的双批量稳定性核验身份不一致")
    return report


def select_configurations(config, run_dir, device):
    run_dir = Path(run_dir)
    if not read_json(run_dir / "调度状态.json")["stopped"]:
        raise ValueError("模型选择需要等待正式训练结束")
    selection_path = run_dir / "选定配置.json"
    if selection_path.exists():
        return read_json(selection_path)["final_candidate"]
    group = config["model_id"]
    rank_weight = config["ranking"]["weight"]
    seed = config["seed"]
    alpha = config["selection"]["alpha"]
    candidates = []
    checks = []
    first = run_path(run_dir, group, rank_weight, seed)
    epoch_directories = sorted(first.glob("epoch_[0-9][0-9]"))
    if not epoch_directories:
        raise ValueError("单模型训练没有完整验证周期")
    for directory in epoch_directories:
        epoch = int(directory.name.split("_")[1])
        metrics = read_json(first / directory.name / "validation_metrics.json")[str(alpha)]
        if metrics.get("quality_pass") is not True or metrics.get("score") is None:
            checks.append({"epoch": epoch, "status": "prediction_quality_failed"})
            continue
        score = metrics["score"]
        candidates.append({
            "group": group, "rank_weight": rank_weight, "seed": seed,
            "epoch": epoch, "alpha": alpha,
            "final_score": float(score["final_score"]),
            "rank_ic": float(score["rank_ic"]["mean"]),
            "annual_excess": float(score["top_decile"]["annual_excess"]),
        })
    comparison = pd.DataFrame(candidates)
    comparison.to_csv(run_dir / "验证周期指标.csv", index=False, encoding="utf-8-sig")
    ordered = sorted(
        candidates,
        key=cmp_to_key(lambda left, right: compare_candidates(
            left, right, config["selection"]["score_tolerance"],
        )),
    )
    choice = None
    for candidate in ordered:
        epoch = candidate["epoch"]
        epoch_dir = first / f"epoch_{epoch:02d}"
        dependence_path = epoch_dir / "dependence_metrics.json"
        if dependence_path.exists():
            dependence_record = read_json(dependence_path)
            if str(seed) not in dependence_record or dependence_record[str(seed)]["epoch"] != epoch:
                raise ValueError("已保存的因子依赖核验身份不一致")
        else:
            dependence_record = {
                str(seed): evaluate_dependence_epoch(
                    config, run_dir, group, rank_weight, seed, epoch, device,
                )
            }
            write_json(dependence_path, dependence_record)
        if dependence_record[str(seed)]["passed"] is not True:
            checks.append({"epoch": epoch, "status": "dependence_failed"})
            write_json(run_dir / "选择核验记录.json", checks)
            continue
        stability = load_or_evaluate_stability_epoch(
            config, run_dir, group, rank_weight, epoch, device,
        )
        if stability["seeds"][str(seed)]["alphas"][str(alpha)]["passed"] is not True:
            checks.append({"epoch": epoch, "status": "stability_failed"})
            write_json(run_dir / "选择核验记录.json", checks)
            continue
        choice = candidate
        checks.append({"epoch": epoch, "status": "passed"})
        break
    write_json(run_dir / "选择核验记录.json", checks)
    if choice is None:
        write_json(run_dir / "选定配置.json", {
            "status": "no_eligible_configuration", "final_candidate": None,
        })
        return None
    write_json(run_dir / "选定配置.json", {
        "status": "selected", "final_candidate": choice,
    })
    return choice


def evaluate_dependence_epoch(config, run_dir, group, rank_weight, seed, epoch, device, split="validation"):
    if group != config["dependence"]["enabled_model"]:
        raise ValueError("因子依赖核验的模型标识不一致")
    reader = CrossSectionInputReader(
        project_path(config, "base_metadata"),
        project_path(config, "input_dir") / "输入元数据.json",
        index_path(config, split), "extended37",
    )
    if split not in ("validation", "local_holdout"):
        raise ValueError("依赖核验只支持验证集与本地留出集")
    eligibility = "selection_eligible" if split == "validation" else "prediction_eligible"
    index = pd.read_parquet(index_path(config, split), columns=[eligibility, "target_date"])
    positions = index.index[index[eligibility].eq(1)].to_numpy(dtype=np.int64)
    expected_rows = config["data"]["selection_eligible"] if split == "validation" else config["data"]["prediction_eligible"][split]
    if len(positions) != expected_rows:
        raise ValueError("依赖程度核验日期范围与选择资格数量不一致")
    checkpoint_path = run_path(run_dir, group, rank_weight, seed) / f"checkpoint_epoch_{epoch:02d}.pt"
    model = load_checkpoint_model(config, run_dir, group, rank_weight, seed, epoch, device)
    model.eval()
    neutral_record = read_json(Path(run_dir) / "dependence_neutral_values.json")
    if neutral_record["config_sha256"] != read_json(Path(run_dir) / "配置快照.json")["config_sha256"]:
        raise ValueError("依赖程度中性值配置摘要不一致")
    neutral_values = np.asarray(neutral_record["values"], dtype=np.float32)
    groups = source_feature_indices(config)
    dates = index["target_date"].to_numpy(dtype=np.int64)[positions]
    predictions = np.empty(len(positions), dtype=np.float64)
    replaced_predictions = np.empty((len(groups), len(positions)), dtype=np.float64)
    batch_size = config["training"]["inference_batch_size"]
    with torch.inference_mode():
        for start in range(0, len(positions), batch_size):
            stop = min(start + batch_size, len(positions))
            batch = reader.read_batch(positions[start:stop])
            features = torch.from_numpy(batch["x"]).to(device=device, dtype=torch.float32)
            original = model(features)
            predictions[start:stop] = original.cpu().numpy().astype(np.float64)
            for group_index, indices in enumerate(groups):
                changed = replace_source_group(features, indices, neutral_values)
                replaced_predictions[group_index, start:stop] = model(changed).cpu().numpy().astype(np.float64)
    records = {}
    for group_index, source_group in enumerate(groups):
        daily = []
        for date in np.unique(dates):
            mask = dates == date
            original = predictions[mask]
            changed = replaced_predictions[group_index, mask]
            original_std = float(original.std(ddof=0))
            change_std = float((original - changed).std(ddof=0))
            ratio = change_std / max(original_std, config["dependence"]["denominator_floor"])
            if not math.isfinite(ratio):
                raise ValueError("依赖程度逐日结果包含非有限值")
            daily.append({
                "trade_date": int(date), "ratio": ratio,
                "original_std": original_std, "change_std": change_std,
                "passed": ratio <= config["dependence"]["limit"],
            })
        pass_count = sum(item["passed"] for item in daily)
        required_dates = math.ceil(config["dependence"]["minimum_passing_date_fraction"] * len(daily))
        records[str(group_index)] = {
            "features": source_group,
            "passing_dates": pass_count,
            "total_dates": len(daily),
            "required_passing_dates": required_dates,
            "mean_ratio": float(np.mean([item["ratio"] for item in daily])),
            "daily": daily,
            "passed": pass_count >= required_dates,
        }
    return {
        "seed": seed, "epoch": epoch, "split": split,
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "config_sha256": neutral_record["config_sha256"],
        "input_sha256": neutral_record["input_sha256"],
        "passed": all(item["passed"] for item in records.values()),
        "source_groups": records,
    }


def evaluate_dependence(config, run_dir, device):
    group = config["model_id"]
    rank_weight = config["ranking"]["weight"]
    first = run_path(run_dir, group, rank_weight, config["seed"])
    epochs = sorted(path.name for path in first.glob("epoch_[0-9][0-9]"))
    if not epochs:
        raise ValueError("单模型缺少已完成验证的周期")
    for name in epochs:
        epoch = int(name.split("_")[1])
        metrics = evaluate_dependence_epoch(
            config, run_dir, group, rank_weight, config["seed"], epoch, device,
        )
        write_json(first / name / "dependence_metrics.json", {str(config["seed"]): metrics})


def load_checkpoint_model(config, run_dir, group, rank_weight, seed, epoch, device):
    path = run_path(run_dir, group, rank_weight, seed) / f"checkpoint_epoch_{epoch:02d}.pt"
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    snapshot = read_json(Path(run_dir) / "配置快照.json")
    expected = {
        "config_sha256": snapshot["config_sha256"], "input_sha256": snapshot["input_sha256"],
        "group": group, "rank_weight": rank_weight, "seed": seed, "epoch": epoch, "input_dim": 37,
    }
    if any(checkpoint[key] != value for key, value in expected.items()):
        raise ValueError("评价检查点身份与当前运行不一致")
    model = model_for_group(config, group, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def evaluate_holdout(config, run_dir, device):
    run_dir = Path(run_dir)
    choice = read_json(Path(run_dir) / "选定配置.json")["final_candidate"]
    if choice is None:
        raise ValueError("留出评价需要通过核验的模型")
    report_path = run_dir / "evaluation" / "本地留出集指标.json"
    if report_path.exists():
        report = read_json(report_path)
        if (report["epoch"], report["seed"], report["alpha"]) != (choice["epoch"], config["seed"], choice["alpha"]):
            raise ValueError("已保存的留出评价身份不一致")
        return report
    model = load_checkpoint_model(
        config, run_dir, choice["group"], choice["rank_weight"],
        config["seed"], choice["epoch"], device,
    )
    evaluation_dir = Path(run_dir) / "evaluation" / "local_holdout"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = evaluation_dir / "local_holdout_raw_predictions.parquet"
    prediction = pd.read_parquet(prediction_path) if prediction_path.exists() else predict_split(config, model, "local_holdout", device)
    metrics = evaluate_prediction(
        config, prediction, "local_holdout", [1.0, choice["alpha"]], evaluation_dir,
    )
    dependence = evaluate_dependence_epoch(
        config, run_dir, choice["group"], choice["rank_weight"], config["seed"],
        choice["epoch"], device, split="local_holdout",
    )
    write_json(evaluation_dir / "dependence_metrics.json", dependence)
    report = {
        "model_id": config["model_id"], "seed": config["seed"],
        "epoch": choice["epoch"], "alpha": choice["alpha"],
        "metrics": metrics,
    }
    write_json(report_path, report)
    return report


def evaluate_competition_test(config, run_dir, device):
    chosen = read_json(Path(run_dir) / "选定配置.json")["final_candidate"]
    if chosen is None:
        raise ValueError("测试预测需要通过核验的模型")
    model = load_checkpoint_model(
        config, run_dir, chosen["group"], chosen["rank_weight"],
        config["selection"]["inference_seed"], chosen["epoch"], device,
    )
    prediction = predict_split(config, model, "competition_test", device)
    smoothed = smooth_predictions(prediction, chosen["alpha"])
    output = Path(run_dir) / "competition_test"
    output.mkdir(parents=True, exist_ok=True)
    smoothed.rename(columns={"smooth_pred": "selected_pred"}).to_parquet(
        output / "合格记录预测.parquet", index=False, compression="zstd"
    )
    missing = pd.read_parquet(project_path(config, "input_dir") / "官方测试覆盖缺失键.parquet")
    if len(missing) != config["data"]["competition_test_uncovered"]:
        raise ValueError("官方测试覆盖缺失键数量不一致")
    missing.to_parquet(output / "覆盖缺失键.parquet", index=False, compression="zstd")
    write_json(output / "覆盖审计.json", {
        "all_rows": config["data"]["rows"]["competition_test"],
        "eligible_predictions": len(smoothed), "uncovered_rows": len(missing),
        "complete_submission": False,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--operation", required=True, choices=("select", "stability", "dependence", "holdout", "competition-test"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--output")
    args = parser.parse_args()
    config, config_sha256 = load_config(args.config)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    run_dir = Path(args.run_dir).resolve()
    if read_json(run_dir / "配置快照.json")["config_sha256"] != config_sha256:
        raise ValueError("实验配置摘要不一致")
    validate_completed_input(config)
    if args.device != "cuda" or not torch.cuda.is_available():
        raise ValueError("正式评价需要可用的 CUDA 设备")
    device = torch.device(args.device)
    if args.operation == "select":
        select_configurations(config, run_dir, device)
    elif args.operation == "stability":
        if args.epoch is None:
            raise ValueError("稳定性核验需要 --epoch")
        rank_weight = config["ranking"]["weight"]
        evaluate_stability_epoch(
            config, run_dir, config["model_id"], rank_weight, args.epoch, device,
            Path(args.output).resolve() if args.output else None,
        )
    else:
        if args.operation == "dependence":
            evaluate_dependence(config, run_dir, device)
        elif args.operation == "holdout":
            evaluate_holdout(config, run_dir, device)
        else:
            evaluate_competition_test(config, run_dir, device)


if __name__ == "__main__":
    main()
