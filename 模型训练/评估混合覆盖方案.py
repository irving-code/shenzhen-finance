import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_first_lstm_result import evaluate_frame
from run_second_score_experiment import smooth_predictions


ROOT = Path(__file__).resolve().parents[1]
STRICT_RUN_DIR = ROOT / "模型训练" / "严格排序损失消融实验" / "20261010_rank_ablation" / "runs" / "rank_0.30" / "seed_42"
STRICT_LOCAL_PATH = STRICT_RUN_DIR / "evaluation" / "local_holdout" / "local_holdout_raw_predictions.parquet"
LGBM_LOCAL_PATH = ROOT / "模型训练" / "LightGBM实验结果" / "本地留出集预测.parquet"
LOCAL_INDEX_PATH = ROOT / "模型预处理" / "横截面扩展输入" / "20261008_横截面排序优化" / "样本索引_local_holdout.parquet"
OUTPUT_DIR = ROOT / "模型训练"
KEY_COLUMNS = ["ts_code", "trade_date"]
MIN_DAILY_OVERLAP = 30
STD_EPSILON = 1e-8
STRICT_ALPHA = 0.3


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def validate_unique(frame, name):
    if frame.duplicated(KEY_COLUMNS).any():
        raise ValueError(f"{name}存在重复股票日期键")
    if frame[KEY_COLUMNS].isna().any().any():
        raise ValueError(f"{name}的股票日期键存在缺失")


def finite_prediction(frame, column, name):
    values = frame[column].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{name}的预测值存在非有限值")


def build_calibration(overlap):
    strict_values = overlap["strict_pred"].to_numpy(dtype=np.float64)
    lgbm_values = overlap["lgbm_pred"].to_numpy(dtype=np.float64)
    global_record = {
        "overlap_count": int(len(overlap)),
        "lgbm_mean": float(lgbm_values.mean()),
        "lgbm_std": float(lgbm_values.std(ddof=0)),
        "strict_mean": float(strict_values.mean()),
        "strict_std": float(strict_values.std(ddof=0)),
    }
    if global_record["lgbm_std"] <= STD_EPSILON or global_record["strict_std"] <= STD_EPSILON:
        raise ValueError("全局校准样本的预测标准差过小")
    daily = {}
    for date, group in overlap.groupby("trade_date", sort=True):
        lgbm = group["lgbm_pred"].to_numpy(dtype=np.float64)
        strict = group["strict_pred"].to_numpy(dtype=np.float64)
        record = {
            "overlap_count": int(len(group)),
            "lgbm_mean": float(lgbm.mean()),
            "lgbm_std": float(lgbm.std(ddof=0)),
            "strict_mean": float(strict.mean()),
            "strict_std": float(strict.std(ddof=0)),
            "valid": bool(len(group) >= MIN_DAILY_OVERLAP and lgbm.std(ddof=0) > STD_EPSILON and strict.std(ddof=0) > STD_EPSILON),
        }
        daily[str(int(date))] = record
    return {"minimum_daily_overlap": MIN_DAILY_OVERLAP, "global": global_record, "daily": daily}


def apply_calibration(frame, calibration):
    result = frame.copy()
    global_record = calibration["global"]
    means_lgbm = []
    stds_lgbm = []
    means_strict = []
    stds_strict = []
    scopes = []
    for date in result["trade_date"].to_numpy(dtype=np.int64):
        daily = calibration["daily"].get(str(int(date)))
        selected = daily if daily is not None and daily["valid"] else global_record
        means_lgbm.append(selected["lgbm_mean"])
        stds_lgbm.append(selected["lgbm_std"])
        means_strict.append(selected["strict_mean"])
        stds_strict.append(selected["strict_std"])
        scopes.append("daily" if selected is daily else "global")
    lgbm = result["lgbm_pred"].to_numpy(dtype=np.float64)
    result["lgbm_pred_calibrated"] = np.asarray(means_strict) + np.asarray(stds_strict) * (lgbm - np.asarray(means_lgbm)) / np.asarray(stds_lgbm)
    result["calibration_scope"] = scopes
    finite_prediction(result, "lgbm_pred_calibrated", "校准后的 LightGBM")
    return result


def metric_row(name, frame, signal_column):
    frame = frame.copy()
    if "prediction_eligible" not in frame.columns:
        frame["prediction_eligible"] = 1
    score = evaluate_frame(frame, signal_column)
    return {
        "model": name,
        "rows": int(score["rows"]),
        "dates": int(score["dates"]),
        "rank_ic": float(score["rank_ic"]["mean"]),
        "icir": float(score["rank_ic"]["icir"]),
        "positive_rank_ic_ratio": float(score["rank_ic"]["positive_ratio"]),
        "annual_excess": float(score["top_decile"]["annual_excess"]),
        "annual_top_decile_return": float(score["top_decile"]["top1_annual_return"]),
        "mean_turnover": float(score["turnover"]["mean_turnover"]),
        "rank_ic_contribution": float(score["rank_ic"]["mean"] * 0.4),
        "annual_excess_contribution": float(score["top_decile"]["annual_excess"] * 0.3),
        "turnover_contribution": float((1.0 - score["turnover"]["mean_turnover"]) * 0.3),
        "final_score": float(score["final_score"]),
    }


def main():
    strict = pd.read_parquet(STRICT_LOCAL_PATH)
    strict = smooth_predictions(strict, STRICT_ALPHA)
    strict["pred"] = strict.pop("smooth_pred")
    lgbm = pd.read_parquet(LGBM_LOCAL_PATH)
    local_index = pd.read_parquet(LOCAL_INDEX_PATH, columns=["ts_code", "target_date", "y_ret_1d", "flag_limit_up"])
    local_index = local_index.rename(columns={"target_date": "trade_date"})
    validate_unique(strict, "严格排序本地预测")
    validate_unique(lgbm, "LightGBM本地预测")
    validate_unique(local_index, "本地留出集索引")
    finite_prediction(strict, "pred", "严格排序模型")
    finite_prediction(lgbm, "pred", "LightGBM")

    strict_eval = strict.merge(local_index, on=KEY_COLUMNS, how="left", suffixes=("", "_index"))
    if strict_eval[["y_ret_1d", "flag_limit_up"]].isna().any().any():
        raise ValueError("严格排序本地预测无法完整关联留出集标签")
    strict_eval = strict_eval.rename(columns={"pred": "strict_pred"})
    lgbm_eval = lgbm.merge(local_index, on=KEY_COLUMNS, how="left", suffixes=("_lgbm", "_index"))
    if lgbm_eval[["y_ret_1d_index", "flag_limit_up"]].isna().any().any():
        raise ValueError("LightGBM本地预测无法完整关联留出集标签")
    lgbm_eval = lgbm_eval.rename(columns={"pred": "lgbm_pred", "y_ret_1d_index": "y_ret_1d"})
    lgbm_eval = lgbm_eval.drop(columns=["y_ret_1d_lgbm"], errors="ignore")
    overlap = strict_eval[[*KEY_COLUMNS, "strict_pred"]].merge(lgbm_eval[[*KEY_COLUMNS, "lgbm_pred"]], on=KEY_COLUMNS, how="inner")
    if len(overlap) != len(strict):
        raise ValueError("两种模型的本地留出集重叠键数量不一致")
    calibration = build_calibration(overlap)
    calibrated = apply_calibration(lgbm_eval[[*KEY_COLUMNS, "lgbm_pred"]], calibration)
    hybrid = lgbm_eval[[*KEY_COLUMNS, "y_ret_1d", "flag_limit_up", "lgbm_pred"]].merge(
        strict_eval[[*KEY_COLUMNS, "strict_pred"]], on=KEY_COLUMNS, how="left"
    ).merge(calibrated[[*KEY_COLUMNS, "lgbm_pred_calibrated", "calibration_scope"]], on=KEY_COLUMNS, how="left")
    hybrid["pred"] = hybrid["strict_pred"].where(hybrid["strict_pred"].notna(), hybrid["lgbm_pred_calibrated"])
    hybrid["prediction_source"] = np.where(hybrid["strict_pred"].notna(), "strict_sort", "lgbm_fallback")
    if hybrid["pred"].isna().any():
        raise ValueError("混合模型存在缺失预测")
    finite_prediction(hybrid, "pred", "混合模型")

    strict_metric = metric_row("严格排序模型（有效序列）", strict_eval.rename(columns={"strict_pred": "pred"}), "pred")
    lgbm_metric = metric_row("LightGBM（完整留出集）", lgbm_eval.rename(columns={"lgbm_pred": "pred"}), "pred")
    hybrid_metric = metric_row("混合模型（严格排序主预测+LightGBM兜底）", hybrid, "pred")
    metrics = pd.DataFrame([strict_metric, lgbm_metric, hybrid_metric])
    metrics.to_csv(OUTPUT_DIR / "混合模型留出集评估.csv", index=False, encoding="utf-8-sig")
    hybrid.to_parquet(OUTPUT_DIR / "混合模型留出集预测.parquet", index=False, compression="zstd")
    write_json(OUTPUT_DIR / "混合模型校准参数.json", calibration)

    coverage = {
        "local_holdout_total_rows": int(len(lgbm_eval)),
        "strict_prediction_rows": int(len(strict_eval)),
        "lgbm_prediction_rows": int(len(lgbm_eval)),
        "strict_coverage_rate": float(len(strict_eval) / len(lgbm_eval)),
        "fallback_rows": int(hybrid["prediction_source"].eq("lgbm_fallback").sum()),
        "hybrid_rows": int(len(hybrid)),
        "hybrid_duplicate_keys": int(hybrid.duplicated(KEY_COLUMNS).sum()),
        "hybrid_null_predictions": int(hybrid["pred"].isna().sum()),
        "hybrid_infinite_predictions": int((~np.isfinite(hybrid["pred"].to_numpy(dtype=np.float64))).sum()),
        "calibration_overlap_rows": int(len(overlap)),
        "daily_calibration_rows": int(sum(item["valid"] for item in calibration["daily"].values())),
        "daily_global_fallback_rows": int((hybrid["calibration_scope"] == "global").sum()),
    }
    write_json(OUTPUT_DIR / "混合模型本地留出集审计.json", coverage)
    report_lines = [
        "# 混合覆盖模型实验报告",
        "",
        "## 实验设置",
        "",
        "本实验使用 2024 年本地留出集作为有标签判定数据。严格排序模型采用 `严格排序损失消融实验/20261010_rank_ablation/runs/rank_0.30/seed_42` 的第 2 周期、alpha=0.3 预测，配置为 37 因子、排序损失权重 0.3、关闭依赖惩罚。LightGBM 采用 `LightGBM实验结果/本地留出集预测.parquet`。两种模型不重新训练。",
        "",
        "预测校准仅使用两种模型在本地留出集的重叠键，按交易日计算均值和标准差；每日重叠数量少于 30 条时使用全局参数。混合模型对严格排序模型已有预测的记录保留严格排序预测，其余记录使用校准后的 LightGBM 预测。",
        "",
        "## 留出集覆盖审计",
        "",
        f"- 留出集完整 LightGBM 键数量：{coverage['local_holdout_total_rows']:,}。",
        f"- 严格排序模型预测数量：{coverage['strict_prediction_rows']:,}，覆盖率：{coverage['strict_coverage_rate']:.6%}。",
        f"- LightGBM 预测数量：{coverage['lgbm_prediction_rows']:,}。",
        f"- 混合模型兜底数量：{coverage['fallback_rows']:,}。",
        f"- 校准重叠样本数量：{coverage['calibration_overlap_rows']:,}，有效交易日参数数量：{coverage['daily_calibration_rows']}。",
        "",
        "## 2024 年本地留出集指标",
        "",
        "```text",
        metrics.to_string(index=False),
        "```",
        "",
        "综合分公式为 `0.4 × Rank IC + 0.3 × 年化超额收益 + 0.3 × (1 - 平均换手率)`。年度 Top10% 收益只作为评价指标，不作为训练标签。",
        "",
        "## 判定结果",
        "",
        f"混合模型完整覆盖 {coverage['hybrid_rows']:,} 条本地留出集记录，重复键为 {coverage['hybrid_duplicate_keys']}，缺失预测为 {coverage['hybrid_null_predictions']}，非有限预测为 {coverage['hybrid_infinite_predictions']}。三项审计均通过。混合模型本地综合分为 {hybrid_metric['final_score']:.8f}，LightGBM 为 {lgbm_metric['final_score']:.8f}，严格排序模型有效序列样本为 {strict_metric['final_score']:.8f}。",
        "",
        "测试集最终提交由后续构建脚本按完整测试集键表生成，并单独检查行数、唯一键、缺失预测和逐日数量。",
    ]
    (OUTPUT_DIR / "混合模型实验报告.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(metrics.to_string(index=False))
    print(json.dumps(coverage, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
