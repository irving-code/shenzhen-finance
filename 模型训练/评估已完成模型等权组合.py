import json
from pathlib import Path

import numpy as np
import pandas as pd

from cross_section_experiment import read_json
from evaluate_cross_section_rank import attach_evaluation
from evaluate_first_lstm_result import evaluate_frame
from run_second_score_experiment import smooth_predictions


ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT = ROOT / "模型训练" / "严格排序损失消融实验" / "20261010_rank_ablation"
CONFIG_PATH = EXPERIMENT / "configs" / "rank_0.00_seed_42.json"
OUTPUT_DIR = EXPERIMENT / "组合初步评估"
SEEDS = (42, 43, 44)
ALPHA = 0.3


def score_record(score):
    rank_ic = float(score["rank_ic"]["mean"])
    annual_excess = float(score["top_decile"]["annual_excess"])
    mean_turnover = float(score["turnover"]["mean_turnover"])
    return {
        "rank_ic": rank_ic,
        "rank_ic_std": float(score["rank_ic"]["std"]),
        "icir": float(score["rank_ic"]["icir"]),
        "positive_rank_ic_ratio": float(score["rank_ic"]["positive_ratio"]),
        "rank_ic_days": int(score["rank_ic"]["days"]),
        "annual_excess": annual_excess,
        "annual_top_decile_return": float(score["top_decile"]["top1_annual_return"]),
        "top_decile_days": int(score["top_decile"]["days"]),
        "mean_turnover": mean_turnover,
        "one_minus_turnover": float(score["turnover"]["one_minus_turnover"]),
        "turnover_transition_days": int(score["turnover"]["transition_days"]),
        "rank_ic_contribution": 0.4 * rank_ic,
        "annual_excess_contribution": 0.3 * annual_excess,
        "turnover_contribution": 0.3 * (1.0 - mean_turnover),
        "final_score": float(score["final_score"]),
        "rows": int(score["rows"]),
        "dates": int(score["dates"]),
    }


def validate_prediction(left, right, seed):
    if len(left) != len(right):
        raise ValueError(f"seed {seed} 预测行数不一致")
    if not np.array_equal(left[["ts_code", "trade_date"]].to_numpy(), right[["ts_code", "trade_date"]].to_numpy()):
        raise ValueError(f"seed {seed} 预测股票日期键不一致")
    if not np.isfinite(right["pred"].to_numpy(dtype=np.float64)).all():
        raise ValueError(f"seed {seed} 预测存在非有限值")


def main():
    config = read_json(CONFIG_PATH)
    prediction_frames = []
    for seed in SEEDS:
        path = EXPERIMENT / "runs" / "rank_0.00" / f"seed_{seed}" / "evaluation" / "local_holdout" / "local_holdout_raw_predictions.parquet"
        if not path.exists():
            raise FileNotFoundError(f"缺少 seed {seed} 的留出集预测：{path}")
        frame = pd.read_parquet(path, columns=["ts_code", "trade_date", "pred", "prediction_eligible"])
        if frame.duplicated(["ts_code", "trade_date"]).any():
            raise ValueError(f"seed {seed} 预测存在重复股票日期键")
        if not prediction_frames:
            if not np.isfinite(frame["pred"].to_numpy(dtype=np.float64)).all():
                raise ValueError(f"seed {seed} 预测存在非有限值")
            prediction_frames.append(frame)
        else:
            validate_prediction(prediction_frames[0], frame, seed)
            prediction_frames.append(frame)

    base = prediction_frames[0][["ts_code", "trade_date", "prediction_eligible"]].copy()
    individual_scores = {}
    for seed, frame in zip(SEEDS, prediction_frames):
        evaluation = attach_evaluation(config, frame, "local_holdout")
        smoothed = smooth_predictions(evaluation, ALPHA)
        individual_scores[f"rank_0.00_seed_{seed}"] = score_record(evaluate_frame(smoothed, "smooth_pred"))

    for seed, frame in zip(SEEDS, prediction_frames):
        base[f"rank_seed_{seed}"] = frame["pred"].to_numpy(dtype=np.float64)
        base[f"rank_seed_{seed}"] = base.groupby("trade_date")[f"rank_seed_{seed}"].rank(method="average", pct=True)
    rank_columns = [f"rank_seed_{seed}" for seed in SEEDS]
    base["pred"] = base[rank_columns].mean(axis=1)
    combined_prediction = base[["ts_code", "trade_date", "pred", "prediction_eligible"]]
    combined_evaluation = attach_evaluation(config, combined_prediction, "local_holdout")
    combined_smoothed = smooth_predictions(combined_evaluation, ALPHA)
    combined_score = score_record(evaluate_frame(combined_smoothed, "smooth_pred"))
    historical_path = ROOT / "模型训练" / "去除因子依赖惩罚实验" / "20261009_local_single_no_dependence" / "evaluation" / "local_holdout" / "local_holdout_metrics.json"
    historical_score = score_record(read_json(historical_path)["0.3"]["score"])
    if (historical_score["rows"], historical_score["dates"]) != (combined_score["rows"], combined_score["dates"]):
        raise ValueError("历史模型与组合的留出集覆盖不一致")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    combined_prediction.to_parquet(OUTPUT_DIR / "三模型横截面排序等权组合预测.parquet", index=False, compression="zstd")
    result = {
        "status": "completed",
        "method": "rank_normalized_equal_weight",
        "components": [f"rank_0.00_seed_{seed}" for seed in SEEDS],
        "rank_normalization": "每个交易日横截面百分位排序",
        "alpha": ALPHA,
        "evaluation_split": "local_holdout",
        "component_scores": individual_scores,
        "combined_score": combined_score,
        "historical_model_score": historical_score,
    }
    (OUTPUT_DIR / "三模型横截面排序等权组合指标.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# 三模型横截面排序等权组合初步评估",
        "",
        "组合由 `rank_0.00` 的随机种子 42、43、44 三个模型构成。每个交易日先对三个模型分别做横截面百分位排序，再取等权平均，最后按现有 `alpha=0.3` 规则平滑。评价区间为 2024 年本地留出集。",
        "",
        "## 细分指标与贡献度",
        "",
        "| 模型 | Rank IC | ICIR | 正 Rank IC 比例 | 年化超额收益 | Top10%年化组合收益 | 平均换手率 | Rank IC贡献 | 收益贡献 | 换手率贡献 | 综合分 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    records = list(individual_scores.items()) + [("三模型等权组合", combined_score), ("既有37因子模型", historical_score)]
    for name, score in records:
        lines.append(
            f"| {name} | {score['rank_ic']:.6f} | {score['icir']:.6f} | {score['positive_rank_ic_ratio']:.2%} | "
            f"{score['annual_excess']:.2%} | {score['annual_top_decile_return']:.2%} | {score['mean_turnover']:.2%} | "
            f"{score['rank_ic_contribution']:.6f} | {score['annual_excess_contribution']:.6f} | {score['turnover_contribution']:.6f} | {score['final_score']:.8f} |"
        )
    lines.extend([
        "",
        "## 评价覆盖",
        "",
        f"- 预测记录：{combined_score['rows']:,} 条。",
        f"- 交易日：{combined_score['dates']:,} 天。",
        f"- Rank IC 有效天数：{combined_score['rank_ic_days']:,} 天。",
        f"- Top10% 超额收益有效天数：{combined_score['top_decile_days']:,} 天。",
        f"- 换手率转移天数：{combined_score['turnover_transition_days']:,} 天。",
        "",
        "## 组合相对单模型均值",
        "",
    ])
    component_mean = {
        key: float(np.mean([score[key] for score in individual_scores.values()]))
        for key in ("rank_ic", "annual_excess", "mean_turnover", "final_score")
    }
    lines.extend([
        f"- 单模型均值 Rank IC：{component_mean['rank_ic']:.6f}；组合变化：{combined_score['rank_ic'] - component_mean['rank_ic']:+.6f}。",
        f"- 单模型均值年化超额收益：{component_mean['annual_excess']:.2%}；组合变化：{combined_score['annual_excess'] - component_mean['annual_excess']:+.2%}。",
        f"- 单模型均值平均换手率：{component_mean['mean_turnover']:.2%}；组合变化：{combined_score['mean_turnover'] - component_mean['mean_turnover']:+.2%}。",
        f"- 单模型均值综合分：{component_mean['final_score']:.8f}；组合变化：{combined_score['final_score'] - component_mean['final_score']:+.8f}。",
        "",
        "## 与既有模型比较",
        "",
        f"既有 37 因子、排序损失权重 0.30、关闭因子依赖训练惩罚模型的留出集综合分为 {historical_score['final_score']:.8f}。本组合相对它的综合分变化为 {combined_score['final_score'] - historical_score['final_score']:+.8f}，Rank IC 变化为 {combined_score['rank_ic'] - historical_score['rank_ic']:+.6f}，年化超额收益变化为 {combined_score['annual_excess'] - historical_score['annual_excess']:+.2%}，平均换手率变化为 {combined_score['mean_turnover'] - historical_score['mean_turnover']:+.2%}。",
    ])
    (OUTPUT_DIR / "三模型横截面排序等权组合初步评估报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
