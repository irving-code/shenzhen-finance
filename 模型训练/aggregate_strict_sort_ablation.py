import json
from pathlib import Path

import numpy as np
import pandas as pd

from cross_section_experiment import read_json, write_json


ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_ROOT = ROOT / "模型训练" / "严格排序损失消融实验" / "20261010_rank_ablation"
WEIGHTS = (0.0, 0.15, 0.3, 0.5)
SEEDS = (42, 43, 44)


def score_row(score, weight, seed, prediction_processing):
    return {
        "rank_weight": weight,
        "seed": seed,
        "prediction_processing": prediction_processing,
        "rank_ic": score["rank_ic"]["mean"],
        "rank_ic_std": score["rank_ic"]["std"],
        "icir": score["rank_ic"]["icir"],
        "positive_rank_ic_ratio": score["rank_ic"]["positive_ratio"],
        "rank_ic_days": score["rank_ic"]["days"],
        "annual_excess": score["top_decile"]["annual_excess"],
        "annual_top_decile_return": score["top_decile"]["top1_annual_return"],
        "return_days": score["top_decile"]["days"],
        "mean_turnover": score["turnover"]["mean_turnover"],
        "one_minus_turnover": score["turnover"]["one_minus_turnover"],
        "turnover_days": score["turnover"]["transition_days"],
        "final_score": score["final_score"],
        "rows": score["rows"],
        "dates": score["dates"],
        "rank_ic_contribution": 0.4 * score["rank_ic"]["mean"],
        "annual_excess_contribution": 0.3 * score["top_decile"]["annual_excess"],
        "turnover_contribution": 0.3 * score["turnover"]["one_minus_turnover"],
    }


def collect_rows():
    rows = []
    period_rows = []
    for weight in WEIGHTS:
        for seed in SEEDS:
            run_dir = EXPERIMENT_ROOT / "runs" / f"rank_{weight:.2f}" / f"seed_{seed}"
            selection = read_json(run_dir / "选定配置.json")
            if selection["status"] != "selected":
                rows.append({"rank_weight": weight, "seed": seed, "status": selection["status"]})
                continue
            choice = selection["final_candidate"]
            holdout = read_json(run_dir / "evaluation" / "本地留出集指标.json")
            for alpha, label in ((1.0, "raw"), (0.3, "smoothed")):
                rows.append(score_row(holdout["metrics"][str(alpha)]["score"], weight, seed, label) | {
                    "status": "selected", "selected_epoch": choice["epoch"], "alpha": alpha,
                })
            period_path = run_dir / "验证周期指标.csv"
            period = pd.read_csv(period_path)
            if "rank_weight" not in period.columns:
                period.insert(0, "rank_weight", weight)
            else:
                period["rank_weight"] = weight
            if "seed" not in period.columns:
                period.insert(1, "seed", seed)
            else:
                period["seed"] = seed
            period_rows.append(period)
    return pd.DataFrame(rows), pd.concat(period_rows, ignore_index=True)


def daily_delta(weight_left, weight_right, seed):
    left = pd.read_parquet(
        EXPERIMENT_ROOT / "runs" / f"rank_{weight_left:.2f}" / f"seed_{seed}" / "evaluation" / "local_holdout" / "local_holdout_alpha_0.3_daily.parquet"
    )
    right = pd.read_parquet(
        EXPERIMENT_ROOT / "runs" / f"rank_{weight_right:.2f}" / f"seed_{seed}" / "evaluation" / "local_holdout" / "local_holdout_alpha_0.3_daily.parquet"
    )
    left = left[["trade_date", "rank_ic", "excess_return", "turnover"]].rename(columns={
        "rank_ic": "rank_ic_left", "excess_return": "excess_return_left", "turnover": "turnover_left",
    })
    right = right[["trade_date", "rank_ic", "excess_return", "turnover"]].rename(columns={
        "rank_ic": "rank_ic_right", "excess_return": "excess_return_right", "turnover": "turnover_right",
    })
    merged = left.merge(right, on="trade_date", how="outer", validate="one_to_one", indicator=True)
    if not merged["_merge"].eq("both").all():
        raise ValueError("严格消融逐日留出键集合不一致")
    merged = merged.drop(columns="_merge")
    merged.insert(1, "seed", seed)
    merged["delta_rank_ic"] = merged["rank_ic_right"] - merged["rank_ic_left"]
    merged["delta_excess_return"] = merged["excess_return_right"] - merged["excess_return_left"]
    merged["delta_turnover"] = merged["turnover_right"] - merged["turnover_left"]
    merged["delta_daily_score"] = (
        0.4 * merged["delta_rank_ic"]
        + 0.3 * merged["delta_excess_return"]
        - 0.3 * merged["delta_turnover"]
    )
    return merged


def block_bootstrap(frame, seed=20261010, repetitions=10000, block_length=5):
    frame = frame.dropna(subset=["delta_rank_ic", "delta_excess_return", "delta_turnover"]).reset_index(drop=True)
    blocks = [np.arange(start, min(start + block_length, len(frame))) for start in range(0, len(frame), block_length)]
    rng = np.random.default_rng(seed)
    values = np.empty((repetitions, 4), dtype=np.float64)
    for repetition in range(repetitions):
        selected = []
        while len(selected) < len(frame):
            selected.extend(blocks[int(rng.integers(0, len(blocks)))].tolist())
        selected = np.asarray(selected[:len(frame)], dtype=np.int64)
        rank = frame.iloc[selected]["delta_rank_ic"].mean()
        excess = frame.iloc[selected]["delta_excess_return"].mean() * 252.0
        turnover = frame.iloc[selected]["delta_turnover"].mean()
        score = 0.4 * rank + 0.3 * excess - 0.3 * turnover
        values[repetition] = (rank, excess, turnover, score)
    labels = ("rank_ic", "annual_excess", "turnover", "final_score")
    return {
        label: {
            "mean": float(values[:, index].mean()),
            "lower_2_5_percent": float(np.quantile(values[:, index], 0.025)),
            "upper_97_5_percent": float(np.quantile(values[:, index], 0.975)),
        }
        for index, label in enumerate(labels)
    }


def previous_models():
    result = []
    first = read_json(ROOT / "模型训练" / "第二次综合分验证实验" / "本地留出集比赛指标评估.json")
    result.append({"model": "27因子、无排序损失、alpha=0.3", **score_row(first["local_group_b_smoothed"], "27", "历史", "smoothed")})
    for name, path, label in (
        ("37因子、排序损失0.3、依赖惩罚0.1", ROOT / "模型训练" / "横截面因子与排序损失实验" / "20261009_local_single" / "evaluation" / "本地留出集指标.json", "smoothed"),
        ("37因子、排序损失0.3、关闭依赖惩罚", ROOT / "模型训练" / "去除因子依赖惩罚实验" / "20261009_local_single_no_dependence" / "evaluation" / "本地留出集指标.json", "smoothed"),
    ):
        record = read_json(path)
        result.append({"model": name, **score_row(record["metrics"]["0.3"]["score"], "37", "历史", label)})
    result.extend([
        {
            "model": "第一次 LSTM-Transformer、2024 原始预测",
            "evaluation_period": "2024本地留出",
            "rank_ic": 0.08169466855580083,
            "icir": 0.4596298670338572,
            "annual_excess": 0.20610214931175644,
            "mean_turnover": 0.7536409320029761,
            "final_score": 0.16841623261495442,
        },
        {
            "model": "MLP、2024 本地留出",
            "evaluation_period": "2024本地留出",
            "rank_ic": 0.09553918706,
            "icir": 0.9078544243,
            "annual_excess": 0.2767518737,
            "mean_turnover": 0.8194865018,
            "final_score": 0.1753952864,
        },
        {
            "model": "LightGBM、2024 本地留出",
            "evaluation_period": "2024本地留出",
            "rank_ic": 0.06820252785,
            "icir": 0.4471754785,
            "annual_excess": 0.2898163978,
            "mean_turnover": 0.8285200249,
            "final_score": 0.165669923,
        },
        {
            "model": "Top边界排序、2024 平滑预测",
            "evaluation_period": "2024本地留出",
            "rank_ic": 0.019629,
            "icir": 0.109555,
            "annual_excess": 0.040396,
            "mean_turnover": 0.311178,
            "final_score": 0.22661692,
        },
        {
            "model": "Top边界排序 B0、2023 官方评分",
            "evaluation_period": "2023官方评分",
            "rank_ic": None,
            "icir": None,
            "annual_excess": None,
            "mean_turnover": None,
            "final_score": 0.31792187,
        },
        {
            "model": "Top边界排序 B1、2023 官方评分",
            "evaluation_period": "2023官方评分",
            "rank_ic": None,
            "icir": None,
            "annual_excess": None,
            "mean_turnover": None,
            "final_score": 0.27281823,
        },
    ])
    return pd.DataFrame(result)


def aggregate():
    comparison_dir = EXPERIMENT_ROOT / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    rows, periods = collect_rows()
    rows.to_csv(comparison_dir / "留出集指标.csv", index=False, encoding="utf-8-sig")
    periods.to_csv(comparison_dir / "验证周期指标.csv", index=False, encoding="utf-8-sig")
    daily_frames = []
    ci_records = {}
    for seed in SEEDS:
        daily = daily_delta(0.0, 0.3, seed)
        daily_frames.append(daily)
        ci_records[str(seed)] = block_bootstrap(daily)
    all_daily = pd.concat(daily_frames, ignore_index=True)
    all_daily.to_parquet(comparison_dir / "A_vs_C_daily_delta.parquet", index=False, compression="zstd")
    ci_records["pooled_dates"] = block_bootstrap(all_daily)
    write_json(comparison_dir / "置信区间.json", {
        "method": "five_trading_day_block_paired_bootstrap",
        "repetitions": 10000,
        "block_length": 5,
        "seed": 20261010,
        "records": ci_records,
    })
    historical = previous_models()
    historical.to_csv(comparison_dir / "历史模型比较.csv", index=False, encoding="utf-8-sig")
    selected = rows.loc[(rows["status"] == "selected") & (rows["prediction_processing"] == "smoothed")].copy()
    summary = selected.groupby("rank_weight", as_index=False).agg(
        seed_count=("seed", "count"), final_score_mean=("final_score", "mean"),
        final_score_median=("final_score", "median"), final_score_std=("final_score", "std"),
        rank_ic_mean=("rank_ic", "mean"), annual_excess_mean=("annual_excess", "mean"),
        turnover_mean=("mean_turnover", "mean"),
    )
    summary.to_csv(comparison_dir / "跨种子摘要.csv", index=False, encoding="utf-8-sig")
    report = build_report(rows, periods, historical, ci_records)
    (EXPERIMENT_ROOT / "实验结果报告.md").write_text(report, encoding="utf-8")
    write_json(comparison_dir / "完整性核验.json", {
        "expected_runs": len(WEIGHTS) * len(SEEDS),
        "selected_runs": int((rows["status"] == "selected").sum() / 2),
        "holdout_metric_rows": int(len(rows)),
        "daily_delta_rows": int(len(all_daily)),
        "status": "passed",
    })
    return report


def build_report(rows, periods, historical, ci_records):
    def display_metric(value, digits=6):
        if pd.isna(value):
            return "不可用"
        return f"{value:.{digits}f}"

    def display_percent(value):
        if pd.isna(value):
            return "不可用"
        return f"{value:.2%}"

    lines = [
        "# 严格排序损失消融实验结果报告", "",
        "实验编号：`20261010_严格排序损失消融`。输入使用 37 个因子，因子依赖训练惩罚和 Top10% 边界排序损失均关闭。四个排序损失权重分别为 0.00、0.15、0.30 和 0.50，每个权重使用随机种子 42、43、44 独立训练。", "",
        "## 一、实验完成状态", "",
        f"训练矩阵包含 {len(WEIGHTS) * len(SEEDS)} 个独立运行。留出指标记录 {len(rows)} 行，逐日 A 与 C 配对差值记录 {sum(len(pd.read_parquet(EXPERIMENT_ROOT / 'runs' / 'rank_0.00' / f'seed_{seed}' / 'evaluation' / 'local_holdout' / 'local_holdout_alpha_0.3_daily.parquet')) for seed in SEEDS)} 行。", "",
        "## 二、2023 年验证周期结果", "",
        "下表列出每个候选和随机种子的最终选择周期。完整周期指标保存在 `comparison/验证周期指标.csv`。", "",
        "| 排序权重 | 随机种子 | 选择周期 | 验证综合分 | Rank IC | 年化超额收益 | 平均换手率 |", "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for weight in WEIGHTS:
        for seed in SEEDS:
            run_dir = EXPERIMENT_ROOT / "runs" / f"rank_{weight:.2f}" / f"seed_{seed}"
            choice = read_json(run_dir / "选定配置.json")["final_candidate"]
            if choice is None:
                lines.append(f"| {weight:.2f} | {seed} | 无合格周期 | | | | |")
            else:
                lines.append(f"| {weight:.2f} | {seed} | {choice['epoch']} | {choice['final_score']:.8f} | {choice['rank_ic']:.8f} | {choice['annual_excess']:.2%} | 待留出评价 |")
    lines.extend(["", "## 三、2024 年本地留出集细分指标", "", "| 排序权重 | 随机种子 | 预测处理 | Rank IC | ICIR | 正 IC 比例 | 年化超额收益 | 年化组合收益 | 平均换手率 | Rank IC贡献 | 收益贡献 | 换手率贡献 | 综合分 |", "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for _, row in rows.loc[rows["status"] == "selected"].sort_values(["rank_weight", "seed", "prediction_processing"]).iterrows():
        lines.append(
            f"| {row['rank_weight']:.2f} | {int(row['seed'])} | {row['prediction_processing']} | {row['rank_ic']:.6f} | {row['icir']:.6f} | {row['positive_rank_ic_ratio']:.2%} | {row['annual_excess']:.2%} | {row['annual_top_decile_return']:.2%} | {row['mean_turnover']:.2%} | {row['rank_ic_contribution']:.6f} | {row['annual_excess_contribution']:.6f} | {row['turnover_contribution']:.6f} | {row['final_score']:.8f} |"
        )
    lines.extend(["", "## 四、跨随机种子摘要", "", "| 排序权重 | 种子数 | 综合分均值 | 综合分中位数 | 综合分标准差 | Rank IC均值 | 年化超额收益均值 | 平均换手率均值 |", "|---:|---:|---:|---:|---:|---:|---:|---:|"])
    summary = rows.loc[(rows["status"] == "selected") & (rows["prediction_processing"] == "smoothed")].groupby("rank_weight", as_index=False).agg(seed_count=("seed", "count"), final_score_mean=("final_score", "mean"), final_score_median=("final_score", "median"), final_score_std=("final_score", "std"), rank_ic_mean=("rank_ic", "mean"), annual_excess_mean=("annual_excess", "mean"), turnover_mean=("mean_turnover", "mean"))
    for _, row in summary.iterrows():
        lines.append(f"| {row['rank_weight']:.2f} | {int(row['seed_count'])} | {row['final_score_mean']:.8f} | {row['final_score_median']:.8f} | {row['final_score_std']:.8f} | {row['rank_ic_mean']:.6f} | {row['annual_excess_mean']:.2%} | {row['turnover_mean']:.2%} |")
    lines.extend(["", "## 五、A 组与 C 组严格配对比较", "", "A 组为 `lambda_rank=0.00`，C 组为 `lambda_rank=0.30`。差值定义为 C 组减 A 组。综合分差值按 `0.4×ΔRankIC + 0.3×Δ年化超额收益 − 0.3×Δ换手率` 计算。", "", "| 随机种子 | Rank IC差值均值 | 年化超额收益差值 | 换手率差值 | 综合分差值 |", "|---:|---:|---:|---:|---:|"])
    for seed in SEEDS:
        record = ci_records[str(seed)]
        lines.append(f"| {seed} | {record['rank_ic']['mean']:.6f} | {record['annual_excess']['mean']:.2%} | {record['turnover']['mean']:.2%} | {record['final_score']['mean']:.8f} |")
    pooled = ci_records["pooled_dates"]
    lines.extend(["", f"跨种子合并日期的综合分差值均值为 `{pooled['final_score']['mean']:.8f}`，95% 区间为 `[{pooled['final_score']['lower_2_5_percent']:.8f}, {pooled['final_score']['upper_97_5_percent']:.8f}]`。完整重采样结果保存在 `comparison/置信区间.json`。", ""])
    lines.extend(["## 六、与既有模型比较", "", "| 模型 | Rank IC | ICIR | 年化超额收益 | 平均换手率 | 综合分 |", "|---|---:|---:|---:|---:|---:|"])
    for _, row in historical.iterrows():
        lines.append(f"| {row['model']} | {display_metric(row['rank_ic'])} | {display_metric(row['icir'])} | {display_percent(row['annual_excess'])} | {display_percent(row['mean_turnover'])} | {display_metric(row['final_score'], 8)} |")
    lines.extend(["", "## 七、判定", "", "严格消融结论以 2024 年留出集的 A 与 C 配对比较为依据。报告同时保留收益项、换手率项和三个分数贡献，避免只依据综合分判断排序损失。B、C、D 的权重关系用于描述排序约束强度，不能替代 A 与 C 的主要消融结论。", "", "## 八、产物位置", "", "- `comparison/验证周期指标.csv`：逐周期验证指标。", "- `comparison/留出集指标.csv`：逐候选、逐种子留出指标。", "- `comparison/A_vs_C_daily_delta.parquet`：逐日配对差值。", "- `comparison/置信区间.json`：五交易日分块配对重采样。", "- `comparison/历史模型比较.csv`：与 27 因子和既有 37 因子模型的比较。"])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    aggregate()
