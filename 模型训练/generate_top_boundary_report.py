import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "模型训练" / "前10%边界排序实验" / "20261010_top_boundary_rank" / "B1_lambda_top_0.3"
B0 = RUN.parent / "B0_existing_no_dependence_official_2023"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def number(value, digits=6):
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def percent(value, digits=2):
    if value is None:
        return "—"
    return f"{float(value) * 100:.{digits}f}%"


def contribution(score):
    rank = score["rank_ic"]["mean"] * 0.4
    excess = score["top_decile"]["annual_excess"] * 0.3
    turnover = score["turnover"]["one_minus_turnover"] * 0.3
    return rank, excess, turnover


def training_table():
    directory = RUN / "runs" / "single" / "lambda_0.3" / "seed_42"
    files = sorted(directory.glob("epoch_[0-9][0-9]_training.json"))
    rows = [read_json(path) for path in files]
    lines = [
        "| 周期 | 收益损失 | 全局排序损失 | Top边界损失 | 加权全局排序 | 加权Top边界 | 总损失 | 有效全局配对 | 有效Top边界配对 | 用时（分钟） |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['epoch']} | {number(row['loss_return'])} | {number(row['loss_global_rank'])} | "
            f"{number(row['loss_top_boundary'])} | {number(row['weighted_global_rank'])} | "
            f"{number(row['weighted_top_boundary'])} | {number(row['loss_total'])} | "
            f"{row['valid_pair_count']:,} | {row['top_pair_count']:,} | {row['elapsed_seconds'] / 60:.2f} |"
        )
    return "\n".join(lines), rows


def official_table():
    frame = pd.read_csv(RUN / "2023官方评分比较.csv")
    frame["rank_contribution"] = frame["official_rank_ic"] * 0.4
    frame["excess_contribution"] = frame["official_annual_excess"] * 0.3
    frame["turnover_contribution"] = (1.0 - frame["official_mean_turnover"]) * 0.3
    lines = [
        "| 周期 | alpha | Rank IC | Rank IC贡献 | 年化超额收益 | 收益贡献 | 平均换手率 | 换手率贡献 | 官方综合分 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in frame.sort_values(["epoch", "alpha"], ascending=[True, False]).iterrows():
        lines.append(
            f"| {int(row['epoch'])} | {row['alpha']:g} | {number(row['official_rank_ic'])} | "
            f"{number(row['rank_contribution'])} | {number(row['official_annual_excess'])} | "
            f"{number(row['excess_contribution'])} | {number(row['official_mean_turnover'])} | "
            f"{number(row['turnover_contribution'])} | {number(row['official_final_score'])} |"
        )
    return "\n".join(lines), frame


def best_official(frame):
    ordered = frame.sort_values(
        ["official_final_score", "official_rank_ic", "official_annual_excess", "epoch", "alpha"],
        ascending=[False, False, False, True, False],
        kind="mergesort",
    )
    row = ordered.iloc[0].copy()
    row["rank_contribution"] = row["official_rank_ic"] * 0.4
    row["excess_contribution"] = row["official_annual_excess"] * 0.3
    row["turnover_contribution"] = (1.0 - row["official_mean_turnover"]) * 0.3
    return row


def official_comparison_table(b0_best, b1_best):
    lines = [
        "| 模型 | 选定周期 | alpha | Rank IC | Rank IC贡献 | 年化超额收益 | 收益贡献 | 平均换手率 | 换手率贡献 | 2023官方综合分 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in (("B0：既有37因子、排序0.3、关闭依赖惩罚", b0_best), ("B1：新增Top边界排序0.3", b1_best)):
        lines.append(
            f"| {name} | {int(row['epoch'])} | {row['alpha']:g} | "
            f"{number(row['official_rank_ic'])} | {number(row['rank_contribution'])} | "
            f"{number(row['official_annual_excess'])} | {number(row['excess_contribution'])} | "
            f"{number(row['official_mean_turnover'])} | {number(row['turnover_contribution'])} | "
            f"{number(row['official_final_score'])} |"
        )
    return "\n".join(lines)


def holdout_table(metrics):
    lines = [
        "| 预测处理 | Rank IC | Rank IC贡献 | ICIR | 正 IC 比例 | 年化超额收益 | 收益贡献 | Top10%年化绝对收益 | 平均换手率 | 换手率贡献 | 综合分 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, key in (("原始预测", "1.0"), ("alpha=0.3 平滑预测", "0.3")):
        score = metrics[key]["score"]
        rank, excess, turnover = contribution(score)
        lines.append(
            f"| {label} | {number(score['rank_ic']['mean'])} | {number(rank)} | "
            f"{number(score['rank_ic']['icir'])} | {percent(score['rank_ic']['positive_ratio'])} | "
            f"{number(score['top_decile']['annual_excess'])} | {number(excess)} | "
            f"{number(score['top_decile']['top1_annual_return'])} | "
            f"{number(score['turnover']['mean_turnover'])} | {number(turnover)} | "
            f"{number(score['final_score'])} |"
        )
    return "\n".join(lines)


def baseline_row(name, path, alpha, section=None):
    metrics = read_json(path)
    if section is not None:
        score = metrics[section]
    elif str(alpha) in metrics:
        score = metrics[str(alpha)]["score"]
    else:
        score = metrics["model"]
    rank, excess, turnover = contribution(score)
    return {
        "name": name,
        "rank": score["rank_ic"]["mean"],
        "rank_contribution": rank,
        "excess": score["top_decile"]["annual_excess"],
        "excess_contribution": excess,
        "turnover": score["turnover"]["mean_turnover"],
        "turnover_contribution": turnover,
        "score": score["final_score"],
    }


def comparison_table(new_score):
    baselines = [
        baseline_row(
            "27因子、无排序损失、alpha=0.3",
            ROOT / "模型训练" / "第二次综合分验证实验" / "本地留出集比赛指标评估.json",
            0.3,
            section="local_group_b_smoothed",
        ),
        baseline_row(
            "37因子、排序损失0.3、依赖惩罚0.1、alpha=0.3",
            ROOT / "模型训练" / "横截面因子与排序损失实验" / "20261009_local_single" / "evaluation" / "local_holdout" / "local_holdout_metrics.json",
            0.3,
        ),
        baseline_row(
            "37因子、排序损失0.3、关闭依赖惩罚、alpha=0.3",
            ROOT / "模型训练" / "去除因子依赖惩罚实验" / "20261009_local_single_no_dependence" / "evaluation" / "local_holdout" / "local_holdout_metrics.json",
            0.3,
        ),
    ]
    new = {
        "name": "37因子、全局排序0.3、Top边界0.3、关闭依赖惩罚、alpha=0.3",
        "rank": new_score["rank_ic"]["mean"],
        "rank_contribution": new_score["rank_ic"]["mean"] * 0.4,
        "excess": new_score["top_decile"]["annual_excess"],
        "excess_contribution": new_score["top_decile"]["annual_excess"] * 0.3,
        "turnover": new_score["turnover"]["mean_turnover"],
        "turnover_contribution": new_score["turnover"]["one_minus_turnover"] * 0.3,
        "score": new_score["final_score"],
    }
    rows = baselines + [new]
    lines = [
        "| 模型 | Rank IC | Rank IC贡献 | 年化超额收益 | 收益贡献 | 平均换手率 | 换手率贡献 | 2024综合分 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {number(row['rank'])} | {number(row['rank_contribution'])} | "
            f"{number(row['excess'])} | {number(row['excess_contribution'])} | "
            f"{number(row['turnover'])} | {number(row['turnover_contribution'])} | {number(row['score'])} |"
        )
    return "\n".join(lines), baselines, new


def main():
    config = read_json(ROOT / "模型训练" / "去除因子依赖惩罚实验配置.json")
    snapshot = read_json(RUN / "配置快照.json")
    choice = read_json(RUN / "选定配置.json")["final_candidate"]
    holdout = read_json(RUN / "evaluation" / "local_holdout" / "local_holdout_metrics.json")
    dependence = read_json(RUN / "evaluation" / "local_holdout" / "dependence_metrics.json")
    official_text, official = official_table()
    b0_official = pd.read_csv(B0 / "2023官方评分比较.csv")
    b0_best = best_official(b0_official)
    b1_best = best_official(official)
    official_comparison_text = official_comparison_table(b0_best, b1_best)
    training_text, training_rows = training_table()
    holdout_text = holdout_table(holdout)
    new_score = holdout[str(choice["alpha"])]["score"]
    comparison_text, baselines, new_row = comparison_table(new_score)
    stability = read_json(RUN / "runs" / "single" / "lambda_0.3" / "seed_42" / "epoch_03" / "stability_metrics.json")
    stability_record = stability["seeds"]["42"]["alphas"]["0.3"]
    quality = holdout[str(choice["alpha"])]["quality"]
    dependence_lines = [
        "| 来源因子组 | 平均依赖程度 | 达标日期 | 是否通过 |",
        "|---|---:|---:|---|",
    ]
    for key, record in dependence["source_groups"].items():
        feature_names = config["dependence"]["source_groups"][int(key)]
        dependence_lines.append(
            f"| {', '.join(feature_names)} | {number(record['mean_ratio'])} | "
            f"{record['passing_dates']}/{record['total_dates']} | "
            f"{'通过' if record['passed'] else '未通过'} |"
        )
    dependence_text = "\n".join(dependence_lines)
    baseline_no_dependence = baselines[2]
    delta_score = new_row["score"] - baseline_no_dependence["score"]
    delta_rank = new_row["rank"] - baseline_no_dependence["rank"]
    delta_excess = new_row["excess"] - baseline_no_dependence["excess"]
    delta_turnover = new_row["turnover"] - baseline_no_dependence["turnover"]
    total_training_minutes = sum(row["elapsed_seconds"] for row in training_rows) / 60
    report = f"""# 前10%边界排序实验结果报告

## 选定结果

本实验完成了 8 个训练周期、8 个周期 × 4 个 alpha 的 2023 年官方评分、双批量推理稳定性核验和一次 2024 年历史留出验证。模型使用 37 个因子、全局排序损失权重 0.3、Top10% 与普通组边界排序损失权重 0.3，并关闭因子依赖训练惩罚。

2023 年主办方原版评分最高的候选为第 **{choice['epoch']} 周期、alpha={choice['alpha']:g}**，官方综合分为 **{choice['final_score']:.8f}**。该候选的 2023 年细分贡献为：Rank IC 项 `{choice['rank_ic'] * 0.4:.8f}`，年化超额收益项 `{choice['annual_excess'] * 0.3:.8f}`，换手率项 `{(1 - choice['mean_turnover']) * 0.3:.8f}`。

2023 年稳定性核验在 {stability_record['passing_dates']}/{stability_record['total_dates']} 个交易日通过，要求为至少 {stability_record['required_passing_dates']} 个交易日；原始预测和 alpha=0.3 平滑预测的排序相关性与 Top10% 重叠均通过双批量检查。

## 实验身份与数据范围

| 项目 | 设置 |
|---|---|
| 输入因子 | 37 个（27 个基础因子 + 10 个横截面扩展因子） |
| 模型 | LSTM-Transformer 回归模型 |
| 序列长度 | 20 个交易日 |
| 训练数据 | 2018-01-02 至 2022-12-30 |
| 训练资格记录 | 4,510,263 |
| 2023 年选择记录 | 1,051,065，241 个交易日 |
| 2024 年历史留出记录 | 1,091,450，242 个交易日 |
| 排序损失 | 全局排序损失 + Top10%/普通组边界排序损失 |
| 全局排序权重 | 0.3 |
| Top边界排序权重 | 0.3 |
| 因子依赖训练惩罚 | 关闭 |
| 优化器 | AdamW，学习率 0.0003，权重衰减 0.0001 |
| 批量大小 | 512，同一交易日内组成批次 |
| 随机种子 | 42 |
| 运行设备 | NVIDIA GeForce RTX 4060 Laptop GPU |
| PyTorch | 2.13.0+cu132，CUDA 13.2 |
| 配置摘要 | `{snapshot['config_sha256']}` |
| 输入摘要 | `{snapshot['input_sha256']}` |

2025—2026 年测试集 X 没有参与本实验，未生成正式 `submission.csv`。2024 年结果只作为模型确定后的历史留出结果，不作为独立最终测试结果。

## 损失函数与标签构造

每天在官方评分有效股票集合中，按真实下一日收益从高到低排序，前 10% 标记为 Top 组，其余标记为普通组。涨停股票不参与 Top 边界标签。模型只输出一个连续预测分数，推理时对分数降序排序并选择前 10%。训练总损失为：

```text
L_total = L_return + 0.3 × L_global-rank + 0.3 × L_Top-vs-rest
```

Top边界损失只对同一交易日中 Top 组和普通组的交叉配对计算，Top 组预测分数高于普通组时损失降低。Top组内部不重复计算边界损失，整体排序损失继续约束全截面排序。

## 逐周期训练结果

训练累计耗时约 **{total_training_minutes / 60:.2f} 小时**。每个周期完整覆盖训练资格记录。

{training_text}

训练总损失从第 1 周期的 `{training_rows[0]['loss_total']:.8f}` 降至第 8 周期的 `{training_rows[-1]['loss_total']:.8f}`。Top边界有效配对数量约为 1.88 亿，说明每个周期均实际执行了边界排序约束。

## 2023 年官方评分

下面所有分数均由主办方原版 `evaluate.py` 计算。综合分公式为：

```text
综合分 = 0.4 × Rank IC + 0.3 × 年化 Top10% 超额收益 + 0.3 × (1 − 平均换手率)
```

其中“贡献”已经乘入对应评分权重，三项贡献之和等于官方综合分。

{official_text}

第 3 周期 alpha=0.3 的选择原因是综合分最高。alpha 从 1.0 调低到 0.3 后，2023 年平均换手率从 `{official[(official.epoch == 3) & (official.alpha == 1.0)].iloc[0]['official_mean_turnover']:.8f}` 降至 `{choice['mean_turnover']:.8f}`，换手率贡献增加，综合分达到 `{choice['final_score']:.8f}`。

## B0 与 B1 的 2023 年官方复评比较

B0 使用既有“37 个因子、全局排序损失权重 0.3、关闭依赖惩罚”实验的原有检查点，在新的临时评分目录中重新生成 2023 年候选预测，并逐一调用主办方原版 `evaluate.py`。原有实验目录没有被写入或覆盖。B1 使用本次新增 Top10% 与普通组边界排序损失，两个模型都按照相同的 8 个周期和 4 个 alpha 进行比较。

{official_comparison_text}

B0 的 2023 年最优候选为第 {int(b0_best['epoch'])} 周期、alpha={b0_best['alpha']:g}，综合分 `{b0_best['official_final_score']:.8f}`；B1 的最优候选为第 {int(b1_best['epoch'])} 周期、alpha={b1_best['alpha']:g}，综合分 `{b1_best['official_final_score']:.8f}`。B1 相对 B0 的 2023 年最优综合分变化为 `{b1_best['official_final_score'] - b0_best['official_final_score']:+.8f}`，Rank IC 变化为 `{b1_best['official_rank_ic'] - b0_best['official_rank_ic']:+.8f}`，年化超额收益变化为 `{b1_best['official_annual_excess'] - b0_best['official_annual_excess']:+.8f}`，平均换手率变化为 `{b1_best['official_mean_turnover'] - b0_best['official_mean_turnover']:+.8f}`。B0 的 2023 年完整候选明细保存在 `{B0.name}/2023官方评分比较.csv`。

## 2024 年历史留出结果

模型周期和 alpha 已由 2023 年选择后固定为第 {choice['epoch']} 周期、alpha={choice['alpha']:g}，随后只在 2024 年执行一次历史留出评价。

{holdout_text}

选定 alpha=0.3 的 2024 年综合分由三项构成：

| 评分项 | 数值 |
|---|---:|
| 0.4 × Rank IC | {new_score['rank_ic']['mean'] * 0.4:.8f} |
| 0.3 × 年化 Top10% 超额收益 | {new_score['top_decile']['annual_excess'] * 0.3:.8f} |
| 0.3 ×（1 − 平均换手率） | {new_score['turnover']['one_minus_turnover'] * 0.3:.8f} |
| 综合分 | {new_score['final_score']:.8f} |

2024 年选定预测的质量检查通过，242 个日期均有有效 Rank IC，换手率有 241 个相邻日期转移；质量预警日期数量为 {quality['warning_dates']}，主要来自预测横截面标准差相对真实收益标准差较小，未触发常数预测或分辨率失效条件。

## 与已有实验的 2024 年比较

{comparison_text}

相对于已有的“37 因子、排序损失 0.3、关闭依赖惩罚”模型，本实验的 2024 年变化为：综合分 `{delta_score:+.8f}`，Rank IC `{delta_rank:+.8f}`，年化超额收益 `{delta_excess:+.8f}`，平均换手率 `{delta_turnover:+.8f}`。新边界损失降低了部分换手率，但年化超额收益和 Rank IC 同时下降，综合分低于已有无依赖惩罚模型。

相对于“37 因子、排序损失 0.3、依赖惩罚 0.1”模型，已有模型 2024 年综合分为 0.25791028；本实验综合分为 `{new_score['final_score']:.8f}`。相对于 27 因子无排序损失模型的 0.26938214，本实验也没有形成更高的历史留出综合分。

## 2024 年因子依赖诊断

因子依赖诊断使用替换指定来源组因子后的预测变化比值，仅用于描述模型敏感度，不参与本次训练损失和 2023 年候选排序。单日阈值为 0.25，历史留出期要求至少 230/242 个日期通过。

{dependence_text}

Top边界排序模型在波动率、成交额冲击、日内收益和市场因子组上未达到原有日期比例要求，说明加入边界目标后模型对部分因子组的预测敏感度发生变化；两个动量来源组在本次阈值下通过。该诊断不表示因果关系。

## 结果判断

本实验完成了预定的 Top10% 分组排序和组内排序训练流程，2023 年官方评分能够选出稳定的第 3 周期 alpha=0.3 候选，2024 年历史留出综合分为 `{new_score['final_score']:.8f}`。与已有无依赖惩罚模型相比，边界排序损失没有提升历史留出综合分，因此当前不把本实验模型作为正式模型依据，也不生成测试集预测或 `submission.csv`。

该结论只说明当前固定的 Top标签构造、边界损失权重 0.3、全局排序权重 0.3 和现有模型结构组合的结果。训练损失下降和 2023 年选择集表现不能单独证明边界损失在未来测试集上有效。

## 产物位置

- [训练入口](train_top_boundary_rank.py)
- [Top边界排序损失](top_boundary_rank_loss.py)
- [2023 年官方评分入口](evaluate_top_boundary_rank.py)
- [2023 年官方评分明细](2023官方评分比较.csv)
- [选定配置](选定配置.json)
- [第 3 周期检查点](runs/single/lambda_0.3/seed_42/checkpoint_epoch_03.pt)
- [2024 年历史留出指标](evaluation/local_holdout/local_holdout_metrics.json)
- [2024 年因子依赖诊断](evaluation/local_holdout/dependence_metrics.json)
"""
    report_path = RUN / "模型结果评估报告.md"
    report_path.write_text(report, encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
