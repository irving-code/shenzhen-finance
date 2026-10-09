import argparse
from pathlib import Path

import pandas as pd

from cross_section_experiment import file_sha256, load_config, project_path, read_json, write_json
from evaluate_cross_section_rank import run_path
from verify_cross_section_rank import run_final_audit


def epoch_records(config, run_dir):
    state = read_json(run_dir / "调度状态.json")
    if not state["stopped"]:
        raise ValueError("正式训练尚未结束")
    directory = run_path(run_dir, config["model_id"], config["ranking"]["weight"], config["seed"])
    records = []
    for epoch in range(1, state["completed_epochs"] + 1):
        record = read_json(directory / f"epoch_{epoch:02d}_training.json")
        output = directory / f"epoch_{epoch:02d}"
        validation = read_json(output / "validation_metrics.json")[str(config["selection"]["alpha"])]
        record["validation_score"] = validation["score"]["final_score"] if validation["score"] else None
        record["validation_quality_pass"] = validation["quality_pass"]
        record["validation_seconds"] = read_json(output / "validation_runtime.json")["elapsed_seconds"]
        records.append(record)
    (run_dir / "evaluation").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(run_dir / "evaluation" / "周期指标.csv", index=False, encoding="utf-8-sig")
    return state, directory, records


def score_values(score):
    return {
        "rank_ic": score["rank_ic"]["mean"],
        "icir": score["rank_ic"]["icir"],
        "annual_excess": score["top_decile"]["annual_excess"],
        "annual_top_decile_return": score["top_decile"]["top1_annual_return"],
        "mean_turnover": score["turnover"]["mean_turnover"],
        "final_score": score["final_score"],
        "rows": score["rows"],
        "dates": score["dates"],
    }


def build_baseline_comparison(config, run_dir, choice, holdout):
    baseline_dir = project_path(config, "baseline_dir")
    baseline_choice = read_json(baseline_dir / "选定配置.json")["final_candidate"]
    baseline_holdout = read_json(baseline_dir / "evaluation" / "本地留出集指标.json")
    processing = ((1.0, "raw"), (choice["alpha"], "smoothed"))
    comparison_rows = []
    daily_frames = []
    monthly_frames = []
    for alpha, label in processing:
        baseline_metrics = baseline_holdout["metrics"][str(alpha)]
        current_metrics = holdout["metrics"][str(alpha)]
        if baseline_metrics["score"] is None or current_metrics["score"] is None:
            raise ValueError("基线比较包含无效评价结果")
        baseline_values = score_values(baseline_metrics["score"])
        current_values = score_values(current_metrics["score"])
        comparison_rows.append({
            "prediction_processing": label,
            "alpha": alpha,
            "baseline_epoch": baseline_choice["epoch"],
            "new_epoch": choice["epoch"],
            **{f"baseline_{key}": value for key, value in baseline_values.items()},
            **{f"new_{key}": value for key, value in current_values.items()},
            **{
                f"delta_{key}": current_values[key] - baseline_values[key]
                for key in current_values
                if key not in ("rows", "dates")
            },
        })
        baseline_daily_path = baseline_dir / "evaluation" / "local_holdout" / f"local_holdout_alpha_{alpha:g}_daily.parquet"
        current_daily_path = Path(run_dir) / "evaluation" / "local_holdout" / f"local_holdout_alpha_{alpha:g}_daily.parquet"
        baseline_daily = pd.read_parquet(baseline_daily_path)
        current_daily = pd.read_parquet(current_daily_path)
        daily = baseline_daily.merge(
            current_daily, on="trade_date", how="outer", suffixes=("_baseline", "_new"),
            validate="one_to_one", indicator=True,
        )
        if len(daily) != 242 or not daily["_merge"].eq("both").all():
            raise ValueError("基线与新版逐日评价日期不完全一致")
        daily = daily.drop(columns="_merge")
        daily.insert(1, "prediction_processing", label)
        daily["delta_rank_ic"] = daily["rank_ic_new"] - daily["rank_ic_baseline"]
        daily["delta_excess_return"] = daily["excess_return_new"] - daily["excess_return_baseline"]
        daily["delta_turnover"] = daily["turnover_new"] - daily["turnover_baseline"]
        daily_frames.append(daily)
        for model_name, daily_source in (("baseline", baseline_daily), ("new", current_daily)):
            monthly = daily_source.copy()
            monthly["month"] = monthly["trade_date"] // 100
            monthly = monthly.groupby("month", sort=True).agg(
                rank_ic_mean=("rank_ic", "mean"),
                excess_return_daily_mean=("excess_return", "mean"),
                turnover_mean=("turnover", "mean"),
                rank_ic_days=("rank_ic", "count"),
                return_days=("excess_return", "count"),
                turnover_days=("turnover", "count"),
            ).reset_index()
            monthly.insert(1, "model", model_name)
            monthly.insert(2, "prediction_processing", label)
            monthly_frames.append(monthly)
    comparison_frame = pd.DataFrame(comparison_rows)
    comparison_dir = Path(run_dir) / "evaluation"
    comparison_frame.to_csv(comparison_dir / "基线比较指标.csv", index=False, encoding="utf-8-sig")
    pd.concat(daily_frames, ignore_index=True).to_parquet(
        comparison_dir / "历史留出逐日比较.parquet", index=False, compression="zstd",
    )
    pd.concat(monthly_frames, ignore_index=True).to_csv(
        comparison_dir / "月度指标.csv", index=False, encoding="utf-8-sig",
    )
    baseline_periods = pd.read_csv(baseline_dir / "evaluation" / "周期指标.csv")
    current_periods = pd.read_csv(comparison_dir / "周期指标.csv")
    period_comparison = baseline_periods.merge(
        current_periods, on="epoch", how="outer", suffixes=("_baseline", "_new"),
        validate="one_to_one", indicator=True,
    )
    period_comparison.to_csv(comparison_dir / "验证周期比较.csv", index=False, encoding="utf-8-sig")
    record = {
        "baseline_run_dir": baseline_dir.relative_to(Path(config["project_root"])).as_posix(),
        "baseline_selected_candidate": baseline_choice,
        "new_selected_candidate": choice,
        "holdout_status": "previously_viewed_exploratory_comparison",
        "input_identity_sha256": read_json(Path(run_dir) / "配置快照.json")["input_sha256"],
        "base_artifact_manifest_sha256": file_sha256(Path(run_dir) / "基线产物摘要.json"),
        "metrics": comparison_rows,
        "daily_rows": len(daily_frames[0]) + len(daily_frames[1]),
        "validation_epoch_rows": len(period_comparison),
    }
    write_json(comparison_dir / "基线比较指标.json", record)
    return record


def report_lines(config, run_dir, state, records):
    environment = read_json(run_dir / "环境记录.json")
    lines = [
        "# 去除因子依赖惩罚实验评估报告", "",
        f"设备：{environment['device']}；随机种子：{config['seed']}；输入：20 个交易日、37 个因子。",
        f"训练样本：{config['data']['training_eligible']:,} 条；验证选择样本：{config['data']['selection_eligible']:,} 条。",
        f"训练周期：{state['completed_epochs']}；停止原因：`{state['stop_reason']}`。",
        f"训练损失：`L_return + {config['ranking']['weight']} × L_rank`；因子依赖训练惩罚已关闭；固定平滑参数：{config['selection']['alpha']}。", "",
        "2023 年数据用于早停和周期选择。2024 年数据已经查看，本报告将其作为历史探索性比较。",
        "新版训练从随机种子 42 的新模型参数开始，完整使用原有 27 个因子和 10 个横截面扩展因子。因子依赖诊断只在模型选定后执行。", "",
        "## 逐周期训练与验证", "",
        "| 周期 | 收益损失 | 排序损失 | 依赖惩罚 | 验证综合分 | 质量检查 | 训练分钟 | 验证分钟 |",
        "|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for record in records:
        score = f"{record['validation_score']:.8f}" if record["validation_score"] is not None else "无法评分"
        quality = "通过" if record["validation_quality_pass"] else "未通过"
        lines.append(
            f"| {record['epoch']} | {record['loss_return']:.8f} | {record['loss_rank']:.8f} | "
            f"{record['dependence_penalty_step_mean']:.8f} | {score} | {quality} | "
            f"{record['elapsed_seconds'] / 60:.2f} | {record['validation_seconds'] / 60:.2f} |"
        )
    lines.extend([
        "", "## 运行环境与耗时", "",
        f"训练计算累计耗时：{sum(item['elapsed_seconds'] for item in records) / 3600:.2f} 小时。",
        f"逐周期验证累计耗时：{sum(item['validation_seconds'] for item in records) / 3600:.2f} 小时。",
        f"Python {environment['python']}，PyTorch {environment['torch']}，CUDA {environment['torch_cuda']}。",
        f"训练峰值显存：{max(item['device_memory_peak_bytes'] for item in records) / 1024**3:.3f} GiB。", "",
    ])
    return lines


def finalize(config, config_sha256, run_dir):
    run_dir = Path(run_dir)
    if read_json(run_dir / "配置快照.json")["config_sha256"] != config_sha256:
        raise ValueError("报告配置摘要不一致")
    state, directory, records = epoch_records(config, run_dir)
    selection = read_json(run_dir / "选定配置.json")
    choice = selection["final_candidate"]
    lines = report_lines(config, run_dir, state, records)
    if selection["status"] == "no_eligible_configuration":
        lines.extend([
            "## 模型选择结果", "",
            "全部训练周期未满足实验方案要求的预测数值质量或双批量推理稳定性检查。因子依赖诊断不参与新版候选筛选。",
            "模型选择状态：`no_eligible_configuration`。留出集评价和官方测试预测按实验方案停止。",
            "逐周期淘汰原因与核验文件记录在 `选择核验记录.json`。", "",
        ])
        checks = read_json(run_dir / "选择核验记录.json")
        for item in checks:
            lines.append(f"- 周期 {item['epoch']}：`{item['status']}`。")
        lines.append("")
        (run_dir / "模型结果评估报告.md").write_text("\n".join(lines), encoding="utf-8")
        write_json(run_dir / "评价状态.json", {
            "status": "no_eligible_configuration", "completed_epochs": state["completed_epochs"],
            "holdout_evaluated": False, "competition_test_predicted": False,
        })
        run_final_audit(config, config_sha256, run_dir)
        return
    if selection["status"] != "selected" or choice is None:
        raise ValueError("模型选择状态无法生成评价报告")
    holdout = read_json(run_dir / "evaluation" / "本地留出集指标.json")
    coverage = read_json(run_dir / "competition_test" / "覆盖审计.json")
    lines.extend([
        "## 模型选择结果", "",
        f"选定周期：{choice['epoch']}；2023 年验证综合分：{choice['final_score']:.8f}。",
        "所选检查点已通过预测数值质量和双批量推理稳定性检查。七组因子依赖程度在选模后计算，不参与候选排序或资格判断。", "",
        "## 2024 年本地留出评价", "",
        "| 预测处理 | Rank IC | ICIR | 年化超额收益 | 换手率 | 综合分 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for alpha, label in ((1.0, "原始预测"), (choice["alpha"], "固定平滑预测")):
        metric = holdout["metrics"][str(alpha)]
        score = metric["score"]
        if score is None:
            lines.append(f"| {label} | 预测质量检查未通过 | | | | |")
        else:
            lines.append(
                f"| {label} | {score['rank_ic']['mean']:.8f} | {score['rank_ic']['icir']:.8f} | "
                f"{score['top_decile']['annual_excess']:.2%} | {score['turnover']['mean_turnover']:.2%} | {score['final_score']:.8f} |"
            )
    selected_metric = holdout["metrics"][str(choice["alpha"])]
    if selected_metric["score"] is not None:
        score = selected_metric["score"]
        lines.extend([
            "", f"评价记录：{score['rows']:,} 条；日期数量：{score['dates']}。",
            f"有效日期：Rank IC 为 {score['rank_ic']['days']}，收益为 {score['top_decile']['days']}，换手率为 {score['turnover']['transition_days']}。",
        ])
        daily = pd.read_parquet(
            run_dir / "evaluation" / "local_holdout" / f"local_holdout_alpha_{choice['alpha']:g}_daily.parquet"
        )
        daily["month"] = daily["trade_date"] // 100
        monthly = daily.groupby("month", sort=True).agg(
            rank_ic_mean=("rank_ic", "mean"), excess_return_daily_mean=("excess_return", "mean"),
            turnover_mean=("turnover", "mean"), rank_ic_days=("rank_ic", "count"),
            return_days=("excess_return", "count"), turnover_days=("turnover", "count"),
        ).reset_index()
        monthly.to_csv(run_dir / "evaluation" / "月度指标.csv", index=False, encoding="utf-8-sig")
    lines.extend([
        "", "## 比赛综合分组成", "",
        "| 预测处理 | Rank IC 贡献 | 年化超额收益贡献 | 换手率贡献 | 综合分 |",
        "|---|---:|---:|---:|---:|",
    ])
    for alpha, label in ((1.0, "原始预测"), (choice["alpha"], "固定平滑预测")):
        score = holdout["metrics"][str(alpha)]["score"]
        if score is not None:
            lines.append(
                f"| {label} | {0.4 * score['rank_ic']['mean']:.8f} | "
                f"{0.3 * score['top_decile']['annual_excess']:.8f} | "
                f"{0.3 * score['turnover']['one_minus_turnover']:.8f} | {score['final_score']:.8f} |"
            )
    lines.extend([
        "", "## 与原实验的 2024 年比较", "",
        "原实验使用同一组 37 个因子、排序损失权重 0.3、固定平滑系数 0.3，选择第 3 周期模型。两个实验的逐日数据使用同一股票日期键和标签。2024 年已经查看，差值用于描述此次实验的历史变化。", "",
        "| 预测处理 | 指标 | 原实验 | 新实验 | 差值 |",
        "|---|---|---:|---:|---:|",
    ])
    comparison = build_baseline_comparison(config, run_dir, choice, holdout)
    for item in comparison["metrics"]:
        for key, label, percent in (
            ("rank_ic", "Rank IC", False),
            ("annual_excess", "Top10% 年化超额收益", True),
            ("annual_top_decile_return", "Top10% 年化组合收益", True),
            ("mean_turnover", "平均换手率", True),
            ("final_score", "综合分", False),
        ):
            base_value = item[f"baseline_{key}"]
            new_value = item[f"new_{key}"]
            delta = item[f"delta_{key}"]
            if percent:
                formatted = f"{base_value:.2%} | {new_value:.2%} | {delta:+.2%}"
            else:
                formatted = f"{base_value:.8f} | {new_value:.8f} | {delta:+.8f}"
            lines.append(f"| {item['prediction_processing']} | {label} | {formatted} |")
    lines.extend([
        "", "## 验证周期与相同周期比较", "",
        f"原实验选中第 {comparison['baseline_selected_candidate']['epoch']} 周期，新实验选中第 {choice['epoch']} 周期。两次实验以相同验证综合分规则选择周期。完整逐周期指标见 `evaluation/验证周期比较.csv`。",
        f"本次与基线使用相同输入摘要：`{comparison['input_identity_sha256']}`。基线配置、程序和评估文件摘要保存在 `基线产物摘要.json`。", "",
        "## 验证集因子依赖程度诊断", "",
        "| 来源组 | 平均依赖程度 | 达标日期数量 | 日期总数 |",
        "|---|---:|---:|---:|",
    ])
    dependence = read_json(directory / f"epoch_{choice['epoch']:02d}" / "dependence_metrics.json")[str(config["seed"])]
    for key, value in dependence["source_groups"].items():
        names = "、".join(config["dependence"]["source_groups"][int(key)])
        lines.append(f"| {names} | {value['mean_ratio']:.8f} | {value['passing_dates']} | {value['total_dates']} |")
    lines.extend([
        "", "## 本地留出集因子依赖程度", "",
        "| 来源组 | 平均依赖程度 | 达标日期数量 | 日期总数 |",
        "|---|---:|---:|---:|",
    ])
    holdout_dependence = read_json(run_dir / "evaluation" / "local_holdout" / "dependence_metrics.json")
    for key, value in holdout_dependence["source_groups"].items():
        names = "、".join(config["dependence"]["source_groups"][int(key)])
        lines.append(f"| {names} | {value['mean_ratio']:.8f} | {value['passing_dates']} | {value['total_dates']} |")
    lines.extend([
        "", "因子依赖比值反映将一组因子替换为训练中位数后，模型预测发生的变化。旧方案的 0.25 比值和 95% 日期比例仅作为参考，不影响本次选模。该诊断不识别因果效应。", "",
        "## 官方测试覆盖", "",
        f"合格记录预测：{coverage['eligible_predictions']:,} 条；缺少合格窗口：{coverage['uncovered_rows']:,} 条。",
        "完整提交状态：`complete_submission=false`。", "",
        "逐周期损失、数值质量、显存和耗时保存在 `evaluation/周期指标.csv`。逐日比较、月度比较及全年基线指标分别保存在 `evaluation/历史留出逐日比较.parquet`、`evaluation/月度指标.csv` 和 `evaluation/基线比较指标.json`。", "",
    ])
    (run_dir / "模型结果评估报告.md").write_text("\n".join(lines), encoding="utf-8")
    run_final_audit(config, config_sha256, run_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    config, config_sha256 = load_config(args.config)
    finalize(config, config_sha256, Path(args.run_dir).resolve())


if __name__ == "__main__":
    main()
