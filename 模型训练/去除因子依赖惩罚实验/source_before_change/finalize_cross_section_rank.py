import argparse
from pathlib import Path

import pandas as pd

from cross_section_experiment import load_config, read_json, write_json
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


def report_lines(config, run_dir, state, records):
    environment = read_json(run_dir / "环境记录.json")
    lines = [
        "# 横截面因子与因子依赖约束模型评估报告", "",
        f"设备：{environment['device']}；随机种子：{config['seed']}；输入：20 个交易日、37 个因子。",
        f"训练样本：{config['data']['training_eligible']:,} 条；验证选择样本：{config['data']['selection_eligible']:,} 条。",
        f"训练周期：{state['completed_epochs']}；停止原因：`{state['stop_reason']}`。",
        f"排序损失权重：{config['ranking']['weight']}；依赖约束权重：{config['dependence']['penalty_weight']}；固定平滑参数：{config['selection']['alpha']}。", "",
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
            "全部训练周期未满足实验方案要求的质量、因子依赖程度与推理稳定性检查。",
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
        "所选检查点已通过原始预测质量、七组因子依赖程度和双批量排序稳定性检查。", "",
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
                f"{score['top_decile']['annual_excess']:.8f} | {score['turnover']['mean_turnover']:.8f} | {score['final_score']:.8f} |"
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
        "", "2024 年数据属于已查看的本地留出集，评价结果用于描述当前模型表现。", "",
        "## 验证集因子依赖程度", "",
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
        "", "依赖程度反映输入替换后的预测敏感度。", "",
        "## 官方测试覆盖", "",
        f"合格记录预测：{coverage['eligible_predictions']:,} 条；缺少合格窗口：{coverage['uncovered_rows']:,} 条。",
        "完整提交状态：`complete_submission=false`。", "",
        "逐周期损失、数值质量、显存和耗时保存在 `evaluation/周期指标.csv`。", "",
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
