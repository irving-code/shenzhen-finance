import json
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
MODEL_ROOT = ROOT / "模型训练"
EXPERIMENT_ROOT = MODEL_ROOT / "严格排序损失消融实验" / "20261010_rank_ablation"
TEST_PATH = ROOT / "因子数据" / "测试集_X_第一次LST-Transformer因子.parquet"
TEST_ROWS = len(pd.read_parquet(TEST_PATH, columns=["ts_code", "trade_date"]))


def score_record(name, period, score, submission_path=None):
    return {
        "model": name,
        "evaluation_period": period,
        "rank_ic": score["rank_ic"]["mean"],
        "icir": score["rank_ic"]["icir"],
        "annual_excess": score["top_decile"]["annual_excess"],
        "mean_turnover": score["turnover"]["mean_turnover"],
        "final_score": score["final_score"],
        "submission_path": str(submission_path) if submission_path else "",
    }


def collect_models():
    models = []
    for weight in (0.0, 0.15, 0.3, 0.5):
        for seed in (42, 43, 44):
            run_dir = EXPERIMENT_ROOT / "runs" / f"rank_{weight:.2f}" / f"seed_{seed}"
            summary_path = run_dir / "运行摘要.json"
            if not summary_path.exists():
                continue
            holdout = json.loads((run_dir / "evaluation" / "本地留出集指标.json").read_text(encoding="utf-8"))
            score = holdout["metrics"]["0.3"]["score"]
            models.append(score_record(f"严格消融 rank={weight:.2f}, seed={seed}", "2024本地留出", score))

    historical = [
        (
            "37因子、排序损失0.3、关闭依赖惩罚",
            MODEL_ROOT / "去除因子依赖惩罚实验" / "20261009_local_single_no_dependence" / "evaluation" / "本地留出集指标.json",
            "0.3",
            None,
        ),
        (
            "37因子、排序损失0.3、依赖惩罚0.1",
            MODEL_ROOT / "横截面因子与排序损失实验" / "20261009_local_single" / "evaluation" / "本地留出集指标.json",
            "0.3",
            None,
        ),
    ]
    for name, path, alpha, submission in historical:
        record = json.loads(path.read_text(encoding="utf-8"))
        models.append(score_record(name, "2024本地留出", record["metrics"][alpha]["score"], submission))

    second = json.loads((MODEL_ROOT / "第二次综合分验证实验" / "本地留出集比赛指标评估.json").read_text(encoding="utf-8"))
    models.append(score_record("27因子、无排序损失、alpha=0.3", "2024本地留出", second["local_group_b_smoothed"], None))

    first = json.loads((MODEL_ROOT / "第一次LST-Transformer实验" / "本地留出集比赛指标评估.json").read_text(encoding="utf-8"))
    models.append(score_record("第一次 LSTM-Transformer", "2024本地留出", first["model"], None))

    models.extend([
        {"model": "MLP", "evaluation_period": "2024本地留出", "rank_ic": 0.09553918706, "icir": 0.9078544243, "annual_excess": 0.2767518737, "mean_turnover": 0.8194865018, "final_score": 0.1753952864, "submission_path": str(MODEL_ROOT / "MLP实验" / "submission.csv")},
        {"model": "LightGBM", "evaluation_period": "2024本地留出", "rank_ic": 0.06820252785, "icir": 0.4471754785, "annual_excess": 0.2898163978, "mean_turnover": 0.8285200249, "final_score": 0.165669923, "submission_path": str(MODEL_ROOT / "LightGBM实验结果" / "submission.csv")},
        {"model": "Top边界排序", "evaluation_period": "2024本地留出", "rank_ic": 0.019629, "icir": 0.109555, "annual_excess": 0.040396, "mean_turnover": 0.311178, "final_score": 0.22661692, "submission_path": ""},
    ])
    return pd.DataFrame(models)


def add_coverage(frame):
    frame = frame.copy()
    frame["submission_rows"] = 0
    frame["complete_test_coverage"] = False
    for index, row in frame.iterrows():
        path = Path(row["submission_path"]) if row["submission_path"] else None
        if path is None or not path.exists():
            continue
        values = pd.read_csv(path, usecols=["ts_code", "trade_date"])
        complete = len(values) == TEST_ROWS and not values.duplicated(["ts_code", "trade_date"]).any()
        frame.loc[index, "submission_rows"] = len(values)
        frame.loc[index, "complete_test_coverage"] = bool(complete)
    return frame


def main():
    models = add_coverage(collect_models())
    models = models.sort_values(["final_score", "complete_test_coverage"], ascending=[False, False]).reset_index(drop=True)
    comparison_path = MODEL_ROOT / "全模型比较.csv"
    models.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    complete = models.loc[models["complete_test_coverage"]].copy()
    if complete.empty:
        raise ValueError("所有候选模型都没有完整覆盖测试集 X")
    selected = complete.iloc[0]
    source = Path(selected["submission_path"])
    target = ROOT / "submission.csv"
    shutil.copyfile(source, target)
    result = {
        "test_rows": TEST_ROWS,
        "highest_local_holdout_model": str(models.iloc[0]["model"]),
        "highest_local_holdout_score": float(models.iloc[0]["final_score"]),
        "selected_submission_model": str(selected["model"]),
        "selected_submission_score": float(selected["final_score"]),
        "selected_submission_source": str(source.relative_to(ROOT)),
        "submission_path": str(target.relative_to(ROOT)),
        "submission_rows": int(selected["submission_rows"]),
        "complete_test_coverage": True,
        "excluded_high_score_models_without_full_coverage": models.loc[~models["complete_test_coverage"]].head(10)["model"].tolist(),
    }
    (MODEL_ROOT / "最终模型选择.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# 全模型比较与最终提交选择", "",
        f"测试集 X 记录数：{TEST_ROWS:,}。候选模型按 2024 年本地留出综合分排序；最终提交还要求提交文件完整覆盖测试集键。", "",
        "| 模型 | 评价周期 | Rank IC | ICIR | 年化超额收益 | 平均换手率 | 综合分 | 提交行数 | 完整覆盖 |", "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in models.iterrows():
        lines.append(f"| {row['model']} | {row['evaluation_period']} | {row['rank_ic']:.6f} | {row['icir']:.6f} | {row['annual_excess']:.2%} | {row['mean_turnover']:.2%} | {row['final_score']:.8f} | {int(row['submission_rows']):,} | {'是' if row['complete_test_coverage'] else '否'} |")
    lines.extend([
        "", f"本地留出综合分最高的模型为 `{result['highest_local_holdout_model']}`，分数 `{result['highest_local_holdout_score']:.8f}`。该模型没有完整测试集提交覆盖，因此最终提交选用综合分最高且完整覆盖测试集的 `{result['selected_submission_model']}`，分数 `{result['selected_submission_score']:.8f}`。", "",
        f"最终文件：`{result['submission_path']}`，行数 `{result['submission_rows']:,}`。",
    ])
    (MODEL_ROOT / "最终模型选择报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
