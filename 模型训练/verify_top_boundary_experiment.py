import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "模型训练" / "前10%边界排序实验" / "20261010_top_boundary_rank" / "B1_lambda_top_0.3"
B0 = RUN.parent / "B0_existing_no_dependence_official_2023"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    status = read_json(RUN / "调度状态.json")
    if status["completed_epochs"] != 8 or status["stopped"] is not True:
        raise ValueError("训练周期未完整结束")
    training_files = sorted(
        (RUN / "runs" / "single" / "lambda_0.3" / "seed_42").glob("epoch_[0-9][0-9]_training.json")
    )
    checkpoints = sorted(
        (RUN / "runs" / "single" / "lambda_0.3" / "seed_42").glob("checkpoint_epoch_[0-9][0-9].pt")
    )
    if len(training_files) != 8 or len(checkpoints) != 8:
        raise ValueError("训练统计或检查点数量不完整")
    training_rows = [read_json(path) for path in training_files]
    if any(row["sample_count"] != 4510263 or row["batch_count"] != 9393 for row in training_rows):
        raise ValueError("训练周期覆盖数量不一致")
    official = pd.read_csv(RUN / "2023官方评分比较.csv")
    if len(official) != 32:
        raise ValueError("2023 年官方候选数量不完整")
    b0_official = pd.read_csv(B0 / "2023官方评分比较.csv")
    if len(b0_official) != 32:
        raise ValueError("B0 2023 年官方候选数量不完整")
    b0_best = b0_official.sort_values(
        ["official_final_score", "official_rank_ic", "official_annual_excess", "epoch", "alpha"],
        ascending=[False, False, False, True, False],
        kind="mergesort",
    ).iloc[0]
    if int(b0_best["epoch"]) != 2 or abs(float(b0_best["alpha"]) - 0.3) > 1e-12:
        raise ValueError("B0 官方复评最优候选与既有选择不一致")
    choice = read_json(RUN / "选定配置.json")["final_candidate"]
    best = official.sort_values(
        ["official_final_score", "official_rank_ic", "official_annual_excess", "epoch", "alpha"],
        ascending=[False, False, False, True, False],
        kind="mergesort",
    ).iloc[0]
    if int(best["epoch"]) != choice["epoch"] or float(best["alpha"]) != choice["alpha"]:
        raise ValueError("选定配置不是官方综合分最高候选")
    stability = read_json(
        RUN / "runs" / "single" / "lambda_0.3" / "seed_42" / "epoch_03" / "stability_metrics.json"
    )
    stability_record = stability["seeds"]["42"]["alphas"]["0.3"]
    if stability_record["passed"] is not True or stability_record["passing_dates"] < 239:
        raise ValueError("选定周期稳定性核验未通过")
    holdout = read_json(RUN / "evaluation" / "local_holdout" / "local_holdout_metrics.json")
    holdout_score = holdout["0.3"]["score"]
    if holdout["0.3"]["quality_pass"] is not True or holdout_score["rows"] != 1091450:
        raise ValueError("2024 年历史留出评价未通过质量或数量核验")
    report_path = RUN / "模型结果评估报告.md"
    if not report_path.exists() or report_path.stat().st_size < 1000:
        raise ValueError("结果报告不存在或内容不完整")
    if list(RUN.rglob("submission.csv")):
        raise ValueError("新实验目录生成了禁止的 submission.csv")
    result = {
        "status": "passed",
        "training_epochs": len(training_files),
        "checkpoints": len(checkpoints),
        "official_2023_candidates": len(official),
        "b0_official_2023_candidates": len(b0_official),
        "b0_best_official_2023": {
            "epoch": int(b0_best["epoch"]),
            "alpha": float(b0_best["alpha"]),
            "final_score": float(b0_best["official_final_score"]),
            "rank_ic": float(b0_best["official_rank_ic"]),
            "annual_excess": float(b0_best["official_annual_excess"]),
            "mean_turnover": float(b0_best["official_mean_turnover"]),
        },
        "selected_candidate": choice,
        "stability": {
            "passed": stability_record["passed"],
            "passing_dates": stability_record["passing_dates"],
            "total_dates": stability_record["total_dates"],
        },
        "holdout_2024": {
            "rows": holdout_score["rows"],
            "dates": holdout_score["dates"],
            "final_score_alpha_0_3": holdout_score["final_score"],
            "quality_pass": holdout["0.3"]["quality_pass"],
        },
        "report_sha256": sha256(report_path),
        "key_files": {
            "official_comparison": sha256(RUN / "2023官方评分比较.csv"),
            "selection": sha256(RUN / "选定配置.json"),
            "holdout_metrics": sha256(RUN / "evaluation" / "local_holdout" / "local_holdout_metrics.json"),
        },
    }
    (RUN / "最终核验.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
