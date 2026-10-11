import argparse
import json
from pathlib import Path

import pandas as pd

from cross_section_experiment import load_config, validate_completed_input, write_json
from evaluate_cross_section_rank import run_path


def select_candidate(config_path, run_dir):
    config, config_hash = load_config(config_path)
    validate_completed_input(config)
    run_dir = Path(run_dir).resolve()
    snapshot = json.loads((run_dir / "配置快照.json").read_text(encoding="utf-8"))
    if snapshot["config_sha256"] != config_hash:
        raise ValueError("运行目录与当前配置摘要不一致")
    comparison_path = run_dir / "2023官方评分比较.csv"
    comparison = pd.read_csv(comparison_path)
    required = {
        "epoch", "alpha", "official_final_score", "official_rank_ic",
        "official_annual_excess", "official_mean_turnover",
    }
    if not required.issubset(comparison.columns) or comparison.empty:
        raise ValueError("2023 年官方评分比较表不完整")
    comparison = comparison.sort_values(
        ["official_final_score", "official_rank_ic", "official_annual_excess", "epoch", "alpha"],
        ascending=[False, False, False, True, False],
        kind="mergesort",
    ).reset_index(drop=True)
    best = comparison.iloc[0]
    final_candidate = {
        "group": config["model_id"],
        "rank_weight": config["ranking"]["weight"],
        "seed": config["seed"],
        "epoch": int(best["epoch"]),
        "alpha": float(best["alpha"]),
        "final_score": float(best["official_final_score"]),
        "rank_ic": float(best["official_rank_ic"]),
        "annual_excess": float(best["official_annual_excess"]),
        "mean_turnover": float(best["official_mean_turnover"]),
        "selection_metric": "organizer_evaluate_2023_final_score",
    }
    write_json(run_dir / "选定配置.json", {
        "status": "selected",
        "selection_source": "2023官方评分比较.csv",
        "final_candidate": final_candidate,
        "candidate_count": len(comparison),
    })
    comparison.to_csv(run_dir / "验证周期指标.csv", index=False, encoding="utf-8-sig")
    return final_candidate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    select_candidate(args.config, args.run_dir)


if __name__ == "__main__":
    main()
