import argparse
import importlib.util
import json
from pathlib import Path

import pandas as pd
import torch

from cross_section_experiment import load_config, validate_completed_input, write_json
from evaluate_cross_section_rank import (
    attach_evaluation,
    evaluate_prediction,
    load_checkpoint_model,
    predict_split,
    run_path,
)
from run_second_score_experiment import smooth_predictions


def official_evaluator(project_root):
    path = Path(project_root) / "赛题五" / "evaluate.py"
    spec = importlib.util.spec_from_file_location("organizer_evaluate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.evaluate


def write_shared_official_inputs(evaluation, output_dir):
    selected = evaluation.loc[evaluation["selection_eligible"].eq(1)].copy()
    if selected.empty:
        raise ValueError("2023 年选择资格记录为空")
    keys = ["ts_code", "trade_date"]
    selected[keys + ["y_ret_1d"]].to_csv(
        output_dir / "测试集_Y.csv", index=False, encoding="utf-8-sig"
    )
    selected[keys + ["flag_limit_up"]].to_csv(
        output_dir / "测试集_X.csv", index=False, encoding="utf-8-sig"
    )


def write_candidate_prediction(evaluation, alpha, output_dir):
    selected = smooth_predictions(evaluation, alpha).loc[
        lambda frame: frame["selection_eligible"].eq(1)
    ].copy()
    if selected.empty:
        raise ValueError("2023 年选择资格记录为空")
    selected[["ts_code", "trade_date"]].assign(
        pred=selected["smooth_pred"].to_numpy()
    ).to_csv(output_dir / "候选预测.csv", index=False, encoding="utf-8-sig")


def evaluate_epochs(config_path, run_dir, alphas, source_run_dir=None):
    config, config_hash = load_config(config_path)
    validate_completed_input(config)
    run_dir = Path(run_dir).resolve()
    source_run_dir = Path(source_run_dir).resolve() if source_run_dir else run_dir
    snapshot = json.loads((source_run_dir / "配置快照.json").read_text(encoding="utf-8"))
    if snapshot["config_sha256"] != config_hash:
        raise ValueError("运行目录与当前配置摘要不一致")
    if not torch.cuda.is_available():
        raise ValueError("正式评价需要可用的 CUDA 设备")
    device = torch.device("cuda")
    evaluate_official = official_evaluator(config["project_root"])
    shared_input_dir = run_dir / "2023官方临时评分目录"
    shared_input_dir.mkdir(parents=True, exist_ok=True)
    rank_weight = config["ranking"]["weight"]
    group = config["model_id"]
    seed = config["seed"]
    source_base = run_path(source_run_dir, group, rank_weight, seed)
    output_base = run_path(run_dir, group, rank_weight, seed)
    checkpoints = sorted(source_base.glob("checkpoint_epoch_[0-9][0-9].pt"))
    if not checkpoints:
        raise ValueError("运行目录没有可评价检查点")
    comparison = []
    for checkpoint in checkpoints:
        epoch = int(checkpoint.stem.split("_")[-1])
        model = load_checkpoint_model(
            config, source_run_dir, group, rank_weight, seed, epoch, device
        )
        prediction = predict_split(config, model, "validation", device)
        epoch_dir = output_base / f"epoch_{epoch:02d}"
        metrics = evaluate_prediction(
            config, prediction, "validation", alphas, epoch_dir, selection_only=True
        )
        evaluation = attach_evaluation(config, prediction, "validation")
        if not (shared_input_dir / "测试集_Y.csv").exists():
            write_shared_official_inputs(evaluation, shared_input_dir)
        for alpha in alphas:
            score_dir = epoch_dir / "official_score" / f"alpha_{alpha:g}"
            score_dir.mkdir(parents=True, exist_ok=True)
            write_candidate_prediction(evaluation, alpha, score_dir)
            score = evaluate_official(
                str(score_dir / "候选预测.csv"), str(shared_input_dir)
            )
            write_json(score_dir / "官方评分.json", score)
            record = {
                "epoch": epoch,
                "alpha": alpha,
                "official_final_score": float(score["final_score"]),
                "official_rank_ic": float(score["ic_mean"]),
                "official_annual_excess": float(score["annual_excess"]),
                "official_mean_turnover": float(score["mean_turnover"]),
                "internal_score": metrics[str(alpha)]["score"],
            }
            comparison.append(record)
        del model
        torch.cuda.empty_cache()
    pd.DataFrame(comparison).to_csv(
        run_dir / "2023官方评分比较.csv", index=False, encoding="utf-8-sig"
    )
    write_json(run_dir / "2023官方评分比较.json", {"candidates": comparison})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--source-run-dir")
    parser.add_argument("--alphas", nargs="+", type=float, default=[1.0, 0.7, 0.5, 0.3])
    args = parser.parse_args()
    if any(alpha <= 0 for alpha in args.alphas):
        raise ValueError("平滑参数必须为正数")
    evaluate_epochs(args.config, args.run_dir, args.alphas, args.source_run_dir)


if __name__ == "__main__":
    main()
