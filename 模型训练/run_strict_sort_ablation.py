import argparse
import hashlib
import json
import shutil
from pathlib import Path

from cross_section_experiment import (
    file_sha256, load_config, project_path, read_json, source_hashes,
    validate_completed_input, write_json,
)
from evaluate_cross_section_rank import evaluate_holdout, select_configurations
from train_cross_section_rank import environment_record, run_training


ROOT = Path(__file__).resolve().parent.parent
BASE_CONFIG = ROOT / "模型训练" / "去除因子依赖惩罚实验配置.json"
EXPERIMENT_ROOT = ROOT / "模型训练" / "严格排序损失消融实验" / "20261010_rank_ablation"
CONFIG_ROOT = EXPERIMENT_ROOT / "configs"
WEIGHTS = (0.0, 0.15, 0.3, 0.5)
SEEDS = (42, 43, 44)


def digest_record(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def build_config(weight, seed):
    config = read_json(BASE_CONFIG)
    config["protocol"].update({
        "schema_version": 6,
        "plan_id": "20261010_严格排序损失消融",
        "fixed_date": "2026-10-10",
        "initialization": "fresh_ablation_matrix_run",
        "candidate_count": 1,
        "run_count": 1,
        "holdout_status": "previously_viewed",
        "matrix_weights": list(WEIGHTS),
        "matrix_seeds": list(SEEDS),
    })
    config["paths"].update({
        "experiment_dir": "模型训练/严格排序损失消融实验/20261010_rank_ablation",
        "temporary_dir": "tmp/严格排序损失消融",
    })
    config["ranking"]["weight"] = weight
    config["seed"] = seed
    config["selection"]["inference_seed"] = seed
    config["dependence"]["training_penalty_enabled"] = False
    config["dependence"]["selection_gate_enabled"] = False
    config["dependence"]["diagnostics_enabled"] = True
    return config


def config_path(weight, seed):
    return CONFIG_ROOT / f"rank_{weight:.2f}_seed_{seed}.json"


def ensure_configs():
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    paths = []
    for weight in WEIGHTS:
        for seed in SEEDS:
            path = config_path(weight, seed)
            config = build_config(weight, seed)
            path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            paths.append(path)
    return paths


def run_path_for(weight, seed):
    return EXPERIMENT_ROOT / "runs" / f"rank_{weight:.2f}" / f"seed_{seed}"


def prepare_run(config_path_value, run_dir):
    config, config_sha256 = load_config(config_path_value)
    input_record = validate_completed_input(config)
    input_sha256 = digest_record(input_record)
    run_dir = Path(run_dir)
    state_path = run_dir / "调度状态.json"
    complete = (
        state_path.exists()
        and read_json(state_path).get("stopped") is True
        and (run_dir / "选定配置.json").exists()
        and (run_dir / "evaluation" / "本地留出集指标.json").exists()
    )
    if complete:
        return config, config_sha256, input_record, True
    if run_dir.exists() and any(run_dir.iterdir()):
        snapshot = read_json(run_dir / "配置快照.json")
        if snapshot["config_sha256"] != config_sha256 or snapshot["input_sha256"] != input_sha256:
            raise ValueError("已有运行目录与当前严格消融配置或输入身份不一致")
        if read_json(run_dir / "verification" / "预检结果.json")["status"] != "passed":
            raise ValueError("已有运行目录的输入核验状态未通过")
        return config, config_sha256, input_record, False
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "verification").mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "配置快照.json", {
        "config_sha256": config_sha256,
        "input_sha256": input_sha256,
        "config": config,
        "source_sha256": source_hashes(config["protocol"]["schema_version"]),
    })
    write_json(run_dir / "环境记录.json", environment_record())
    write_json(run_dir / "输入接入记录.json", input_record)
    write_json(run_dir / "verification" / "预检结果.json", {
        "status": "passed",
        "type": "strict_ablation_input_identity",
        "config_sha256": config_sha256,
        "input_sha256": input_sha256,
        "input_record_sha256": file_sha256(run_dir / "输入接入记录.json"),
    })
    neutral_source = ROOT / "模型训练" / "去除因子依赖惩罚实验" / "20261009_local_single_no_dependence" / "dependence_neutral_values.json"
    neutral = read_json(neutral_source)
    write_json(run_dir / "dependence_neutral_values.json", {
        "config_sha256": config_sha256,
        "input_sha256": input_sha256,
        "method": neutral["method"],
        "source_rows": neutral["source_rows"],
        "values": neutral["values"],
    })
    return config, config_sha256, input_record, False


def run_one(config_path_value):
    config, config_sha256, input_record, complete = prepare_run(
        config_path_value,
        run_path_for(
            read_json(config_path_value)["ranking"]["weight"],
            read_json(config_path_value)["seed"],
        ),
    )
    run_dir = run_path_for(config["ranking"]["weight"], config["seed"])
    if complete:
        return read_json(run_dir / "运行摘要.json")
    resume = (run_dir / "调度状态.json").exists() or any((run_dir / "runs").glob("*"))
    run_training(config, config_sha256, input_record, run_dir, resume)
    choice = select_configurations(config, run_dir, __import__("torch").device("cuda"))
    if choice is not None:
        evaluate_holdout(config, run_dir, __import__("torch").device("cuda"))
    holdout = read_json(run_dir / "evaluation" / "本地留出集指标.json") if choice is not None else None
    summary = {
        "weight": config["ranking"]["weight"],
        "seed": config["seed"],
        "run_dir": run_dir.relative_to(ROOT).as_posix(),
        "config_sha256": config_sha256,
        "input_sha256": digest_record(input_record),
        "selection": read_json(run_dir / "选定配置.json"),
        "holdout": holdout,
    }
    write_json(run_dir / "运行摘要.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    ensure_configs()
    paths = [config_path(weight, seed) for weight in WEIGHTS for seed in SEEDS]
    if args.only:
        selected = set(args.only)
        paths = [path for path in paths if path.stem in selected]
    for path in paths:
        print(json.dumps({"phase": "start", "config": str(path)}, ensure_ascii=False), flush=True)
        result = run_one(path)
        print(json.dumps({"phase": "completed", "weight": result["weight"], "seed": result["seed"]}, ensure_ascii=False), flush=True)
    expected = [run_path_for(weight, seed) / "运行摘要.json" for weight in WEIGHTS for seed in SEEDS]
    if all(path.exists() for path in expected):
        from aggregate_strict_sort_ablation import aggregate
        aggregate()
    else:
        print(json.dumps({"phase": "aggregation_deferred", "completed_runs": sum(path.exists() for path in expected), "expected_runs": len(expected)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
