import argparse
import json
import os
from pathlib import Path

import torch

from cross_section_experiment import load_config, read_json, source_hashes, validate_completed_input, write_json
from evaluate_cross_section_rank import evaluate_competition_test, evaluate_holdout, select_configurations
from finalize_cross_section_rank import finalize
from train_cross_section_rank import run_training, set_determinism
from verify_cross_section_rank import run_preflight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config, config_sha256 = load_config(args.config)
    run_dir = Path(args.run_dir).resolve()
    set_determinism(config["seed"])
    if not torch.cuda.is_available():
        raise ValueError("本机 CUDA 设备不可用")
    if args.resume:
        snapshot = read_json(run_dir / "配置快照.json")
        if snapshot["config_sha256"] != config_sha256 or snapshot["source_sha256"] != source_hashes():
            raise ValueError("恢复运行的配置或程序摘要不一致")
        if read_json(run_dir / "verification" / "预检结果.json")["status"] != "passed":
            raise ValueError("恢复运行需要已通过的真实数据预检")
    else:
        print(json.dumps({"phase": "preflight", "pid": os.getpid(), "run_dir": str(run_dir)}, ensure_ascii=False), flush=True)
        run_preflight(config, config_sha256, run_dir)
    input_record = validate_completed_input(config)
    write_json(run_dir / "执行状态.json", {"phase": "training", "pid": os.getpid()})
    run_training(config, config_sha256, input_record, run_dir, args.resume)
    torch.cuda.empty_cache()
    device = torch.device("cuda")
    write_json(run_dir / "执行状态.json", {"phase": "selection", "pid": os.getpid()})
    choice = select_configurations(config, run_dir, device)
    if choice is not None:
        write_json(run_dir / "执行状态.json", {"phase": "holdout", "pid": os.getpid()})
        evaluate_holdout(config, run_dir, device)
        write_json(run_dir / "执行状态.json", {"phase": "competition_test", "pid": os.getpid()})
        evaluate_competition_test(config, run_dir, device)
    finalize(config, config_sha256, run_dir)
    write_json(run_dir / "执行状态.json", {
        "phase": "completed", "model_selected": choice is not None,
        "report": "模型结果评估报告.md", "pid": os.getpid(),
    })
    print(json.dumps({"phase": "completed", "report": str(run_dir / "模型结果评估报告.md")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
