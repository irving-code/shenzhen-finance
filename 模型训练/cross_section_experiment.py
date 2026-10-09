import hashlib
import json
import os
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


SPLITS = ("train_fit", "validation", "local_holdout", "competition_test")


def source_hashes(schema_version=4):
    directory = Path(__file__).resolve().parent
    names = (
        "cross_section_experiment.py", "cross_section_input.py", "dependence_constraint.py",
        "ranking_loss.py", "lstm_transformer_small.py", "train_cross_section_rank.py",
        "evaluate_cross_section_rank.py", "verify_cross_section_rank.py",
        "finalize_cross_section_rank.py", "coordinate_cross_section_rank.py",
        "evaluate_first_lstm_result.py", "run_second_score_experiment.py",
    )
    if schema_version == 5:
        names = (*names, "audit_cross_section_delivery.py")
    return {name: file_sha256(directory / name) for name in names}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def temporary_path(target):
    root = Path(__file__).resolve().parent.parent
    target = Path(target).resolve()
    if target.is_relative_to(root / "模型训练" / "去除因子依赖惩罚实验"):
        temporary_dir = "去除因子依赖惩罚实验"
    else:
        temporary_dir = "横截面排序实验"
    directory = root / "tmp" / temporary_dir
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{Path(target).name}.{uuid.uuid4().hex}.writing"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_path(path)
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_config(path):
    path = Path(path).resolve()
    config = read_json(path)
    required = {
        "project_root", "protocol", "paths", "data", "features", "splits", "model", "training",
        "sampling", "ranking", "dependence", "model_id", "feature_mode", "seed", "selection", "quality",
        "execution", "verification",
    }
    if set(config) != required:
        raise ValueError("训练配置区域缺失或包含未知区域")
    root = path.parent.parent.resolve()
    if path.parent != root / "模型训练":
        raise ValueError("训练配置文件与项目根目录不一致")
    config["project_root"] = str(root)
    if config["model_id"] != "single" or config["feature_mode"] != "extended37" or config["seed"] != 42:
        raise ValueError("单模型、37 通道输入或随机种子与计划不一致")
    schema_version = config["protocol"]["schema_version"]
    expected_protocols = {
        4: "20261009_本机横截面依赖约束",
        5: "20261009_去除因子依赖惩罚",
    }
    if schema_version not in expected_protocols or config["protocol"]["plan_id"] != expected_protocols[schema_version]:
        raise ValueError("实验方案版本或编号不一致")
    if config["protocol"]["execution_status"] != "authorized_to_run":
        raise ValueError("本机训练配置状态未启用")
    if config["protocol"]["candidate_count"] != 1 or config["protocol"]["run_count"] != 1:
        raise ValueError("训练运行数量与单模型方案不一致")
    if config["dependence"]["enabled_model"] != "single" or len(config["dependence"]["source_groups"]) != 7:
        raise ValueError("因子依赖约束定义不完整")
    if config["execution"]["mode"] != "local_single_gpu" or config["execution"]["max_concurrent_training_tasks"] != 1:
        raise ValueError("本机单任务执行设置不一致")
    if config["selection"]["alpha"] != 0.3:
        raise ValueError("平滑参数与计划不一致")
    if config["training"]["device"] != "cuda" or config["training"]["batch_size"] != 512:
        raise ValueError("训练设备或批次上限与计划不一致")
    if config["ranking"]["weight"] != 0.3 or config["training"]["early_stopping_count_start_epoch"] != 6:
        raise ValueError("排序损失权重或早停起始周期与计划不一致")
    if (config["training"]["min_epochs"], config["training"]["max_epochs"], config["training"]["early_stopping_patience"]) != (6, 12, 3):
        raise ValueError("周期范围或早停耐心值与计划不一致")
    if config["training"]["early_stopping_alpha"] != config["selection"]["alpha"] or config["selection"]["inference_seed"] != config["seed"]:
        raise ValueError("训练、选择和推理的固定参数不一致")
    if (config["ranking"]["beta"], config["ranking"]["tau"]) != (0.02, 0.02):
        raise ValueError("收益损失或排序损失参数与计划不一致")
    if schema_version == 4:
        if (config["dependence"]["limit"], config["dependence"]["penalty_weight"], config["dependence"]["window_scope"]) != (0.25, 0.1, "all_20_timesteps"):
            raise ValueError("历史方案因子依赖约束参数与计划不一致")
    else:
        dependence = config["dependence"]
        if (
            config["protocol"]["fixed_date"] != "2026-10-09"
            or config["protocol"]["holdout_status"] != "previously_viewed"
            or config["protocol"]["initialization"] != "fresh_local_single_model"
            or config["paths"]["experiment_dir"] != "模型训练/去除因子依赖惩罚实验"
            or config["paths"]["temporary_dir"] != "tmp/去除因子依赖惩罚实验"
            or config["paths"]["baseline_dir"] != "模型训练/横截面因子与排序损失实验/20261009_local_single"
            or dependence["limit"] != 0.25
            or dependence["penalty_weight"] != 0.0
            or dependence["window_scope"] != "all_20_timesteps"
            or dependence["training_penalty_enabled"] is not False
            or dependence["selection_gate_enabled"] is not False
            or dependence["diagnostics_enabled"] is not True
            or dependence["minimum_passing_date_fraction"] != 0.95
        ):
            raise ValueError("新版依赖诊断或惩罚开关与实验方案不一致")
    if config["data"]["build_id"] != "20261008_横截面排序优化":
        raise ValueError("训练构建编号与验收构建不一致")
    return config, file_sha256(path)


def training_dependence_enabled(config, group):
    if group != config["dependence"]["enabled_model"]:
        return False
    if config["protocol"]["schema_version"] == 4:
        return True
    return config["dependence"]["training_penalty_enabled"] is True


def selection_dependence_gate_enabled(config):
    if config["protocol"]["schema_version"] == 4:
        return True
    return config["dependence"]["selection_gate_enabled"] is True


def project_path(config, name):
    return Path(config["project_root"]) / config["paths"][name]


def input_path(config, filename):
    return project_path(config, "input_dir") / filename


def index_path(config, split):
    if split not in SPLITS:
        raise ValueError(f"未知数据分区: {split}")
    return input_path(config, f"样本索引_{split}.parquet")


def validate_completed_input(config):
    input_dir = project_path(config, "input_dir")
    base_path = project_path(config, "base_metadata")
    manifest = read_json(input_dir / "数据交付清单.json")
    metadata_path = input_dir / "输入元数据.json"
    metadata = read_json(metadata_path)
    verification_path = input_dir / "数据核验结果.json"
    verification = read_json(verification_path)
    base = read_json(base_path)
    build_id = config["data"]["build_id"]
    if manifest["status"] != "complete" or manifest["data_verified"] is not True:
        raise ValueError("数据交付状态不完整")
    if metadata["status"] != "complete" or metadata["data_verified"] is not True:
        raise ValueError("输入元数据状态不完整")
    if verification["status"] != "passed":
        raise ValueError("数据核验状态未通过")
    if any(item["build_id"] != build_id for item in (manifest, metadata, verification)):
        raise ValueError("数据构建编号不一致")
    hashes = {
        "metadata": file_sha256(metadata_path),
        "verification": file_sha256(verification_path),
        "report": file_sha256(input_dir / "预处理报告.md"),
        "feature_store": file_sha256(input_dir / "新增特征存储.dat"),
        "parameter": file_sha256(input_dir / "处理参数_train_fit.json"),
        "base_store": file_sha256(base_path.parent / base["feature_store_file"]),
        "base_metadata": file_sha256(base_path),
        "stock_metadata": file_sha256(base_path.parent / base["stock_metadata_file"]),
        "factor_dates": file_sha256(project_path(config, "factor_dates")),
    }
    if hashes["metadata"] != manifest["metadata_sha256"] or hashes["metadata"] != config["data"]["metadata_sha256"]:
        raise ValueError("输入元数据摘要不一致")
    for key, manifest_key in (("verification", "verification_sha256"), ("report", "report_sha256"), ("feature_store", "feature_store_sha256")):
        if hashes[key] != manifest[manifest_key]:
            raise ValueError(f"{key} 文件摘要不一致")
    if hashes["parameter"] != metadata["parameter_sha256"]:
        raise ValueError("处理参数摘要不一致")
    if hashes["base_store"] != metadata["source_sha256"]["base_store"]:
        raise ValueError("原有特征存储摘要不一致")
    if base["window_length"] != metadata["window_length"] or base["window_length"] != 20:
        raise ValueError("窗口长度不一致")
    if base["feature_count"] != 27 or metadata["extra_feature_count"] != 10:
        raise ValueError("模型输入通道数量不一致")
    if metadata["feature_columns"] != config["features"]["extra_columns"]:
        raise ValueError("新增因子列顺序不一致")
    if tuple(base["feature_store_shape"]) != (9499950, 27) or tuple(metadata["feature_store_shape"]) != (9499950, 10):
        raise ValueError("特征存储形状不一致")
    if base["feature_dtype"] != "float32" or metadata["feature_dtype"] != "float32":
        raise ValueError("特征存储类型不一致")
    if (base_path.parent / base["feature_store_file"]).stat().st_size != 9499950 * 27 * 4:
        raise ValueError("原有特征存储大小不一致")
    if (input_dir / "新增特征存储.dat").stat().st_size != 9499950 * 10 * 4:
        raise ValueError("新增特征存储大小不一致")
    index_records = {}
    for split in SPLITS:
        path = index_path(config, split)
        digest = file_sha256(path)
        if digest != manifest["index_sha256"][split]:
            raise ValueError(f"{split} 索引摘要不一致")
        index_file = pq.ParquetFile(path)
        rows = index_file.metadata.num_rows
        if rows != config["data"]["rows"][split]:
            raise ValueError(f"{split} 索引行数不一致")
        frame = pd.read_parquet(path, columns=[
            "ts_code", "target_date", "prediction_eligible", "training_eligible",
            "selection_eligible", "label_realization_date", "y_ret_1d"
        ] if split != "competition_test" else [
            "ts_code", "target_date", "prediction_eligible", "training_eligible",
            "selection_eligible", "label_realization_date"
        ])
        if frame.duplicated(["ts_code", "target_date"]).any():
            raise ValueError(f"{split} 索引有重复股票日期键")
        prediction_rows = int(frame["prediction_eligible"].sum())
        if prediction_rows != config["data"]["prediction_eligible"][split]:
            raise ValueError(f"{split} 预测资格数量不一致")
        training_rows = int(frame["training_eligible"].sum())
        selection_rows = int(frame["selection_eligible"].sum())
        if split == "train_fit" and training_rows != config["data"]["training_eligible"]:
            raise ValueError("训练资格数量不一致")
        if split == "validation" and selection_rows != config["data"]["selection_eligible"]:
            raise ValueError("验证选择资格数量不一致")
        if split not in ("train_fit",) and training_rows:
            raise ValueError(f"{split} 存在训练资格")
        if split not in ("validation",) and selection_rows:
            raise ValueError(f"{split} 存在验证选择资格")
        index_records[split] = {
            "rows": rows, "prediction_eligible": prediction_rows,
            "training_eligible": training_rows, "selection_eligible": selection_rows,
            "sha256": digest,
        }
    return {"build_id": build_id, "file_sha256": hashes, "indexes": index_records}
