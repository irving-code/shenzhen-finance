import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from cross_section_experiment import (
    file_sha256, index_path, load_config, project_path, read_json, temporary_path, write_json,
)
from verify_cross_section_rank import run_final_audit


def audit_delivery(config, config_sha256, run_dir):
    run_dir = Path(run_dir)
    if read_json(run_dir / "配置快照.json")["config_sha256"] != config_sha256:
        raise ValueError("交付核验配置摘要不一致")
    output = run_dir / "competition_test"
    keys = ["ts_code", "target_date"]
    criteria = {
        "sequence_available": "序列可用性未通过",
        "sequence_features_finite": "窗口特征有限性未通过",
        "sequence_inside_valid": "窗口有效区间资格未通过",
        "sample_eligible": "样本资格未通过",
        "target_inside_valid_interval": "目标日有效区间资格未通过",
        "target_current_market_valid": "目标日市场有效性未通过",
    }
    columns = keys + list(criteria) + [
        "sequence_length", "history_available_count_20d", "prediction_eligible",
    ]
    source_path = index_path(config, "competition_test")
    source = pd.read_parquet(source_path, columns=columns)
    missing = source.loc[source["prediction_eligible"].eq(0)].reset_index(drop=True)
    original_missing = pd.read_parquet(project_path(config, "input_dir") / "官方测试覆盖缺失键.parquet")
    actual_keys = missing[keys].sort_values(keys).reset_index(drop=True)
    expected_keys = original_missing[keys].sort_values(keys).reset_index(drop=True)
    if len(missing) != config["data"]["competition_test_uncovered"] or not actual_keys.equals(expected_keys):
        raise ValueError("缺失记录与已验收数据资格不一致")
    if not missing[list(criteria)].isin([0, 1]).all().all():
        raise ValueError("资格字段出现未定义数值")
    missing["资格未通过字段"] = missing.apply(
        lambda row: "；".join(name for name in criteria if row[name] == 0), axis=1,
    )
    missing["资格原因"] = missing.apply(
        lambda row: "；".join(label for name, label in criteria.items() if row[name] == 0), axis=1,
    )
    if missing["资格原因"].eq("").any():
        raise ValueError("存在无法由原始资格字段解释的缺失记录")
    prediction = pd.read_parquet(output / "合格记录预测.parquet", columns=[
        "ts_code", "trade_date", "selected_pred",
    ])
    if not np.isfinite(prediction["selected_pred"]).all():
        raise ValueError("交付预测存在非有限值")
    predicted_keys = prediction.rename(columns={"trade_date": "target_date"})[keys]
    delivered_keys = pd.concat([predicted_keys, missing[keys]], ignore_index=True)
    if delivered_keys.duplicated(keys).any():
        raise ValueError("预测记录与缺失记录出现重复股票日期键")
    joined = delivered_keys.merge(source[keys], on=keys, how="outer", indicator=True, validate="one_to_one")
    if len(joined) != config["data"]["rows"]["competition_test"] or not joined["_merge"].eq("both").all():
        raise ValueError("预测记录与缺失记录未完整覆盖官方测试键")
    destination = output / "覆盖缺失键.parquet"
    temporary = temporary_path(destination)
    missing.to_parquet(temporary, index=False, compression="zstd")
    if not pd.read_parquet(temporary).equals(missing):
        raise ValueError("缺失资格记录写入后不一致")
    os.replace(temporary, destination)
    counts = {name: int(missing[name].eq(0).sum()) for name in criteria}
    coverage = read_json(output / "覆盖审计.json")
    coverage["uncovered_reason_counts"] = counts
    coverage["uncovered_reason_counts_overlap"] = True
    write_json(output / "覆盖审计.json", coverage)
    result = {
        "status": "passed", "config_sha256": config_sha256,
        "source_index_sha256": file_sha256(source_path),
        "audit_program_sha256": file_sha256(__file__),
        "annotated_missing_sha256": file_sha256(destination),
        "all_test_keys": len(joined), "predicted_keys": len(predicted_keys),
        "missing_keys": len(missing), "uncovered_reason_counts": counts,
        "uncovered_reason_counts_overlap": True,
    }
    write_json(output / "覆盖资格核验.json", result)
    run_final_audit(config, config_sha256, run_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    config, config_sha256 = load_config(args.config)
    audit_delivery(config, config_sha256, Path(args.run_dir).resolve())


if __name__ == "__main__":
    main()
