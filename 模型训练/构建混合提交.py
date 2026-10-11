import json
from pathlib import Path

import numpy as np
import pandas as pd

from 评估混合覆盖方案 import apply_calibration


ROOT = Path(__file__).resolve().parents[1]
TEST_KEYS_PATH = ROOT / "模型预处理" / "步骤5_标准化处理" / "测试集_标准化.parquet"
STRICT_TEST_PATH = ROOT / "模型训练" / "严格排序损失消融实验" / "20261010_rank_ablation" / "runs" / "rank_0.30" / "seed_42" / "competition_test" / "合格记录预测.parquet"
LGBM_TEST_PATH = ROOT / "模型训练" / "LightGBM实验结果" / "测试集预测.parquet"
CALIBRATION_PATH = ROOT / "模型训练" / "混合模型校准参数.json"
SUBMISSION_PATH = ROOT / "submission_hybrid.csv"
AUDIT_PATH = ROOT / "模型训练" / "混合模型覆盖审计.json"
REPORT_PATH = ROOT / "模型训练" / "混合模型实验报告.md"
KEY_COLUMNS = ["ts_code", "trade_date"]


def validate_unique(frame, name):
    if frame.duplicated(KEY_COLUMNS).any():
        raise ValueError(f"{name}存在重复股票日期键")
    if frame[KEY_COLUMNS].isna().any().any():
        raise ValueError(f"{name}的股票日期键存在缺失")


def main():
    test_keys = pd.read_parquet(TEST_KEYS_PATH, columns=KEY_COLUMNS)
    strict = pd.read_parquet(STRICT_TEST_PATH)
    if "selected_pred" in strict.columns:
        strict["pred"] = strict.pop("selected_pred")
    lgbm = pd.read_parquet(LGBM_TEST_PATH)
    calibration = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    validate_unique(test_keys, "测试集完整键表")
    validate_unique(strict, "严格排序测试预测")
    validate_unique(lgbm, "LightGBM测试预测")
    if not np.isfinite(strict["pred"].to_numpy(dtype=np.float64)).all():
        raise ValueError("严格排序测试预测存在非有限值")
    if not np.isfinite(lgbm["pred"].to_numpy(dtype=np.float64)).all():
        raise ValueError("LightGBM测试预测存在非有限值")

    test_key_set = pd.MultiIndex.from_frame(test_keys[KEY_COLUMNS])
    lgbm_key_set = pd.MultiIndex.from_frame(lgbm[KEY_COLUMNS])
    strict_key_set = pd.MultiIndex.from_frame(strict[KEY_COLUMNS])
    if len(test_key_set.difference(lgbm_key_set)) != 0 or len(lgbm_key_set.difference(test_key_set)) != 0:
        raise ValueError("LightGBM测试预测键集合与完整测试集不一致")
    if len(strict_key_set.difference(test_key_set)) != 0:
        raise ValueError("严格排序测试预测存在测试集之外的键")

    lgbm_input = lgbm.rename(columns={"pred": "lgbm_pred"})
    calibrated = apply_calibration(lgbm_input[[*KEY_COLUMNS, "lgbm_pred"]], calibration)
    strict_input = strict.rename(columns={"pred": "strict_pred"})
    result = test_keys.merge(strict_input, on=KEY_COLUMNS, how="left").merge(
        calibrated[[*KEY_COLUMNS, "lgbm_pred_calibrated", "calibration_scope"]], on=KEY_COLUMNS, how="left"
    )
    result["pred"] = result["strict_pred"].where(result["strict_pred"].notna(), result["lgbm_pred_calibrated"])
    result["prediction_source"] = np.where(result["strict_pred"].notna(), "strict_sort", "lgbm_fallback")
    result = result.sort_values(KEY_COLUMNS, kind="mergesort").reset_index(drop=True)
    if result["pred"].isna().any():
        raise ValueError("混合测试预测存在缺失值")
    if not np.isfinite(result["pred"].to_numpy(dtype=np.float64)).all():
        raise ValueError("混合测试预测存在非有限值")

    expected_daily = test_keys.groupby("trade_date", sort=True).size().rename("expected_rows")
    actual_daily = result.groupby("trade_date", sort=True).size().rename("actual_rows")
    daily = pd.concat([expected_daily, actual_daily], axis=1).fillna(0).astype(int)
    daily_mismatch = daily.loc[daily["expected_rows"] != daily["actual_rows"]]
    audit = {
        "test_rows": int(len(test_keys)),
        "strict_prediction_rows": int(len(strict)),
        "lgbm_prediction_rows": int(len(lgbm)),
        "strict_coverage_rate": float(len(strict) / len(test_keys)),
        "fallback_rows": int(result["prediction_source"].eq("lgbm_fallback").sum()),
        "final_rows": int(len(result)),
        "unique_key_rows": int(result[KEY_COLUMNS].drop_duplicates().shape[0]),
        "duplicate_key_rows": int(result.duplicated(KEY_COLUMNS).sum()),
        "null_prediction_rows": int(result["pred"].isna().sum()),
        "infinite_prediction_rows": int((~np.isfinite(result["pred"].to_numpy(dtype=np.float64))).sum()),
        "daily_count_rows": int(len(daily)),
        "daily_count_mismatch_rows": int(len(daily_mismatch)),
        "calibration_scope_counts": {str(key): int(value) for key, value in result["calibration_scope"].value_counts().to_dict().items()},
    }
    if audit["final_rows"] != audit["test_rows"] or audit["unique_key_rows"] != audit["test_rows"] or audit["duplicate_key_rows"] != 0 or audit["null_prediction_rows"] != 0 or audit["infinite_prediction_rows"] != 0 or audit["daily_count_mismatch_rows"] != 0:
        raise ValueError(f"混合提交覆盖审计未通过: {audit}")
    result[KEY_COLUMNS + ["pred"]].to_csv(SUBMISSION_PATH, index=False, encoding="utf-8-sig", float_format="%.12g")
    AUDIT_PATH.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    section = [
        "",
        "## 测试集提交审计",
        "",
        f"测试集完整键表共 {audit['test_rows']:,} 条记录。严格排序模型覆盖 {audit['strict_prediction_rows']:,} 条，LightGBM 兜底 {audit['fallback_rows']:,} 条，混合提交最终保留 {audit['final_rows']:,} 条。",
        "",
        f"独立读取审计结果：唯一键 {audit['unique_key_rows']:,} 条，重复键 {audit['duplicate_key_rows']} 条，缺失预测 {audit['null_prediction_rows']} 条，非有限预测 {audit['infinite_prediction_rows']} 条，逐日数量不一致 {audit['daily_count_mismatch_rows']} 天。全部通过。",
        "",
        f"提交文件为 `{SUBMISSION_PATH.name}`，校准参数中的测试期交易日使用全局参数，来源计数为 `{audit['calibration_scope_counts']}`。",
    ]
    report = REPORT_PATH.read_text(encoding="utf-8") if REPORT_PATH.exists() else "# 混合覆盖模型实验报告\n"
    if "## 测试集提交审计" not in report:
        REPORT_PATH.write_text(report.rstrip() + "\n" + "\n".join(section) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(f"submission={SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
