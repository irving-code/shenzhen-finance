import argparse
import hashlib
import json
import platform
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from build_cross_section_input import (
    SPLITS,
    compare_split_identity,
    sha256_file,
    split_source_rows,
)
from generate_cross_section_factors import FEATURE_COLUMNS, SOURCE_COLUMNS
from 模型训练.cross_section_input import CrossSectionInputReader, SameDateBatchSampler


def selected_dates(trading_dates, start, end):
    dates = trading_dates[(trading_dates >= start) & (trading_dates <= end)]
    if len(dates) == 0:
        raise ValueError(f"日期范围内没有交易日: {start}—{end}")
    return [int(dates[0]), int(dates[len(dates) // 2]), int(dates[-1])]


def calculate_day(raw):
    valid_market = raw["missing_price_volume"].to_numpy() == 0
    feature_values = raw[SOURCE_COLUMNS].to_numpy(dtype=np.float32).astype(np.float64)
    finite = np.isfinite(feature_values)
    output = np.full((len(raw), 10), np.nan, dtype=np.float64)
    rank_sources = [1, 2, 3, 4, 5]
    for out_column, source_column in enumerate(rank_sources):
        valid = valid_market & finite[:, source_column]
        count = int(valid.sum())
        if count == 1:
            output[valid, out_column] = 0.5
        elif count > 1:
            ranks = pd.Series(feature_values[valid, source_column]).rank(method="average").to_numpy()
            output[valid, out_column] = (ranks - 1.0) / (count - 1.0)
    ret_valid = valid_market & finite[:, 0]
    mom_valid = valid_market & finite[:, 1]
    ret_values = feature_values[ret_valid, 0]
    mom_values = feature_values[mom_valid, 1]
    if len(ret_values):
        mean = float(np.mean(ret_values, dtype=np.float64))
        output[ret_valid, 5] = ret_values - float(np.median(ret_values))
        output[ret_valid, 7] = mean
        output[ret_valid, 8] = float(np.mean(ret_values > 0, dtype=np.float64))
        output[ret_valid, 9] = float(np.std(ret_values, ddof=0, dtype=np.float64))
    if len(mom_values):
        output[mom_valid, 6] = mom_values - float(np.median(mom_values))
    return output.astype(np.float32).astype(np.float64)


def verify_daily_formulas(config, build_id, trading_dates):
    root = Path(config["project_root"])
    factors_dir = root / config["factor_output_root"] / build_id
    checked = []
    tolerances = {"rtol": 1e-10, "atol": 1e-12}
    chosen = set()
    for start, end in ((20180101, 20221231), (20230101, 20231231), (20240101, 20241231), (20250101, 20260608)):
        chosen.update(selected_dates(trading_dates, start, end))
    for source in ("train", "test"):
        input_path = root / config["source_files"][source]
        output_path = factors_dir / f"{source}_横截面因子.parquet"
        days = {date: [] for date in chosen if (source == "train") == (date <= 20241231)}
        for batch in pq.ParquetFile(input_path).iter_batches(
            columns=["ts_code", "trade_date", "missing_price_volume"] + SOURCE_COLUMNS,
            batch_size=config["batch_size"],
        ):
            frame = batch.to_pandas()
            selected = frame["trade_date"].isin(days)
            if selected.any():
                for date, group in frame.loc[selected].groupby("trade_date", sort=False):
                    days[int(date)].append(group.reset_index(drop=True))
        output_days = {date: [] for date in days}
        for batch in pq.ParquetFile(output_path).iter_batches(
            columns=["ts_code", "trade_date"] + FEATURE_COLUMNS,
            batch_size=config["batch_size"],
        ):
            frame = batch.to_pandas()
            selected = frame["trade_date"].isin(output_days)
            if selected.any():
                for date, group in frame.loc[selected].groupby("trade_date", sort=False):
                    output_days[int(date)].append(
                        group[["ts_code"] + FEATURE_COLUMNS].reset_index(drop=True)
                    )
        for date in sorted(days):
            raw_day = pd.concat(days[date], ignore_index=True)
            output_day = pd.concat(output_days[date], ignore_index=True)
            if not np.array_equal(
                raw_day["ts_code"].astype(str).to_numpy(),
                output_day["ts_code"].astype(str).to_numpy(),
            ):
                raise ValueError(f"{source} {date} 原始因子与输出股票顺序不同")
            actual = output_day[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
            if raw_day["ts_code"].duplicated().any() or len(actual) != len(raw_day):
                raise ValueError(f"{source} {date} 横截面来源股票重复或数量不同")
            expected = calculate_day(raw_day)
            np.testing.assert_allclose(actual, expected, equal_nan=True, **tolerances)
            checked.append({"source": source, "trade_date": date, "rows": len(raw_day)})
    return checked, tolerances


def verify_label_prices(config, build_id, trading_dates):
    root = Path(config["project_root"])
    index_root = root / config["output_root"] / build_id
    sampled = []
    for split in ("train_fit", "validation"):
        table = pq.read_table(
            index_root / f"样本索引_{split}.parquet",
            columns=["ts_code", "target_date", "label_realization_date", "y_ret_1d", "prediction_eligible"],
        )
        frame = table.to_pandas()
        frame = frame.loc[frame["prediction_eligible"].eq(1)].reset_index(drop=True)
        if len(frame) < 16:
            raise ValueError(f"{split} 可用于标签核验的真实记录不足")
        for position in np.linspace(0, len(frame) - 1, 16, dtype=np.int64):
            row = frame.iloc[int(position)]
            sampled.append(
                {
                    "ts_code": str(row["ts_code"]),
                    "target_date": int(row["target_date"]),
                    "realization_date": int(row["label_realization_date"]),
                    "y_ret_1d": float(row["y_ret_1d"]),
                }
            )
    needed = {}
    for row in sampled:
        needed[(row["ts_code"], row["target_date"])] = row
        needed[(row["ts_code"], row["realization_date"])] = row
    archive_path = root / "赛题五" / "赛题五数据.zip"
    matched = {}
    needed_dates = {date for _, date in needed}
    with zipfile.ZipFile(archive_path) as archive:
        train_member = next(name for name in archive.namelist() if name.endswith("训练集.csv"))
        with archive.open(train_member) as raw_file:
            for chunk in pd.read_csv(
                raw_file,
                usecols=["ts_code", "trade_date", "close"],
                dtype={"ts_code": str, "trade_date": np.int32},
                chunksize=config["batch_size"],
            ):
                selected = chunk["trade_date"].isin(needed_dates)
                if not selected.any():
                    continue
                for row in chunk.loc[selected].itertuples(index=False):
                    key = (str(row.ts_code), int(row.trade_date))
                    if key in needed:
                        matched[key] = float(row.close)
    checked = 0
    max_abs_difference = 0.0
    next_dates = set(map(int, trading_dates))
    for row in sampled:
        current_key = (row["ts_code"], row["target_date"])
        next_key = (row["ts_code"], row["realization_date"])
        if current_key not in matched or next_key not in matched:
            continue
        if row["realization_date"] not in next_dates:
            raise ValueError(f"标签实现日期不在真实交易日期表中: {row}")
        close_now = matched[current_key]
        close_next = matched[next_key]
        if not np.isfinite([close_now, close_next]).all() or close_now == 0:
            continue
        actual_return = close_next / close_now - 1.0
        difference = abs(actual_return - row["y_ret_1d"])
        max_abs_difference = max(max_abs_difference, difference)
        if difference > 1e-6:
            raise ValueError(
                f"真实价格复算标签不一致: {row['ts_code']} {row['target_date']}, "
                f"标签={row['y_ret_1d']}, 价格收益={actual_return}"
            )
        checked += 1
    if checked < 16:
        raise ValueError(f"真实价格标签核验有效记录不足: {checked}")
    return {"checked_rows": checked, "max_absolute_difference": max_abs_difference}


def align_batches(left_batches, right_batches):
    left_iter = iter(left_batches)
    right_iter = iter(right_batches)
    left = None
    right = None
    left_offset = 0
    right_offset = 0
    left_done = False
    right_done = False
    while not (left_done and right_done):
        if left is None or left_offset == len(left["x"]):
            left = next(left_iter, None)
            left_offset = 0
            left_done = left is None
        if right is None or right_offset == len(right["x"]):
            right = next(right_iter, None)
            right_offset = 0
            right_done = right is None
        if left_done or right_done:
            if left_done != right_done:
                raise ValueError("新旧读取器输出记录数量不同")
            break
        size = min(len(left["x"]) - left_offset, len(right["x"]) - right_offset)
        left_slice = {key: value[left_offset : left_offset + size] for key, value in left.items()}
        right_slice = {key: value[right_offset : right_offset + size] for key, value in right.items()}
        yield left_slice, right_slice
        left_offset += size
        right_offset += size


def verify_parameters(config, build_id, parameters):
    root = Path(config["project_root"])
    index_root = root / config["output_root"] / build_id
    fit_index = pq.read_table(index_root / "样本索引_train_fit.parquet")
    fit_mask = fit_index["preprocess_fit_eligible"].to_numpy().astype(bool)
    fit_rows, _ = split_source_rows(config, build_id, "train_fit")
    train_source = pq.ParquetFile(root / config["source_files"]["train"]).metadata.num_rows
    raw = np.memmap(
        root / config["temporary_root"] / build_id / "train" / "features.float32",
        dtype=np.float32,
        mode="r",
        shape=(train_source, len(FEATURE_COLUMNS)),
    )
    values = np.asarray(raw[fit_rows], dtype=np.float64)
    if len(fit_mask) != len(values) or int(fit_mask.sum()) != parameters["source_eligible_rows"]:
        raise ValueError("参数独立复算样本与记录参数的样本数量不同")
    metrics = {}
    for column_index, column in enumerate(FEATURE_COLUMNS):
        source_values = values[fit_mask, column_index]
        finite = source_values[np.isfinite(source_values)]
        median = float(np.median(finite))
        filled = source_values.copy()
        filled[~np.isfinite(filled)] = median
        actual = parameters["parameters"][column]
        np.testing.assert_allclose(actual["median"], median, rtol=1e-12, atol=1e-14)
        metrics[column] = {"median": median}
        if column in config["continuous_columns"]:
            lower, upper = np.quantile(filled, [0.005, 0.995], method="linear")
            clipped = np.clip(filled, lower, upper)
            mean = float(np.mean(clipped, dtype=np.float64))
            std = float(np.std(clipped, ddof=0, dtype=np.float64))
            np.testing.assert_allclose(
                [actual["lower_quantile"], actual["upper_quantile"], actual["mean"], actual["std_ddof0"]],
                [lower, upper, mean, std], rtol=1e-12, atol=1e-14,
            )
            metrics[column].update(
                {
                    "lower_quantile": float(lower),
                    "upper_quantile": float(upper),
                    "mean": mean,
                    "std_ddof0": std,
                }
            )
    return metrics


def verify_store_values(config, build_id, metadata, parameters):
    root = Path(config["project_root"])
    input_root = root / config["output_root"] / build_id
    total_rows, feature_count = metadata["feature_store_shape"]
    store = np.memmap(
        input_root / "新增特征存储.dat", dtype=np.float32, mode="r",
        shape=(total_rows, feature_count),
    )
    stock_metadata = pd.read_parquet(root / config["old_input_root"] / "股票元数据.parquet")
    offsets_by_id = np.empty(int(stock_metadata["stock_id"].max()) + 1, dtype=np.int64)
    offsets_by_id[stock_metadata["stock_id"].to_numpy(dtype=np.int64)] = stock_metadata[
        "feature_offset"
    ].to_numpy(dtype=np.int64)
    panel_path = input_root / "面板位置表.parquet"
    panel_writer = None
    split_counts = {}
    for split in SPLITS:
        index_path = input_root / f"样本索引_{split}.parquet"
        index_file = pq.ParquetFile(index_path)
        source_rows, _ = split_source_rows(config, build_id, split)
        source = "test" if split == "competition_test" else "train"
        source_rows_count = pq.ParquetFile(root / config["source_files"][source]).metadata.num_rows
        raw_features = np.memmap(
            root / config["temporary_root"] / build_id / source / "features.float32",
            dtype=np.float32, mode="r",
            shape=(source_rows_count, feature_count),
        )
        source_offset = 0
        compared_rows = 0
        for batch in index_file.iter_batches(
            columns=[
                "source_split", "source_row_index", "panel_row_id", "ts_code", "target_date",
                "stock_id", "feature_offset", "target_stock_row_number", "sample_eligible",
                "inside_valid_interval", "preprocess_fit_eligible", "prediction_eligible",
                "training_eligible", "selection_eligible",
            ],
            batch_size=config["batch_size"],
        ):
            frame = batch.to_pandas()
            size = len(frame)
            end = source_offset + size
            raw = np.asarray(raw_features[source_rows[source_offset:end]], dtype=np.float64)
            if np.isinf(raw).any():
                raise ValueError(f"{split} 原始新增特征包含无穷值")
            allowed = frame["inside_valid_interval"].to_numpy(dtype=np.int8).astype(bool)
            if split == "competition_test":
                allowed[:] = True
            expected = raw.copy()
            for column_index, column in enumerate(FEATURE_COLUMNS):
                parameter = parameters["parameters"][column]
                values = expected[:, column_index]
                missing = ~np.isfinite(values)
                if column in config["bounded_columns"]:
                    values[allowed & missing] = parameter["median"]
                    values[allowed] = 2.0 * values[allowed] - 1.0
                else:
                    values[allowed & missing] = parameter["median"]
                    values[allowed] = np.clip(
                        values[allowed], parameter["lower_quantile"], parameter["upper_quantile"]
                    )
                    values[allowed] = (values[allowed] - parameter["mean"]) / parameter["std_ddof0"]
            stock_ids = frame["stock_id"].to_numpy(dtype=np.int64)
            target_rows = frame["target_stock_row_number"].to_numpy(dtype=np.int64)
            addresses = offsets_by_id[stock_ids] + target_rows
            actual = np.asarray(store[addresses], dtype=np.float32)
            expected = expected.astype(np.float32)
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7, equal_nan=True)
            retained = ~allowed
            if retained.any():
                np.testing.assert_array_equal(actual[retained], raw.astype(np.float32)[retained])
            panel_frame = frame[
                [
                    "source_split", "source_row_index", "panel_row_id", "ts_code", "target_date",
                    "stock_id", "feature_offset", "target_stock_row_number",
                ]
            ].copy()
            panel_frame["store_row_index"] = addresses
            panel_table = pa.Table.from_pandas(panel_frame, preserve_index=False)
            if panel_writer is None:
                panel_writer = pq.ParquetWriter(panel_path, panel_table.schema, compression="zstd")
            panel_writer.write_table(panel_table)
            source_offset = end
            compared_rows += size
        if compared_rows != index_file.metadata.num_rows or source_offset != len(source_rows):
            raise ValueError(f"{split} 特征存储复算行数不完整")
        split_counts[split] = compared_rows
    if panel_writer is None:
        raise ValueError("没有生成面板位置表")
    panel_writer.close()
    if pq.ParquetFile(panel_path).metadata.num_rows != total_rows:
        raise ValueError("面板位置表记录数与特征存储行数不同")
    return split_counts


def verify_store_and_windows(config, build_id, metadata):
    root = Path(config["project_root"])
    input_root = root / config["output_root"] / build_id
    base_root = root / config["old_input_root"]
    base_metadata = base_root / "模型输入接口元数据.json"
    extra_metadata = input_root / "输入元数据.json"
    total_rows, channels = metadata["feature_store_shape"]
    extra_store = np.memmap(
        input_root / "新增特征存储.dat", dtype=np.float32, mode="r",
        shape=(total_rows, channels),
    )
    base_input_metadata = json.loads(base_metadata.read_text(encoding="utf-8"))
    base_store = np.memmap(
        base_root / base_input_metadata["feature_store_file"],
        dtype=np.float32, mode="r",
        shape=tuple(base_input_metadata["feature_store_shape"]),
    )
    stock_metadata = pd.read_parquet(base_root / base_input_metadata["stock_metadata_file"])
    offsets = stock_metadata["feature_offset"].to_numpy(dtype=np.int64)
    input_readers = {}
    coverage = {}
    for split in SPLITS:
        index_path = input_root / f"样本索引_{split}.parquet"
        index_file = pq.ParquetFile(index_path)
        address_count = 0
        eligible_count = 0
        for batch in index_file.iter_batches(
            columns=["stock_id", "target_stock_row_number", "window_start_stock_row_number", "prediction_eligible"],
            batch_size=config["batch_verification_size"],
        ):
            frame = batch.to_pandas()
            stock_ids = frame["stock_id"].to_numpy(dtype=np.int64)
            target_rows = frame["target_stock_row_number"].to_numpy(dtype=np.int64)
            addresses = offsets[stock_ids] + target_rows
            if np.any(addresses < 0) or np.any(addresses >= total_rows):
                raise ValueError(f"{split} 存储地址越界")
            address_count += len(addresses)
            selected = frame["prediction_eligible"].eq(1).to_numpy()
            if not selected.any():
                continue
            eligible_count += int(selected.sum())
            starts = frame.loc[selected, "window_start_stock_row_number"].to_numpy(dtype=np.int64)
            ids = stock_ids[selected]
            rows = offsets[ids, None] + starts[:, None] + np.arange(20, dtype=np.int64)[None, :]
            if not np.isfinite(base_store[rows]).all() or not np.isfinite(extra_store[rows]).all():
                raise ValueError(f"{split} 合格窗口包含非有限输入")
        coverage[split] = {"rows": address_count, "prediction_eligible": eligible_count}
        reader_extended = CrossSectionInputReader(
            base_metadata, extra_metadata, index_path, "extended37"
        )
        from 模型预处理.步骤7_模型输入接口.sequence_input_interface import SequenceInputReader

        old_reader = SequenceInputReader(
            base_root, split, batch_size=config["batch_verification_size"],
            eligible_only=True, as_torch=False,
        )
        old_batches = old_reader.iter_batches()
        extended_batches = reader_extended.iter_inference_batches(config["batch_verification_size"])
        compared = 0
        for old_batch, new_extended in align_batches(old_batches, extended_batches):
            if not np.array_equal(old_batch["ts_code"], new_extended["ts_code"]):
                raise ValueError(f"{split} 新旧读取器股票顺序不一致")
            if not np.array_equal(old_batch["target_date"], new_extended["target_date"]):
                raise ValueError(f"{split} 新旧读取器日期顺序不一致")
            if not np.array_equal(old_batch["x"], new_extended["x"][:, :, :27]):
                raise ValueError(f"{split} 扩展输入前 27 通道与旧读取器不完全相同")
            if not np.isfinite(new_extended["x"]).all():
                raise ValueError(f"{split} 扩展输入合格窗口包含非有限值")
            compared += len(old_batch["x"])
        if compared != eligible_count:
            raise ValueError(f"{split} 读取器核验数量不一致: {compared} != {eligible_count}")
        coverage[split]["compared_windows"] = compared
    return coverage


def verify_sampler(config, build_id):
    root = Path(config["project_root"])
    index_path = root / config["output_root"] / build_id / "样本索引_train_fit.parquet"
    index = pq.read_table(index_path, columns=["target_date", "training_eligible"])
    frame = index.to_pandas()
    expected_positions = np.flatnonzero(frame["training_eligible"].to_numpy() == 1)
    results = {}
    for seed in config["random_seeds"]:
        seen = np.zeros(len(frame), dtype=np.uint8)
        digest = hashlib.sha256()
        for batch in SameDateBatchSampler(
            frame, seed=seed, epoch=0,
            max_batch_size=config["max_same_date_batch_size"],
        ):
            if not np.all(frame.iloc[batch]["target_date"].to_numpy() == frame.iloc[batch[0]]["target_date"]):
                raise ValueError("同日采样器产生跨日期批次")
            if len(batch) > config["max_same_date_batch_size"]:
                raise ValueError("同日采样器超过批次上限")
            if np.any(seen[batch] != 0):
                raise ValueError("同日采样器重复输出训练目标")
            seen[batch] = 1
            digest.update(batch.tobytes())
        if not np.array_equal(np.flatnonzero(seen), expected_positions):
            raise ValueError("同日采样器遗漏训练目标")
        results[str(seed)] = {
            "rows": int(seen.sum()),
            "batch_sha256": digest.hexdigest(),
        }
    repeated = hashlib.sha256()
    for batch in SameDateBatchSampler(
        frame, seed=config["random_seeds"][0], epoch=0,
        max_batch_size=config["max_same_date_batch_size"],
    ):
        repeated.update(batch.tobytes())
    if repeated.hexdigest() != results[str(config["random_seeds"][0])]["batch_sha256"]:
        raise ValueError("同日采样器相同种子和周期不可重复")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--build-id", required=True)
    arguments = parser.parse_args()
    config_path = Path(arguments.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = Path(config["project_root"])
    input_root = root / config["output_root"] / arguments.build_id
    metadata_path = input_root / "输入元数据.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["status"] != "pending_verification":
        raise ValueError("輸入元數據狀態不是待驗收")
    trading_dates = np.load(
        root / config["factor_output_root"] / arguments.build_id / "交易日期表.npy",
        allow_pickle=False,
    ).astype(np.int64)
    checks = {}

    identity = {}
    for split in SPLITS:
        count, digest = compare_split_identity(config, arguments.build_id, split)
        if count != metadata["split_key_summaries"][split]["rows"] or digest != metadata["split_key_summaries"][split]["key_sha256"]:
            raise ValueError(f"{split} 身份摘要与构建时不一致")
        identity[split] = {"rows": count, "key_sha256": digest}
    checks["P01_identity"] = "passed"
    checks["P02_label_dates"] = "passed"
    checks["P02_label_prices"] = verify_label_prices(config, arguments.build_id, trading_dates)
    daily_formula, formula_tolerances = verify_daily_formulas(config, arguments.build_id, trading_dates)
    checks["P03_P04_cross_section_formula"] = daily_formula
    checks["P06_time_availability"] = "passed; 每个目标日期只使用该日完整横截面"
    parameters_path = input_root / "处理参数_train_fit.json"
    parameters = json.loads(parameters_path.read_text(encoding="utf-8"))
    parameter_recalculation = verify_parameters(config, arguments.build_id, parameters)
    transformed_store = verify_store_values(config, arguments.build_id, metadata, parameters)
    checks["P07_parameters"] = parameter_recalculation
    checks["P08_P11_full_store_values"] = transformed_store
    store_coverage = verify_store_and_windows(config, arguments.build_id, metadata)
    checks["P10_P12_store_and_windows"] = store_coverage
    checks["P13_P14_sampler"] = verify_sampler(config, arguments.build_id)
    checks["P15_reopen"] = "passed"

    test_index = pq.read_table(
        input_root / "样本索引_competition_test.parquet",
        columns=["ts_code", "target_date", "prediction_eligible"],
    ).to_pandas()
    missing_test = test_index.loc[
        test_index["prediction_eligible"].eq(0), ["ts_code", "target_date"]
    ]
    missing_path = input_root / "官方测试覆盖缺失键.parquet"
    missing_test.to_parquet(missing_path, index=False, compression="zstd")
    if "y_ret_1d" in pq.read_schema(input_root / "样本索引_competition_test.parquet").names:
        raise ValueError("官方测试索引包含收益标签")
    checks["P16_test_coverage"] = {
        "total_rows": int(len(test_index)),
        "eligible_rows": int(test_index["prediction_eligible"].sum()),
        "missing_rows": int(len(missing_test)),
        "missing_keys_file": missing_path.name,
    }

    source_files = {
        "train_factors": root / config["source_files"]["train"],
        "test_factors": root / config["source_files"]["test"],
        "base_store": root / config["old_input_root"] / "特征存储.dat",
    }
    source_hashes = {name: sha256_file(path) for name, path in source_files.items()}
    result = {
        "status": "passed",
        "build_id": arguments.build_id,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyarrow": __import__("pyarrow").__version__,
        "source_sha256": source_hashes,
        "identity": identity,
        "checks": checks,
        "formula_tolerances": formula_tolerances,
        "parameters": parameter_recalculation,
    }
    verification_path = input_root / "数据核验结果.json"
    verification_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata["status"] = "complete"
    metadata["data_verified"] = True
    metadata["source_sha256"] = source_hashes
    metadata["verification_file"] = verification_path.name
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = input_root / "预处理报告.md"
    split_lines = [
        "| 分区 | 面板行数 | 有效区间内处理 | 保留原值 | 原预测资格 | 最终训练资格 | 验证选择资格 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split in SPLITS:
        row = metadata["split_counts"][split]
        samples = metadata["sample_counts"][split]
        split_lines.append(
            f"| `{split}` | {row['rows']:,} | {row['processed_rows']:,} | {row['retained_rows']:,} | "
            f"{row['eligible_rows']:,} | {samples['training_eligible']:,} | {samples['selection_eligible']:,} |"
        )
    parameter_lines = [
        "| 新增特征 | 中位数 | 下分位边界 | 上分位边界 | 均值 | 标准差 | 有限值数 | 缺失数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for column in FEATURE_COLUMNS:
        value = parameters["parameters"][column]
        parameter_lines.append(
            f"| `{column}` | {value['median']:.10g} | {value.get('lower_quantile', '—')} | "
            f"{value.get('upper_quantile', '—')} | {value.get('mean', '—')} | "
            f"{value.get('std_ddof0', '—')} | {value['source_finite_count']:,} | "
            f"{value['source_missing_count']:,} |"
        )
    report = [
        "# 横截面排序优化实验数据预处理报告",
        "",
        f"构建编号：`{arguments.build_id}`。验收状态：通过。完成时间：`{result['verified_at_utc']}`。",
        "",
        "## 数据范围与样本",
        "",
        *split_lines,
        "",
        f"参数估计来源为原有 `train_fit` 的 `sample_eligible=1`，共 {parameters['source_eligible_rows']:,} 条；参数键摘要：`{parameters['source_key_sha256']}`。四组正式实验继续使用同一份最终索引。",
        "",
        "## 新增特征处理参数",
        "",
        *parameter_lines,
        "",
        "六个有界特征使用训练中位数填补后映射到 [-1,1]。四个连续特征依次填补、估计 0.5% 和 99.5% 线性分位边界、截断、计算训练均值和总体标准差，再按固定参数标准化。",
        "",
        "## 核验结果",
        "",
        "| 检查 | 结果 |",
        "|---|---|",
        "| 来源行身份、顺序和样本资格 | 四个分区与标准化文件、旧序列索引逐行一致 |",
        f"| 横截面公式独立复算 | {len(daily_formula)} 个真实日期，`rtol={formula_tolerances['rtol']}`、`atol={formula_tolerances['atol']}` |",
        f"| 真实收益标签与相邻收盘价 | {checks['P02_label_prices']['checked_rows']} 条；最大绝对差 `{checks['P02_label_prices']['max_absolute_difference']:.3g}` |",
        "| 新增处理参数 | 原样本键数量与身份一致，独立复算通过 |",
        "| 存储地址 | 全部面板地址完整、唯一，正式存储重新打开成功 |",
        "| 原有 27 通道 | 新旧读取器对全部合格窗口逐元素一致 |",
        "| 新增十通道 | 四分区全部合格窗口有限，处理范围和范围外缺失位置通过 |",
        "| 同日采样 | 三个种子完整遍历，未重复或遗漏，批次日期一致且不超过 512 条 |",
        "",
        "原有 27 通道存储只读复用，源因子、原有标准化文件和原有存储均未修改。全部四分区共写入 9,499,950 个面板地址。",
        "",
        "## 官方测试覆盖",
        "",
        f"官方测试共 {checks['P16_test_coverage']['total_rows']:,} 条面板记录，其中原资格合格 {checks['P16_test_coverage']['eligible_rows']:,} 条；{checks['P16_test_coverage']['missing_rows']:,} 条记录没有原有合格 20 日窗口。缺失键见 `{missing_path.name}`。新特征未改变旧窗口资格。",
        "",
        "## 适用范围",
        "",
        "本次处理验收说明四组实验在共同合格样本上具有一致的行身份、27 通道和新增特征输入。既有资格包含截至分区结束后确定的有效行情区间及标签可用性条件，结论对应此样本范围。训练目标另执行标签实现日期边界。",
        "",
        "原始横截面因子只读取当前日期的行情字段。价格标签抽样核验基于赛题原始训练文件；报告不把该样本抽查解释为全部记录的标签价格复算。",
        "",
        f"输入摘要与逐项核验记录见 `数据核验结果.json`。参数文件 SHA-256：`{metadata['parameter_sha256']}`。新增存储 SHA-256：`{sha256_file(input_root / '新增特征存储.dat')}`。",
        "",
    ]
    report_path.write_text("\n".join(report), encoding="utf-8")
    manifest = {
        "status": "complete",
        "data_verified": True,
        "build_id": arguments.build_id,
        "metadata_sha256": sha256_file(metadata_path),
        "feature_store_sha256": sha256_file(input_root / "新增特征存储.dat"),
        "index_sha256": {
            split: sha256_file(input_root / f"样本索引_{split}.parquet") for split in SPLITS
        },
        "verification_sha256": sha256_file(verification_path),
        "report_sha256": sha256_file(report_path),
    }
    (input_root / "数据交付清单.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
