import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


SPLITS = ["train_fit", "validation", "local_holdout", "competition_test"]
STD_NAMES = {
    "train_fit": "训练集_train_fit_标准化.parquet",
    "validation": "验证集_标准化.parquet",
    "local_holdout": "本地留出集_标准化.parquet",
    "competition_test": "测试集_标准化.parquet",
}
INDEX_NAMES = {split: f"序列索引_{split}.parquet" for split in SPLITS}
IDENTITY_COLUMNS = ["ts_code", "trade_date"]
PARAMETER_COLUMNS = ["sample_eligible", "inside_valid_interval", "limit_up"]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        while block := file.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def key_stream(parquet_path, columns, date_range=None):
    parquet_file = pq.ParquetFile(parquet_path)
    for batch in parquet_file.iter_batches(columns=columns, batch_size=200_000):
        codes = batch.column(0).to_numpy(zero_copy_only=False).astype(str)
        dates = batch.column(1).to_numpy(zero_copy_only=False).astype(np.int64)
        if date_range is not None:
            keep = (dates >= date_range[0]) & (dates <= date_range[1])
            codes = codes[keep]
            dates = dates[keep]
        yield codes, dates


def compare_key_streams(left_stream, right_stream, context):
    left_iter = iter(left_stream)
    right_iter = iter(right_stream)
    left_batch = None
    right_batch = None
    left_offset = 0
    right_offset = 0
    count = 0
    digest = hashlib.sha256()
    left_done = False
    right_done = False
    while not (left_done and right_done):
        if left_batch is None or left_offset == len(left_batch[0]):
            left_batch = next(left_iter, None)
            left_offset = 0
            left_done = left_batch is None
        if right_batch is None or right_offset == len(right_batch[0]):
            right_batch = next(right_iter, None)
            right_offset = 0
            right_done = right_batch is None
        if left_done or right_done:
            if left_done != right_done:
                raise ValueError(f"{context} 两侧身份记录数量不同")
            break
        size = min(
            len(left_batch[0]) - left_offset,
            len(right_batch[0]) - right_offset,
        )
        left_codes = left_batch[0][left_offset : left_offset + size]
        right_codes = right_batch[0][right_offset : right_offset + size]
        left_dates = left_batch[1][left_offset : left_offset + size]
        right_dates = right_batch[1][right_offset : right_offset + size]
        if not np.array_equal(left_codes, right_codes) or not np.array_equal(left_dates, right_dates):
            mismatch = np.flatnonzero((left_codes != right_codes) | (left_dates != right_dates))
            position = int(mismatch[0])
            raise ValueError(
                f"{context} 股票日期身份不一致，记录位置 {count + position}: "
                f"{left_codes[position]} {left_dates[position]} 与 "
                f"{right_codes[position]} {right_dates[position]}"
            )
        hash_frame = pd.DataFrame({"ts_code": left_codes, "trade_date": left_dates})
        digest.update(pd.util.hash_pandas_object(hash_frame, index=False).values.tobytes())
        left_offset += size
        right_offset += size
        count += size
    return count, digest.hexdigest()


def split_factor_path(config, build_id, split):
    source = "test" if split == "competition_test" else "train"
    return (
        Path(config["project_root"])
        / config["factor_output_root"]
        / build_id
        / f"{source}_横截面因子.parquet"
    )


def split_source_rows(config, build_id, split):
    root = Path(config["project_root"])
    source = "test" if split == "competition_test" else "train"
    factor_path = split_factor_path(config, build_id, split)
    row_count = pq.ParquetFile(factor_path).metadata.num_rows
    temp_dir = root / config["temporary_root"] / build_id / source
    date_map = np.memmap(temp_dir / "trade_date.int32", dtype=np.int32, mode="r", shape=(row_count,))
    date_range = config["split_date_ranges"][split]
    rows = np.flatnonzero((date_map >= date_range[0]) & (date_map <= date_range[1]))
    return rows.astype(np.int64), date_map


def compare_split_identity(config, build_id, split):
    root = Path(config["project_root"])
    standard_path = root / config["standardized_root"] / STD_NAMES[split]
    index_path = root / config["sequence_index_root"] / INDEX_NAMES[split]
    factor_path = split_factor_path(config, build_id, split)
    date_range = config["split_date_ranges"][split]
    standard_keys = key_stream(standard_path, IDENTITY_COLUMNS)
    index_keys = key_stream(index_path, ["ts_code", "target_date"])
    factor_keys = key_stream(factor_path, ["ts_code", "trade_date"], date_range)
    count, key_digest = compare_key_streams(standard_keys, index_keys, f"{split} 标准化文件与序列索引")
    factor_count, factor_digest = compare_key_streams(
        key_stream(factor_path, ["ts_code", "trade_date"], date_range),
        key_stream(standard_path, IDENTITY_COLUMNS),
        f"{split} 横截面因子与标准化文件",
    )
    if count != factor_count or key_digest != factor_digest:
        raise ValueError(f"{split} 因子来源与序列索引股票日期摘要不同")
    return count, key_digest


def index_table_for_split(config, split, dates, date_to_next, train_end, validation_bounds):
    root = Path(config["project_root"])
    standard_path = root / config["standardized_root"] / STD_NAMES[split]
    sequence_path = root / config["sequence_index_root"] / INDEX_NAMES[split]
    standard = pq.read_table(
        standard_path,
        columns=["sample_eligible", "inside_valid_interval", "limit_up"],
    )
    sequence = pq.read_table(sequence_path)
    if standard.num_rows != sequence.num_rows:
        raise ValueError(f"{split} 标准化文件与序列索引行数不同")
    sample = standard["sample_eligible"].to_numpy().astype(np.int8)
    sequence_sample = sequence["sample_eligible"].to_numpy().astype(np.int8)
    if not np.array_equal(sample, sequence_sample):
        raise ValueError(f"{split} sample_eligible 与旧序列索引不同")
    inside = standard["inside_valid_interval"].to_numpy().astype(np.int8)
    if "target_inside_valid_interval" in sequence.column_names:
        if not np.array_equal(
            inside,
            sequence["target_inside_valid_interval"].to_numpy().astype(np.int8),
        ):
            raise ValueError(f"{split} inside_valid_interval 与旧序列索引不同")
    limit_up = standard["limit_up"].to_numpy(zero_copy_only=False).astype(np.float32)
    target_dates = sequence["target_date"].to_numpy().astype(np.int64)
    indices = np.searchsorted(dates, target_dates)
    known = indices < len(dates)
    next_dates = np.full(len(target_dates), -1, dtype=np.int64)
    valid_indices = np.flatnonzero(known)
    exact = dates[indices[known]] == target_dates[known]
    next_dates[valid_indices[exact]] = date_to_next[indices[known][exact]]
    labels = sequence["y_ret_1d"].to_numpy(zero_copy_only=False) if "y_ret_1d" in sequence.column_names else np.full(len(target_dates), np.nan)
    prediction = sequence["prediction_eligible"].to_numpy().astype(np.int8)
    if split == "train_fit":
        training = (
            (prediction == 1)
            & np.isfinite(labels)
            & (next_dates >= 0)
            & (next_dates <= train_end)
        ).astype(np.int8)
    else:
        training = np.zeros(len(target_dates), dtype=np.int8)
    if split == "validation":
        selection = (
            (prediction == 1)
            & np.isfinite(labels)
            & (next_dates >= validation_bounds[0])
            & (next_dates <= validation_bounds[1])
        ).astype(np.int8)
    else:
        selection = np.zeros(len(target_dates), dtype=np.int8)

    result = sequence.append_column("inside_valid_interval", pa.array(inside, type=pa.int8()))
    fit_eligible = (sample == 1) & (split == "train_fit")
    result = result.append_column(
        "preprocess_fit_eligible",
        pa.array(fit_eligible.astype(np.int8), type=pa.int8()),
    )
    result = result.append_column("training_eligible", pa.array(training, type=pa.int8()))
    result = result.append_column("selection_eligible", pa.array(selection, type=pa.int8()))
    result = result.append_column("label_realization_date", pa.array(np.where(next_dates >= 0, next_dates, 0), mask=next_dates < 0, type=pa.int64()))
    result = result.append_column("flag_limit_up", pa.array(limit_up, type=pa.float32()))
    return result


def estimate_parameters(raw_fit, fit_mask, config, fit_key_digest):
    if int(fit_mask.sum()) != 4_514_536:
        raise ValueError(f"参数估计样本数量不是原有 4,514,536 条: {int(fit_mask.sum())}")
    fit_values = np.asarray(raw_fit[fit_mask], dtype=np.float64)
    if np.isinf(fit_values).any():
        raise ValueError("参数估计样本包含无穷值")
    parameters = {}
    for column_index, column in enumerate(config["feature_columns"]):
        values = fit_values[:, column_index]
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            raise ValueError(f"参数估计样本中的特征全部缺失: {column}")
        median = float(np.median(finite_values))
        filled = values.copy()
        filled[~np.isfinite(filled)] = median
        parameter = {
            "median": median,
            "source_rows": int(values.size),
            "source_finite_count": int(finite_values.size),
            "source_missing_count": int((~np.isfinite(values)).sum()),
        }
        if column in config["continuous_columns"]:
            lower, upper = np.quantile(filled, [0.005, 0.995], method="linear")
            clipped = np.clip(filled, lower, upper)
            mean = float(np.mean(clipped, dtype=np.float64))
            std = float(np.std(clipped, ddof=0, dtype=np.float64))
            if not np.isfinite([lower, upper, mean, std]).all() or lower > upper or std == 0:
                raise ValueError(f"连续特征处理参数无效: {column}")
            parameter.update(
                {
                    "lower_quantile": float(lower),
                    "upper_quantile": float(upper),
                    "quantile_interpolation": "linear",
                    "mean": mean,
                    "std_ddof0": std,
                }
            )
        else:
            if np.any((finite_values < 0) | (finite_values > 1)):
                raise ValueError(f"有界特征原始值超出 [0,1]: {column}")
        parameters[column] = parameter
    return {
        "source_split": "train_fit",
        "source_eligible_rows": int(fit_mask.sum()),
        "source_key_sha256": fit_key_digest,
        "parameters": parameters,
    }


def transform_batch(values, inside, split, config, parameters):
    transformed = np.asarray(values, dtype=np.float64).copy()
    if np.isinf(transformed).any():
        raise ValueError(f"{split} 原始新增特征包含正无穷或负无穷")
    allowed = inside.astype(bool) | (split == "competition_test")
    for index, column in enumerate(config["feature_columns"]):
        parameter = parameters[column]
        data = transformed[:, index]
        finite = np.isfinite(data)
        if column in config["bounded_columns"]:
            data[allowed & ~finite] = parameter["median"]
            data[allowed] = 2.0 * data[allowed] - 1.0
        else:
            data[allowed & ~finite] = parameter["median"]
            data[allowed] = np.clip(
                data[allowed], parameter["lower_quantile"], parameter["upper_quantile"]
            )
            data[allowed] = (
                data[allowed] - parameter["mean"]
            ) / parameter["std_ddof0"]
        transformed[:, index] = data
    if np.isinf(transformed[allowed]).any():
        raise ValueError(f"{split} 允许处理范围内仍有无穷值")
    return transformed.astype(np.float32), allowed


def build(config, build_id, operation):
    root = Path(config["project_root"])
    input_dir = root / config["output_root"] / build_id
    factors_dir = root / config["factor_output_root"] / build_id
    old_root = root / config["old_input_root"]
    stock_metadata = pq.read_table(old_root / "股票元数据.parquet").to_pandas()
    old_metadata = json.loads((old_root / "模型输入接口元数据.json").read_text(encoding="utf-8"))
    if old_metadata["feature_count"] != 27 or old_metadata["window_length"] != 20:
        raise ValueError("原模型输入元数据与预处理方案不符")
    stock_code_values = pa.array(stock_metadata["ts_code"].astype(str).to_numpy())
    stock_id_values = stock_metadata["stock_id"].to_numpy(dtype=np.int32)
    stock_offset_by_id = np.empty(int(stock_id_values.max()) + 1, dtype=np.int64)
    stock_offset_by_id[stock_id_values] = stock_metadata["feature_offset"].to_numpy(dtype=np.int64)
    source_meta = {
        "train": pq.ParquetFile(config_path_root(config) / config["source_files"]["train"]).metadata.num_rows,
        "test": pq.ParquetFile(config_path_root(config) / config["source_files"]["test"]).metadata.num_rows,
    }
    if operation == "check-input":
        for split in SPLITS:
            count, key_digest = compare_split_identity(config, build_id, split)
            print(f"{split}: rows={count}, key_sha256={key_digest}")
        return

    factor_audit = json.loads((factors_dir / "生成审计.json").read_text(encoding="utf-8"))
    dates = np.load(factors_dir / "交易日期表.npy", allow_pickle=False).astype(np.int64)
    date_to_next = np.full(len(dates), -1, dtype=np.int64)
    date_to_next[:-1] = dates[1:]
    train_dates = dates[(dates >= 20180101) & (dates <= 20221231)]
    validation_dates = dates[(dates >= 20230101) & (dates <= 20231231)]
    if not len(train_dates) or not len(validation_dates):
        raise ValueError("真实交易日期未覆盖训练或验证区间")
    train_end = int(train_dates[-1])
    validation_bounds = (int(validation_dates[0]), int(validation_dates[-1]))
    index_dir = input_dir
    index_dir.mkdir(parents=True, exist_ok=True)
    key_summaries = {}
    index_paths = {}
    row_maps = {}
    date_maps = {}
    for split in SPLITS:
        count, key_digest = compare_split_identity(config, build_id, split)
        key_summaries[split] = {"rows": count, "key_sha256": key_digest}
        rows, date_map = split_source_rows(config, build_id, split)
        if len(rows) != count:
            raise ValueError(f"{split} 原始因子来源行数与标准化分区不同")
        row_maps[split] = rows
        date_maps[split] = date_map
        index_table = index_table_for_split(
            config, split, dates, date_to_next, train_end, validation_bounds
        )
        stock_row_positions = pc.index_in(index_table["ts_code"], value_set=stock_code_values)
        if stock_row_positions.null_count:
            raise ValueError(f"{split} 存在未知股票代码")
        stock_ids = stock_id_values[stock_row_positions.to_numpy().astype(np.int64)]
        feature_offsets = stock_offset_by_id[stock_ids]
        store_rows = feature_offsets + index_table["target_stock_row_number"].to_numpy().astype(np.int64)
        index_table = index_table.append_column("stock_id", pa.array(stock_ids, type=pa.int32()))
        index_table = index_table.append_column("feature_offset", pa.array(feature_offsets, type=pa.int64()))
        index_table = index_table.append_column("store_row_index", pa.array(store_rows, type=pa.int64()))
        index_path = index_dir / f"样本索引_{split}.parquet"
        pq.write_table(index_table, index_path, compression="zstd")
        index_paths[split] = index_path
        del index_table, stock_row_positions, stock_ids, feature_offsets, store_rows

    train_fit_index = pq.read_table(index_paths["train_fit"])
    fit_mask = train_fit_index["preprocess_fit_eligible"].to_numpy().astype(bool)
    if int(fit_mask.sum()) != 4_514_536:
        raise ValueError("新增参数估计资格数量与原有步骤 3—5 不同")
    train_temp = root / config["temporary_root"] / build_id / "train"
    train_count = source_meta["train"]
    raw_train = np.memmap(
        train_temp / "features.float32", dtype=np.float32,
        mode="r", shape=(train_count, len(config["feature_columns"])),
    )
    fit_rows = row_maps["train_fit"]
    if not np.array_equal(date_maps["train_fit"][fit_rows], train_fit_index["target_date"].to_numpy().astype(np.int32)):
        raise ValueError("train_fit 原始因子与序列索引日期位置不同")
    fit_values = np.asarray(raw_train[fit_rows], dtype=np.float32)
    fit_key_hasher = hashlib.sha256()
    for batch in pq.ParquetFile(index_paths["train_fit"]).iter_batches(
        columns=["ts_code", "target_date", "preprocess_fit_eligible"], batch_size=200_000
    ):
        frame = batch.to_pandas()
        selected = frame.loc[frame["preprocess_fit_eligible"].eq(1), ["ts_code", "target_date"]]
        fit_key_hasher.update(pd.util.hash_pandas_object(selected, index=False).values.tobytes())
    fit_key_digest = fit_key_hasher.hexdigest()
    parameters = estimate_parameters(fit_values, fit_mask, config, fit_key_digest)
    del train_fit_index, fit_values
    parameters["date_start"] = int(train_dates[0])
    parameters["date_end"] = train_end
    parameters["fit_rule"] = "train_fit sample_eligible=1"
    parameters["feature_columns"] = config["feature_columns"]
    (input_dir / "处理参数_train_fit.json").write_text(
        json.dumps(parameters, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    total_rows = int(stock_metadata["stock_length"].sum())
    store_path = input_dir / "新增特征存储.dat"
    store = np.memmap(
        store_path, dtype=np.float32, mode="w+",
        shape=(total_rows, len(config["feature_columns"])),
    )
    written = np.zeros(total_rows, dtype=np.bool_)
    transformed_counts = {}
    source_by_split = {
        split: ("test" if split == "competition_test" else "train") for split in SPLITS
    }
    for split in SPLITS:
        index_table = pq.read_table(index_paths[split])
        count = index_table.num_rows
        stock_ids = index_table["stock_id"].to_numpy().astype(np.int64)
        offsets = stock_offset_by_id[stock_ids]
        target_rows = index_table["target_stock_row_number"].to_numpy().astype(np.int64)
        addresses = offsets + target_rows
        if np.any(addresses < 0) or np.any(addresses >= total_rows):
            raise ValueError(f"{split} 存储地址越界")
        if len(np.unique(addresses)) != len(addresses):
            raise ValueError(f"{split} 存储地址重复")
        source_name = source_by_split[split]
        source_count = source_meta[source_name]
        source_temp = root / config["temporary_root"] / build_id / source_name
        raw_features = np.memmap(
            source_temp / "features.float32", dtype=np.float32, mode="r",
            shape=(source_count, len(config["feature_columns"])),
        )
        source_rows = row_maps[split]
        if len(source_rows) != count:
            raise ValueError(f"{split} 存储位置与索引数量不同")
        inside = index_table["inside_valid_interval"].to_numpy().astype(np.int8)
        split_processed = 0
        batch_size = config["batch_size"]
        for start in range(0, count, batch_size):
            end = min(start + batch_size, count)
            values, allowed = transform_batch(
                raw_features[source_rows[start:end]], inside[start:end], split,
                config, parameters["parameters"],
            )
            destination = addresses[start:end]
            if written[destination].any():
                raise ValueError(f"{split} 地址先前已写入")
            store[destination] = values
            written[destination] = True
            split_processed += int(allowed.sum())
        transformed_counts[split] = {
            "rows": count,
            "processed_rows": split_processed,
            "retained_rows": count - split_processed,
            "eligible_rows": int(index_table["prediction_eligible"].to_numpy().sum()),
            "training_rows": int(index_table["training_eligible"].to_numpy().sum()),
            "selection_rows": int(index_table["selection_eligible"].to_numpy().sum()),
        }
    if not written.all():
        raise ValueError(f"特征存储存在未写入地址: {int((~written).sum())}")
    store.flush()
    del store

    split_counts = {}
    for split in SPLITS:
        index_table = pq.read_table(index_paths[split])
        training_flags = index_table["training_eligible"].to_numpy()
        positions = np.flatnonzero(training_flags == 1)
        position_frame = pd.DataFrame(
            {"position": np.arange(len(positions), dtype=np.int64), "index_row_position": positions}
        )
        position_frame.to_parquet(index_dir / f"训练位置_{split}.parquet", index=False, compression="zstd")
        if split == "train_fit":
            selected = index_table["target_date"].to_numpy().astype(np.int64)[positions]
            date_index = []
            unique_dates, starts, counts = np.unique(
                selected, return_index=True, return_counts=True
            )
            for date, start, count in zip(unique_dates, starts, counts):
                date_index.append(
                    {
                        "target_date": int(date),
                        "start_position": int(start),
                        "end_position_exclusive": int(start + count),
                        "eligible_count": int(count),
                    }
                )
            pd.DataFrame(date_index).to_parquet(
                index_dir / "日期索引_train_fit.parquet", index=False, compression="zstd"
            )
        split_counts[split] = {
            "rows": index_table.num_rows,
            "prediction_eligible": int(index_table["prediction_eligible"].to_numpy().sum()),
            "training_eligible": int(training_flags.sum()),
            "selection_eligible": int(index_table["selection_eligible"].to_numpy().sum()),
        }
        del index_table, training_flags, positions
    metadata = {
        "status": "pending_verification",
        "build_id": build_id,
        "window_length": 20,
        "base_feature_count": 27,
        "extra_feature_count": len(config["feature_columns"]),
        "feature_columns": config["feature_columns"],
        "feature_dtype": "float32",
        "feature_store_shape": [total_rows, len(config["feature_columns"])],
        "split_key_summaries": key_summaries,
        "source_factor_audit": factor_audit,
        "split_counts": transformed_counts,
        "sample_counts": split_counts,
        "parameter_sha256": sha256_file(input_dir / "处理参数_train_fit.json"),
    }
    (input_dir / "输入元数据.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


def config_path_root(config):
    return Path(config["project_root"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--build-id", required=True)
    parser.add_argument("--operation", required=True, choices=["check-input", "build"])
    arguments = parser.parse_args()
    config_path = Path(arguments.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if arguments.operation == "check-input":
        build(config, arguments.build_id, "check-input")
    else:
        build(config, arguments.build_id, "build")


if __name__ == "__main__":
    main()
