import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


IDENTITY_COLUMNS = ["ts_code", "trade_date"]
SOURCE_COLUMNS = [
    "ret_1d",
    "mom_5d",
    "mom_20d",
    "volatility_5d",
    "amount_shock_5d",
    "intraday_ret",
]
RANK_COLUMNS = [
    "mom_5d",
    "mom_20d",
    "volatility_5d",
    "amount_shock_5d",
    "intraday_ret",
]
FEATURE_COLUMNS = [
    "csrank_mom_5d",
    "csrank_mom_20d",
    "csrank_volatility_5d",
    "csrank_amount_shock_5d",
    "csrank_intraday_ret",
    "excess_ret_1d",
    "excess_mom_5d",
    "market_mean_ret_1d",
    "market_breadth_1d",
    "market_dispersion_1d",
]


def calculate_source(config, source_name, build_id):
    root = Path(config["project_root"])
    source_path = root / config["source_files"][source_name]
    temporary_dir = root / config["temporary_root"] / build_id / source_name
    output_dir = root / config["factor_output_root"] / build_id
    temporary_dir.mkdir(parents=True, exist_ok=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    row_count = pq.ParquetFile(source_path).metadata.num_rows
    batch_size = config["batch_size"]
    date_path = temporary_dir / "trade_date.int32"
    source_path_mem = temporary_dir / "source_values.float32"
    valid_path = temporary_dir / "market_valid.uint8"
    output_path = temporary_dir / "features.float32"
    dates = np.memmap(date_path, dtype=np.int32, mode="w+", shape=(row_count,))
    source_values = np.memmap(
        source_path_mem, dtype=np.float32, mode="w+", shape=(row_count, len(SOURCE_COLUMNS))
    )
    market_valid = np.memmap(valid_path, dtype=np.uint8, mode="w+", shape=(row_count,))
    output_values = np.memmap(
        output_path, dtype=np.float32, mode="w+", shape=(row_count, len(FEATURE_COLUMNS))
    )

    source_file = pq.ParquetFile(source_path)
    offset = 0
    previous_code = None
    previous_date = None
    for batch in source_file.iter_batches(
        columns=IDENTITY_COLUMNS + ["missing_price_volume"] + SOURCE_COLUMNS,
        batch_size=batch_size,
    ):
        frame = batch.to_pandas()
        codes = frame["ts_code"].astype(str).to_numpy()
        batch_dates = frame["trade_date"].to_numpy(dtype=np.int32)
        if previous_code is not None and len(codes) and codes[0] < previous_code:
            raise ValueError(f"{source_name} 股票代码顺序未递增")
        if len(codes):
            boundaries = np.flatnonzero(codes[1:] != codes[:-1]) + 1
            starts = np.concatenate(([0], boundaries))
            ends = np.concatenate((boundaries, [len(codes)]))
            for group_start, group_end in zip(starts, ends):
                group_dates = batch_dates[group_start:group_end]
                if np.any(group_dates[1:] <= group_dates[:-1]):
                    raise ValueError(f"{source_name} 同一股票的交易日期重复或未递增")
            if previous_code == codes[0] and previous_date is not None and batch_dates[0] <= previous_date:
                raise ValueError(f"{source_name} 相邻批次的同一股票日期重复或未递增")
            previous_code = codes[-1]
            previous_date = int(batch_dates[-1])
        if frame.duplicated(IDENTITY_COLUMNS).any():
            raise ValueError(f"{source_name} 批次内股票日期键重复")
        end = offset + len(frame)
        dates[offset:end] = batch_dates
        source_values[offset:end] = frame[SOURCE_COLUMNS].to_numpy(dtype=np.float32)
        market_valid[offset:end] = frame["missing_price_volume"].eq(0).to_numpy(dtype=np.uint8)
        offset = end
    if offset != row_count:
        raise ValueError(f"{source_name} 读取行数与 Parquet 元数据不一致")
    dates.flush()
    source_values.flush()
    market_valid.flush()

    order = np.argsort(dates, kind="stable")
    sorted_dates = dates[order]
    unique_dates, date_starts = np.unique(sorted_dates, return_index=True)
    audit_rows = []
    rank_output_columns = [
        (source_column, SOURCE_COLUMNS.index(source_column))
        for source_column in RANK_COLUMNS
    ]
    for date_index, trade_date in enumerate(unique_dates):
        start = int(date_starts[date_index])
        end = int(date_starts[date_index + 1]) if date_index + 1 < len(date_starts) else row_count
        positions = order[start:end]
        day_values = np.asarray(source_values[positions], dtype=np.float64)
        day_valid = np.asarray(market_valid[positions], dtype=bool)
        finite = np.isfinite(day_values)
        daily = np.full((len(positions), len(FEATURE_COLUMNS)), np.nan, dtype=np.float64)
        for output_index, (source_column, value_index) in enumerate(rank_output_columns):
            valid = day_valid & finite[:, value_index]
            valid_values = day_values[valid, value_index]
            count = int(valid.sum())
            if count == 1:
                daily[valid, output_index] = 0.5
            elif count > 1:
                daily[valid, output_index] = (
                    pd.Series(valid_values).rank(method="average").to_numpy() - 1.0
                ) / (count - 1.0)
            audit_rows.append(
                {
                    "source": source_name,
                    "trade_date": int(trade_date),
                    "source_factor": source_column,
                    "source_rows": len(positions),
                    "valid_count": count,
                    "missing_count": int((~finite[:, value_index]).sum()),
                    "invalid_market_count": int((~day_valid).sum()),
                    "unique_value_count": int(np.unique(valid_values).size),
                }
            )

        ret_valid = day_valid & finite[:, 0]
        mom_valid = day_valid & finite[:, 1]
        ret_values = day_values[ret_valid, 0]
        mom_values = day_values[mom_valid, 1]
        if ret_values.size:
            ret_median = float(np.median(ret_values))
            ret_mean = float(np.mean(ret_values, dtype=np.float64))
            ret_dispersion = float(np.std(ret_values, ddof=0, dtype=np.float64))
            daily[ret_valid, 5] = ret_values - ret_median
            daily[ret_valid, 7] = ret_mean
            daily[ret_valid, 8] = np.mean(ret_values > 0, dtype=np.float64)
            daily[ret_valid, 9] = ret_dispersion
        if mom_values.size:
            daily[mom_valid, 6] = mom_values - float(np.median(mom_values))
        if np.isinf(daily).any():
            raise ValueError(f"{source_name} {trade_date} 横截面计算产生非有限数值")
        output_values[positions] = daily.astype(np.float32)

    output_values.flush()
    factor_path = output_dir / f"{source_name}_横截面因子.parquet"
    writer = None
    source_row_index = 0
    for batch in source_file.iter_batches(columns=IDENTITY_COLUMNS, batch_size=batch_size):
        frame = batch.to_pandas()
        end = source_row_index + len(frame)
        result = frame.copy()
        result.insert(0, "raw_source", source_name)
        result.insert(1, "raw_source_row_index", np.arange(source_row_index, end, dtype=np.int64))
        feature_frame = pd.DataFrame(
            np.asarray(output_values[source_row_index:end]), columns=FEATURE_COLUMNS
        )
        result = pd.concat([result.reset_index(drop=True), feature_frame], axis=1)
        table = pa.Table.from_pandas(result, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(factor_path, table.schema, compression="zstd")
        writer.write_table(table)
        source_row_index = end
    if writer is None or source_row_index != row_count:
        raise ValueError(f"{source_name} 新增因子输出不完整")
    writer.close()
    return factor_path, audit_rows, unique_dates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--build-id", required=True)
    arguments = parser.parse_args()
    config_path = Path(arguments.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if FEATURE_COLUMNS != config["feature_columns"]:
        raise ValueError("配置中的新增特征顺序与生成器定义不一致")
    results = {}
    all_audit = []
    all_dates = []
    for source_name in ("train", "test"):
        factor_path, audit_rows, dates = calculate_source(config, source_name, arguments.build_id)
        results[source_name] = str(factor_path)
        all_audit.extend(audit_rows)
        all_dates.append(dates)
    output_dir = Path(config["project_root"]) / config["factor_output_root"] / arguments.build_id
    statistics = pd.DataFrame(all_audit)
    statistics.to_parquet(output_dir / "每日集合统计.parquet", index=False, compression="zstd")
    trading_dates = np.unique(np.concatenate(all_dates)).astype(np.int32)
    np.save(output_dir / "交易日期表.npy", trading_dates, allow_pickle=False)
    audit = {
        "status": "complete",
        "feature_columns": FEATURE_COLUMNS,
        "rank_source_columns": RANK_COLUMNS,
        "source_files": results,
        "date_count": int(len(trading_dates)),
        "date_start": int(trading_dates[0]),
        "date_end": int(trading_dates[-1]),
        "statistics_rows": int(len(statistics)),
        "calculation_dtype": "float64",
        "output_dtype": "float32",
    }
    (output_dir / "生成审计.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
