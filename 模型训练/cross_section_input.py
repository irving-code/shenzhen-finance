import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


class CrossSectionInputReader:
    def __init__(self, base_metadata_path, extra_metadata_path, index_path, feature_mode):
        self.base_metadata_path = Path(base_metadata_path)
        self.extra_metadata_path = Path(extra_metadata_path)
        self.index_path = Path(index_path)
        self.feature_mode = feature_mode
        if feature_mode not in {"base27", "extended37"}:
            raise ValueError(f"未知特征模式: {feature_mode}")
        self.base_metadata = json.loads(self.base_metadata_path.read_text(encoding="utf-8"))
        self.extra_metadata = json.loads(self.extra_metadata_path.read_text(encoding="utf-8"))
        if self.extra_metadata["status"] not in {"pending_verification", "complete"}:
            raise ValueError("新增特征数据状态不允许读取")
        self.stock_metadata = pd.read_parquet(
            self.base_metadata_path.parent / self.base_metadata["stock_metadata_file"]
        )
        self.stock_offsets = self.stock_metadata["feature_offset"].to_numpy(dtype=np.int64)
        base_shape = tuple(self.base_metadata["feature_store_shape"])
        self.base_store = np.memmap(
            self.base_metadata_path.parent / self.base_metadata["feature_store_file"],
            dtype=np.dtype(self.base_metadata["feature_dtype"]),
            mode="r",
            shape=base_shape,
        )
        extra_shape = tuple(self.extra_metadata["feature_store_shape"])
        self.extra_store = np.memmap(
            self.extra_metadata_path.parent / "新增特征存储.dat",
            dtype=np.dtype(self.extra_metadata["feature_dtype"]),
            mode="r",
            shape=extra_shape,
        )
        self.index_file = pq.ParquetFile(self.index_path)
        index_columns = [
            "stock_id",
            "ts_code",
            "target_date",
            "panel_row_id",
            "feature_offset",
            "target_stock_row_number",
            "store_row_index",
            "window_start_stock_row_number",
            "prediction_eligible",
        ]
        if "y_ret_1d" in self.index_file.schema_arrow.names:
            index_columns.append("y_ret_1d")
        self.index_table = pq.read_table(self.index_path, columns=index_columns)

    def read_batch(self, index_positions):
        positions = np.asarray(index_positions, dtype=np.int64)
        if positions.ndim != 1 or np.any(positions < 0):
            raise ValueError("索引位置必须是一维非负整数数组")
        if np.any(positions >= self.index_table.num_rows):
            raise ValueError("索引位置超出样本索引范围")
        frame = self.index_table.take(pa.array(positions)).to_pandas()
        batch = self._read_frame(frame)
        batch["index_row_position"] = positions
        return batch

    def _read_frame(self, frame):
        if frame.empty:
            raise ValueError("模型输入批次不能为空")
        if not frame["window_start_stock_row_number"].notna().all():
            raise ValueError("模型输入批次包含不完整窗口")
        stock_ids = frame["stock_id"].to_numpy(dtype=np.int64)
        starts = frame["window_start_stock_row_number"].to_numpy(dtype=np.int64)
        if np.any(stock_ids < 0) or np.any(stock_ids >= len(self.stock_metadata)):
            raise ValueError("股票身份超出股票元数据范围")
        stock_rows = self.stock_metadata.iloc[stock_ids]
        if not np.array_equal(stock_rows["stock_id"].to_numpy(dtype=np.int64), stock_ids):
            raise ValueError("股票身份与股票元数据不一致")
        if not np.array_equal(stock_rows["ts_code"].astype(str).to_numpy(), frame["ts_code"].astype(str).to_numpy()):
            raise ValueError("股票代码与股票元数据不一致")
        offsets = self.stock_offsets[stock_ids]
        targets = frame["target_stock_row_number"].to_numpy(dtype=np.int64)
        lengths = stock_rows["stock_length"].to_numpy(dtype=np.int64)
        if np.any(starts < 0) or np.any(starts + 19 != targets) or np.any(targets >= lengths):
            raise ValueError("模型窗口超出对应股票的有效行数")
        if not np.array_equal(frame["feature_offset"].to_numpy(dtype=np.int64), offsets):
            raise ValueError("窗口特征偏移与股票元数据不一致")
        if not np.array_equal(frame["store_row_index"].to_numpy(dtype=np.int64), offsets + targets):
            raise ValueError("窗口目标存储地址不一致")
        addresses = offsets[:, None] + starts[:, None] + np.arange(20, dtype=np.int64)[None, :]
        base = np.asarray(self.base_store[addresses], dtype=np.float32)
        if base.shape != (len(frame), 20, 27):
            raise ValueError(f"原有模型输入形状异常: {base.shape}")
        if self.feature_mode == "extended37":
            extra = np.asarray(self.extra_store[addresses], dtype=np.float32)
            if extra.shape != (len(frame), 20, 10):
                raise ValueError(f"新增模型输入形状异常: {extra.shape}")
            features = np.concatenate((base, extra), axis=2)
        else:
            features = base
        if frame["prediction_eligible"].eq(1).all() and not np.isfinite(features).all():
            raise ValueError("合格模型窗口包含非有限输入")
        batch = {
            "x": features,
            "ts_code": frame["ts_code"].astype(str).to_numpy(),
            "target_date": frame["target_date"].to_numpy(dtype=np.int64),
            "panel_row_id": frame["panel_row_id"].to_numpy(dtype=np.int64),
            "prediction_eligible": frame["prediction_eligible"].to_numpy(dtype=np.int8),
        }
        if "y_ret_1d" in frame.columns and self.index_path.name != "样本索引_competition_test.parquet":
            batch["y"] = frame["y_ret_1d"].to_numpy(dtype=np.float32)
        return batch

    def iter_inference_batches(self, batch_size=4096):
        if batch_size <= 0:
            raise ValueError("批次大小必须为正整数")
        columns = [
            "stock_id",
            "ts_code",
            "target_date",
            "panel_row_id",
            "feature_offset",
            "target_stock_row_number",
            "store_row_index",
            "window_start_stock_row_number",
            "prediction_eligible",
        ]
        if self.index_path.name != "样本索引_competition_test.parquet":
            columns.append("y_ret_1d")
        for record_batch in self.index_file.iter_batches(columns=columns, batch_size=batch_size):
            frame = record_batch.to_pandas()
            frame = frame.loc[frame["prediction_eligible"].eq(1)].reset_index(drop=True)
            if not frame.empty:
                yield self._read_frame(frame)


class SameDateBatchSampler:
    def __init__(self, index, seed, epoch, max_batch_size=512):
        if max_batch_size <= 0:
            raise ValueError("同日批次上限必须为正整数")
        frame = index.to_pandas() if isinstance(index, pa.Table) else index.copy()
        eligible = frame.loc[frame["training_eligible"].eq(1)]
        if eligible.empty:
            raise ValueError("训练索引没有合格目标记录")
        self.max_batch_size = int(max_batch_size)
        self.seed = int(seed)
        self.epoch = int(epoch)
        if self.epoch < 1:
            raise ValueError("训练周期从 1 开始")
        self.groups = {
            int(date): group.sort_values(
                ["ts_code"], kind="stable"
            ).index.to_numpy(dtype=np.int64)
            for date, group in eligible.groupby("target_date", sort=True)
        }

    def __iter__(self):
        date_generator = np.random.default_rng(
            np.random.SeedSequence([self.seed, self.epoch, 20261008, 0])
        )
        dates = np.asarray(list(self.groups), dtype=np.int64)
        for date_position in date_generator.permutation(len(dates)):
            date = int(dates[date_position])
            positions = self.groups[date]
            stock_generator = np.random.default_rng(
                np.random.SeedSequence([self.seed, self.epoch, date, 20261008, 1])
            )
            shuffled = stock_generator.permutation(positions)
            batch_count = math.ceil(len(shuffled) / self.max_batch_size)
            for batch in np.array_split(shuffled, batch_count):
                if len(batch) and len(batch) <= self.max_batch_size:
                    yield batch
                else:
                    raise ValueError(f"{date} 同日批次大小异常")

