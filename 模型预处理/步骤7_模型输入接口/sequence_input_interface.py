import json
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


class SequenceInputReader:
    def __init__(
        self,
        root_dir,
        split,
        batch_size=1024,
        eligible_only=True,
        as_torch=False
    ):
        self.root_dir = os.fspath(root_dir)
        self.split = split
        self.batch_size = int(batch_size)
        self.eligible_only = bool(eligible_only)
        self.as_torch = bool(as_torch)
        if self.batch_size <= 0:
            raise ValueError('batch_size 必须为正整数')

        metadata_path = os.path.join(self.root_dir, '模型输入接口元数据.json')
        with open(metadata_path, 'r', encoding='utf-8') as file:
            self.metadata = json.load(file)
        if split not in self.metadata['split_rows']:
            raise ValueError(f'未知分区: {split}')

        stock_path = os.path.join(
            self.root_dir, self.metadata['stock_metadata_file']
        )
        stock_frame = pd.read_parquet(stock_path)
        self.stock_id_map = dict(zip(
            stock_frame['ts_code'].astype(str),
            stock_frame['stock_id'].to_numpy(dtype=np.int32)
        ))
        self.feature_offsets = stock_frame[
            'feature_offset'
        ].to_numpy(dtype=np.int64)
        shape = tuple(self.metadata['feature_store_shape'])
        self.feature_store = np.memmap(
            os.path.join(self.root_dir, self.metadata['feature_store_file']),
            dtype=np.dtype(self.metadata['feature_dtype']),
            mode='r', shape=shape
        )
        index_name = {
            'train_fit': '序列索引_train_fit.parquet',
            'validation': '序列索引_validation.parquet',
            'local_holdout': '序列索引_local_holdout.parquet',
            'competition_test': '序列索引_competition_test.parquet'
        }[split]
        self.index_file = pq.ParquetFile(os.path.join(
            self.root_dir, '..', '步骤6_序列构造', index_name
        ))
        self.feature_columns = list(self.metadata['feature_columns'])
        self.window_length = int(self.metadata['window_length'])
        self._eligible_count = int(
            self.metadata['split_prediction_eligible'][split]
        )

    def __len__(self):
        if self.eligible_only:
            return self._eligible_count
        return int(self.metadata['split_rows'][self.split])

    def _build_features(self, frame):
        codes = frame['ts_code'].astype(str).tolist()
        stock_ids = np.asarray(
            [self.stock_id_map[code] for code in codes], dtype=np.int64
        )
        starts = frame['window_start_stock_row_number'].array
        if any(pd.isna(value) for value in starts):
            raise ValueError('可读取批次包含窗口不足的目标日')
        starts = np.asarray([int(value) for value in starts], dtype=np.int64)
        offsets = self.feature_offsets[stock_ids]
        row_numbers = (
            offsets[:, None]
            + starts[:, None]
            + np.arange(self.window_length, dtype=np.int64)[None, :]
        )
        features = np.asarray(self.feature_store[row_numbers], dtype=np.float32)
        expected_shape = (len(frame), self.window_length, len(self.feature_columns))
        if features.shape != expected_shape:
            raise ValueError(f'模型输入形状异常: {features.shape}')
        if not np.isfinite(features).all():
            raise ValueError('可预测序列包含非有限输入因子')
        return features

    def _convert_to_torch(self, batch):
        import torch

        batch['x'] = torch.from_numpy(batch['x'])
        if 'y' in batch:
            batch['y'] = torch.from_numpy(batch['y'])
        return batch

    def _make_batch(self, frame):
        features = self._build_features(frame)
        batch = {
            'x': features,
            'ts_code': frame['ts_code'].astype(str).to_numpy(),
            'target_date': frame['target_date'].to_numpy(dtype=np.int64),
            'panel_row_id': frame['panel_row_id'].to_numpy(dtype=np.int64),
            'prediction_eligible': frame[
                'prediction_eligible'
            ].to_numpy(dtype=np.int8)
        }
        if 'y_ret_1d' in frame.columns:
            batch['y'] = frame['y_ret_1d'].to_numpy(dtype=np.float32)
        if self.as_torch:
            batch = self._convert_to_torch(batch)
        return batch

    def iter_batches(self):
        columns = [
            'ts_code', 'target_date', 'panel_row_id',
            'window_start_stock_row_number', 'prediction_eligible'
        ]
        if self.split != 'competition_test':
            columns.append('y_ret_1d')
        for record_batch in self.index_file.iter_batches(
            columns=columns, batch_size=self.batch_size
        ):
            frame = record_batch.to_pandas()
            if self.eligible_only:
                frame = frame.loc[frame['prediction_eligible'].eq(1)].reset_index(
                    drop=True
                )
            if len(frame) == 0:
                continue
            yield self._make_batch(frame)

    def __iter__(self):
        return self.iter_batches()
