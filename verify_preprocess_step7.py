import json
import os
import sys
from collections import OrderedDict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

BASE = r'E:\ChatGPT\ChatGPT_Project\深圳金融比赛赛题五'
STEP5_DIR = os.path.join(BASE, '模型预处理', '步骤5_标准化处理')
STEP6_DIR = os.path.join(BASE, '模型预处理', '步骤6_序列构造')
STEP7_DIR = os.path.join(BASE, '模型预处理', '步骤7_模型输入接口')
sys.path.insert(0, STEP7_DIR)
from sequence_input_interface import SequenceInputReader

SPLITS = OrderedDict([
    ('train_fit', {
        'data': os.path.join(STEP5_DIR, '训练集_train_fit_标准化.parquet'),
        'index': os.path.join(STEP6_DIR, '序列索引_train_fit.parquet'),
        'has_label': True
    }),
    ('validation', {
        'data': os.path.join(STEP5_DIR, '验证集_标准化.parquet'),
        'index': os.path.join(STEP6_DIR, '序列索引_validation.parquet'),
        'has_label': True
    }),
    ('local_holdout', {
        'data': os.path.join(STEP5_DIR, '本地留出集_标准化.parquet'),
        'index': os.path.join(STEP6_DIR, '序列索引_local_holdout.parquet'),
        'has_label': True
    }),
    ('competition_test', {
        'data': os.path.join(STEP5_DIR, '测试集_标准化.parquet'),
        'index': os.path.join(STEP6_DIR, '序列索引_competition_test.parquet'),
        'has_label': False
    })
])


def first_eligible_index(index_path):
    index_file = pq.ParquetFile(index_path)
    for record_batch in index_file.iter_batches(
        columns=[
            'source_row_index', 'ts_code', 'target_stock_row_number',
            'window_start_stock_row_number', 'prediction_eligible'
        ], batch_size=50_000
    ):
        frame = record_batch.to_pandas()
        eligible = frame.loc[frame['prediction_eligible'].eq(1)]
        if len(eligible):
            return eligible.iloc[0]
    raise ValueError('序列索引中没有可预测样本')


def compare_one_source_row(split, config, reader, index_row):
    source_row = int(index_row['source_row_index'])
    source_file = pq.ParquetFile(config['data'])
    needed = reader.feature_columns
    start = 0
    for record_batch in source_file.iter_batches(
        columns=needed, batch_size=50_000
    ):
        frame = record_batch.to_pandas()
        end = start + len(frame)
        if start <= source_row < end:
            source_values = frame.iloc[source_row - start].to_numpy(dtype=np.float64)
            code = str(index_row['ts_code'])
            stock_id = reader.stock_id_map[code]
            stock_row = int(index_row['target_stock_row_number'])
            offset = int(reader.feature_offsets[stock_id])
            stored_values = np.asarray(
                reader.feature_store[offset + stock_row], dtype=np.float64
            )
            finite = np.isfinite(source_values) & np.isfinite(stored_values)
            if np.any(finite):
                error = float(np.max(np.abs(
                    source_values[finite] - stored_values[finite]
                )))
            else:
                error = 0.0
            if error > 5e-6:
                raise ValueError(f'{split} 特征存储与来源行不一致: {error}')
            return error
        start = end
    raise ValueError(f'{split} 找不到来源行')


def main():
    metadata_path = os.path.join(STEP7_DIR, '模型输入接口元数据.json')
    with open(metadata_path, 'r', encoding='utf-8') as file:
        metadata = json.load(file)
    stock_frame = pd.read_parquet(
        os.path.join(STEP7_DIR, metadata['stock_metadata_file'])
    )
    if int(stock_frame['stock_length'].sum()) != metadata['total_rows']:
        raise ValueError('股票元数据长度总和异常')

    results = OrderedDict()
    for split, config in SPLITS.items():
        reader = SequenceInputReader(
            STEP7_DIR, split, batch_size=8, eligible_only=True, as_torch=False
        )
        batch = next(iter(reader.iter_batches()))
        if batch['x'].shape[0] <= 0 or batch['x'].shape[0] > 8 or batch['x'].shape[1:] != (
            metadata['window_length'], metadata['feature_count']
        ):
            raise ValueError(f'{split} 输入形状异常: {batch["x"].shape}')
        if batch['x'].dtype != np.float32:
            raise ValueError(f'{split} 输入数据类型异常: {batch["x"].dtype}')
        if not np.isfinite(batch['x']).all():
            raise ValueError(f'{split} 可预测批次存在非有限输入')
        if config['has_label'] != ('y' in batch):
            raise ValueError(f'{split} 标签字段存在性异常')
        if 'y' in batch and batch['y'].shape != (batch['x'].shape[0],):
            raise ValueError(f'{split} 标签形状异常')
        first_row = first_eligible_index(config['index'])
        source_error = compare_one_source_row(split, config, reader, first_row)
        results[split] = OrderedDict([
            ('reader_length', len(reader)),
            ('batch_shape', list(batch['x'].shape)),
            ('dtype', str(batch['x'].dtype)),
            ('has_label', 'y' in batch),
            ('first_source_row_max_abs_error', source_error)
        ])

    torch_reader = SequenceInputReader(
        STEP7_DIR, 'train_fit', batch_size=4, eligible_only=True, as_torch=True
    )
    torch_batch = next(iter(torch_reader.iter_batches()))
    if torch_batch['x'].shape[0] <= 0 or torch_batch['x'].shape[0] > 4 or tuple(
        torch_batch['x'].shape[1:]
    ) != (metadata['window_length'], metadata['feature_count']):
        raise ValueError('PyTorch 输入形状异常')
    results['torch_adapter'] = OrderedDict([
        ('x_type', str(type(torch_batch['x']))),
        ('x_shape', list(torch_batch['x'].shape)),
        ('y_shape', list(torch_batch['y'].shape))
    ])
    result = OrderedDict([
        ('all_pass', True),
        ('results', results),
        ('feature_store_shape', metadata['feature_store_shape']),
        ('feature_store_dtype', metadata['feature_dtype'])
    ])
    output_path = os.path.join(STEP7_DIR, '步骤7_独立核验结果.json')
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(result, file, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
