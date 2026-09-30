import json
import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

BASE = r'E:\ChatGPT\ChatGPT_Project\深圳金融比赛赛题五'
STEP5_DIR = os.path.join(BASE, '模型预处理', '步骤5_标准化处理')
STEP6_DIR = os.path.join(BASE, '模型预处理', '步骤6_序列构造')
OUTPUT_DIR = os.path.join(BASE, '模型预处理', '步骤7_模型输入接口')
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 200_000
FEATURE_DTYPE = np.float32
CONTINUOUS_COLUMNS = [
    'ret_1d', 'mom_3d', 'mom_5d', 'mom_10d', 'mom_20d', 'mom_20d_skip1',
    'volatility_5d', 'volatility_20d', 'range_1d', 'gap_1d',
    'true_range_1d', 'intraday_ret', 'close_position', 'body_ratio',
    'upper_shadow', 'lower_shadow', 'log_amount', 'log_volume',
    'delta_log_amount', 'delta_log_volume', 'amount_shock_5d',
    'amount_shock_20d'
]
DISCRETE_COLUMNS = [
    'body_direction', 'limit_up', 'limit_down', 'missing_price_volume'
]
MODEL_FEATURE_COLUMNS = CONTINUOUS_COLUMNS + DISCRETE_COLUMNS + [
    'history_available_ratio_20d'
]
SPLITS = OrderedDict([
    ('train_fit', {
        'data': os.path.join(STEP5_DIR, '训练集_train_fit_标准化.parquet'),
        'index': os.path.join(STEP6_DIR, '序列索引_train_fit.parquet')
    }),
    ('validation', {
        'data': os.path.join(STEP5_DIR, '验证集_标准化.parquet'),
        'index': os.path.join(STEP6_DIR, '序列索引_validation.parquet')
    }),
    ('local_holdout', {
        'data': os.path.join(STEP5_DIR, '本地留出集_标准化.parquet'),
        'index': os.path.join(STEP6_DIR, '序列索引_local_holdout.parquet')
    }),
    ('competition_test', {
        'data': os.path.join(STEP5_DIR, '测试集_标准化.parquet'),
        'index': os.path.join(STEP6_DIR, '序列索引_competition_test.parquet')
    })
])


def collect_stock_metadata():
    stock_ids = OrderedDict()
    stock_lengths = OrderedDict()
    row_counts = OrderedDict()
    for split, paths in SPLITS.items():
        index_file = pq.ParquetFile(paths['index'])
        row_counts[split] = index_file.metadata.num_rows
        for record_batch in index_file.iter_batches(
            columns=['ts_code', 'target_stock_row_number'], batch_size=BATCH_SIZE
        ):
            frame = record_batch.to_pandas()
            codes = frame['ts_code'].astype(str)
            positions = frame['target_stock_row_number'].to_numpy(dtype=np.int64)
            for code in codes.drop_duplicates().tolist():
                if code not in stock_ids:
                    stock_ids[code] = len(stock_ids)
            group_max = frame.assign(_position=positions).groupby(
                'ts_code', sort=False
            )['_position'].max()
            for code, max_position in group_max.items():
                code = str(code)
                stock_lengths[code] = max(
                    stock_lengths.get(code, 0), int(max_position) + 1
                )

    missing_lengths = [code for code in stock_ids if code not in stock_lengths]
    if missing_lengths:
        raise ValueError(f'股票元数据缺少长度: {missing_lengths[:5]}')
    rows = []
    offset = 0
    for code, stock_id in stock_ids.items():
        length = stock_lengths[code]
        rows.append({
            'ts_code': code,
            'stock_id': stock_id,
            'feature_offset': offset,
            'stock_length': length
        })
        offset += length
    stock_frame = pd.DataFrame(rows)
    stock_frame.to_parquet(
        os.path.join(OUTPUT_DIR, '股票元数据.parquet'),
        index=False, compression='zstd'
    )
    return stock_frame, row_counts, offset


def collect_source_maps(stock_frame, row_counts):
    stock_id_map = dict(zip(
        stock_frame['ts_code'].astype(str), stock_frame['stock_id'].astype(np.int32)
    ))
    source_stock_ids = {}
    source_stock_rows = {}
    for split, paths in SPLITS.items():
        source_stock_ids[split] = np.empty(row_counts[split], dtype=np.int32)
        source_stock_rows[split] = np.empty(row_counts[split], dtype=np.int64)
        row_start = 0
        index_file = pq.ParquetFile(paths['index'])
        for record_batch in index_file.iter_batches(
            columns=['source_row_index', 'ts_code', 'target_stock_row_number'],
            batch_size=BATCH_SIZE
        ):
            frame = record_batch.to_pandas()
            n = len(frame)
            source_rows = frame['source_row_index'].to_numpy(dtype=np.int64)
            expected_rows = np.arange(row_start, row_start + n, dtype=np.int64)
            if np.any(source_rows != expected_rows):
                raise ValueError(f'{split} 来源行号不连续')
            codes = frame['ts_code'].astype(str).tolist()
            source_stock_ids[split][row_start:row_start + n] = np.asarray(
                [stock_id_map[code] for code in codes], dtype=np.int32
            )
            source_stock_rows[split][row_start:row_start + n] = frame[
                'target_stock_row_number'
            ].to_numpy(dtype=np.int64)
            row_start += n
        if row_start != row_counts[split]:
            raise ValueError(f'{split} 来源行数映射不完整')
    return source_stock_ids, source_stock_rows


def write_feature_store(stock_frame, row_counts, source_stock_ids, source_stock_rows):
    total_rows = int(stock_frame['stock_length'].sum())
    feature_path = os.path.join(OUTPUT_DIR, '特征存储.dat')
    feature_store = np.memmap(
        feature_path, dtype=FEATURE_DTYPE, mode='w+',
        shape=(total_rows, len(MODEL_FEATURE_COLUMNS))
    )
    offsets = stock_frame['feature_offset'].to_numpy(dtype=np.int64)
    filled = np.zeros(total_rows, dtype=np.bool_)
    max_abs_cast_error = 0.0
    written_rows = 0
    for split, paths in SPLITS.items():
        data_file = pq.ParquetFile(paths['data'])
        row_start = 0
        for record_batch in data_file.iter_batches(
            columns=MODEL_FEATURE_COLUMNS, batch_size=BATCH_SIZE
        ):
            frame = record_batch.to_pandas()
            n = len(frame)
            stock_ids = source_stock_ids[split][row_start:row_start + n]
            stock_rows = source_stock_rows[split][row_start:row_start + n]
            destination_rows = offsets[stock_ids] + stock_rows
            if np.any(filled[destination_rows]):
                raise ValueError(f'{split} 特征存储出现重复写入')
            values = frame.to_numpy(dtype=np.float64)
            cast_values = values.astype(FEATURE_DTYPE)
            finite_difference = np.isfinite(values) & np.isfinite(cast_values)
            if np.any(finite_difference):
                difference = np.abs(
                    values[finite_difference] - cast_values[finite_difference]
                )
                max_abs_cast_error = max(max_abs_cast_error, float(difference.max()))
            feature_store[destination_rows] = cast_values
            filled[destination_rows] = True
            written_rows += n
            row_start += n
        if row_start != row_counts[split]:
            raise ValueError(f'{split} 特征写入行数异常')
    if written_rows != total_rows or not bool(filled.all()):
        raise ValueError('特征存储存在未写入行')
    feature_store.flush()
    del feature_store
    return total_rows, max_abs_cast_error


def main():
    stock_frame, row_counts, total_rows = collect_stock_metadata()
    source_stock_ids, source_stock_rows = collect_source_maps(stock_frame, row_counts)
    written_rows, max_abs_cast_error = write_feature_store(
        stock_frame, row_counts, source_stock_ids, source_stock_rows
    )
    index_result_path = os.path.join(STEP6_DIR, '步骤6_序列构造结果.json')
    with open(index_result_path, 'r', encoding='utf-8') as file:
        index_result = json.load(file)
    metadata = OrderedDict([
        ('window_length', 20),
        ('feature_count', len(MODEL_FEATURE_COLUMNS)),
        ('feature_columns', MODEL_FEATURE_COLUMNS),
        ('feature_dtype', 'float32'),
        ('feature_store_file', '特征存储.dat'),
        ('feature_store_shape', [written_rows, len(MODEL_FEATURE_COLUMNS)]),
        ('stock_metadata_file', '股票元数据.parquet'),
        ('stock_count', int(len(stock_frame))),
        ('total_rows', int(written_rows)),
        ('max_abs_float32_cast_error', max_abs_cast_error),
        ('split_rows', row_counts),
        ('split_prediction_eligible', {
            split: index_result['splits'][split]['prediction_eligible']
            for split in SPLITS
        }),
        ('source_files_modified', False)
    ])
    metadata_path = os.path.join(OUTPUT_DIR, '模型输入接口元数据.json')
    with open(metadata_path, 'w', encoding='utf-8') as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    report_lines = [
        '# 第七步：模型输入接口构建报告',
        '',
        '## 处理结果',
        '',
        f"- 特征存储形状：`({written_rows:,}, {len(MODEL_FEATURE_COLUMNS)})`。",
        '- 特征数据类型：`float32`。',
        f'- 股票数量：{len(stock_frame):,}。',
        f'- `float64` 转换为 `float32` 的最大绝对误差：{max_abs_cast_error:.12g}。',
        '- 存储布局：同一支股票的所有交易日连续排列，序列索引使用股票内行号提取窗口。',
        '',
        '## 分区接口',
        '',
        '| 分区 | 输入行数 | 可预测样本 | 标签 |',
        '|---|---:|---:|---|'
    ]
    for split in SPLITS:
        label = '有' if split != 'competition_test' else '无'
        report_lines.append(
            f"| `{split}` | {row_counts[split]:,} | "
            f"{metadata['split_prediction_eligible'][split]:,} | {label} |"
        )
    report_lines.extend([
        '',
        '## 使用方式',
        '',
        '- 使用 `sequence_input_interface.py` 的 `SequenceInputReader` 按批读取。',
        '- 每个 `x` 的形状为 `(batch_size, 20, 27)`。',
        '- 训练、验证和本地留出集的批次包含 `y`；官方测试集批次不包含 `y`。',
        '- `as_torch=True` 时，接口将 `x` 和 `y` 转换为 PyTorch 张量。',
        '',
        '## 数据保护',
        '',
        '- 原始因子文件未修改。',
        '- 第五步标准化文件未修改。',
        '- 第六步序列索引只读使用。'
    ])
    with open(os.path.join(OUTPUT_DIR, '步骤7_模型输入接口构建报告.md'), 'w', encoding='utf-8') as file:
        file.write('\n'.join(report_lines) + '\n')


if __name__ == '__main__':
    main()
