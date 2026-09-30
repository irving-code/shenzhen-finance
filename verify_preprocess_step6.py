import json
import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

BASE = r'E:\ChatGPT\ChatGPT_Project\深圳金融比赛赛题五'
STEP5_DIR = os.path.join(BASE, '模型预处理', '步骤5_标准化处理')
STEP6_DIR = os.path.join(BASE, '模型预处理', '步骤6_序列构造')
BATCH_SIZE = 200_000

INPUTS = OrderedDict([
    ('train_fit', os.path.join(STEP5_DIR, '训练集_train_fit_标准化.parquet')),
    ('validation', os.path.join(STEP5_DIR, '验证集_标准化.parquet')),
    ('local_holdout', os.path.join(STEP5_DIR, '本地留出集_标准化.parquet')),
    ('competition_test', os.path.join(STEP5_DIR, '测试集_标准化.parquet'))
])
INDEXES = OrderedDict([
    ('train_fit', os.path.join(STEP6_DIR, '序列索引_train_fit.parquet')),
    ('validation', os.path.join(STEP6_DIR, '序列索引_validation.parquet')),
    ('local_holdout', os.path.join(STEP6_DIR, '序列索引_local_holdout.parquet')),
    ('competition_test', os.path.join(STEP6_DIR, '序列索引_competition_test.parquet'))
])
WINDOW_LENGTH = 20


def check_sequence_file(split, path, expected_rows, global_start):
    parquet_file = pq.ParquetFile(path)
    required = {
        'source_split', 'source_row_index', 'panel_row_id', 'ts_code',
        'target_date', 'window_start_panel_row_id', 'window_end_panel_row_id',
        'window_start_stock_row_number', 'window_end_stock_row_number',
        'target_stock_row_number', 'sequence_available',
        'sequence_features_finite', 'sequence_inside_valid', 'sample_eligible',
        'prediction_eligible'
    }
    missing = sorted(required - set(parquet_file.schema_arrow.names))
    if missing:
        raise ValueError(f'{split} 序列索引缺少字段: {missing}')
    if parquet_file.metadata.num_rows != expected_rows:
        raise ValueError(f'{split} 序列索引行数异常')

    counts = OrderedDict([
        ('rows', 0), ('available', 0), ('finite', 0), ('inside', 0),
        ('eligible', 0), ('rule_error', 0), ('row_id_error', 0),
        ('stock_row_error', 0), ('split_error', 0)
    ])
    first_row = global_start
    for record_batch in parquet_file.iter_batches(batch_size=BATCH_SIZE):
        frame = record_batch.to_pandas()
        n = len(frame)
        expected_ids = np.arange(first_row, first_row + n, dtype=np.int64)
        panel_ids = frame['panel_row_id'].to_numpy(dtype=np.int64)
        counts['rows'] += n
        counts['available'] += int(frame['sequence_available'].sum())
        counts['finite'] += int(frame['sequence_features_finite'].sum())
        counts['inside'] += int(frame['sequence_inside_valid'].sum())
        counts['eligible'] += int(frame['prediction_eligible'].sum())
        counts['row_id_error'] += int(np.count_nonzero(panel_ids != expected_ids))
        counts['split_error'] += int(
            frame['source_split'].astype(str).ne(split).sum()
        )
        counts['row_id_error'] += int(
            np.count_nonzero(
                frame['window_end_panel_row_id'].to_numpy(dtype=np.int64) != panel_ids
            )
        )

        available = frame['sequence_available'].to_numpy(dtype=np.int8)
        finite = frame['sequence_features_finite'].to_numpy(dtype=np.int8)
        inside = frame['sequence_inside_valid'].to_numpy(dtype=np.int8)
        sample = frame['sample_eligible'].to_numpy(dtype=np.int8)
        eligible = frame['prediction_eligible'].to_numpy(dtype=np.int8)
        target_stock = frame['target_stock_row_number'].to_numpy(dtype=np.int64)
        end_stock = frame['window_end_stock_row_number'].to_numpy(dtype=np.int64)
        counts['stock_row_error'] += int(np.count_nonzero(end_stock != target_stock))
        if split == 'competition_test':
            expected_eligible = (available == 1) & (finite == 1)
        else:
            expected_eligible = (
                (sample == 1) & (available == 1) &
                (inside == 1) & (finite == 1)
            )
        counts['rule_error'] += int(np.count_nonzero(eligible != expected_eligible))

        start_panel = frame['window_start_panel_row_id'].array
        start_stock = frame['window_start_stock_row_number'].array
        for i in range(n):
            if available[i] == 1:
                if pd.isna(start_panel[i]) or pd.isna(start_stock[i]):
                    counts['row_id_error'] += 1
                elif int(start_stock[i]) != int(target_stock[i]) - WINDOW_LENGTH + 1:
                    counts['stock_row_error'] += 1
            else:
                if not pd.isna(start_panel[i]) or not pd.isna(start_stock[i]):
                    counts['row_id_error'] += 1
                if finite[i] != 0 or inside[i] != 0:
                    counts['rule_error'] += 1
        first_row += n
    return counts


def check_panel_index(path, expected_rows):
    parquet_file = pq.ParquetFile(path)
    required = {
        'panel_row_id', 'source_split', 'source_row_index', 'ts_code',
        'trade_date', 'stock_row_number'
    }
    missing = sorted(required - set(parquet_file.schema_arrow.names))
    if missing:
        raise ValueError(f'面板行索引缺少字段: {missing}')
    if parquet_file.metadata.num_rows != expected_rows:
        raise ValueError('面板行索引行数异常')

    errors = OrderedDict([
        ('panel_row_id', 0), ('source_row_index', 0),
        ('stock_row_number', 0), ('date_order', 0), ('source_split', 0)
    ])
    global_row = 0
    source_row = {split: 0 for split in INPUTS}
    stock_next = {}
    stock_last_date = {}
    for record_batch in parquet_file.iter_batches(
        columns=list(required), batch_size=BATCH_SIZE
    ):
        frame = record_batch.to_pandas()
        for row in frame.itertuples(index=False):
            row_dict = row._asdict()
            if int(row_dict['panel_row_id']) != global_row:
                errors['panel_row_id'] += 1
            split = str(row_dict['source_split'])
            if split not in source_row:
                errors['source_split'] += 1
                source_row[split] = 0
            if int(row_dict['source_row_index']) != source_row[split]:
                errors['source_row_index'] += 1
            source_row[split] += 1
            code = str(row_dict['ts_code'])
            expected_stock_row = stock_next.get(code, 0)
            if int(row_dict['stock_row_number']) != expected_stock_row:
                errors['stock_row_number'] += 1
            stock_next[code] = expected_stock_row + 1
            date_value = int(row_dict['trade_date'])
            if code in stock_last_date and date_value <= stock_last_date[code]:
                errors['date_order'] += 1
            stock_last_date[code] = date_value
            global_row += 1
    return errors, len(stock_next)


def main():
    panel_path = os.path.join(STEP6_DIR, '面板行索引.parquet')
    expected_rows = OrderedDict([
        (split, pq.ParquetFile(path).metadata.num_rows)
        for split, path in INPUTS.items()
    ])
    global_start = 0
    sequence_checks = OrderedDict()
    for split, path in INDEXES.items():
        sequence_checks[split] = check_sequence_file(
            split, path, expected_rows[split], global_start
        )
        global_start += expected_rows[split]
    expected_total = sum(expected_rows.values())
    panel_errors, stock_count = check_panel_index(panel_path, expected_total)

    all_pass = all(
        all(value == 0 for key, value in checks.items() if key.endswith('error'))
        for checks in sequence_checks.values()
    ) and all(value == 0 for value in panel_errors.values())
    result = OrderedDict([
        ('all_pass', bool(all_pass)),
        ('expected_rows', expected_rows),
        ('panel_rows', expected_total),
        ('stock_count', stock_count),
        ('sequence_checks', sequence_checks),
        ('panel_errors', panel_errors)
    ])
    output_path = os.path.join(STEP6_DIR, '步骤6_独立核验结果.json')
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    if not all_pass:
        raise ValueError('第六步独立核验未通过')


if __name__ == '__main__':
    main()
