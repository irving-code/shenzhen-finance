import json
import os

import numpy as np
import pyarrow.parquet as pq

BASE = r'E:\ChatGPT\ChatGPT_Project\深圳金融比赛赛题五'
DATA_DIR = os.path.join(BASE, '因子数据')
OUTPUT_DIR = os.path.join(BASE, '模型预处理', '步骤3_缺失值处理')
FACTOR_COLUMNS = [
    'ret_1d', 'mom_3d', 'mom_5d', 'mom_10d', 'mom_20d', 'mom_20d_skip1',
    'volatility_5d', 'volatility_20d', 'range_1d', 'gap_1d',
    'true_range_1d', 'intraday_ret', 'close_position', 'body_direction',
    'body_ratio', 'upper_shadow', 'lower_shadow', 'log_amount', 'log_volume',
    'delta_log_amount', 'delta_log_volume', 'amount_shock_5d',
    'amount_shock_20d', 'limit_up', 'limit_down', 'missing_price_volume',
    'history_available_count_20d'
]
FILL_COLUMNS = [c for c in FACTOR_COLUMNS if c not in {'missing_price_volume', 'history_available_count_20d'}]


def split_for_date(value):
    value = int(value)
    if value <= 20221231:
        return 'train_fit'
    if value <= 20231231:
        return 'validation'
    if value <= 20241231:
        return 'local_holdout'
    return 'competition_test'


def count_input_rows(path):
    counts = {'train_fit': 0, 'validation': 0, 'local_holdout': 0, 'competition_test': 0}
    for batch in pq.ParquetFile(path).iter_batches(columns=['trade_date'], batch_size=200_000):
        dates = batch.to_pandas()['trade_date']
        for value in dates:
            counts[split_for_date(value)] += 1
    return counts


def check_output(path, expected_rows, train_output):
    parquet = pq.ParquetFile(path)
    checks = {
        'rows': parquet.metadata.num_rows,
        'expected_rows': expected_rows,
        'remaining_missing_eligible': 0,
        'training_outside_filled': 0,
        'invalid_status': 0,
        'nonfinite_filled_factor': 0,
        'first_key': None,
        'last_key': None
    }
    columns = ['ts_code', 'trade_date', 'inside_valid_interval', 'sample_eligible',
               'filled_factor_count', 'missing_price_volume',
               'history_available_count_20d'] + FILL_COLUMNS
    for batch in parquet.iter_batches(columns=columns, batch_size=200_000):
        frame = batch.to_pandas()
        if checks['first_key'] is None:
            checks['first_key'] = [str(frame.iloc[0]['ts_code']), int(frame.iloc[0]['trade_date'])]
        checks['last_key'] = [str(frame.iloc[-1]['ts_code']), int(frame.iloc[-1]['trade_date'])]
        eligible = frame['sample_eligible'].eq(1).to_numpy()
        missing = frame[FILL_COLUMNS].isna().to_numpy().sum(axis=1)
        checks['remaining_missing_eligible'] += int(missing[eligible].sum())
        if train_output:
            checks['training_outside_filled'] += int(
                ((frame['filled_factor_count'] > 0) & (frame['inside_valid_interval'] == 0)).sum()
            )
        checks['invalid_status'] += int(
            ((~frame['missing_price_volume'].isin([0, 1])) |
             (frame['history_available_count_20d'] < 0) |
             (frame['history_available_count_20d'] > 20)).sum()
        )
        checks['nonfinite_filled_factor'] += int(
            (~np.isfinite(frame.loc[eligible, FILL_COLUMNS].to_numpy(dtype=float))).sum()
        )
    return checks


def main():
    train_input = os.path.join(DATA_DIR, '训练集_第一次LST-Transformer因子.parquet')
    test_input = os.path.join(DATA_DIR, '测试集_X_第一次LST-Transformer因子.parquet')
    train_counts = count_input_rows(train_input)
    test_counts = count_input_rows(test_input)
    expected = {
        'train_fit': train_counts['train_fit'],
        'validation': train_counts['validation'],
        'local_holdout': train_counts['local_holdout'],
        'competition_test': test_counts['competition_test']
    }
    outputs = {
        'train_fit': os.path.join(OUTPUT_DIR, '训练集_train_fit_填充后.parquet'),
        'validation': os.path.join(OUTPUT_DIR, '验证集_填充后.parquet'),
        'local_holdout': os.path.join(OUTPUT_DIR, '本地留出集_填充后.parquet'),
        'competition_test': os.path.join(OUTPUT_DIR, '测试集_填充后.parquet')
    }
    result = {'expected_rows': expected, 'checks': {}}
    for name, path in outputs.items():
        result['checks'][name] = check_output(path, expected[name], name != 'competition_test')
    result['all_pass'] = all(
        value['rows'] == value['expected_rows']
        and value['remaining_missing_eligible'] == 0
        and value['training_outside_filled'] == 0
        and value['invalid_status'] == 0
        and value['nonfinite_filled_factor'] == 0
        for value in result['checks'].values()
    )
    output_path = os.path.join(OUTPUT_DIR, '步骤3_独立核验结果.json')
    with open(output_path, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False))
    if not result['all_pass']:
        raise ValueError('第三步独立核验未通过')


if __name__ == '__main__':
    main()
