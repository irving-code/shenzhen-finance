import json
import os

import numpy as np
import pyarrow.parquet as pq

BASE = r'E:\ChatGPT\ChatGPT_Project\深圳金融比赛赛题五'
INPUT_DIR = os.path.join(BASE, '模型预处理', '步骤4_异常值处理')
OUTPUT_DIR = os.path.join(BASE, '模型预处理', '步骤5_标准化处理')
CONTINUOUS_COLUMNS = [
    'ret_1d', 'mom_3d', 'mom_5d', 'mom_10d', 'mom_20d', 'mom_20d_skip1',
    'volatility_5d', 'volatility_20d', 'range_1d', 'gap_1d',
    'true_range_1d', 'intraday_ret', 'close_position', 'body_ratio',
    'upper_shadow', 'lower_shadow', 'log_amount', 'log_volume',
    'delta_log_amount', 'delta_log_volume', 'amount_shock_5d',
    'amount_shock_20d'
]

INPUTS = {
    'train_fit': os.path.join(INPUT_DIR, 'train_fit_异常值处理.parquet'),
    'validation': os.path.join(INPUT_DIR, 'validation_异常值处理.parquet'),
    'local_holdout': os.path.join(INPUT_DIR, 'local_holdout_异常值处理.parquet'),
    'competition_test': os.path.join(INPUT_DIR, 'competition_test_异常值处理.parquet')
}
OUTPUTS = {
    'train_fit': os.path.join(OUTPUT_DIR, '训练集_train_fit_标准化.parquet'),
    'validation': os.path.join(OUTPUT_DIR, '验证集_标准化.parquet'),
    'local_holdout': os.path.join(OUTPUT_DIR, '本地留出集_标准化.parquet'),
    'competition_test': os.path.join(OUTPUT_DIR, '测试集_标准化.parquet')
}


def check_split(name):
    input_file = pq.ParquetFile(INPUTS[name])
    output_file = pq.ParquetFile(OUTPUTS[name])
    result = {
        'input_rows': input_file.metadata.num_rows,
        'output_rows': output_file.metadata.num_rows,
        'eligible_rows': 0,
        'eligible_nonfinite': 0,
        'outside_standardized_cells': 0,
        'ratio_invalid': 0,
        'mean_max_abs': None,
        'std_max_abs_error': None,
        'label_max_abs_diff': 0.0,
        'test_has_label': 'y_ret_1d' in output_file.schema_arrow.names
    }
    running_means = np.zeros(len(CONTINUOUS_COLUMNS), dtype=np.float64)
    running_m2 = np.zeros(len(CONTINUOUS_COLUMNS), dtype=np.float64)
    counts = np.zeros(len(CONTINUOUS_COLUMNS), dtype=np.int64)
    output_columns = ['ts_code', 'trade_date', 'inside_valid_interval', 'sample_eligible',
                      'standardized_factor_count', 'history_available_ratio_20d'] + CONTINUOUS_COLUMNS
    if name == 'train_fit':
        output_columns.append('y_ret_1d')
    for batch in output_file.iter_batches(columns=output_columns, batch_size=200_000):
        frame = batch.to_pandas()
        eligible = frame['sample_eligible'].eq(1).to_numpy()
        values = frame.loc[eligible, CONTINUOUS_COLUMNS].to_numpy(dtype=float)
        result['eligible_rows'] += int(eligible.sum())
        result['eligible_nonfinite'] += int((~np.isfinite(values)).sum())
        if name != 'competition_test':
            result['outside_standardized_cells'] += int(
                (frame['standardized_factor_count'].to_numpy() * (~frame['inside_valid_interval'].eq(1).to_numpy())).sum()
            )
        ratio = frame['history_available_ratio_20d'].dropna().to_numpy(dtype=float)
        result['ratio_invalid'] += int(((ratio < 0) | (ratio > 1) | (~np.isfinite(ratio))).sum())
        if values.size:
            for index in range(values.shape[1]):
                current = values[:, index]
                current = current[np.isfinite(current)]
                if current.size == 0:
                    continue
                batch_count = current.size
                batch_mean = float(current.mean())
                batch_m2 = float(((current - batch_mean) ** 2).sum())
                previous_count = counts[index]
                total_count = previous_count + batch_count
                delta = batch_mean - running_means[index]
                running_means[index] += delta * batch_count / total_count
                running_m2[index] += batch_m2 + delta * delta * previous_count * batch_count / total_count
                counts[index] = total_count
    means = running_means
    stds = np.sqrt(running_m2 / counts)
    result['mean_max_abs'] = float(np.max(np.abs(means)))
    result['std_max_abs_error'] = float(np.max(np.abs(stds - 1)))
    result['rows_match'] = result['input_rows'] == result['output_rows']
    result['pass'] = (
        result['rows_match']
        and result['eligible_nonfinite'] == 0
        and result['ratio_invalid'] == 0
        and result['outside_standardized_cells'] == 0
        and (name != 'train_fit' or result['mean_max_abs'] < 1e-5)
        and (name != 'train_fit' or result['std_max_abs_error'] < 1e-5)
        and (name != 'competition_test' or not result['test_has_label'])
    )
    return result


def check_labels():
    input_file = pq.ParquetFile(INPUTS['train_fit'])
    output_file = pq.ParquetFile(OUTPUTS['train_fit'])
    input_batches = input_file.iter_batches(columns=['y_ret_1d'], batch_size=200_000)
    output_batches = output_file.iter_batches(columns=['y_ret_1d'], batch_size=200_000)
    maximum = 0.0
    for input_batch, output_batch in zip(input_batches, output_batches):
        left = input_batch.to_pandas()['y_ret_1d'].to_numpy(dtype=float)
        right = output_batch.to_pandas()['y_ret_1d'].to_numpy(dtype=float)
        diff = np.abs(left - right)
        diff[np.isnan(diff)] = 0
        maximum = max(maximum, float(np.max(diff)))
    return maximum


def main():
    result = {'checks': {name: check_split(name) for name in INPUTS}}
    result['checks']['train_fit']['label_max_abs_diff'] = check_labels()
    result['checks']['train_fit']['pass'] = (
        result['checks']['train_fit']['pass']
        and result['checks']['train_fit']['label_max_abs_diff'] == 0
    )
    result['all_pass'] = all(item['pass'] for item in result['checks'].values())
    output_path = os.path.join(OUTPUT_DIR, '步骤5_独立核验结果.json')
    with open(output_path, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False))
    if not result['all_pass']:
        raise ValueError('第五步独立核验未通过')


if __name__ == '__main__':
    main()
