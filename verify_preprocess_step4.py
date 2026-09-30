import json
import os

import numpy as np
import pyarrow.parquet as pq

BASE = r'E:\ChatGPT\ChatGPT_Project\深圳金融比赛赛题五'
INPUT_DIR = os.path.join(BASE, '模型预处理', '步骤3_缺失值处理')
OUTPUT_DIR = os.path.join(BASE, '模型预处理', '步骤4_异常值处理')
FILL_COLUMNS = [
    'ret_1d', 'mom_3d', 'mom_5d', 'mom_10d', 'mom_20d', 'mom_20d_skip1',
    'volatility_5d', 'volatility_20d', 'range_1d', 'gap_1d',
    'true_range_1d', 'intraday_ret', 'close_position', 'body_ratio',
    'upper_shadow', 'lower_shadow', 'log_amount', 'log_volume',
    'delta_log_amount', 'delta_log_volume', 'amount_shock_5d',
    'amount_shock_20d'
]
DISCRETE_COLUMNS = ['body_direction', 'limit_up', 'limit_down', 'missing_price_volume', 'history_available_count_20d']

INPUTS = {
    'train_fit': os.path.join(INPUT_DIR, '训练集_train_fit_填充后.parquet'),
    'validation': os.path.join(INPUT_DIR, '验证集_填充后.parquet'),
    'local_holdout': os.path.join(INPUT_DIR, '本地留出集_填充后.parquet'),
    'competition_test': os.path.join(INPUT_DIR, '测试集_填充后.parquet')
}
OUTPUTS = {
    name: os.path.join(OUTPUT_DIR, f'{name}_异常值处理.parquet')
    for name in INPUTS
}


def check_output(name, input_path, output_path):
    input_rows = pq.ParquetFile(input_path).metadata.num_rows
    output_file = pq.ParquetFile(output_path)
    result = {
        'input_rows': input_rows,
        'output_rows': output_file.metadata.num_rows,
        'eligible_nonfinite': 0,
        'invalid_discrete': 0,
        'outside_clipped': 0,
        'clip_count': 0,
        'first_key': None,
        'last_key': None
    }
    columns = ['ts_code', 'trade_date', 'inside_valid_interval', 'sample_eligible',
               'clipped_factor_count'] + FILL_COLUMNS + DISCRETE_COLUMNS
    for batch in output_file.iter_batches(columns=columns, batch_size=200_000):
        frame = batch.to_pandas()
        if result['first_key'] is None:
            result['first_key'] = [str(frame.iloc[0]['ts_code']), int(frame.iloc[0]['trade_date'])]
        result['last_key'] = [str(frame.iloc[-1]['ts_code']), int(frame.iloc[-1]['trade_date'])]
        eligible = frame['sample_eligible'].eq(1).to_numpy()
        result['eligible_nonfinite'] += int((~np.isfinite(frame.loc[eligible, FILL_COLUMNS].to_numpy(dtype=float))).sum())
        eligible_frame = frame.loc[eligible]
        result['invalid_discrete'] += int((~eligible_frame['body_direction'].isin([-1, 0, 1])).sum())
        result['invalid_discrete'] += int((~eligible_frame['limit_up'].isin([0, 1])).sum())
        result['invalid_discrete'] += int((~eligible_frame['limit_down'].isin([0, 1])).sum())
        result['invalid_discrete'] += int((~eligible_frame['missing_price_volume'].isin([0, 1])).sum())
        result['invalid_discrete'] += int(((eligible_frame['history_available_count_20d'] < 0) | (eligible_frame['history_available_count_20d'] > 20)).sum())
        result['clip_count'] += int(frame['clipped_factor_count'].sum())
        if name != 'competition_test':
            result['outside_clipped'] += int(
                ((frame['clipped_factor_count'] > 0) & (frame['inside_valid_interval'] == 0)).sum()
            )
    result['rows_match'] = result['input_rows'] == result['output_rows']
    result['pass'] = (
        result['rows_match']
        and result['eligible_nonfinite'] == 0
        and result['invalid_discrete'] == 0
        and result['outside_clipped'] == 0
    )
    return result


def main():
    result = {'checks': {}}
    for name in INPUTS:
        result['checks'][name] = check_output(name, INPUTS[name], OUTPUTS[name])
    result['all_pass'] = all(item['pass'] for item in result['checks'].values())
    output_path = os.path.join(OUTPUT_DIR, '步骤4_独立核验结果.json')
    with open(output_path, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False))
    if not result['all_pass']:
        raise ValueError('第四步独立核验未通过')


if __name__ == '__main__':
    main()
