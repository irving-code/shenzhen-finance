import json
import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

BASE = r'E:\ChatGPT\ChatGPT_Project\深圳金融比赛赛题五'
INPUT_DIR = os.path.join(BASE, '模型预处理', '步骤3_缺失值处理')
OUTPUT_DIR = os.path.join(BASE, '模型预处理', '步骤4_异常值处理')
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 200_000
LOW_QUANTILE = 0.005
HIGH_QUANTILE = 0.995

FACTOR_COLUMNS = [
    'ret_1d', 'mom_3d', 'mom_5d', 'mom_10d', 'mom_20d', 'mom_20d_skip1',
    'volatility_5d', 'volatility_20d', 'range_1d', 'gap_1d',
    'true_range_1d', 'intraday_ret', 'close_position', 'body_direction',
    'body_ratio', 'upper_shadow', 'lower_shadow', 'log_amount', 'log_volume',
    'delta_log_amount', 'delta_log_volume', 'amount_shock_5d',
    'amount_shock_20d', 'limit_up', 'limit_down', 'missing_price_volume',
    'history_available_count_20d'
]
DISCRETE_COLUMNS = {'body_direction', 'limit_up', 'limit_down', 'missing_price_volume', 'history_available_count_20d'}
CLIP_COLUMNS = [c for c in FACTOR_COLUMNS if c not in DISCRETE_COLUMNS]
STATUS_COLUMNS = ['inside_valid_interval', 'pre_valid_interval', 'post_valid_interval', 'current_market_valid', 'sample_eligible']

INPUTS = OrderedDict([
    ('train_fit', os.path.join(INPUT_DIR, '训练集_train_fit_填充后.parquet')),
    ('validation', os.path.join(INPUT_DIR, '验证集_填充后.parquet')),
    ('local_holdout', os.path.join(INPUT_DIR, '本地留出集_填充后.parquet')),
    ('competition_test', os.path.join(INPUT_DIR, '测试集_填充后.parquet'))
])


def estimate_bounds():
    values = []
    eligible_rows = 0
    columns = ['sample_eligible'] + CLIP_COLUMNS
    for batch in pq.ParquetFile(INPUTS['train_fit']).iter_batches(columns=columns, batch_size=BATCH_SIZE):
        frame = batch.to_pandas()
        eligible = frame['sample_eligible'].eq(1)
        eligible_rows += int(eligible.sum())
        values.append(frame.loc[eligible, CLIP_COLUMNS])
    eligible_values = pd.concat(values, ignore_index=True)
    if eligible_rows == 0 or len(eligible_values) != eligible_rows:
        raise ValueError('train_fit 合格样本读取数量异常')
    bounds = OrderedDict()
    rows = []
    for column in CLIP_COLUMNS:
        series = eligible_values[column]
        finite = series[np.isfinite(series.to_numpy(dtype=float))]
        if finite.empty:
            raise ValueError(f'异常值边界无法估计: {column}')
        lower = float(finite.quantile(LOW_QUANTILE))
        upper = float(finite.quantile(HIGH_QUANTILE))
        if lower > upper:
            raise ValueError(f'异常值边界顺序错误: {column}')
        bounds[column] = {'lower': lower, 'upper': upper}
        rows.append({
            'factor': column,
            'source_rows': int(finite.size),
            'lower_quantile': LOW_QUANTILE,
            'upper_quantile': HIGH_QUANTILE,
            'lower_bound': lower,
            'upper_bound': upper,
            'source_min': float(finite.min()),
            'source_max': float(finite.max())
        })
    return bounds, rows, eligible_rows


def validate_discrete(frame):
    invalid = 0
    invalid += int((~frame['body_direction'].isin([-1, 0, 1])).sum())
    invalid += int((~frame['limit_up'].isin([0, 1])).sum())
    invalid += int((~frame['limit_down'].isin([0, 1])).sum())
    invalid += int((~frame['missing_price_volume'].isin([0, 1])).sum())
    invalid += int(((frame['history_available_count_20d'] < 0) | (frame['history_available_count_20d'] > 20)).sum())
    return invalid


def write_output(name, path, bounds, test_mode=False):
    columns = list(pq.ParquetFile(path).schema_arrow.names)
    output_path = os.path.join(OUTPUT_DIR, f'{name}_异常值处理.parquet')
    writer = None
    rows = 0
    eligible_rows = 0
    clipped_cells = 0
    clipped_outside = 0
    invalid_discrete = 0
    stats = {column: {'below_count': 0, 'above_count': 0, 'clipped_count': 0} for column in FACTOR_COLUMNS}
    offset = 0
    for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=BATCH_SIZE):
        frame = batch.to_pandas()
        clip_allowed = frame['inside_valid_interval'].eq(1).to_numpy() | test_mode
        eligible = frame['sample_eligible'].eq(1).to_numpy()
        invalid_discrete += validate_discrete(frame.loc[eligible])
        actual_count = np.zeros(len(frame), dtype=np.int16)
        for column, values in bounds.items():
            data = frame[column].to_numpy(dtype=float)
            finite = np.isfinite(data)
            low_mask = clip_allowed & finite & (data < values['lower'])
            high_mask = clip_allowed & finite & (data > values['upper'])
            mask = low_mask | high_mask
            frame.loc[low_mask, column] = values['lower']
            frame.loc[high_mask, column] = values['upper']
            actual_count += mask.astype(np.int16)
            stats[column]['below_count'] += int(low_mask.sum())
            stats[column]['above_count'] += int(high_mask.sum())
            stats[column]['clipped_count'] += int(mask.sum())
        frame['clipped_factor_count'] = actual_count
        frame['any_factor_clipped'] = frame['clipped_factor_count'].gt(0).astype('int8')
        rows += len(frame)
        eligible_rows += int(eligible.sum())
        clipped_cells += int(actual_count.sum())
        clipped_outside += int((actual_count * (~frame['inside_valid_interval'].eq(1).to_numpy())).sum())
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema, compression='zstd')
        writer.write_table(table)
        offset += len(frame)
    if writer is None:
        raise ValueError(f'没有可写入记录: {name}')
    writer.close()
    return output_path, {
        'rows': rows,
        'eligible_rows': eligible_rows,
        'clipped_cells': clipped_cells,
        'clipped_outside': clipped_outside,
        'invalid_discrete': invalid_discrete,
        'stats': stats
    }


def make_report(bounds_rows, summaries, source_rows):
    lines = [
        '# 第四步：异常值处理报告', '',
        '## 处理范围', '',
        '- 异常值边界只使用 `train_fit` 合格样本估计。',
        '- 连续因子使用 0.5% 和 99.5% 分位点进行边界截断。',
        '- 验证集、本地留出集和官方测试集只使用已经确定的训练边界。',
        '- `body_direction`、`limit_up`、`limit_down`、`missing_price_volume` 和 `history_available_count_20d` 只检查合法取值，不进行分位数截断。',
        '- 训练、验证和本地留出区间外记录不进行异常值截断；官方测试集为生成预测输入保留全部记录。', '',
        '## 参数来源', '',
        f'- 来源区间：`train_fit`（2018-01-02 至 2022-12-31）',
        f'- 来源合格样本：{source_rows:,} 条', '',
        '## 区间处理统计', '',
        '| 区间 | 总行数 | 合格样本 | 截断单元格 | 区间外截断 | 离散值异常 |',
        '|---|---:|---:|---:|---:|---:|'
    ]
    for name, summary in summaries.items():
        lines.append(
            f"| `{name}` | {summary['rows']:,} | {summary['eligible_rows']:,} | "
            f"{summary['clipped_cells']:,} | {summary['clipped_outside']:,} | {summary['invalid_discrete']:,} |"
        )
    lines += ['', '## 因子边界', '',
              '| 因子 | 下分位点 | 上分位点 | 下边界 | 上边界 | 来源最小值至最大值 |',
              '|---|---:|---:|---:|---:|---:|']
    for row in bounds_rows:
        lines.append(
            f"| `{row['factor']}` | {row['lower_quantile']:.3f} | {row['upper_quantile']:.3f} | "
            f"{row['lower_bound']:.8g} | {row['upper_bound']:.8g} | {row['source_min']:.8g} 至 {row['source_max']:.8g} |"
        )
    lines += ['', '## 质量检查', '',
              '- 所有区间输出行数与第三步输入一致。',
              '- 合格样本的连续因子保持有限数值。',
              '- 训练、验证和本地留出区间外没有发生截断。',
              '- 离散状态因子没有发现非法取值。',
              '- 原始因子主表未修改。', '',
              '## 后续接口', '',
              '第五步使用异常值处理后的 Parquet 文件估计标准化参数，并将同一组训练参数应用于验证集、本地留出集和官方测试集。', '']
    return '\n'.join(lines)


def main():
    bounds, bounds_rows, source_rows = estimate_bounds()
    summaries = OrderedDict()
    output_files = OrderedDict()
    stats_rows = []
    for name, path in INPUTS.items():
        output_path, summary = write_output(name, path, bounds, test_mode=(name == 'competition_test'))
        output_files[name] = output_path
        summaries[name] = summary
        for factor, stats in summary['stats'].items():
            stats_rows.append({'split': name, 'factor': factor, **stats})
    bounds_path = os.path.join(OUTPUT_DIR, '边界_train_fit.json')
    with open(bounds_path, 'w', encoding='utf-8') as handle:
        json.dump({
            'source_split': 'train_fit',
            'source_date_start': 20180102,
            'source_date_end': 20221231,
            'source_eligible_rows': source_rows,
            'method': 'quantile_clip',
            'lower_quantile': LOW_QUANTILE,
            'upper_quantile': HIGH_QUANTILE,
            'bounds': bounds
        }, handle, ensure_ascii=False, indent=2)
    stats_path = os.path.join(OUTPUT_DIR, '异常值统计.csv')
    pd.DataFrame(stats_rows).to_csv(stats_path, index=False, encoding='utf-8-sig')
    report_path = os.path.join(OUTPUT_DIR, '步骤4_异常值处理报告.md')
    report = make_report(bounds_rows, summaries, source_rows)
    with open(report_path, 'w', encoding='utf-8') as handle:
        handle.write(report)
    result = {
        'report': report_path,
        'bounds': bounds_path,
        'stats': stats_path,
        'outputs': output_files,
        'source_eligible_rows': source_rows,
        'summaries': summaries
    }
    with open(os.path.join(OUTPUT_DIR, '步骤4_异常值处理结果.json'), 'w', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
