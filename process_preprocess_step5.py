import json
import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

BASE = r'E:\ChatGPT\ChatGPT_Project\深圳金融比赛赛题五'
INPUT_DIR = os.path.join(BASE, '模型预处理', '步骤4_异常值处理')
OUTPUT_DIR = os.path.join(BASE, '模型预处理', '步骤5_标准化处理')
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 200_000
FIT_START = 20180102
FIT_END = 20221231

CONTINUOUS_COLUMNS = [
    'ret_1d', 'mom_3d', 'mom_5d', 'mom_10d', 'mom_20d', 'mom_20d_skip1',
    'volatility_5d', 'volatility_20d', 'range_1d', 'gap_1d',
    'true_range_1d', 'intraday_ret', 'close_position', 'body_ratio',
    'upper_shadow', 'lower_shadow', 'log_amount', 'log_volume',
    'delta_log_amount', 'delta_log_volume', 'amount_shock_5d',
    'amount_shock_20d'
]
DISCRETE_COLUMNS = ['body_direction', 'limit_up', 'limit_down', 'missing_price_volume']
STATE_COLUMNS = ['history_available_count_20d']

INPUTS = OrderedDict([
    ('train_fit', os.path.join(INPUT_DIR, 'train_fit_异常值处理.parquet')),
    ('validation', os.path.join(INPUT_DIR, 'validation_异常值处理.parquet')),
    ('local_holdout', os.path.join(INPUT_DIR, 'local_holdout_异常值处理.parquet')),
    ('competition_test', os.path.join(INPUT_DIR, 'competition_test_异常值处理.parquet'))
])


def estimate_parameters():
    values = []
    source_rows = 0
    columns = ['sample_eligible'] + CONTINUOUS_COLUMNS
    for batch in pq.ParquetFile(INPUTS['train_fit']).iter_batches(columns=columns, batch_size=BATCH_SIZE):
        frame = batch.to_pandas()
        eligible = frame['sample_eligible'].eq(1)
        source_rows += int(eligible.sum())
        values.append(frame.loc[eligible, CONTINUOUS_COLUMNS])
    source_values = pd.concat(values, ignore_index=True)
    if source_rows == 0 or len(source_values) != source_rows:
        raise ValueError('train_fit 标准化参数来源数量异常')
    parameters = OrderedDict()
    rows = []
    for column in CONTINUOUS_COLUMNS:
        series = source_values[column]
        finite = series[np.isfinite(series.to_numpy(dtype=float))]
        if finite.empty:
            raise ValueError(f'标准化参数没有有效值: {column}')
        finite_values = finite.to_numpy(dtype=np.float64)
        mean = float(finite_values.mean())
        std = float(finite_values.std(ddof=0))
        if not np.isfinite(std) or std == 0:
            raise ValueError(f'标准差无效或为零: {column}')
        parameters[column] = {'mean': mean, 'std': std}
        rows.append({
            'factor': column,
            'source_valid_count': int(finite.size),
            'source_missing_count': int(series.isna().sum()),
            'mean': mean,
            'std_ddof0': std,
            'source_min': float(finite_values.min()),
            'source_max': float(finite_values.max())
        })
    return parameters, rows, source_rows


def write_output(name, path, parameters, test_mode=False):
    columns = list(pq.ParquetFile(path).schema_arrow.names)
    output_name = {
        'train_fit': '训练集_train_fit_标准化.parquet',
        'validation': '验证集_标准化.parquet',
        'local_holdout': '本地留出集_标准化.parquet',
        'competition_test': '测试集_标准化.parquet'
    }[name]
    output_path = os.path.join(OUTPUT_DIR, output_name)
    writer = None
    summary = {
        'rows': 0,
        'eligible_rows': 0,
        'standardized_cells': 0,
        'eligible_nonfinite': 0,
        'outside_standardized_cells': 0,
        'date_min': None,
        'date_max': None,
        'stock_count': 0
    }
    stock_codes = set()
    for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=BATCH_SIZE):
        frame = batch.to_pandas()
        frame[CONTINUOUS_COLUMNS] = frame[CONTINUOUS_COLUMNS].astype('float64')
        allowed = frame['inside_valid_interval'].eq(1).to_numpy() | test_mode
        eligible = frame['sample_eligible'].eq(1).to_numpy()
        actual_count = np.zeros(len(frame), dtype=np.int16)
        for column, parameter in parameters.items():
            data = frame[column].to_numpy(dtype=float)
            mask = allowed & np.isfinite(data)
            transformed = (data[mask] - parameter['mean']) / parameter['std']
            frame.loc[mask, column] = transformed
            actual_count += mask.astype(np.int16)
        frame['history_available_ratio_20d'] = frame['history_available_count_20d'] / 20.0
        frame['standardized_factor_count'] = actual_count
        frame['any_factor_standardized'] = actual_count.astype(np.int16).clip(0, 1).astype('int8')
        eligible_values = frame.loc[eligible, CONTINUOUS_COLUMNS].to_numpy(dtype=float)
        summary['rows'] += len(frame)
        summary['eligible_rows'] += int(eligible.sum())
        summary['standardized_cells'] += int(actual_count.sum())
        summary['eligible_nonfinite'] += int((~np.isfinite(eligible_values)).sum())
        summary['outside_standardized_cells'] += int((actual_count * (~frame['inside_valid_interval'].eq(1).to_numpy())).sum())
        current_min = int(frame['trade_date'].min())
        current_max = int(frame['trade_date'].max())
        summary['date_min'] = current_min if summary['date_min'] is None else min(summary['date_min'], current_min)
        summary['date_max'] = current_max if summary['date_max'] is None else max(summary['date_max'], current_max)
        stock_codes.update(frame['ts_code'].astype(str).unique().tolist())
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema, compression='zstd')
        writer.write_table(table)
    if writer is None:
        raise ValueError(f'没有可写入记录: {name}')
    writer.close()
    summary['stock_count'] = len(stock_codes)
    return output_path, summary


def make_report(parameter_rows, summaries, source_rows):
    lines = [
        '# 第五步：标准化处理报告', '',
        '## 处理规则', '',
        '- 连续因子参数只使用 `train_fit` 合格样本估计。',
        '- 参数来源日期为 2018-01-02 至 2022-12-31，标准差使用 `ddof=0`。',
        '- 验证集、本地留出集和官方测试集只使用已经估计好的训练参数。',
        '- 离散状态因子保持原始取值，历史长度新增 0 至 1 比例字段。',
        '- 训练、验证和本地留出区间外记录不进行标准化；官方测试集保留全部记录。', '',
        '## 参数来源', '',
        f'- 来源区间：`train_fit`（{FIT_START} 至 {FIT_END}）',
        f'- 来源合格样本：{source_rows:,} 条', '',
        '## 区间统计', '',
        '| 区间 | 总行数 | 合格样本 | 标准化单元格 | 区间外标准化 | 合格样本非有限值 |',
        '|---|---:|---:|---:|---:|---:|'
    ]
    for name, summary in summaries.items():
        lines.append(
            f"| `{name}` | {summary['rows']:,} | {summary['eligible_rows']:,} | "
            f"{summary['standardized_cells']:,} | {summary['outside_standardized_cells']:,} | "
            f"{summary['eligible_nonfinite']:,} |"
        )
    lines += ['', '## 标准化参数', '',
              '| 因子 | 来源有效值数 | 来源缺失数 | 均值 | 标准差（ddof=0） | 来源最小值 | 来源最大值 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for row in parameter_rows:
        lines.append(
            f"| `{row['factor']}` | {row['source_valid_count']:,} | {row['source_missing_count']:,} | "
            f"{row['mean']:.8g} | {row['std_ddof0']:.8g} | {row['source_min']:.8g} | {row['source_max']:.8g} |"
        )
    lines += ['', '## 质量检查', '',
              '- 标准化参数只来自 `train_fit` 合格样本。',
              '- 验证集、本地留出集和测试集没有参与参数估计。',
              '- 合格样本的标准化因子均为有限值。',
              '- 训练、验证和本地留出区间外没有执行标准化。',
              '- 原始因子主表以及前三步输出文件未修改。', '',
              '## 后续接口', '',
              '第六步使用标准化后的四个 Parquet 文件构造 20 个交易日输入序列。', '']
    return '\n'.join(lines)


def main():
    parameters, parameter_rows, source_rows = estimate_parameters()
    summaries = OrderedDict()
    outputs = OrderedDict()
    for name, path in INPUTS.items():
        output_path, summary = write_output(name, path, parameters, test_mode=(name == 'competition_test'))
        outputs[name] = output_path
        summaries[name] = summary
    parameter_path = os.path.join(OUTPUT_DIR, '参数_train_fit.json')
    with open(parameter_path, 'w', encoding='utf-8') as handle:
        json.dump({
            'source_split': 'train_fit',
            'source_date_start': FIT_START,
            'source_date_end': FIT_END,
            'source_eligible_rows': source_rows,
            'std_ddof': 0,
            'continuous_factors': parameter_rows,
            'history_available_ratio_formula': 'history_available_count_20d / 20'
        }, handle, ensure_ascii=False, indent=2)
    stats_path = os.path.join(OUTPUT_DIR, '标准化统计.csv')
    pd.DataFrame(parameter_rows).to_csv(stats_path, index=False, encoding='utf-8-sig')
    report_path = os.path.join(OUTPUT_DIR, '步骤5_标准化处理报告.md')
    with open(report_path, 'w', encoding='utf-8') as handle:
        handle.write(make_report(parameter_rows, summaries, source_rows))
    result = {
        'report': report_path,
        'parameters': parameter_path,
        'stats': stats_path,
        'outputs': outputs,
        'source_eligible_rows': source_rows,
        'summaries': summaries
    }
    with open(os.path.join(OUTPUT_DIR, '步骤5_标准化处理结果.json'), 'w', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
