import json
import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

BASE = r'E:\ChatGPT\ChatGPT_Project\深圳金融比赛赛题五'
DATA_DIR = os.path.join(BASE, '因子数据')
REPORT_DIR = os.path.join(BASE, '模型预处理')
os.makedirs(REPORT_DIR, exist_ok=True)

FILES = OrderedDict([
    ('train', os.path.join(DATA_DIR, '训练集_第一次LST-Transformer因子.parquet')),
    ('test', os.path.join(DATA_DIR, '测试集_X_第一次LST-Transformer因子.parquet')),
])
FACTOR_COLS = [
    'ret_1d', 'mom_3d', 'mom_5d', 'mom_10d', 'mom_20d', 'mom_20d_skip1',
    'volatility_5d', 'volatility_20d', 'range_1d', 'gap_1d', 'true_range_1d',
    'intraday_ret', 'close_position', 'body_direction', 'body_ratio',
    'upper_shadow', 'lower_shadow', 'log_amount', 'log_volume',
    'delta_log_amount', 'delta_log_volume', 'amount_shock_5d',
    'amount_shock_20d', 'limit_up', 'limit_down', 'missing_price_volume',
    'history_available_count_20d'
]
EXPECTED_TRAIN = ['ts_code', 'trade_date'] + FACTOR_COLS + ['y_ret_1d']
EXPECTED_TEST = ['ts_code', 'trade_date'] + FACTOR_COLS


def inspect_file(path, expected_columns, is_train):
    pf = pq.ParquetFile(path)
    columns = pf.schema_arrow.names
    result = OrderedDict([
        ('path', path), ('columns', columns), ('columns_match', columns == expected_columns),
        ('rows', 0), ('stocks', set()), ('dates', set()), ('duplicate_keys', 0),
        ('order_violations', 0), ('factor_missing', OrderedDict((c, 0) for c in FACTOR_COLS)),
        ('factor_inf', OrderedDict((c, 0) for c in FACTOR_COLS)),
        ('invalid_binary', OrderedDict()), ('invalid_body_direction', 0),
        ('invalid_history_count', 0), ('label_missing', 0), ('label_inf', 0),
        ('date_min', None), ('date_max', None), ('last_key', None)
    ])
    for batch in pf.iter_batches(batch_size=100_000):
        frame = batch.to_pandas()
        result['rows'] += len(frame)
        result['stocks'].update(frame['ts_code'].astype(str).unique().tolist())
        result['dates'].update(frame['trade_date'].astype(int).unique().tolist())
        keys = list(zip(frame['ts_code'].astype(str), frame['trade_date'].astype(int)))
        if result['last_key'] is not None and keys and keys[0] <= tuple(result['last_key']):
            result['order_violations'] += 1
        for prev, curr in zip(keys, keys[1:]):
            if curr <= prev:
                result['duplicate_keys'] += int(curr == prev)
                result['order_violations'] += int(curr < prev)
        if keys:
            result['last_key'] = list(keys[-1])
        dmin = int(frame['trade_date'].min())
        dmax = int(frame['trade_date'].max())
        result['date_min'] = dmin if result['date_min'] is None else min(result['date_min'], dmin)
        result['date_max'] = dmax if result['date_max'] is None else max(result['date_max'], dmax)
        for col in FACTOR_COLS:
            values = frame[col].to_numpy(dtype='float64', na_value=np.nan)
            result['factor_missing'][col] += int(np.isnan(values).sum())
            result['factor_inf'][col] += int(np.isinf(values).sum())
        for col in ['limit_up', 'limit_down', 'missing_price_volume']:
            invalid = ~frame[col].isin([0.0, 1.0]) & frame[col].notna()
            result['invalid_binary'][col] = result['invalid_binary'].get(col, 0) + int(invalid.sum())
        bd = frame['body_direction'].dropna()
        result['invalid_body_direction'] += int((~bd.isin([-1.0, 0.0, 1.0])).sum())
        hc = frame['history_available_count_20d'].dropna()
        result['invalid_history_count'] += int(((hc < 0) | (hc > 20)).sum())
        if is_train:
            result['label_missing'] += int(frame['y_ret_1d'].isna().sum())
            result['label_inf'] += int(np.isinf(frame['y_ret_1d'].to_numpy(dtype='float64', na_value=np.nan)).sum())
    result['stocks'] = len(result['stocks'])
    result['dates'] = len(result['dates'])
    result['factor_missing_rate'] = {k: v / result['rows'] for k, v in result['factor_missing'].items()}
    result.pop('last_key')
    return result


def make_report(results):
    lines = [
        '# 第一步：数据质量检查报告',
        '',
        '## 检查范围',
        '',
        '本报告只检查统一未填补因子主表，不执行缺失值填充、异常值截断、标准化或序列构造。',
        '',
        '## 规模与字段',
        '',
        '| 数据集 | 行数 | 股票数 | 日期数 | 日期范围 | 字段结构 | 重复键 | 顺序异常 |',
        '|---|---:|---:|---:|---|---|---:|---:|',
    ]
    for name, r in results.items():
        lines.append(f"| {name} | {r['rows']:,} | {r['stocks']:,} | {r['dates']:,} | {r['date_min']} 至 {r['date_max']} | {'通过' if r['columns_match'] else '异常'} | {r['duplicate_keys']} | {r['order_violations']} |")
    lines += ['', '## 因子缺失率', '', '| 因子 | 训练集 | 测试集 |', '|---|---:|---:|']
    for col in FACTOR_COLS:
        lines.append(f"| `{col}` | {results['train']['factor_missing_rate'][col]:.4%} | {results['test']['factor_missing_rate'][col]:.4%} |")
    lines += ['', '## 取值检查', '', '| 检查项 | 训练集 | 测试集 |', '|---|---:|---:|']
    for col in ['limit_up', 'limit_down', 'missing_price_volume']:
        lines.append(f"| `{col}` 非 0/1 数量 | {results['train']['invalid_binary'][col]} | {results['test']['invalid_binary'][col]} |")
    lines.append(f"| `body_direction` 非 -1/0/1 数量 | {results['train']['invalid_body_direction']} | {results['test']['invalid_body_direction']} |")
    lines.append(f"| `history_available_count_20d` 超出 0 至 20 数量 | {results['train']['invalid_history_count']} | {results['test']['invalid_history_count']} |")
    lines.append(f"| 因子正负无穷数量 | {sum(results['train']['factor_inf'].values())} | {sum(results['test']['factor_inf'].values())} |")
    lines += ['', '## 标签检查', '', f"- 训练集 `y_ret_1d` 缺失：{results['train']['label_missing']:,} 条。", f"- 训练集 `y_ret_1d` 正负无穷：{results['train']['label_inf']:,} 条。", '- 测试集不包含 `y_ret_1d`。', '', '## 结论', '', '- 统一因子主表的结构、规模、键唯一性和时间顺序已完成检查。', '- 因子空值按生成规则保留，后续模型预处理时再处理。', '- 当前报告不改变任何数据文件。']
    return '\n'.join(lines) + '\n'


def main():
    results = OrderedDict()
    results['train'] = inspect_file(FILES['train'], EXPECTED_TRAIN, True)
    results['test'] = inspect_file(FILES['test'], EXPECTED_TEST, False)
    json_path = os.path.join(REPORT_DIR, '步骤1_数据质量检查结果.json')
    md_path = os.path.join(REPORT_DIR, '步骤1_数据质量检查报告.md')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(make_report(results))
    print(json.dumps({'report': md_path, 'json': json_path, 'train_rows': results['train']['rows'], 'test_rows': results['test']['rows']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
