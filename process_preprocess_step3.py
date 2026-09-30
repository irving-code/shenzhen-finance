import json
import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

BASE = r'E:\ChatGPT\ChatGPT_Project\深圳金融比赛赛题五'
DATA_DIR = os.path.join(BASE, '因子数据')
REPORT_DIR = os.path.join(BASE, '模型预处理', '步骤3_缺失值处理')
os.makedirs(REPORT_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(DATA_DIR, '训练集_第一次LST-Transformer因子.parquet')
TEST_PATH = os.path.join(DATA_DIR, '测试集_X_第一次LST-Transformer因子.parquet')
HISTORY_THRESHOLD = 16
BATCH_SIZE = 200_000
FIT_START = 20180102
FIT_END = 20221231
VALIDATION_START = 20230101
VALIDATION_END = 20231231
HOLDOUT_START = 20240101
HOLDOUT_END = 20241231
TEST_START = 20250101
TEST_END = 20260608
TRAIN_SPLITS = {
    'train_fit': (FIT_START, FIT_END),
    'validation': (VALIDATION_START, VALIDATION_END),
    'local_holdout': (HOLDOUT_START, HOLDOUT_END)
}

FACTOR_COLUMNS = [
    'ret_1d', 'mom_3d', 'mom_5d', 'mom_10d', 'mom_20d', 'mom_20d_skip1',
    'volatility_5d', 'volatility_20d', 'range_1d', 'gap_1d',
    'true_range_1d', 'intraday_ret', 'close_position', 'body_direction',
    'body_ratio', 'upper_shadow', 'lower_shadow', 'log_amount', 'log_volume',
    'delta_log_amount', 'delta_log_volume', 'amount_shock_5d',
    'amount_shock_20d', 'limit_up', 'limit_down', 'missing_price_volume',
    'history_available_count_20d'
]
STATUS_COLUMNS = {'missing_price_volume', 'history_available_count_20d'}
FILL_COLUMNS = [c for c in FACTOR_COLUMNS if c not in STATUS_COLUMNS]
META_COLUMNS = ['ts_code', 'trade_date', 'missing_price_volume', 'history_available_count_20d']


def split_for_date(value):
    value = int(value)
    if FIT_START <= value <= FIT_END:
        return 'train_fit'
    if VALIDATION_START <= value <= VALIDATION_END:
        return 'validation'
    if HOLDOUT_START <= value <= HOLDOUT_END:
        return 'local_holdout'
    if TEST_START <= value <= TEST_END:
        return 'competition_test'
    raise ValueError(f'日期不在预定区间内: {value}')


def collect_boundaries(paths, maximum_date=None):
    boundaries = {}
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(
            columns=['ts_code', 'trade_date', 'missing_price_volume'],
            batch_size=BATCH_SIZE
        ):
            frame = batch.to_pandas()
            valid = frame[frame['missing_price_volume'] == 0]
            if maximum_date is not None:
                valid = valid[valid['trade_date'] <= maximum_date]
            for code, group in valid.groupby('ts_code', sort=False):
                code = str(code)
                first = int(group['trade_date'].min())
                last = int(group['trade_date'].max())
                if code not in boundaries:
                    boundaries[code] = [first, last]
                else:
                    boundaries[code][0] = min(boundaries[code][0], first)
                    boundaries[code][1] = max(boundaries[code][1], last)
    return boundaries


def build_metadata(path, boundaries, has_label, test_mode=False):
    columns = list(META_COLUMNS)
    if has_label:
        columns.append('y_ret_1d')
    batches = []
    for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=BATCH_SIZE):
        batches.append(batch.to_pandas())
    meta = pd.concat(batches, ignore_index=True)
    if has_label:
        meta['split'] = meta['trade_date'].map(split_for_date)
        if not meta['split'].isin(TRAIN_SPLITS).all():
            raise ValueError('训练主表包含官方测试区间的记录')
        meta['first_valid_date'] = np.nan
        meta['last_valid_date'] = np.nan
        for split, split_boundaries in boundaries.items():
            selected = meta['split'].eq(split)
            first_map = {code: values[0] for code, values in split_boundaries.items()}
            last_map = {code: values[1] for code, values in split_boundaries.items()}
            meta.loc[selected, 'first_valid_date'] = meta.loc[selected, 'ts_code'].astype(str).map(first_map).to_numpy()
            meta.loc[selected, 'last_valid_date'] = meta.loc[selected, 'ts_code'].astype(str).map(last_map).to_numpy()
    else:
        if not meta['trade_date'].map(split_for_date).eq('competition_test').all():
            raise ValueError('官方测试主表包含训练或验证区间的记录')
        first_map = {code: values[0] for code, values in boundaries.items()}
        last_map = {code: values[1] for code, values in boundaries.items()}
        meta['first_valid_date'] = meta['ts_code'].astype(str).map(first_map)
        meta['last_valid_date'] = meta['ts_code'].astype(str).map(last_map)
    meta['current_market_valid'] = meta['missing_price_volume'].eq(0)
    meta['inside_valid_interval'] = meta['trade_date'].between(
        meta['first_valid_date'], meta['last_valid_date']
    )
    meta['pre_valid_interval'] = meta['trade_date'] < meta['first_valid_date']
    meta['post_valid_interval'] = meta['trade_date'] > meta['last_valid_date']
    if test_mode:
        meta['window_inside_valid'] = True
    else:
        window_start = meta.groupby('ts_code', sort=False)['trade_date'].shift(19)
        meta['window_inside_valid'] = (
            meta['inside_valid_interval']
            & window_start.ge(meta['first_valid_date'])
        )
    meta['history_16_valid'] = meta['history_available_count_20d'].ge(HISTORY_THRESHOLD)
    if has_label:
        meta['label_valid'] = meta['y_ret_1d'].notna()
        meta['sample_eligible'] = (
            meta['inside_valid_interval']
            & meta['label_valid']
            & meta['history_16_valid']
            & meta['window_inside_valid']
        )
    else:
        meta['label_valid'] = False
        meta['sample_eligible'] = meta['inside_valid_interval'] & meta['history_16_valid']
    return meta


def estimate_fill_values():
    train_boundaries = {
        split: collect_boundaries([TRAIN_PATH], maximum_date=end)
        for split, (_, end) in TRAIN_SPLITS.items()
    }
    train_meta = build_metadata(TRAIN_PATH, train_boundaries, has_label=True)
    fit_source = train_meta['split'].eq('train_fit') & train_meta['sample_eligible']
    if not train_meta.loc[fit_source, 'trade_date'].between(FIT_START, FIT_END).all():
        raise ValueError('填充参数来源包含拟合区间外的日期')
    source_rows = int(fit_source.sum())
    if source_rows == 0:
        raise ValueError('拟合区间内没有合格样本')
    eligible = fit_source.to_numpy()
    values = []
    offset = 0
    for batch in pq.ParquetFile(TRAIN_PATH).iter_batches(
        columns=FILL_COLUMNS, batch_size=BATCH_SIZE
    ):
        frame = batch.to_pandas()
        end = offset + len(frame)
        mask = eligible[offset:end]
        values.append(frame.loc[mask])
        offset = end
    eligible_values = pd.concat(values, ignore_index=True)
    if len(eligible_values) != source_rows:
        raise ValueError('填充参数来源样本数量与筛选结果不一致')
    fill_values = OrderedDict()
    parameter_rows = []
    for column in FILL_COLUMNS:
        series = eligible_values[column]
        valid = series.dropna()
        if valid.empty:
            raise ValueError(f'合格 train_fit 样本中因子没有有效值: {column}')
        fill_value = float(valid.median())
        fill_values[column] = fill_value
        parameter_rows.append({
            'factor': column,
            'train_valid_count': int(valid.size),
            'train_missing_count': int(series.isna().sum()),
            'train_missing_rate': float(series.isna().mean()),
            'fill_value': fill_value
        })
        if parameter_rows[-1]['train_valid_count'] + parameter_rows[-1]['train_missing_count'] != source_rows:
            raise ValueError(f'填充参数样本数量不一致: {column}')
    status_missing = train_meta.loc[eligible, list(STATUS_COLUMNS)].isna().sum()
    if int(status_missing.sum()) != 0:
        raise ValueError(f'状态因子存在空值: {status_missing.to_dict()}')
    source_audit = {
        'split': 'train_fit',
        'date_start': FIT_START,
        'date_end': FIT_END,
        'eligible_rows': source_rows,
        'total_rows': int(train_meta['split'].eq('train_fit').sum()),
        'stock_count': int(train_meta.loc[fit_source, 'ts_code'].nunique())
    }
    return train_meta, fill_values, parameter_rows, source_audit


def write_split(path, meta, fill_values, output_name, input_has_label, split_filter=None, test_mode=False):
    columns = list(pq.ParquetFile(path).schema_arrow.names)
    writer = None
    stats = {column: {'before_missing': 0, 'after_missing': 0, 'filled_count': 0, 'outside_missing': 0} for column in FACTOR_COLUMNS}
    summary = {
        'rows': 0,
        'eligible': 0,
        'inside': 0,
        'pre': 0,
        'post': 0,
        'history_16': 0,
        'window_inside': 0,
        'filled_cells': 0,
        'outside_filled_cells': 0,
        'eligible_remaining_missing': 0,
        'date_min': None,
        'date_max': None,
        'stock_count': 0
    }
    stock_codes = set()
    offset = 0
    for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=BATCH_SIZE):
        frame = batch.to_pandas()
        batch_meta = meta.iloc[offset:offset + len(frame)].copy()
        offset += len(frame)
        if split_filter is not None:
            keep = frame['trade_date'].map(split_for_date).eq(split_filter).to_numpy()
        else:
            keep = np.ones(len(frame), dtype=bool)
        if not keep.any():
            continue
        frame = frame.loc[keep].reset_index(drop=True)
        batch_meta = batch_meta.loc[keep].reset_index(drop=True)
        fill_allowed = batch_meta['inside_valid_interval'].to_numpy() | test_mode
        original_missing = frame[FILL_COLUMNS].isna()
        actual_filled_count = np.zeros(len(frame), dtype=np.int16)
        for column, value in fill_values.items():
            mask = fill_allowed & original_missing[column].to_numpy()
            frame.loc[mask, column] = value
            actual_filled_count += mask.astype(np.int16)
            stats[column]['before_missing'] += int(original_missing[column].sum())
            stats[column]['after_missing'] += int(frame[column].isna().sum())
            stats[column]['filled_count'] += int(mask.sum())
            stats[column]['outside_missing'] += int((~fill_allowed & original_missing[column].to_numpy()).sum())
        frame['inside_valid_interval'] = batch_meta['inside_valid_interval'].astype('int8').to_numpy()
        frame['pre_valid_interval'] = batch_meta['pre_valid_interval'].astype('int8').to_numpy()
        frame['post_valid_interval'] = batch_meta['post_valid_interval'].astype('int8').to_numpy()
        frame['current_market_valid'] = batch_meta['current_market_valid'].astype('int8').to_numpy()
        frame['sample_eligible'] = batch_meta['sample_eligible'].astype('int8').to_numpy()
        frame['filled_factor_count'] = actual_filled_count
        frame['any_factor_filled'] = frame['filled_factor_count'].gt(0).astype('int8')
        summary['rows'] += len(frame)
        summary['eligible'] += int(batch_meta['sample_eligible'].sum())
        summary['inside'] += int(batch_meta['inside_valid_interval'].sum())
        summary['pre'] += int(batch_meta['pre_valid_interval'].sum())
        summary['post'] += int(batch_meta['post_valid_interval'].sum())
        summary['history_16'] += int(batch_meta['history_16_valid'].sum())
        summary['window_inside'] += int(batch_meta['window_inside_valid'].sum())
        summary['filled_cells'] += int(frame['filled_factor_count'].sum())
        summary['outside_filled_cells'] += int((frame['filled_factor_count'].to_numpy() * (~batch_meta['inside_valid_interval'].to_numpy())).sum())
        remaining_missing = frame[FILL_COLUMNS].isna().to_numpy().sum(axis=1)
        summary['eligible_remaining_missing'] += int(remaining_missing[batch_meta['sample_eligible'].to_numpy()].sum())
        summary['date_min'] = int(frame['trade_date'].min()) if summary['date_min'] is None else min(summary['date_min'], int(frame['trade_date'].min()))
        summary['date_max'] = int(frame['trade_date'].max()) if summary['date_max'] is None else max(summary['date_max'], int(frame['trade_date'].max()))
        stock_codes.update(frame['ts_code'].astype(str).unique().tolist())
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(output_name, table.schema, compression='zstd')
        writer.write_table(table)
    if writer is None:
        raise ValueError(f'没有可写入记录: {output_name}')
    writer.close()
    summary['stock_count'] = len(stock_codes)
    return summary, stats


def make_report(summaries, parameter_rows, stats_rows, checks):
    lines = [
        '# 第三步：缺失值处理报告', '',
        '## 处理规则', '',
        '- 有效行情区间、首个有效日和最后有效日按 `ts_code` 独立确定。',
        '- `train_fit`、`validation` 和 `local_holdout` 只填充有效行情区间内部的因子空值。',
        '- 填充参数只使用 `train_fit` 合格样本的因子中位数。',
        '- 合格训练目标日要求标签有效、历史 20 日至少 16 日完整，且 20 日窗口位于股票有效行情区间内。',
        '- 官方测试集使用训练参数处理模型输入，同时保留全部记录和区间状态。', '',
        '## 区间统计', '',
        '| 输出区间 | 行数 | 股票数 | 有效区间内 | 区间前 | 区间后 | 16日历史有效 | 20日窗口有效 | 资格样本 | 填充单元格 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|'
    ]
    for name, summary in summaries.items():
        lines.append(
            f"| `{name}` | {summary['rows']:,} | {summary['stock_count']:,} | {summary['inside']:,} | "
            f"{summary['pre']:,} | {summary['post']:,} | {summary['history_16']:,} | "
            f"{summary['window_inside']:,} | {summary['eligible']:,} | {summary['filled_cells']:,} |"
        )
    lines += ['', '## 填充参数', '',
              '填充参数全部从 `train_fit` 合格样本估计，普通因子使用中位数。', '',
              '| 因子 | 有效值数量 | 训练缺失数量 | 训练缺失率 | 填充值 |',
              '|---|---:|---:|---:|---:|']
    for row in parameter_rows:
        lines.append(
            f"| `{row['factor']}` | {row['train_valid_count']:,} | {row['train_missing_count']:,} | "
            f"{row['train_missing_rate']:.4%} | {row['fill_value']:.8g} |"
        )
    lines += ['', '## 缺失率变化', '',
              '| 区间 | 因子 | 填充前缺失 | 填充后缺失 | 实际填充 | 区间外保留缺失 |',
              '|---|---|---:|---:|---:|---:|']
    for row in stats_rows:
        lines.append(
            f"| `{row['split']}` | `{row['factor']}` | {row['before_missing']:,} | "
            f"{row['after_missing']:,} | {row['filled_count']:,} | {row['outside_missing']:,} |"
        )
    lines += ['', '## 质量检查', '']
    for key, value in checks.items():
        lines.append(f'- {key}：{value}')
    lines += ['', '## 后续接口', '',
              '训练、验证和本地留出的序列构造使用 `sample_eligible=1` 的目标日；官方测试集保留全部记录并按相同的股票级历史规则构造输入。',
              '统一未填补因子主表保持原样。', '']
    return '\n'.join(lines)


def main():
    train_meta, fill_values, parameter_rows, source_audit = estimate_fill_values()
    combined_boundaries = collect_boundaries([TRAIN_PATH, TEST_PATH])
    test_meta = build_metadata(TEST_PATH, combined_boundaries, has_label=False, test_mode=True)
    fill_path = os.path.join(REPORT_DIR, '参数_train_fit.json')
    with open(fill_path, 'w', encoding='utf-8') as handle:
        json.dump({
            'source': 'train_fit sample_eligible rows only',
            'source_date_start': FIT_START,
            'source_date_end': FIT_END,
            'source_total_rows': source_audit['total_rows'],
            'source_eligible_rows': source_audit['eligible_rows'],
            'source_stock_count': source_audit['stock_count'],
            'history_threshold': HISTORY_THRESHOLD,
            'fill_method': 'median',
            'factors': parameter_rows
        }, handle, ensure_ascii=False, indent=2)

    jobs = [
        ('train_fit', TRAIN_PATH, train_meta, True, '训练集_train_fit_填充后.parquet', False),
        ('validation', TRAIN_PATH, train_meta, True, '验证集_填充后.parquet', False),
        ('local_holdout', TRAIN_PATH, train_meta, True, '本地留出集_填充后.parquet', False),
        ('competition_test', TEST_PATH, test_meta, False, '测试集_填充后.parquet', True)
    ]
    summaries = OrderedDict()
    all_stats = []
    for name, path, meta, has_label, output_file, test_mode in jobs:
        output_path = os.path.join(REPORT_DIR, output_file)
        summary, stats = write_split(
            path, meta, fill_values, output_path, has_label,
            split_filter=None if test_mode else name, test_mode=test_mode
        )
        summaries[name] = summary
        for factor, row in stats.items():
            all_stats.append({'split': name, 'factor': factor, **row})

    stats_path = os.path.join(REPORT_DIR, '填充前后缺失率.csv')
    pd.DataFrame(all_stats).to_csv(stats_path, index=False, encoding='utf-8-sig')
    checks = {
        '填充参数来源': '仅 train_fit 合格样本',
        '填充参数来源日期': f'{FIT_START} 至 {FIT_END}',
        '填充参数来源合格样本数': str(source_audit['eligible_rows']),
        '验证集参数参与估计': '否',
        '本地留出集参数参与估计': '否',
        '官方测试集参数参与估计': '否',
        '股票边界': '按 ts_code 独立计算',
        '训练区间外填充数量': str(sum(s['outside_filled_cells'] for n, s in summaries.items() if n != 'competition_test')),
        '统一因子主表': '未修改',
        '测试集记录': f"保留 {summaries['competition_test']['rows']:,} 条",
        '普通因子合格训练记录空值': str(summaries['train_fit']['eligible_remaining_missing'])
    }
    report = make_report(summaries, parameter_rows, all_stats, checks)
    report_path = os.path.join(REPORT_DIR, '步骤3_缺失值处理报告.md')
    with open(report_path, 'w', encoding='utf-8') as handle:
        handle.write(report)
    result = {
        'report': report_path,
        'parameters': fill_path,
        'missing_stats': stats_path,
        'summaries': summaries,
        'checks': checks
    }
    with open(os.path.join(REPORT_DIR, '步骤3_缺失值处理结果.json'), 'w', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
