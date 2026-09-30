import json
import os
from collections import OrderedDict

import pandas as pd
import pyarrow.parquet as pq

BASE = r'E:\ChatGPT\ChatGPT_Project\深圳金融比赛赛题五'
DATA_DIR = os.path.join(BASE, '因子数据')
REPORT_DIR = os.path.join(BASE, '模型预处理')
os.makedirs(REPORT_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(DATA_DIR, '训练集_第一次LST-Transformer因子.parquet')
TEST_PATH = os.path.join(DATA_DIR, '测试集_X_第一次LST-Transformer因子.parquet')
HISTORY_THRESHOLD = 16


def split_date(date):
    year = int(date) // 10000
    if year <= 2022:
        return 'train_fit'
    if year == 2023:
        return 'validation'
    if year == 2024:
        return 'local_holdout'
    return 'competition_test'


def collect_boundaries(path):
    boundaries = {}
    columns = ['ts_code', 'trade_date', 'missing_price_volume']
    for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=200_000):
        frame = batch.to_pandas()
        valid = frame[frame['missing_price_volume'] == 0]
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


def inspect_train(boundaries):
    rows = []
    columns = ['ts_code', 'trade_date', 'history_available_count_20d', 'missing_price_volume', 'y_ret_1d']
    for batch in pq.ParquetFile(TRAIN_PATH).iter_batches(columns=columns, batch_size=200_000):
        frame = batch.to_pandas()
        frame['split'] = frame['trade_date'].map(split_date)
        frame['first_valid_date'] = frame['ts_code'].map(lambda x: boundaries.get(str(x), [None, None])[0])
        frame['last_valid_date'] = frame['ts_code'].map(lambda x: boundaries.get(str(x), [None, None])[1])
        frame['inside_valid_interval'] = frame['trade_date'].between(frame['first_valid_date'], frame['last_valid_date'])
        frame['pre_interval'] = frame['trade_date'] < frame['first_valid_date']
        frame['post_interval'] = frame['trade_date'] > frame['last_valid_date']
        frame['label_valid'] = frame['y_ret_1d'].notna()
        frame['history_16_valid'] = frame['history_available_count_20d'] >= HISTORY_THRESHOLD
        frame['current_market_valid'] = frame['missing_price_volume'] == 0
        frame['eligible'] = frame['inside_valid_interval'] & frame['label_valid'] & frame['history_16_valid']
        grouped = frame.groupby(['trade_date', 'split'], as_index=False).agg(
            rows=('trade_date', 'size'),
            pre_interval=('pre_interval', 'sum'),
            post_interval=('post_interval', 'sum'),
            inside_valid_interval=('inside_valid_interval', 'sum'),
            label_valid=('label_valid', 'sum'),
            history_16_valid=('history_16_valid', 'sum'),
            current_market_valid=('current_market_valid', 'sum'),
            eligible=('eligible', 'sum')
        )
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True).groupby(['trade_date', 'split'], as_index=False).sum()


def inspect_test(boundaries):
    rows = []
    columns = ['ts_code', 'trade_date', 'history_available_count_20d', 'missing_price_volume']
    for batch in pq.ParquetFile(TEST_PATH).iter_batches(columns=columns, batch_size=200_000):
        frame = batch.to_pandas()
        frame['split'] = frame['trade_date'].map(split_date)
        frame['first_valid_date'] = frame['ts_code'].map(lambda x: boundaries.get(str(x), [None, None])[0])
        frame['last_valid_date'] = frame['ts_code'].map(lambda x: boundaries.get(str(x), [None, None])[1])
        frame['inside_valid_interval'] = frame['trade_date'].between(frame['first_valid_date'], frame['last_valid_date'])
        frame['pre_interval'] = frame['trade_date'] < frame['first_valid_date']
        frame['post_interval'] = frame['trade_date'] > frame['last_valid_date']
        frame['history_16_valid'] = frame['history_available_count_20d'] >= HISTORY_THRESHOLD
        frame['current_market_valid'] = frame['missing_price_volume'] == 0
        frame['eligible'] = frame['inside_valid_interval'] & frame['history_16_valid']
        grouped = frame.groupby(['trade_date', 'split'], as_index=False).agg(
            rows=('trade_date', 'size'),
            pre_interval=('pre_interval', 'sum'),
            post_interval=('post_interval', 'sum'),
            inside_valid_interval=('inside_valid_interval', 'sum'),
            history_16_valid=('history_16_valid', 'sum'),
            current_market_valid=('current_market_valid', 'sum'),
            eligible=('eligible', 'sum')
        )
        rows.append(grouped)
    daily = pd.concat(rows, ignore_index=True).groupby(['trade_date', 'split'], as_index=False).sum()
    daily['label_valid'] = 0
    return daily[['trade_date', 'split', 'rows', 'pre_interval', 'post_interval', 'inside_valid_interval', 'label_valid', 'history_16_valid', 'current_market_valid', 'eligible']]


def make_summary(daily):
    summary = daily.groupby('split').agg(
        dates=('trade_date', 'nunique'),
        rows=('rows', 'sum'),
        pre_interval=('pre_interval', 'sum'),
        post_interval=('post_interval', 'sum'),
        inside_valid_interval=('inside_valid_interval', 'sum'),
        label_valid=('label_valid', 'sum'),
        history_16_valid=('history_16_valid', 'sum'),
        current_market_valid=('current_market_valid', 'sum'),
        eligible=('eligible', 'sum')
    ).reset_index()
    summary['eligible_rate'] = summary['eligible'] / summary['rows']
    return summary


def report_text(summary):
    lines = [
        '# 第二步：时间划分与样本资格报告（修订版）', '',
        '## 时间划分', '',
        '| 区间 | 日期范围 | 用途 |', '|---|---|---|',
        '| `train_fit` | 2018-01-02 至 2022-12-31 | 拟合模型、估计填充和标准化参数 |',
        '| `validation` | 2023-01-01 至 2023-12-31 | 选择超参数和训练轮数 |',
        '| `local_holdout` | 2024-01-01 至 2024-12-31 | 本地留出评价 |',
        '| `competition_test` | 2025-01-02 至 2026-06-08 | 最终生成提交预测 |', '',
        '## 修订后的样本资格规则', '',
        '样本资格按 `ts_code` 独立判断，不设置全市场统一的有效起始日期。一只股票的缺失区间不影响其他股票使用各自有效区间内的记录。', '',
        '训练、验证和本地留出样本同时满足以下条件：', '',
        '1. 目标日位于该股票首个有效行情日和最后有效行情日之间；',
        '2. `y_ret_1d` 不为空；',
        f'3. 最近 20 个交易日中至少 {HISTORY_THRESHOLD} 天的六个量价字段完整。', '',
        '有效行情区间内部的缺失进入模型输入阶段填充，并保留 `missing_price_volume`。首个有效行情日前和最后有效行情日后的记录不进入训练样本。', '',
        '官方测试集保留全部记录用于生成预测，区间边界和历史有效性只作为数据状态信息。', '',
        '## 区间统计', '',
        '| 区间 | 交易日数 | 总记录数 | 首个有效日前 | 最后有效日后 | 有效区间内 | 标签有效 | 16日历史有效 | 训练资格样本 | 资格比例 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|'
    ]
    for _, r in summary.iterrows():
        label = '—' if r['split'] == 'competition_test' else f"{int(r['label_valid']):,}"
        lines.append(f"| `{r['split']}` | {int(r['dates']):,} | {int(r['rows']):,} | {int(r['pre_interval']):,} | {int(r['post_interval']):,} | {int(r['inside_valid_interval']):,} | {label} | {int(r['history_16_valid']):,} | {int(r['eligible']):,} | {r['eligible_rate']:.2%} |")
    lines += ['', '## 后续使用规则', '', '- 填充参数、异常值边界和标准化参数只从 `train_fit` 的资格样本估计。', '- 验证集和本地留出集只使用已经拟合好的训练参数。', '- 样本序列构造时，以目标日 t 为末端，使用 t-19 至 t 的 20 个交易日。', f'- 输入窗口允许内部缺失，但要求最近 20 个交易日中至少 {HISTORY_THRESHOLD} 天六个量价字段完整。', '- 目标标签使用目标日记录中的 `y_ret_1d`，对应 t+1 日收益率。']
    return '\n'.join(lines) + '\n'


def main():
    train_boundaries = collect_boundaries(TRAIN_PATH)
    combined_boundaries = collect_boundaries(TRAIN_PATH)
    test_boundaries = collect_boundaries(TEST_PATH)
    for code, values in test_boundaries.items():
        if code not in combined_boundaries:
            combined_boundaries[code] = values
        else:
            combined_boundaries[code][0] = min(combined_boundaries[code][0], values[0])
            combined_boundaries[code][1] = max(combined_boundaries[code][1], values[1])
    train_daily = inspect_train(train_boundaries)
    test_daily = inspect_test(combined_boundaries)
    daily = pd.concat([train_daily, test_daily], ignore_index=True).sort_values(['trade_date', 'split']).reset_index(drop=True)
    summary = make_summary(daily)
    summary_path = os.path.join(REPORT_DIR, '步骤2_修订版区间样本统计.csv')
    daily_path = os.path.join(REPORT_DIR, '步骤2_修订版每日样本统计.csv')
    json_path = os.path.join(REPORT_DIR, '步骤2_修订版时间划分与样本资格.json')
    md_path = os.path.join(REPORT_DIR, '步骤2_修订版时间划分与样本资格报告.md')
    summary.to_csv(summary_path, index=False, encoding='utf-8-sig')
    daily.to_csv(daily_path, index=False, encoding='utf-8-sig')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({'summary': summary.to_dict(orient='records'), 'history_threshold': HISTORY_THRESHOLD, 'rules': ['inside_valid_interval', 'label_valid', f'history_available_count_20d >= {HISTORY_THRESHOLD}'], 'daily_rows': len(daily)}, f, ensure_ascii=False, indent=2)
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(report_text(summary))
    print(json.dumps({'report': md_path, 'summary': summary_path, 'daily': daily_path}, ensure_ascii=False))


if __name__ == '__main__':
    main()
