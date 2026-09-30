import json
import os
from collections import OrderedDict

import numpy as np
import pandas as pd

BASE = r'E:\ChatGPT\ChatGPT_Project\深圳金融比赛赛题五'
RUN_DIR = os.path.join(BASE, '模型训练', '第一次LST-Transformer实验')
STEP5_PATH = os.path.join(
    BASE, '模型预处理', '步骤5_标准化处理', '本地留出集_标准化.parquet'
)
INDEX_PATH = os.path.join(
    BASE, '模型预处理', '步骤6_序列构造', '序列索引_local_holdout.parquet'
)
PREDICTION_PATH = os.path.join(RUN_DIR, 'local_holdout_predictions.parquet')


def daily_rank_ic(frame, signal_column):
    values = []
    for _, group in frame.groupby('trade_date', sort=True):
        if len(group) < 30:
            continue
        signal_rank = group[signal_column].rank(method='average').to_numpy()
        target_rank = group['y_ret_1d'].rank(method='average').to_numpy()
        if signal_rank.std() == 0 or target_rank.std() == 0:
            continue
        values.append(float(np.corrcoef(signal_rank, target_rank)[0, 1]))
    values = np.asarray(values, dtype=np.float64)
    return OrderedDict([
        ('mean', float(values.mean())),
        ('std', float(values.std(ddof=1))),
        ('icir', float(values.mean() / values.std(ddof=1))),
        ('positive_ratio', float((values > 0).mean())),
        ('days', int(len(values)))
    ])


def top_decile_excess(frame, signal_column):
    values = []
    top_returns = []
    for _, group in frame.groupby('trade_date', sort=True):
        valid = group.loc[
            group['limit_up'].eq(0) & group['y_ret_1d'].notna()
        ].sort_values(signal_column, ascending=False)
        if len(valid) < 100:
            continue
        top_count = max(len(valid) // 10, 1)
        top_return = float(valid['y_ret_1d'].iloc[:top_count].mean())
        market_return = float(valid['y_ret_1d'].mean())
        values.append(top_return - market_return)
        top_returns.append(top_return)
    return OrderedDict([
        ('annual_excess', float(np.mean(values) * 252)),
        ('top1_annual_return', float(np.mean(top_returns) * 252)),
        ('days', int(len(values)))
    ])


def turnover(frame, signal_column):
    previous = None
    values = []
    for _, group in frame.groupby('trade_date', sort=True):
        valid = group.loc[group['limit_up'].eq(0)].sort_values(
            signal_column, ascending=False
        )
        if len(valid) < 100:
            previous = None
            continue
        top_count = max(len(valid) // 10, 1)
        current = set(valid['ts_code'].iloc[:top_count])
        if previous:
            values.append(1.0 - len(current & previous) / len(current | previous))
        previous = current
    mean_turnover = float(np.mean(values))
    return OrderedDict([
        ('mean_turnover', mean_turnover),
        ('one_minus_turnover', 1.0 - mean_turnover),
        ('transition_days', int(len(values)))
    ])


def score(frame, signal_column):
    rank = daily_rank_ic(frame, signal_column)
    top = top_decile_excess(frame, signal_column)
    trade = turnover(frame, signal_column)
    final = (
        rank['mean'] * 0.4
        + top['annual_excess'] * 0.3
        + trade['one_minus_turnover'] * 0.3
    )
    return OrderedDict([
        ('rank_ic', rank),
        ('top_decile', top),
        ('turnover', trade),
        ('final_score', float(final))
    ])


def main():
    prediction = pd.read_parquet(PREDICTION_PATH)
    index = pd.read_parquet(INDEX_PATH, columns=[
        'ts_code', 'target_date', 'prediction_eligible', 'source_row_index'
    ])
    eligible_index = index.loc[index['prediction_eligible'].eq(1)].reset_index(drop=True)
    if len(prediction) != len(eligible_index):
        raise ValueError('本地预测行数与序列索引合格样本数不一致')
    if not np.array_equal(
        prediction['trade_date'].to_numpy(dtype=np.int64),
        eligible_index['target_date'].to_numpy(dtype=np.int64)
    ):
        raise ValueError('本地预测日期与序列索引不一致')

    factors = pd.read_parquet(STEP5_PATH, columns=['limit_up', 'ret_1d'])
    source_rows = eligible_index['source_row_index'].to_numpy(dtype=np.int64)
    evaluation = pd.DataFrame({
        'ts_code': eligible_index['ts_code'].astype(str).to_numpy(),
        'trade_date': prediction['trade_date'].to_numpy(dtype=np.int64),
        'pred': prediction['pred'].to_numpy(dtype=np.float64),
        'y_ret_1d': prediction['y_ret_1d'].to_numpy(dtype=np.float64),
        'limit_up': factors.iloc[source_rows]['limit_up'].to_numpy(dtype=np.float64),
        'ret_1d': factors.iloc[source_rows]['ret_1d'].to_numpy(dtype=np.float64)
    })
    evaluation.to_parquet(
        os.path.join(RUN_DIR, 'local_holdout_predictions_with_code.parquet'),
        index=False, compression='zstd'
    )
    model_score = score(evaluation, 'pred')
    baseline_score = score(evaluation, 'ret_1d')
    result = OrderedDict([
        ('rows', int(len(evaluation))),
        ('dates', int(evaluation['trade_date'].nunique())),
        ('model', model_score),
        ('naive_ret_1d_baseline', baseline_score),
        ('model_minus_baseline', OrderedDict([
            ('rank_ic_mean', model_score['rank_ic']['mean'] - baseline_score['rank_ic']['mean']),
            ('annual_excess', model_score['top_decile']['annual_excess'] - baseline_score['top_decile']['annual_excess']),
            ('one_minus_turnover', model_score['turnover']['one_minus_turnover'] - baseline_score['turnover']['one_minus_turnover']),
            ('final_score', model_score['final_score'] - baseline_score['final_score'])
        ]))
    ])
    result_path = os.path.join(RUN_DIR, '本地留出集比赛指标评估.json')
    with open(result_path, 'w', encoding='utf-8') as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    report_lines = [
        '# 第一次 LST-Transformer 本地留出集效果评估',
        '',
        f"- 评估样本：{len(evaluation):,} 条。",
        f"- 交易日数量：{evaluation['trade_date'].nunique():,} 天。",
        '',
        '## 模型指标',
        '',
        f"- 日 Rank IC 均值：{model_score['rank_ic']['mean']:.8f}。",
        f"- ICIR：{model_score['rank_ic']['icir']:.8f}。",
        f"- Rank IC 为正占比：{model_score['rank_ic']['positive_ratio']:.2%}。",
        f"- 前 10% 组合年化超额收益：{model_score['top_decile']['annual_excess']:.8f}。",
        f"- `1-换手率`：{model_score['turnover']['one_minus_turnover']:.8f}。",
        f"- 按比赛权重计算的本地综合分：{model_score['final_score']:.8f}。",
        '',
        '## 朴素基线',
        '',
        '使用当前日 `ret_1d` 因子作为排序信号，作为简单动量基线。',
        f"- 基线日 Rank IC 均值：{baseline_score['rank_ic']['mean']:.8f}。",
        f"- 基线前 10% 组合年化超额收益：{baseline_score['top_decile']['annual_excess']:.8f}。",
        f"- 基线 `1-换手率`：{baseline_score['turnover']['one_minus_turnover']:.8f}。",
        f"- 基线本地综合分：{baseline_score['final_score']:.8f}。",
        '',
        '## 判断',
        '',
        '模型在验证集和本地留出集均保持正的日 Rank IC，具备继续实验价值。训练损失继续下降而验证 Rank IC 从第一周期后下降，说明当前首轮训练已出现过拟合，第一周期模型应作为后续比较基准。',
        '',
        '完整数值见 `本地留出集比赛指标评估.json`，带股票代码的预测见 `local_holdout_predictions_with_code.parquet`。'
    ]
    with open(os.path.join(RUN_DIR, '本地留出集效果评估报告.md'), 'w', encoding='utf-8') as file:
        file.write('\n'.join(report_lines) + '\n')


if __name__ == '__main__':
    main()
