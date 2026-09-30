import json
import os
import time
from collections import OrderedDict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import rankdata


BASE = Path(r'E:\ChatGPT\ChatGPT_Project\深圳金融比赛赛题五')
DATA_DIR = BASE / '因子数据'
OUTPUT_DIR = BASE / '模型训练' / 'LightGBM实验结果'
TRAIN_PATH = DATA_DIR / '训练集_第一次LST-Transformer因子.parquet'
TEST_PATH = DATA_DIR / '测试集_X_第一次LST-Transformer因子.parquet'
BATCH_SIZE = 200_000
HISTORY_THRESHOLD = 16
SEED = 42
NUM_THREADS = 8
MAX_ROUNDS = 400
CHECKPOINT_INTERVAL = 50

FEATURE_COLUMNS = [
    'ret_1d', 'mom_3d', 'mom_5d', 'mom_10d', 'mom_20d', 'mom_20d_skip1',
    'volatility_5d', 'volatility_20d', 'range_1d', 'gap_1d',
    'true_range_1d', 'intraday_ret', 'close_position', 'body_direction',
    'body_ratio', 'upper_shadow', 'lower_shadow', 'log_amount', 'log_volume',
    'delta_log_amount', 'delta_log_volume', 'amount_shock_5d',
    'amount_shock_20d', 'limit_up', 'limit_down', 'missing_price_volume',
    'history_available_count_20d'
]

SPLITS = OrderedDict([
    ('train_fit', (20180102, 20221231)),
    ('validation', (20230101, 20231231)),
    ('local_holdout', (20240101, 20241231)),
    ('competition_test', (20250101, 20260608))
])

CANDIDATES = [
    OrderedDict([
        ('name', 'lgb_01'), ('learning_rate', 0.03), ('num_leaves', 31),
        ('min_data_in_leaf', 300), ('feature_fraction', 1.0),
        ('bagging_fraction', 1.0), ('lambda_l2', 1.0)
    ]),
    OrderedDict([
        ('name', 'lgb_02'), ('learning_rate', 0.05), ('num_leaves', 63),
        ('min_data_in_leaf', 300), ('feature_fraction', 0.8),
        ('bagging_fraction', 0.8), ('lambda_l2', 10.0)
    ]),
    OrderedDict([
        ('name', 'lgb_03'), ('learning_rate', 0.03), ('num_leaves', 127),
        ('min_data_in_leaf', 1000), ('feature_fraction', 0.8),
        ('bagging_fraction', 0.8), ('lambda_l2', 10.0)
    ]),
    OrderedDict([
        ('name', 'lgb_04'), ('learning_rate', 0.05), ('num_leaves', 31),
        ('min_data_in_leaf', 100), ('feature_fraction', 1.0),
        ('bagging_fraction', 1.0), ('lambda_l2', 1.0)
    ])
]


def collect_boundaries(path):
    boundaries = {}
    for batch in pq.ParquetFile(path).iter_batches(
        columns=['ts_code', 'trade_date', 'missing_price_volume'],
        batch_size=BATCH_SIZE
    ):
        frame = batch.to_pandas()
        valid = frame.loc[frame['missing_price_volume'].eq(0)]
        for code, group in valid.groupby('ts_code', sort=False):
            code = str(code)
            first_date = int(group['trade_date'].min())
            last_date = int(group['trade_date'].max())
            if code not in boundaries:
                boundaries[code] = [first_date, last_date]
            else:
                boundaries[code][0] = min(boundaries[code][0], first_date)
                boundaries[code][1] = max(boundaries[code][1], last_date)
    return boundaries


def merge_boundaries(left, right):
    merged = {code: values[:] for code, values in left.items()}
    for code, values in right.items():
        if code not in merged:
            merged[code] = values[:]
        else:
            merged[code][0] = min(merged[code][0], values[0])
            merged[code][1] = max(merged[code][1], values[1])
    return merged


def split_for_dates(dates):
    result = np.full(len(dates), '', dtype='<U20')
    for name, (start, end) in SPLITS.items():
        mask = (dates >= start) & (dates <= end)
        result[mask] = name
    if (result == '').any():
        raise ValueError('存在不属于预定区间的日期')
    return result


def eligibility_mask(frame, boundaries, has_label):
    codes = frame['ts_code'].astype(str)
    first = codes.map({code: values[0] for code, values in boundaries.items()}).to_numpy()
    last = codes.map({code: values[1] for code, values in boundaries.items()}).to_numpy()
    dates = frame['trade_date'].to_numpy(dtype=np.int32)
    inside = (dates >= first) & (dates <= last)
    history_valid = frame['history_available_count_20d'].to_numpy(dtype=float) >= HISTORY_THRESHOLD
    if has_label:
        labels = frame['y_ret_1d'].to_numpy(dtype=float)
        label_valid = np.isfinite(labels)
        return inside & history_valid & label_valid
    return inside & history_valid


def read_training_data(boundaries):
    columns = ['ts_code', 'trade_date'] + FEATURE_COLUMNS + ['y_ret_1d']
    arrays = {
        name: {'x': [], 'y': [], 'dates': [], 'codes': [], 'limit_up': []}
        for name in ('train_fit', 'validation', 'local_holdout')
    }
    counts = OrderedDict((name, 0) for name in arrays)
    for batch in pq.ParquetFile(TRAIN_PATH).iter_batches(
        columns=columns, batch_size=BATCH_SIZE
    ):
        frame = batch.to_pandas()
        split_labels = split_for_dates(frame['trade_date'].to_numpy(dtype=np.int32))
        eligible = eligibility_mask(frame, boundaries, has_label=True)
        for name in arrays:
            mask = (split_labels == name) & eligible
            if not mask.any():
                continue
            selected = frame.loc[mask]
            arrays[name]['x'].append(selected[FEATURE_COLUMNS].to_numpy(dtype=np.float32))
            arrays[name]['y'].append(selected['y_ret_1d'].to_numpy(dtype=np.float32))
            arrays[name]['dates'].append(selected['trade_date'].to_numpy(dtype=np.int32))
            arrays[name]['codes'].append(selected['ts_code'].astype(str).to_numpy())
            arrays[name]['limit_up'].append(selected['limit_up'].to_numpy(dtype=np.float32))
            counts[name] += len(selected)
    result = {}
    for name, parts in arrays.items():
        result[name] = {
            'x': np.concatenate(parts['x'], axis=0),
            'y': np.concatenate(parts['y'], axis=0),
            'dates': np.concatenate(parts['dates'], axis=0),
            'codes': np.concatenate(parts['codes'], axis=0),
            'limit_up': np.concatenate(parts['limit_up'], axis=0)
        }
    return result, counts


def read_test_data(boundaries):
    columns = ['ts_code', 'trade_date'] + FEATURE_COLUMNS
    parts = {'x': [], 'dates': [], 'codes': []}
    for batch in pq.ParquetFile(TEST_PATH).iter_batches(
        columns=columns, batch_size=BATCH_SIZE
    ):
        frame = batch.to_pandas()
        parts['x'].append(frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32))
        parts['dates'].append(frame['trade_date'].to_numpy(dtype=np.int32))
        parts['codes'].append(frame['ts_code'].astype(str).to_numpy())
        eligibility_mask(frame, boundaries, has_label=False)
    return {
        'x': np.concatenate(parts['x'], axis=0),
        'dates': np.concatenate(parts['dates'], axis=0),
        'codes': np.concatenate(parts['codes'], axis=0)
    }


def daily_metrics(predictions, targets, dates, codes, limit_up):
    frame = pd.DataFrame({
        'trade_date': dates,
        'pred': predictions,
        'target': targets,
        'ts_code': codes,
        'limit_up': limit_up
    })
    rank_ics = []
    excess_returns = []
    daily_top_sets = []
    for trade_date, group in frame.groupby('trade_date', sort=True):
        valid = group[np.isfinite(group['pred']) & np.isfinite(group['target'])]
        if len(valid) >= 30:
            prediction_rank = rankdata(valid['pred'].to_numpy(dtype=float), method='average')
            target_rank = rankdata(valid['target'].to_numpy(dtype=float), method='average')
            prediction_std = prediction_rank.std()
            target_std = target_rank.std()
            if prediction_std > 0 and target_std > 0:
                rank_ics.append(float(np.corrcoef(prediction_rank, target_rank)[0, 1]))
        investable = group[
            (group['limit_up'] == 0)
            & np.isfinite(group['pred'])
            & np.isfinite(group['target'])
        ].sort_values('pred', ascending=False)
        if len(investable) >= 100:
            top_count = max(len(investable) // 10, 1)
            excess_returns.append(
                float(investable['target'].iloc[:top_count].mean() - investable['target'].mean())
            )
            daily_top_sets.append(set(investable['ts_code'].iloc[:top_count]))
    if not rank_ics:
        raise ValueError('没有足够的交易日计算 Rank IC')
    rank_ic_mean = float(np.mean(rank_ics))
    rank_ic_std = float(np.std(rank_ics, ddof=1)) if len(rank_ics) > 1 else 0.0
    turnovers = []
    for previous, current in zip(daily_top_sets, daily_top_sets[1:]):
        union = previous | current
        if union:
            turnovers.append(1.0 - len(previous & current) / len(union))
    mean_turnover = float(np.mean(turnovers)) if turnovers else 0.0
    annual_excess = float(np.mean(excess_returns) * 252) if excess_returns else 0.0
    return OrderedDict([
        ('daily_rank_ic_mean', rank_ic_mean),
        ('daily_rank_ic_std', rank_ic_std),
        ('icir', rank_ic_mean / rank_ic_std if rank_ic_std > 0 else 0.0),
        ('ic_positive_ratio', float(np.mean(np.asarray(rank_ics) > 0))),
        ('annual_top_decile_excess', annual_excess),
        ('mean_turnover', mean_turnover),
        ('official_score', rank_ic_mean * 0.4 + annual_excess * 0.3 + (1.0 - mean_turnover) * 0.3),
        ('rank_ic_days', len(rank_ics)),
        ('top_decile_days', len(excess_returns)),
        ('turnover_days', len(turnovers))
    ])


def model_parameters(candidate):
    return {
        'objective': 'regression',
        'metric': 'l2',
        'learning_rate': candidate['learning_rate'],
        'num_leaves': candidate['num_leaves'],
        'min_data_in_leaf': candidate['min_data_in_leaf'],
        'feature_fraction': candidate['feature_fraction'],
        'bagging_fraction': candidate['bagging_fraction'],
        'bagging_freq': 1,
        'max_depth': -1,
        'lambda_l1': 0.0,
        'lambda_l2': candidate['lambda_l2'],
        'max_bin': 255,
        'feature_pre_filter': False,
        'verbosity': -1,
        'seed': SEED,
        'data_random_seed': SEED,
        'feature_fraction_seed': SEED,
        'bagging_seed': SEED,
        'num_threads': NUM_THREADS,
        'force_col_wise': True,
        'deterministic': True
    }


def save_prediction(path, data, predictions, include_target):
    result = pd.DataFrame({
        'ts_code': data['codes'],
        'trade_date': data['dates'],
        'pred': predictions.astype(np.float32)
    })
    if include_target:
        result['y_ret_1d'] = data['y']
    result.to_parquet(path, index=False, compression='zstd')


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    train_boundaries = collect_boundaries(TRAIN_PATH)
    test_boundaries = collect_boundaries(TEST_PATH)
    all_boundaries = merge_boundaries(train_boundaries, test_boundaries)
    data, counts = read_training_data(train_boundaries)
    if any(value == 0 for value in counts.values()):
        raise ValueError(f'存在空训练分区: {counts}')
    validation = data['validation']
    train_dataset = lgb.Dataset(
        data['train_fit']['x'],
        label=data['train_fit']['y'],
        feature_name=FEATURE_COLUMNS,
        free_raw_data=True
    )
    validation_dataset = lgb.Dataset(
        validation['x'],
        label=validation['y'],
        reference=train_dataset,
        feature_name=FEATURE_COLUMNS,
        free_raw_data=True
    )
    candidate_results = []
    training_log = []
    best_record = None
    best_model_text = None
    for candidate in CANDIDATES:
        print(json.dumps({'event': 'candidate_start', 'candidate': candidate['name']}, ensure_ascii=False))
        params = model_parameters(candidate)
        candidate_start = time.perf_counter()
        booster = lgb.train(
            params,
            train_dataset,
            num_boost_round=MAX_ROUNDS,
            valid_sets=[validation_dataset],
            valid_names=['validation'],
            callbacks=[lgb.log_evaluation(CHECKPOINT_INTERVAL)]
        )
        best_checkpoint = None
        for iteration in range(CHECKPOINT_INTERVAL, booster.current_iteration() + 1, CHECKPOINT_INTERVAL):
            predictions = booster.predict(validation['x'], num_iteration=iteration)
            metrics = daily_metrics(
                predictions, validation['y'], validation['dates'],
                validation['codes'], validation['limit_up']
            )
            record = OrderedDict([
                ('candidate', candidate['name']),
                ('iteration', iteration),
                ('validation_l2', float(np.mean((predictions - validation['y']) ** 2))),
                *metrics.items()
            ])
            training_log.append(record)
            if best_checkpoint is None or (
                record['daily_rank_ic_mean'], -record['validation_l2']
            ) > (
                best_checkpoint['daily_rank_ic_mean'], -best_checkpoint['validation_l2']
            ):
                best_checkpoint = record
        if best_checkpoint is None:
            raise ValueError(f'候选模型没有生成检查点: {candidate["name"]}')
        candidate_record = OrderedDict(candidate)
        candidate_record.update({
            'best_iteration': best_checkpoint['iteration'],
            'validation_l2': best_checkpoint['validation_l2'],
            'validation_daily_rank_ic_mean': best_checkpoint['daily_rank_ic_mean'],
            'validation_daily_rank_ic_std': best_checkpoint['daily_rank_ic_std'],
            'validation_icir': best_checkpoint['icir'],
            'validation_ic_positive_ratio': best_checkpoint['ic_positive_ratio'],
            'validation_annual_top_decile_excess': best_checkpoint['annual_top_decile_excess'],
            'validation_mean_turnover': best_checkpoint['mean_turnover'],
            'validation_official_score': best_checkpoint['official_score'],
            'seconds': time.perf_counter() - candidate_start
        })
        candidate_results.append(candidate_record)
        if best_record is None or (
            candidate_record['validation_daily_rank_ic_mean'],
            -candidate_record['validation_l2']
        ) > (
            best_record['validation_daily_rank_ic_mean'],
            -best_record['validation_l2']
        ):
            best_record = candidate_record
            best_model_text = booster.model_to_string(
                num_iteration=best_checkpoint['iteration']
            )
        print(json.dumps({'event': 'candidate_end', **candidate_record}, ensure_ascii=False))
    if best_record is None or best_model_text is None:
        raise ValueError('没有确定最佳模型')
    (OUTPUT_DIR / '最佳模型.txt').write_text(best_model_text, encoding='utf-8')
    (OUTPUT_DIR / 'best_model.txt').write_text(best_model_text, encoding='utf-8')
    final_model = lgb.Booster(model_str=best_model_text)
    validation_predictions = final_model.predict(
        validation['x'], num_iteration=int(best_record['best_iteration'])
    )
    holdout_predictions = final_model.predict(
        data['local_holdout']['x'], num_iteration=int(best_record['best_iteration'])
    )
    holdout_metrics = daily_metrics(
        holdout_predictions, data['local_holdout']['y'], data['local_holdout']['dates'],
        data['local_holdout']['codes'], data['local_holdout']['limit_up']
    )
    test_data = read_test_data(all_boundaries)
    test_predictions = final_model.predict(
        test_data['x'], num_iteration=int(best_record['best_iteration'])
    )
    if not np.isfinite(test_predictions).all():
        raise ValueError('测试预测存在非有限值')
    save_prediction(
        OUTPUT_DIR / '验证集预测.parquet', validation, validation_predictions, include_target=True
    )
    save_prediction(
        OUTPUT_DIR / '本地留出集预测.parquet', data['local_holdout'], holdout_predictions, include_target=True
    )
    save_prediction(
        OUTPUT_DIR / '测试集预测.parquet', test_data, test_predictions, include_target=False
    )
    submission = pd.DataFrame({
        'ts_code': test_data['codes'],
        'trade_date': test_data['dates'],
        'pred': test_predictions.astype(np.float32)
    })
    submission.to_csv(OUTPUT_DIR / 'submission.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(candidate_results).to_csv(
        OUTPUT_DIR / '候选配置结果.csv', index=False, encoding='utf-8-sig'
    )
    pd.DataFrame(training_log).to_csv(
        OUTPUT_DIR / '训练日志.csv', index=False, encoding='utf-8-sig'
    )
    config = OrderedDict([
        ('experiment_name', 'LightGBM实验'),
        ('feature_columns', FEATURE_COLUMNS),
        ('data_source', {
            'train': str(TRAIN_PATH),
            'test': str(TEST_PATH),
            'missing_value_rule': '保留NaN并使用LightGBM原生缺失值处理'
        }),
        ('sample_rule', {
            'history_threshold': HISTORY_THRESHOLD,
            'train_validation_holdout': 'inside_valid_interval and label_valid and history_available_count_20d >= 16',
            'competition_test': '保留全部记录'
        }),
        ('split_counts', counts),
        ('best_candidate', best_record),
        ('local_holdout', holdout_metrics),
        ('test_rows', int(len(test_predictions))),
        ('seconds', time.perf_counter() - started)
    ])
    with open(OUTPUT_DIR / '配置.json', 'w', encoding='utf-8') as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
    report_lines = [
        '# LightGBM 实验报告', '',
        f"- 最佳候选：`{best_record['name']}`。",
        f"- 最佳迭代轮数：`{best_record['best_iteration']}`。",
        f"- 训练样本数：`{counts['train_fit']:,}`。",
        f"- 验证样本数：`{counts['validation']:,}`。",
        f"- 本地留出样本数：`{counts['local_holdout']:,}`。",
        '', '## 验证集结果', '',
        f"- 每日 Rank IC 均值：`{best_record['validation_daily_rank_ic_mean']:.10g}`。",
        f"- 验证集 ICIR：`{best_record['validation_icir']:.10g}`。",
        f"- 验证集 Top 10% 年化超额收益：`{best_record['validation_annual_top_decile_excess']:.10g}`。",
        f"- 验证集平均换手率：`{best_record['validation_mean_turnover']:.10g}`。",
        f"- 验证集综合分数：`{best_record['validation_official_score']:.10g}`。",
        '', '## 本地留出集结果', '',
        f"- 每日 Rank IC 均值：`{holdout_metrics['daily_rank_ic_mean']:.10g}`。",
        f"- ICIR：`{holdout_metrics['icir']:.10g}`。",
        f"- Top 10% 年化超额收益：`{holdout_metrics['annual_top_decile_excess']:.10g}`。",
        f"- 平均换手率：`{holdout_metrics['mean_turnover']:.10g}`。",
        f"- 综合分数：`{holdout_metrics['official_score']:.10g}`。",
        '', '## 测试集输出', '',
        f"- 预测行数：`{len(test_predictions):,}`。",
        f"- 预测文件：`测试集预测.parquet`。",
        f"- 提交文件：`submission.csv`。", ''
    ]
    (OUTPUT_DIR / '实验报告.md').write_text('\n'.join(report_lines), encoding='utf-8')
    print(json.dumps({
        'event': 'complete',
        'output_dir': str(OUTPUT_DIR),
        'best_candidate': best_record['name'],
        'best_iteration': best_record['best_iteration'],
        'validation_rank_ic': best_record['validation_daily_rank_ic_mean'],
        'holdout_rank_ic': holdout_metrics['daily_rank_ic_mean'],
        'test_rows': len(test_predictions),
        'seconds': time.perf_counter() - started
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
