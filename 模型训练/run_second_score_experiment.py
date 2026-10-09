import json
import os
import re
import sys
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE, '模型训练')
RUN_DIR = os.path.join(MODEL_DIR, '第二次综合分验证实验')
FIRST_RUN_DIR = os.path.join(MODEL_DIR, '第一次LST-Transformer实验')
CONFIG_PATH = os.path.join(MODEL_DIR, '第一次LST-Transformer实验配置.json')
STEP7_DIR = os.path.join(BASE, '模型预处理', '步骤7_模型输入接口')
STEP5_DIR = os.path.join(BASE, '模型预处理', '步骤5_标准化处理')
STEP6_DIR = os.path.join(BASE, '模型预处理', '步骤6_序列构造')
sys.path.insert(0, STEP7_DIR)
sys.path.insert(0, MODEL_DIR)
from sequence_input_interface import SequenceInputReader
from lstm_transformer_small import LSTMTransformerRegressor
from evaluate_first_lstm_result import evaluate_frame


def make_model(config, device):
    return LSTMTransformerRegressor(
        input_dim=config['input']['feature_count'],
        sequence_length=config['input']['sequence_length'],
        projection_dim=config['model']['input_projection_dim'],
        lstm_hidden_dim=config['model']['lstm_hidden_dim'],
        lstm_layers=config['model']['lstm_layers'],
        transformer_layers=config['model']['transformer_layers'],
        transformer_heads=config['model']['transformer_heads'],
        transformer_feedforward_dim=config['model']['transformer_feedforward_dim'],
        dropout=config['model']['dropout']
    ).to(device)


def read_evaluation_data(split):
    index_path = os.path.join(STEP6_DIR, f'序列索引_{split}.parquet')
    index = pd.read_parquet(index_path, columns=[
        'ts_code', 'target_date', 'source_row_index', 'prediction_eligible',
        'y_ret_1d'
    ])
    eligible = index.loc[index['prediction_eligible'].eq(1)].reset_index(drop=True)
    factors = pd.read_parquet(
        os.path.join(STEP5_DIR, f'{"验证集" if split == "validation" else "本地留出集"}_标准化.parquet'),
        columns=['limit_up']
    )
    source_rows = eligible['source_row_index'].to_numpy(dtype=np.int64)
    if source_rows.min() < 0 or source_rows.max() >= len(factors):
        raise ValueError(f'{split} source_row_index 超出标准化数据范围')
    frame = pd.DataFrame({
        'ts_code': eligible['ts_code'].astype(str).to_numpy(),
        'trade_date': eligible['target_date'].to_numpy(dtype=np.int64),
        'y_ret_1d': eligible['y_ret_1d'].to_numpy(dtype=np.float64),
        'flag_limit_up': factors.iloc[source_rows]['limit_up'].to_numpy(dtype=np.float64),
        'prediction_eligible': eligible['prediction_eligible'].to_numpy(dtype=np.int8)
    })
    if frame.duplicated(['ts_code', 'trade_date']).any():
        raise ValueError(f'{split} 评价键存在重复')
    if not np.isfinite(frame['flag_limit_up']).all():
        raise ValueError(f'{split} 涨停标记包含非有限值')
    return frame


def predict_checkpoint(checkpoint_path, split, config, device, batch_size):
    model = make_model(config, device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    reader = SequenceInputReader(
        STEP7_DIR, split, batch_size=batch_size, eligible_only=True
    )
    codes = []
    dates = []
    targets = []
    predictions = []
    with torch.no_grad():
        for batch in reader.iter_batches():
            x = torch.from_numpy(batch['x']).to(device)
            output = model(x).detach().cpu().numpy().astype(np.float64)
            codes.append(batch['ts_code'].astype(str))
            dates.append(batch['target_date'].astype(np.int64))
            if 'y' in batch:
                targets.append(batch['y'].astype(np.float64))
            predictions.append(output)
    result = pd.DataFrame({
        'ts_code': np.concatenate(codes),
        'trade_date': np.concatenate(dates),
        'pred': np.concatenate(predictions),
        'prediction_eligible': np.ones(len(np.concatenate(predictions)), dtype=np.int8)
    })
    if targets:
        result['y_ret_1d'] = np.concatenate(targets)
    if result.duplicated(['ts_code', 'trade_date']).any():
        raise ValueError(f'{split} 预测键存在重复')
    if not np.isfinite(result['pred']).all():
        raise ValueError(f'{split} 预测值包含非有限值')
    return result


def attach_evaluation(prediction, evaluation_data):
    if len(prediction) != len(evaluation_data):
        raise ValueError('预测行数与评价样本数量不一致')
    left_keys = prediction[['ts_code', 'trade_date']].reset_index(drop=True)
    right_keys = evaluation_data[['ts_code', 'trade_date']].reset_index(drop=True)
    if not left_keys.equals(right_keys):
        raise ValueError('预测键顺序与序列索引顺序不一致')
    frame = prediction.copy()
    frame['y_ret_1d'] = evaluation_data['y_ret_1d'].to_numpy(dtype=np.float64)
    frame['flag_limit_up'] = evaluation_data['flag_limit_up'].to_numpy(dtype=np.float64)
    frame['prediction_eligible'] = evaluation_data['prediction_eligible'].to_numpy(dtype=np.int8)
    return frame


def smooth_predictions(frame, alpha):
    if not 0 < alpha <= 1:
        raise ValueError('alpha 必须在 (0, 1] 范围内')
    ordered = frame.sort_values(
        ['ts_code', 'trade_date'], kind='mergesort'
    ).reset_index(drop=True)
    smooth = np.empty(len(ordered), dtype=np.float64)
    for _, positions in ordered.groupby('ts_code', sort=False).groups.items():
        indices = np.asarray(positions, dtype=np.int64)
        raw = ordered.iloc[indices]['pred'].to_numpy(dtype=np.float64)
        smooth[indices[0]] = raw[0]
        for position in range(1, len(indices)):
            smooth[indices[position]] = (
                alpha * raw[position] + (1.0 - alpha) * smooth[indices[position - 1]]
            )
    ordered['smooth_pred'] = smooth
    return ordered


def metric_record(epoch, metrics):
    return OrderedDict([
        ('epoch', epoch),
        ('validation_rank_ic_mean', metrics['rank_ic']['mean']),
        ('validation_rank_ic_std', metrics['rank_ic']['std']),
        ('validation_icir', metrics['rank_ic']['icir']),
        ('validation_ic_positive_ratio', metrics['rank_ic']['positive_ratio']),
        ('validation_rank_ic_days', metrics['rank_ic']['days']),
        ('validation_annual_excess', metrics['top_decile']['annual_excess']),
        ('validation_top_decile_days', metrics['top_decile']['days']),
        ('validation_mean_turnover', metrics['turnover']['mean_turnover']),
        ('validation_one_minus_turnover', metrics['turnover']['one_minus_turnover']),
        ('validation_turnover_transition_days', metrics['turnover']['transition_days']),
        ('validation_final_score', metrics['final_score'])
    ])


def choose_epoch(records, metric_name):
    if metric_name == 'rank_ic':
        ordered = sorted(records, key=lambda item: (
            item['validation_rank_ic_mean'],
            -item['validation_mean_turnover'],
            -item['epoch']
        ), reverse=True)
    elif metric_name == 'final_score':
        ordered = sorted(records, key=lambda item: (
            item['validation_final_score'],
            -item['validation_mean_turnover'],
            item['validation_rank_ic_mean'],
            -item['epoch']
        ), reverse=True)
    else:
        raise ValueError(f'未知选模指标: {metric_name}')
    return ordered[0]


def main():
    with open(CONFIG_PATH, 'r', encoding='utf-8') as file:
        config = json.load(file)
    os.makedirs(RUN_DIR, exist_ok=True)
    batch_size = 4096
    device = torch.device('cpu')
    torch.set_num_threads(8)
    runtime = {
        'source_experiment': '第一次LST-Transformer实验',
        'checkpoint_source': FIRST_RUN_DIR,
        'batch_size': batch_size,
        'device': str(device),
        'random_seed': config['optimization']['random_seed'],
        'smoothing_alphas': [1.0, 0.9, 0.7, 0.5, 0.3]
    }
    with open(os.path.join(RUN_DIR, '运行配置.json'), 'w', encoding='utf-8') as file:
        json.dump(runtime, file, ensure_ascii=False, indent=2)

    validation_data = read_evaluation_data('validation')
    checkpoint_paths = []
    for name in os.listdir(FIRST_RUN_DIR):
        match = re.fullmatch(r'checkpoint_epoch_(\d{2})\.pt', name)
        if match:
            checkpoint_paths.append((int(match.group(1)), os.path.join(FIRST_RUN_DIR, name)))
    checkpoint_paths.sort()
    if not checkpoint_paths:
        raise ValueError('没有找到第一次实验的训练周期检查点')

    records = []
    for epoch, checkpoint_path in checkpoint_paths:
        prediction = predict_checkpoint(checkpoint_path, 'validation', config, device, batch_size)
        frame = attach_evaluation(prediction, validation_data)
        prediction.to_parquet(
            os.path.join(RUN_DIR, f'validation_predictions_epoch_{epoch:02d}.parquet'),
            index=False, compression='zstd'
        )
        frame.to_parquet(
            os.path.join(RUN_DIR, f'validation_evaluation_epoch_{epoch:02d}.parquet'),
            index=False, compression='zstd'
        )
        metrics = evaluate_frame(frame)
        record = metric_record(epoch, metrics)
        records.append(record)
        pd.DataFrame(records).to_csv(
            os.path.join(RUN_DIR, '验证集各周期比赛指标.csv'),
            index=False, encoding='utf-8-sig'
        )
        print(json.dumps(record, ensure_ascii=False))

    group_a = choose_epoch(records, 'rank_ic')
    group_b = choose_epoch(records, 'final_score')
    for source_epoch, target_name in [
        (group_a['epoch'], '最佳模型_按RankIC.pt'),
        (group_b['epoch'], '最佳模型_按综合分.pt')
    ]:
        source = torch.load(
            os.path.join(FIRST_RUN_DIR, f'checkpoint_epoch_{source_epoch:02d}.pt'),
            map_location='cpu', weights_only=False
        )
        torch.save(source, os.path.join(RUN_DIR, target_name))

    best_b_path = os.path.join(
        RUN_DIR, f'validation_evaluation_epoch_{group_b["epoch"]:02d}.parquet'
    )
    best_b_frame = pd.read_parquet(best_b_path)
    smoothing_records = []
    smoothed_validation = {}
    for alpha in runtime['smoothing_alphas']:
        candidate = smooth_predictions(best_b_frame, alpha)
        metrics = evaluate_frame(candidate, 'smooth_pred')
        smoothing_records.append(OrderedDict([
            ('epoch', group_b['epoch']),
            ('alpha', alpha),
            ('rank_ic_mean', metrics['rank_ic']['mean']),
            ('icir', metrics['rank_ic']['icir']),
            ('annual_excess', metrics['top_decile']['annual_excess']),
            ('mean_turnover', metrics['turnover']['mean_turnover']),
            ('final_score', metrics['final_score'])
        ]))
        smoothed_validation[alpha] = candidate
    smoothing_table = pd.DataFrame(smoothing_records)
    smoothing_table.to_csv(
        os.path.join(RUN_DIR, '验证集平滑参数比较.csv'), index=False, encoding='utf-8-sig'
    )
    best_alpha = float(smoothing_table.sort_values(
        ['final_score', 'mean_turnover', 'rank_ic_mean', 'alpha'],
        ascending=[False, True, False, False]
    ).iloc[0]['alpha'])
    smoothed_validation[best_alpha].to_parquet(
        os.path.join(RUN_DIR, 'validation_predictions_best_smoothed.parquet'),
        index=False, compression='zstd'
    )

    local_data = read_evaluation_data('local_holdout')
    local_a_prediction = predict_checkpoint(
        os.path.join(FIRST_RUN_DIR, f'checkpoint_epoch_{group_a["epoch"]:02d}.pt'),
        'local_holdout', config, device, batch_size
    )
    local_b_prediction = predict_checkpoint(
        os.path.join(FIRST_RUN_DIR, f'checkpoint_epoch_{group_b["epoch"]:02d}.pt'),
        'local_holdout', config, device, batch_size
    )
    local_a = attach_evaluation(local_a_prediction, local_data)
    local_b = attach_evaluation(local_b_prediction, local_data)
    local_a_metrics = evaluate_frame(local_a)
    local_b_metrics = evaluate_frame(local_b)
    local_b_smoothed = smooth_predictions(local_b, best_alpha)
    local_b_smoothed_metrics = evaluate_frame(local_b_smoothed, 'smooth_pred')
    local_b.to_parquet(os.path.join(RUN_DIR, '本地留出集原始预测.parquet'), index=False, compression='zstd')
    local_b_smoothed.to_parquet(os.path.join(RUN_DIR, '本地留出集平滑预测.parquet'), index=False, compression='zstd')

    result = OrderedDict([
        ('group_a_best_epoch', group_a['epoch']),
        ('group_b_best_epoch', group_b['epoch']),
        ('best_alpha', best_alpha),
        ('validation_group_a', group_a),
        ('validation_group_b', group_b),
        ('local_group_a', local_a_metrics),
        ('local_group_b', local_b_metrics),
        ('local_group_b_smoothed', local_b_smoothed_metrics)
    ])
    with open(os.path.join(RUN_DIR, '本地留出集比赛指标评估.json'), 'w', encoding='utf-8') as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    report = [
        '# 第二次综合分验证实验报告', '',
        f'- 实验组 A 按验证集 Rank IC 选择周期：第 {group_a["epoch"]} 周期。',
        f'- 实验组 B 按验证集综合分选择周期：第 {group_b["epoch"]} 周期。',
        f'- 验证集选择的平滑参数：alpha = {best_alpha:g}。', '',
        '## 验证集结果', '',
        '| 实验组 | 周期 | Rank IC | 年化超额收益 | 平均换手率 | 综合分 |',
        '|---|---:|---:|---:|---:|---:|',
        f'| A：Rank IC 选模 | {group_a["epoch"]} | {group_a["validation_rank_ic_mean"]:.8f} | {group_a["validation_annual_excess"]:.8f} | {group_a["validation_mean_turnover"]:.8f} | {group_a["validation_final_score"]:.8f} |',
        f'| B：综合分选模 | {group_b["epoch"]} | {group_b["validation_rank_ic_mean"]:.8f} | {group_b["validation_annual_excess"]:.8f} | {group_b["validation_mean_turnover"]:.8f} | {group_b["validation_final_score"]:.8f} |',
        '', '## 本地留出集结果', '',
        '| 实验组 | Rank IC | 年化超额收益 | 平均换手率 | 综合分 |',
        '|---|---:|---:|---:|---:|',
        f'| A：Rank IC 选模 | {local_a_metrics["rank_ic"]["mean"]:.8f} | {local_a_metrics["top_decile"]["annual_excess"]:.8f} | {local_a_metrics["turnover"]["mean_turnover"]:.8f} | {local_a_metrics["final_score"]:.8f} |',
        f'| B：综合分选模 | {local_b_metrics["rank_ic"]["mean"]:.8f} | {local_b_metrics["top_decile"]["annual_excess"]:.8f} | {local_b_metrics["turnover"]["mean_turnover"]:.8f} | {local_b_metrics["final_score"]:.8f} |',
        f'| C：B 加平滑 alpha={best_alpha:g} | {local_b_smoothed_metrics["rank_ic"]["mean"]:.8f} | {local_b_smoothed_metrics["top_decile"]["annual_excess"]:.8f} | {local_b_smoothed_metrics["turnover"]["mean_turnover"]:.8f} | {local_b_smoothed_metrics["final_score"]:.8f} |',
        '',
        '每周期预测、评价表、检查点、平滑参数比较和完整指标见本目录中的 Parquet、CSV、PT 与 JSON 文件。'
    ]
    with open(os.path.join(RUN_DIR, '实验报告.md'), 'w', encoding='utf-8') as file:
        file.write('\n'.join(report) + '\n')


if __name__ == '__main__':
    main()
