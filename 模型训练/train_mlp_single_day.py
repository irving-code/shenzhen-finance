import json
import os
import random
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from scipy.stats import rankdata
from torch import nn


BASE = Path(r'E:\ChatGPT\ChatGPT_Project\深圳金融比赛赛题五')
STEP5_DIR = BASE / '模型预处理' / '步骤5_标准化处理'
STEP7_DIR = BASE / '模型预处理' / '步骤7_模型输入接口'
OUTPUT_DIR = BASE / '模型训练' / 'MLP实验'
BATCH_SIZE = 4096
PARQUET_BATCH_SIZE = 200_000
SEED = 42

SPLIT_FILES = OrderedDict([
    ('train_fit', STEP5_DIR / '训练集_train_fit_标准化.parquet'),
    ('validation', STEP5_DIR / '验证集_标准化.parquet'),
    ('local_holdout', STEP5_DIR / '本地留出集_标准化.parquet'),
    ('competition_test', STEP5_DIR / '测试集_标准化.parquet')
])

with open(STEP7_DIR / '模型输入接口元数据.json', 'r', encoding='utf-8') as file:
    FEATURE_COLUMNS = json.load(file)['feature_columns']


class SingleDayMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def frame_columns(has_label):
    columns = [
        'ts_code', 'trade_date', 'sample_eligible', 'limit_up'
    ] + FEATURE_COLUMNS
    if has_label:
        columns.append('y_ret_1d')
    return columns


def iter_data(path, has_label, batch_size=PARQUET_BATCH_SIZE):
    parquet_file = pq.ParquetFile(path)
    columns = frame_columns(has_label)
    for record_batch in parquet_file.iter_batches(
        columns=columns,
        batch_size=batch_size
    ):
        frame = record_batch.to_pandas()
        frame = frame.loc[frame['sample_eligible'].eq(1)].reset_index(drop=True)
        if len(frame) == 0:
            continue
        features = frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        if not np.isfinite(features).all():
            raise ValueError(f'存在非有限输入因子: {path}')
        if has_label:
            labels = frame['y_ret_1d'].to_numpy(dtype=np.float32)
            if not np.isfinite(labels).all():
                raise ValueError(f'存在非有限标签: {path}')
        else:
            labels = None
        yield frame, features, labels


def iter_torch_batches(path, has_label, shuffle, device):
    for frame, features, labels in iter_data(path, has_label):
        order = np.arange(len(features))
        if shuffle:
            np.random.shuffle(order)
        for start in range(0, len(order), BATCH_SIZE):
            selected = order[start:start + BATCH_SIZE]
            x = torch.from_numpy(features[selected]).to(device)
            batch = {
                'x': x,
                'frame': frame.iloc[selected].reset_index(drop=True)
            }
            if labels is not None:
                batch['y'] = torch.from_numpy(labels[selected]).to(device)
            yield batch


def train_epoch(model, path, optimizer, loss_function, device):
    model.train()
    loss_sum = 0.0
    row_count = 0
    for batch in iter_torch_batches(path, True, True, device):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch['x'])
        loss = loss_function(prediction, batch['y'])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        rows = len(batch['y'])
        loss_sum += float(loss.detach().item()) * rows
        row_count += rows
    if row_count == 0:
        raise ValueError(f'训练分区没有有效样本: {path}')
    return loss_sum / row_count, row_count


def predict_split(model, path, device):
    model.eval()
    predictions = []
    targets = []
    dates = []
    codes = []
    limit_up = []
    losses = []
    loss_function = nn.SmoothL1Loss(beta=0.02)
    with torch.no_grad():
        for batch in iter_torch_batches(path, True, False, device):
            prediction = model(batch['x'])
            losses.append(
                float(loss_function(prediction, batch['y']).item()) * len(batch['y'])
            )
            predictions.append(prediction.cpu().numpy())
            targets.append(batch['y'].cpu().numpy())
            dates.append(batch['frame']['trade_date'].to_numpy(dtype=np.int32))
            codes.append(batch['frame']['ts_code'].astype(str).to_numpy())
            limit_up.append(batch['frame']['limit_up'].to_numpy(dtype=np.float32))
    if not predictions:
        raise ValueError(f'评估分区没有有效样本: {path}')
    target_values = np.concatenate(targets)
    return {
        'predictions': np.concatenate(predictions),
        'targets': target_values,
        'dates': np.concatenate(dates),
        'codes': np.concatenate(codes),
        'limit_up': np.concatenate(limit_up),
        'loss': float(sum(losses) / len(target_values))
    }


def predict_test(model, path, device):
    model.eval()
    predictions = []
    dates = []
    codes = []
    with torch.no_grad():
        for batch in iter_torch_batches(path, False, False, device):
            predictions.append(model(batch['x']).cpu().numpy())
            dates.append(batch['frame']['trade_date'].to_numpy(dtype=np.int32))
            codes.append(batch['frame']['ts_code'].astype(str).to_numpy())
    if not predictions:
        raise ValueError(f'测试分区没有有效样本: {path}')
    return {
        'predictions': np.concatenate(predictions),
        'dates': np.concatenate(dates),
        'codes': np.concatenate(codes)
    }


def daily_metrics(result):
    frame = pd.DataFrame({
        'trade_date': result['dates'],
        'prediction': result['predictions'],
        'target': result['targets'],
        'ts_code': result['codes'],
        'limit_up': result['limit_up']
    })
    rank_ics = []
    excess_returns = []
    top_sets = []
    for trade_date, group in frame.groupby('trade_date', sort=True):
        valid = group[np.isfinite(group['prediction']) & np.isfinite(group['target'])]
        if len(valid) >= 30:
            prediction_rank = rankdata(valid['prediction'].to_numpy(dtype=float))
            target_rank = rankdata(valid['target'].to_numpy(dtype=float))
            if prediction_rank.std() > 0 and target_rank.std() > 0:
                rank_ics.append(float(np.corrcoef(prediction_rank, target_rank)[0, 1]))
        investable = group[
            (group['limit_up'] == 0)
            & np.isfinite(group['prediction'])
            & np.isfinite(group['target'])
        ].sort_values('prediction', ascending=False)
        if len(investable) >= 100:
            top_count = max(len(investable) // 10, 1)
            excess_returns.append(
                float(
                    investable['target'].iloc[:top_count].mean()
                    - investable['target'].mean()
                )
            )
            top_sets.append(set(investable['ts_code'].iloc[:top_count]))
    if not rank_ics:
        raise ValueError('没有足够的交易日计算 Rank IC')
    turnovers = []
    for previous, current in zip(top_sets, top_sets[1:]):
        union = previous | current
        if union:
            turnovers.append(1.0 - len(previous & current) / len(union))
    rank_ic_mean = float(np.mean(rank_ics))
    rank_ic_std = float(np.std(rank_ics, ddof=1)) if len(rank_ics) > 1 else 0.0
    annual_excess = float(np.mean(excess_returns) * 252) if excess_returns else 0.0
    mean_turnover = float(np.mean(turnovers)) if turnovers else 0.0
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


def save_prediction(path, result, include_target):
    output = pd.DataFrame({
        'ts_code': result['codes'],
        'trade_date': result['dates'],
        'pred': result['predictions'].astype(np.float32)
    })
    if include_target:
        output['y_ret_1d'] = result['targets']
    output.to_parquet(path, index=False, compression='zstd')


def main():
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SingleDayMLP(len(FEATURE_COLUMNS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    loss_function = nn.SmoothL1Loss(beta=0.02)
    train_path = SPLIT_FILES['train_fit']
    validation_path = SPLIT_FILES['validation']
    started = time.perf_counter()
    best_validation = None
    best_epoch = None
    history = []
    patience = 4
    patience_count = 0
    for epoch in range(1, 21):
        train_loss, train_rows = train_epoch(
            model, train_path, optimizer, loss_function, device
        )
        validation = predict_split(model, validation_path, device)
        validation_metrics = daily_metrics(validation)
        record = OrderedDict([
            ('epoch', epoch),
            ('train_loss', train_loss),
            ('train_rows', train_rows),
            ('validation_loss', validation['loss']),
            *[(f'validation_{key}', value) for key, value in validation_metrics.items()]
        ])
        history.append(record)
        pd.DataFrame(history).to_csv(
            OUTPUT_DIR / '训练日志.csv', index=False, encoding='utf-8-sig'
        )
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'validation': dict(validation_metrics),
                'feature_columns': FEATURE_COLUMNS
            },
            OUTPUT_DIR / f'checkpoint_epoch_{epoch:02d}.pt'
        )
        if best_validation is None or (
            validation_metrics['daily_rank_ic_mean'],
            -validation['loss']
        ) > (
            best_validation['daily_rank_ic_mean'],
            -best_validation['loss']
        ):
            best_validation = {
                **validation_metrics,
                'loss': validation['loss'],
                'epoch': epoch
            }
            best_epoch = epoch
            patience_count = 0
            torch.save(model.state_dict(), OUTPUT_DIR / '最佳模型.pt')
        else:
            patience_count += 1
        print(json.dumps({'event': 'epoch_end', **record}, ensure_ascii=False))
        if patience_count >= patience:
            break
    if best_epoch is None:
        raise ValueError('没有确定最佳模型')
    model.load_state_dict(torch.load(OUTPUT_DIR / '最佳模型.pt', map_location=device))
    validation = predict_split(model, validation_path, device)
    local = predict_split(model, SPLIT_FILES['local_holdout'], device)
    local_metrics = daily_metrics(local)
    test = predict_test(model, SPLIT_FILES['competition_test'], device)
    if not np.isfinite(test['predictions']).all():
        raise ValueError('测试预测存在非有限值')
    save_prediction(OUTPUT_DIR / '验证集预测.parquet', validation, True)
    save_prediction(OUTPUT_DIR / '本地留出集预测.parquet', local, True)
    test_output = pd.DataFrame({
        'ts_code': test['codes'],
        'trade_date': test['dates'],
        'pred': test['predictions'].astype(np.float32)
    })
    test_output.to_parquet(OUTPUT_DIR / '测试集预测.parquet', index=False, compression='zstd')
    test_output.to_csv(OUTPUT_DIR / 'submission.csv', index=False, encoding='utf-8-sig')
    config = OrderedDict([
        ('experiment_name', '单日 27 因子 MLP 实验'),
        ('feature_columns', FEATURE_COLUMNS),
        ('input_shape', [len(FEATURE_COLUMNS)]),
        ('data_source', {key: str(value) for key, value in SPLIT_FILES.items()}),
        ('sample_rule', 'sample_eligible == 1'),
        ('model', {
            'hidden_layers': [128, 64],
            'activation': 'GELU',
            'dropout': 0.1,
            'loss': 'SmoothL1Loss(beta=0.02)',
            'optimizer': 'AdamW',
            'learning_rate': 3e-4,
            'weight_decay': 1e-4,
            'max_epochs': 20,
            'early_stopping_patience': patience,
            'seed': SEED
        }),
        ('best_epoch', best_epoch),
        ('best_validation', best_validation),
        ('local_holdout', local_metrics),
        ('test_rows', int(len(test['predictions']))),
        ('device', str(device)),
        ('seconds', time.perf_counter() - started)
    ])
    with open(OUTPUT_DIR / '配置.json', 'w', encoding='utf-8') as file:
        json.dump(config, file, ensure_ascii=False, indent=2)
    report = [
        '# 单日 27 因子 MLP 实验报告',
        '',
        f'- 输入特征数：`{len(FEATURE_COLUMNS)}`。',
        f'- 最佳周期：`{best_epoch}`。',
        f'- 运行设备：`{device}`。',
        '',
        '## 验证集结果',
        '',
        f"- 每日 Rank IC 均值：`{best_validation['daily_rank_ic_mean']:.10g}`。",
        f"- ICIR：`{best_validation['icir']:.10g}`。",
        f"- Top 10% 年化超额收益：`{best_validation['annual_top_decile_excess']:.10g}`。",
        f"- 平均换手率：`{best_validation['mean_turnover']:.10g}`。",
        f"- 综合分数：`{best_validation['official_score']:.10g}`。",
        '',
        '## 本地留出集结果',
        '',
        f"- 每日 Rank IC 均值：`{local_metrics['daily_rank_ic_mean']:.10g}`。",
        f"- ICIR：`{local_metrics['icir']:.10g}`。",
        f"- Top 10% 年化超额收益：`{local_metrics['annual_top_decile_excess']:.10g}`。",
        f"- 平均换手率：`{local_metrics['mean_turnover']:.10g}`。",
        f"- 综合分数：`{local_metrics['official_score']:.10g}`。",
        '',
        '## 输出文件',
        '',
        '- `最佳模型.pt`',
        '- `验证集预测.parquet`',
        '- `本地留出集预测.parquet`',
        '- `测试集预测.parquet`',
        '- `submission.csv`'
    ]
    (OUTPUT_DIR / '实验报告.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    print(json.dumps({
        'event': 'complete',
        'output_dir': str(OUTPUT_DIR),
        'best_epoch': best_epoch,
        'validation_rank_ic': best_validation['daily_rank_ic_mean'],
        'holdout_rank_ic': local_metrics['daily_rank_ic_mean'],
        'test_rows': len(test['predictions']),
        'seconds': time.perf_counter() - started
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
