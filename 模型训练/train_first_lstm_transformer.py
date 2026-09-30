import argparse
import json
import os
import random
import sys
import time
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from torch import nn

BASE = r'E:\ChatGPT\ChatGPT_Project\深圳金融比赛赛题五'
STEP7_DIR = os.path.join(BASE, '模型预处理', '步骤7_模型输入接口')
MODEL_DIR = os.path.join(BASE, '模型训练')
RUN_DIR = os.path.join(MODEL_DIR, '第一次LST-Transformer实验')
CONFIG_PATH = os.path.join(MODEL_DIR, '第一次LST-Transformer实验配置.json')
sys.path.insert(0, STEP7_DIR)
sys.path.insert(0, MODEL_DIR)
from sequence_input_interface import SequenceInputReader
from lstm_transformer_small import LSTMTransformerRegressor


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rank_ic_mean(predictions, targets, dates):
    frame = pd.DataFrame({
        'trade_date': dates,
        'prediction': predictions,
        'target': targets
    })
    daily_values = []
    for _, group in frame.groupby('trade_date', sort=True):
        if len(group) < 30:
            continue
        prediction_rank = group['prediction'].rank(method='average').to_numpy()
        target_rank = group['target'].rank(method='average').to_numpy()
        prediction_std = prediction_rank.std()
        target_std = target_rank.std()
        if prediction_std == 0 or target_std == 0:
            continue
        daily_values.append(float(np.corrcoef(prediction_rank, target_rank)[0, 1]))
    if not daily_values:
        raise ValueError('没有足够的交易日计算Rank IC')
    return float(np.mean(daily_values)), float(np.std(daily_values, ddof=1)), len(daily_values)


def run_train_epoch(model, reader, optimizer, loss_function, device, clip_norm, max_batches=None):
    model.train()
    total_loss = 0.0
    total_rows = 0
    batch_count = 0
    start_time = time.perf_counter()
    for batch in reader.iter_batches():
        x = torch.from_numpy(batch['x']).to(device)
        y = torch.from_numpy(batch['y']).to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x)
        loss = loss_function(prediction, y)
        if not torch.isfinite(loss):
            raise ValueError('训练损失出现非有限值')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        rows = int(x.shape[0])
        total_loss += float(loss.detach().item()) * rows
        total_rows += rows
        batch_count += 1
        if max_batches is not None and batch_count >= max_batches:
            break
    if total_rows == 0:
        raise ValueError('训练集没有可用样本')
    elapsed = time.perf_counter() - start_time
    return OrderedDict([
        ('loss', total_loss / total_rows),
        ('rows', total_rows),
        ('batches', batch_count),
        ('seconds', elapsed),
        ('rows_per_second', total_rows / elapsed)
    ])


def evaluate_model(model, reader, loss_function, device, max_batches=None):
    model.eval()
    losses = []
    rows = []
    with torch.no_grad():
        for batch_index, batch in enumerate(reader.iter_batches(), start=1):
            x = torch.from_numpy(batch['x']).to(device)
            y = torch.from_numpy(batch['y']).to(device)
            prediction = model(x)
            loss = loss_function(prediction, y)
            losses.append(float(loss.detach().item()) * len(y))
            rows.append({
                'ts_code': batch['ts_code'],
                'trade_date': batch['target_date'],
                'prediction': prediction.cpu().numpy(),
                'target': batch['y']
            })
            if max_batches is not None and batch_index >= max_batches:
                break
    if not rows:
        raise ValueError('评估集没有可用样本')
    dates = np.concatenate([item['trade_date'] for item in rows])
    ts_codes = np.concatenate([item['ts_code'] for item in rows])
    predictions = np.concatenate([item['prediction'] for item in rows])
    targets = np.concatenate([item['target'] for item in rows])
    rank_ic, rank_ic_std, rank_ic_days = rank_ic_mean(predictions, targets, dates)
    return OrderedDict([
        ('loss', float(sum(losses) / len(targets))),
        ('rows', int(len(targets))),
        ('daily_rank_ic_mean', rank_ic),
        ('daily_rank_ic_std', rank_ic_std),
        ('rank_ic_days', rank_ic_days),
        ('predictions', predictions),
        ('targets', targets),
        ('dates', dates),
        ('ts_codes', ts_codes)
    ])


def write_checkpoint(path, model, optimizer, epoch, validation):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'validation': dict(validation)
    }
    torch.save(checkpoint, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--max-train-batches', type=int, default=None)
    parser.add_argument('--max-validation-batches', type=int, default=None)
    parser.add_argument('--skip-final-eval', action='store_true')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--torch-threads', type=int, default=None)
    args = parser.parse_args()

    with open(CONFIG_PATH, 'r', encoding='utf-8') as file:
        config = json.load(file)
    seed = int(config['optimization']['random_seed'])
    set_seed(seed)
    if args.torch_threads is not None:
        if args.torch_threads <= 0:
            raise ValueError('torch-threads 必须为正整数')
        torch.set_num_threads(args.torch_threads)
        torch.set_num_interop_threads(args.torch_threads)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('请求使用CUDA，但当前环境不可用')
    device = torch.device(args.device)
    batch_size = args.batch_size or int(config['input']['batch_size'])
    epochs = args.epochs or int(config['optimization']['max_epochs'])
    os.makedirs(RUN_DIR, exist_ok=True)
    with open(os.path.join(RUN_DIR, '运行配置.json'), 'w', encoding='utf-8') as file:
        runtime_config = dict(config)
        runtime_config['runtime'] = {
            'batch_size': batch_size,
            'epochs': epochs,
            'device': str(device),
            'torch_threads': args.torch_threads,
            'max_train_batches': args.max_train_batches,
            'max_validation_batches': args.max_validation_batches
        }
        json.dump(runtime_config, file, ensure_ascii=False, indent=2)

    model = LSTMTransformerRegressor(
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
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['optimization']['learning_rate'],
        weight_decay=config['optimization']['weight_decay']
    )
    loss_function = nn.SmoothL1Loss(beta=0.02)
    train_reader = SequenceInputReader(
        STEP7_DIR, 'train_fit', batch_size=batch_size, eligible_only=True
    )
    validation_reader = SequenceInputReader(
        STEP7_DIR, 'validation', batch_size=batch_size, eligible_only=True
    )
    history = []
    best_metric = -np.inf
    best_epoch = None
    patience_count = 0
    log_path = os.path.join(RUN_DIR, '训练日志.csv')
    for epoch in range(1, epochs + 1):
        train_result = run_train_epoch(
            model, train_reader, optimizer, loss_function, device,
            config['optimization']['gradient_clip_norm'], args.max_train_batches
        )
        validation_result = evaluate_model(
            model, validation_reader, loss_function, device
            , args.max_validation_batches
        )
        record = OrderedDict([
            ('epoch', epoch),
            ('train_loss', train_result['loss']),
            ('train_rows', train_result['rows']),
            ('train_batches', train_result['batches']),
            ('train_seconds', train_result['seconds']),
            ('train_rows_per_second', train_result['rows_per_second']),
            ('validation_loss', validation_result['loss']),
            ('validation_daily_rank_ic_mean', validation_result['daily_rank_ic_mean']),
            ('validation_daily_rank_ic_std', validation_result['daily_rank_ic_std']),
            ('validation_rank_ic_days', validation_result['rank_ic_days'])
        ])
        history.append(record)
        pd.DataFrame(history).to_csv(log_path, index=False, encoding='utf-8-sig')
        write_checkpoint(
            os.path.join(RUN_DIR, f'checkpoint_epoch_{epoch:02d}.pt'),
            model, optimizer, epoch, record
        )
        metric = validation_result['daily_rank_ic_mean']
        if metric > best_metric:
            best_metric = metric
            best_epoch = epoch
            patience_count = 0
            write_checkpoint(
                os.path.join(RUN_DIR, '最佳模型.pt'),
                model, optimizer, epoch, record
            )
        else:
            patience_count += 1
        print(json.dumps(record, ensure_ascii=False))
        if patience_count >= config['optimization']['early_stopping_patience']:
            break

    if best_epoch is None:
        raise ValueError('没有生成最佳模型')
    if args.skip_final_eval:
        return
    best_checkpoint = torch.load(
        os.path.join(RUN_DIR, '最佳模型.pt'), map_location=device, weights_only=False
    )
    model.load_state_dict(best_checkpoint['model_state_dict'])
    local_reader = SequenceInputReader(
        STEP7_DIR, 'local_holdout', batch_size=batch_size, eligible_only=True
    )
    local_result = evaluate_model(model, local_reader, loss_function, device)
    local_frame = pd.DataFrame({
        'ts_code': local_result['ts_codes'],
        'trade_date': local_result['dates'],
        'pred': local_result['predictions'],
        'y_ret_1d': local_result['targets']
    })
    local_frame.to_parquet(
        os.path.join(RUN_DIR, 'local_holdout_predictions.parquet'),
        index=False, compression='zstd'
    )
    result = OrderedDict([
        ('best_epoch', best_epoch),
        ('best_validation_daily_rank_ic_mean', best_metric),
        ('local_holdout', OrderedDict([
            ('loss', local_result['loss']),
            ('rows', local_result['rows']),
            ('daily_rank_ic_mean', local_result['daily_rank_ic_mean']),
            ('daily_rank_ic_std', local_result['daily_rank_ic_std']),
            ('rank_ic_days', local_result['rank_ic_days'])
        ])),
        ('history', history),
        ('device', str(device)),
        ('batch_size', batch_size),
        ('epochs_completed', len(history))
    ])
    with open(os.path.join(RUN_DIR, '训练结果.json'), 'w', encoding='utf-8') as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    report_lines = [
        '# 第一次 LST-Transformer 实际训练报告',
        '',
        f"- 运行设备：`{device}`。",
        f'- 批量大小：`{batch_size}`。',
        f'- 完成周期：{len(history)}。',
        f'- 最佳周期：{best_epoch}。',
        f"- 最佳验证集日 Rank IC 均值：{best_metric:.10g}。",
        f"- 本地留出集日 Rank IC 均值：{local_result['daily_rank_ic_mean']:.10g}。",
        f"- 本地留出集 Smooth L1 损失：{local_result['loss']:.10g}。",
        '',
        '详细训练过程见 `训练日志.csv`，最佳模型见 `最佳模型.pt`，本地留出集预测见 `local_holdout_predictions.parquet`。'
    ]
    with open(os.path.join(RUN_DIR, '训练报告.md'), 'w', encoding='utf-8') as file:
        file.write('\n'.join(report_lines) + '\n')


if __name__ == '__main__':
    main()
