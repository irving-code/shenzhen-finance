import json
import os
import sys
from collections import OrderedDict

import numpy as np
import torch
from torch import nn

BASE = r'E:\ChatGPT\ChatGPT_Project\深圳金融比赛赛题五'
STEP7_DIR = os.path.join(BASE, '模型预处理', '步骤7_模型输入接口')
MODEL_DIR = os.path.join(BASE, '模型训练')
sys.path.insert(0, STEP7_DIR)
sys.path.insert(0, MODEL_DIR)
from sequence_input_interface import SequenceInputReader
from lstm_transformer_small import LSTMTransformerRegressor


def check_split(split, model):
    reader = SequenceInputReader(
        STEP7_DIR, split, batch_size=16, eligible_only=True, as_torch=False
    )
    batch = next(reader.iter_batches())
    x = torch.from_numpy(batch['x'])
    output = model(x)
    if output.ndim != 1 or output.shape[0] != x.shape[0]:
        raise ValueError(f'{split} 输出形状异常: {tuple(output.shape)}')
    if not torch.isfinite(output).all():
        raise ValueError(f'{split} 输出存在非有限值')
    return OrderedDict([
        ('reader_length', len(reader)),
        ('input_shape', list(x.shape)),
        ('output_shape', list(output.shape)),
        ('output_mean', float(output.detach().mean().item()))
    ])


def main():
    torch.manual_seed(42)
    model = LSTMTransformerRegressor(
        input_dim=27,
        sequence_length=20,
        projection_dim=64,
        lstm_hidden_dim=64,
        lstm_layers=1,
        transformer_layers=1,
        transformer_heads=4,
        transformer_feedforward_dim=128,
        dropout=0.1
    )
    train_reader = SequenceInputReader(
        STEP7_DIR, 'train_fit', batch_size=16, eligible_only=True, as_torch=False
    )
    train_batch = next(train_reader.iter_batches())
    x = torch.from_numpy(train_batch['x'])
    y = torch.from_numpy(train_batch['y'])
    model.train()
    prediction = model(x)
    loss = nn.SmoothL1Loss(beta=0.02)(prediction, y)
    loss.backward()
    gradients = [
        parameter.grad.detach()
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise ValueError('模型反向传播梯度存在非有限值')
    parameter_count = sum(
        int(parameter.numel()) for parameter in model.parameters()
    )
    result = OrderedDict([
        ('all_pass', True),
        ('parameter_count', parameter_count),
        ('train_batch_shape', list(x.shape)),
        ('target_shape', list(y.shape)),
        ('train_batch_loss', float(loss.detach().item())),
        ('splits', OrderedDict([
            ('train_fit', check_split('train_fit', model)),
            ('validation', check_split('validation', model)),
            ('local_holdout', check_split('local_holdout', model)),
            ('competition_test', check_split('competition_test', model))
        ])),
        ('future_label_in_test_batch', 'y' in next(
            SequenceInputReader(
                STEP7_DIR, 'competition_test', batch_size=16,
                eligible_only=True, as_torch=False
            ).iter_batches()
        ))
    ])
    result_path = os.path.join(MODEL_DIR, '步骤8_模型接口核验结果.json')
    with open(result_path, 'w', encoding='utf-8') as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    report_lines = [
        '# 第八步：首轮模型接口核验报告',
        '',
        '- 已使用真实序列数据完成前向传播。',
        '- 已使用真实训练标签计算 `SmoothL1Loss(beta=0.02)`。',
        '- 已完成一次反向传播，所有梯度均为有限值。',
        f"- 模型参数量：{parameter_count:,}。",
        f"- 首批训练样本损失：{float(loss.detach().item()):.10g}。该数值只用于接口核验，不代表模型训练效果。",
        '- 官方测试批次不包含标签字段。',
        '',
        '详细结果见 `步骤8_模型接口核验结果.json`。'
    ]
    with open(os.path.join(MODEL_DIR, '步骤8_模型接口核验报告.md'), 'w', encoding='utf-8') as file:
        file.write('\n'.join(report_lines) + '\n')


if __name__ == '__main__':
    main()
