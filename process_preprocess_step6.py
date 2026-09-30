import json
import os
from collections import OrderedDict, deque

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

BASE = r'E:\ChatGPT\ChatGPT_Project\深圳金融比赛赛题五'
INPUT_DIR = os.path.join(BASE, '模型预处理', '步骤5_标准化处理')
OUTPUT_DIR = os.path.join(BASE, '模型预处理', '步骤6_序列构造')
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 200_000
WINDOW_LENGTH = 20
HISTORY_LENGTH = WINDOW_LENGTH - 1

CONTINUOUS_COLUMNS = [
    'ret_1d', 'mom_3d', 'mom_5d', 'mom_10d', 'mom_20d', 'mom_20d_skip1',
    'volatility_5d', 'volatility_20d', 'range_1d', 'gap_1d',
    'true_range_1d', 'intraday_ret', 'close_position', 'body_ratio',
    'upper_shadow', 'lower_shadow', 'log_amount', 'log_volume',
    'delta_log_amount', 'delta_log_volume', 'amount_shock_5d',
    'amount_shock_20d'
]
DISCRETE_COLUMNS = [
    'body_direction', 'limit_up', 'limit_down', 'missing_price_volume'
]
MODEL_FEATURE_COLUMNS = CONTINUOUS_COLUMNS + DISCRETE_COLUMNS + [
    'history_available_ratio_20d'
]
META_COLUMNS = [
    'ts_code', 'trade_date', 'history_available_count_20d',
    'history_available_ratio_20d', 'inside_valid_interval',
    'current_market_valid', 'sample_eligible'
]

INPUTS = OrderedDict([
    ('train_fit', os.path.join(INPUT_DIR, '训练集_train_fit_标准化.parquet')),
    ('validation', os.path.join(INPUT_DIR, '验证集_标准化.parquet')),
    ('local_holdout', os.path.join(INPUT_DIR, '本地留出集_标准化.parquet')),
    ('competition_test', os.path.join(INPUT_DIR, '测试集_标准化.parquet'))
])
INDEX_NAMES = {
    'train_fit': '序列索引_train_fit.parquet',
    'validation': '序列索引_validation.parquet',
    'local_holdout': '序列索引_local_holdout.parquet',
    'competition_test': '序列索引_competition_test.parquet'
}
LABELED_SPLITS = {'train_fit', 'validation', 'local_holdout'}


def create_sequence_frame(split, frame, global_start, source_start, stock_states):
    codes = frame['ts_code'].astype(str).to_numpy()
    dates = pd.to_numeric(frame['trade_date'], errors='raise').to_numpy(dtype=np.int64)
    history_count = pd.to_numeric(
        frame['history_available_count_20d'], errors='raise'
    ).to_numpy(dtype=np.float64)
    history_ratio = pd.to_numeric(
        frame['history_available_ratio_20d'], errors='raise'
    ).to_numpy(dtype=np.float64)
    inside = pd.to_numeric(
        frame['inside_valid_interval'], errors='raise'
    ).to_numpy(dtype=np.int64)
    current_market = pd.to_numeric(
        frame['current_market_valid'], errors='raise'
    ).to_numpy(dtype=np.int64)
    sample_eligible = pd.to_numeric(
        frame['sample_eligible'], errors='raise'
    ).to_numpy(dtype=np.int64)
    feature_values = frame.loc[:, MODEL_FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    feature_finite = np.isfinite(feature_values).all(axis=1)
    labels = None
    if 'y_ret_1d' in frame.columns:
        labels = pd.to_numeric(frame['y_ret_1d'], errors='raise').to_numpy(dtype=np.float64)

    source_rows = np.arange(
        source_start, source_start + len(frame), dtype=np.int64
    )
    start_panel_rows = []
    start_stock_rows = []
    end_rows = []
    start_dates = []
    available = np.zeros(len(frame), dtype=np.int8)
    sequence_finite = np.zeros(len(frame), dtype=np.int8)
    sequence_inside = np.zeros(len(frame), dtype=np.int8)
    prediction_eligible = np.zeros(len(frame), dtype=np.int8)
    stock_row_number = np.zeros(len(frame), dtype=np.int64)
    panel_rows = []

    global_row = global_start
    for i, code in enumerate(codes):
        date_value = int(dates[i])
        stock_state = stock_states.get(code)
        if stock_state is None:
            stock_state = {
                'history': deque(maxlen=HISTORY_LENGTH),
                'inside_sum': 0,
                'finite_sum': 0,
                'stock_position': 0,
                'last_date': -1
            }
            stock_states[code] = stock_state
        elif date_value <= stock_state['last_date']:
            raise ValueError(
                f'同一支股票的交易日期不是严格递增: ts_code={code}, '
                f'previous={stock_state["last_date"]}, current={date_value}'
            )

        history = stock_state['history']
        inside_sum = stock_state['inside_sum']
        finite_sum = stock_state['finite_sum']
        stock_position = stock_state['stock_position']
        stock_row_number[i] = stock_position
        has_window = len(history) == HISTORY_LENGTH
        if has_window:
            first_row = history[0]
            start_panel_rows.append(first_row[0])
            start_stock_rows.append(first_row[2])
            start_dates.append(first_row[1])
            available[i] = 1
            sequence_inside[i] = int(inside_sum + int(inside[i]) == WINDOW_LENGTH)
            sequence_finite[i] = int(finite_sum + int(feature_finite[i]) == WINDOW_LENGTH)
        else:
            start_panel_rows.append(None)
            start_stock_rows.append(None)
            start_dates.append(None)

        if split in LABELED_SPLITS:
            prediction_eligible[i] = int(
                sample_eligible[i] == 1
                and available[i] == 1
                and sequence_inside[i] == 1
                and sequence_finite[i] == 1
            )
        else:
            prediction_eligible[i] = int(
                available[i] == 1 and sequence_finite[i] == 1
            )

        end_rows.append(global_row)
        panel_rows.append({
            'panel_row_id': global_row,
            'source_split': split,
            'source_row_index': int(source_rows[i]),
            'ts_code': code,
            'trade_date': date_value,
            'stock_row_number': int(stock_position),
            'sample_eligible': int(sample_eligible[i]),
            'inside_valid_interval': int(inside[i]),
            'current_market_valid': int(current_market[i])
        })

        if len(history) == HISTORY_LENGTH:
            old = history.popleft()
            inside_sum -= old[3]
            finite_sum -= old[4]
        history.append((
            global_row, date_value, stock_position,
            int(inside[i]), int(feature_finite[i])
        ))
        inside_sum += int(inside[i])
        finite_sum += int(feature_finite[i])
        stock_state['inside_sum'] = inside_sum
        stock_state['finite_sum'] = finite_sum
        stock_state['last_date'] = date_value
        stock_state['stock_position'] = stock_position + 1
        global_row += 1

    sequence_frame = pd.DataFrame({
        'source_split': split,
        'source_row_index': source_rows,
        'panel_row_id': np.asarray(end_rows, dtype=np.int64),
        'ts_code': codes,
        'target_date': dates,
        'window_start_panel_row_id': pd.array(start_panel_rows, dtype='Int64'),
        'window_end_panel_row_id': np.asarray(end_rows, dtype=np.int64),
        'window_start_stock_row_number': pd.array(start_stock_rows, dtype='Int64'),
        'window_end_stock_row_number': stock_row_number,
        'window_start_date': pd.array(start_dates, dtype='Int64'),
        'window_end_date': dates,
        'target_stock_row_number': stock_row_number,
        'sequence_length': np.full(len(frame), WINDOW_LENGTH, dtype=np.int16),
        'sequence_available': available,
        'sequence_features_finite': sequence_finite,
        'sequence_inside_valid': sequence_inside,
        'sample_eligible': sample_eligible.astype(np.int8),
        'target_inside_valid_interval': inside.astype(np.int8),
        'target_current_market_valid': current_market.astype(np.int8),
        'history_available_count_20d': history_count,
        'history_available_ratio_20d': history_ratio,
        'prediction_eligible': prediction_eligible
    })
    if labels is not None:
        sequence_frame['y_ret_1d'] = labels
    return sequence_frame, pd.DataFrame(panel_rows), stock_states, global_row


def write_table(writer, frame):
    table = pa.Table.from_pandas(frame, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(
            frame.attrs['output_path'], table.schema, compression='zstd'
        )
    writer.write_table(table)
    return writer


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    panel_path = os.path.join(OUTPUT_DIR, '面板行索引.parquet')
    result_path = os.path.join(OUTPUT_DIR, '步骤6_序列构造结果.json')
    report_path = os.path.join(OUTPUT_DIR, '步骤6_序列构造报告.md')
    output_paths = {
        split: os.path.join(OUTPUT_DIR, name) for split, name in INDEX_NAMES.items()
    }

    panel_writer = None
    sequence_writers = {split: None for split in INPUTS}
    stock_states = {}
    stats = OrderedDict()
    global_row = 0
    for split, path in INPUTS.items():
        parquet_file = pq.ParquetFile(path)
        names = set(parquet_file.schema_arrow.names)
        required = set(META_COLUMNS + MODEL_FEATURE_COLUMNS)
        missing = sorted(required - names)
        if missing:
            raise ValueError(f'{split} 缺少必要字段: {missing}')
        read_columns = META_COLUMNS + MODEL_FEATURE_COLUMNS
        if 'y_ret_1d' in names:
            read_columns.append('y_ret_1d')
        split_stats = OrderedDict([
            ('rows', 0),
            ('sample_eligible', 0),
            ('sequence_available', 0),
            ('sequence_features_finite', 0),
            ('sequence_inside_valid', 0),
            ('prediction_eligible', 0),
            ('first_target_date', None),
            ('last_target_date', None)
        ])
        source_row_start = 0
        for record_batch in parquet_file.iter_batches(
            columns=read_columns, batch_size=BATCH_SIZE
        ):
            frame = record_batch.to_pandas()
            sequence_frame, panel_frame, stock_states, global_row = create_sequence_frame(
                split, frame, global_row, source_row_start, stock_states
            )
            source_row_start += len(frame)
            sequence_frame.attrs['output_path'] = output_paths[split]
            panel_frame.attrs['output_path'] = panel_path
            sequence_writers[split] = write_table(
                sequence_writers[split], sequence_frame
            )
            panel_writer = write_table(panel_writer, panel_frame)
            split_stats['rows'] += len(sequence_frame)
            split_stats['sample_eligible'] += int(sequence_frame['sample_eligible'].sum())
            split_stats['sequence_available'] += int(sequence_frame['sequence_available'].sum())
            split_stats['sequence_features_finite'] += int(
                sequence_frame['sequence_features_finite'].sum()
            )
            split_stats['sequence_inside_valid'] += int(
                sequence_frame['sequence_inside_valid'].sum()
            )
            split_stats['prediction_eligible'] += int(
                sequence_frame['prediction_eligible'].sum()
            )
            first_date = int(sequence_frame['target_date'].iloc[0])
            last_date = int(sequence_frame['target_date'].iloc[-1])
            if split_stats['first_target_date'] is None:
                split_stats['first_target_date'] = first_date
            split_stats['last_target_date'] = last_date
        stats[split] = split_stats

    if panel_writer is not None:
        panel_writer.close()
    for writer in sequence_writers.values():
        if writer is not None:
            writer.close()

    result = OrderedDict([
        ('window_length', WINDOW_LENGTH),
        ('history_length', HISTORY_LENGTH),
        ('model_feature_count', len(MODEL_FEATURE_COLUMNS)),
        ('model_feature_columns', MODEL_FEATURE_COLUMNS),
        ('panel_row_count', global_row),
        ('splits', stats),
        ('outputs', OrderedDict([
            ('panel_index', panel_path),
            ('sequence_indices', output_paths)
        ])),
        ('rules', OrderedDict([
            ('labeled_prediction_eligible', 'sample_eligible=1 and sequence_available=1 and sequence_inside_valid=1 and sequence_features_finite=1'),
            ('test_prediction_eligible', 'sequence_available=1 and sequence_features_finite=1'),
            ('cross_stock_window', False),
            ('original_factor_files_modified', False),
            ('step5_standardized_files_modified', False)
        ]))
    ])
    with open(result_path, 'w', encoding='utf-8') as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    lines = [
        '# 第六步：序列构造报告',
        '',
        '## 处理规则',
        '',
        '- 每个目标日使用同一支股票向前连续 20 条面板记录。',
        '- 序列特征共 27 个：22 个标准化连续因子、4 个离散因子和 `history_available_ratio_20d`。',
        '- 训练、验证和本地留出集要求原有样本合格、窗口完整、窗口内有效区间完整且输入因子有限。',
        '- 官方测试集保留全部目标日，只将窗口完整且输入因子有限的目标日标记为可预测。',
        '- 序列索引通过股票内行号、全局面板行号和来源文件行号连接第五步标准化文件，模型读取时按批提取窗口。',
        '',
        '## 分区统计',
        '',
        '| 分区 | 总行数 | 原有合格样本 | 完整窗口 | 窗口输入有限 | 窗口有效区间完整 | 可预测目标日 |',
        '|---|---:|---:|---:|---:|---:|---:|'
    ]
    for split, split_stats in stats.items():
        lines.append(
            f"| `{split}` | {split_stats['rows']:,} | {split_stats['sample_eligible']:,} | "
            f"{split_stats['sequence_available']:,} | {split_stats['sequence_features_finite']:,} | "
            f"{split_stats['sequence_inside_valid']:,} | {split_stats['prediction_eligible']:,} |"
        )
    lines.extend([
        '',
        '## 输出文件',
        '',
        '- `序列索引_train_fit.parquet`',
        '- `序列索引_validation.parquet`',
        '- `序列索引_local_holdout.parquet`',
        '- `序列索引_competition_test.parquet`',
        '- `面板行索引.parquet`',
        '- `步骤6_序列构造结果.json`',
        '',
        '## 读取接口',
        '',
        '- `window_start_stock_row_number` 和 `window_end_stock_row_number` 表示同一支股票内 20 条连续面板行的起止位置。',
        '- `window_start_panel_row_id` 和 `window_end_panel_row_id` 仅用于跨文件审计，不用于直接切片。',
        '- `source_split` 与 `source_row_index` 用于定位标准化 Parquet 文件中的目标日。',
        '- `prediction_eligible=1` 才进入训练、验证、本地评估或官方测试预测。',
        '- 窗口不足的行保留在索引中，模型输入接口应跳过这些行。',
        '',
        '## 数据保护',
        '',
        '- 原始因子文件未修改。',
        '- 第五步标准化文件未修改。',
        '- 本步骤只在 `模型预处理/步骤6_序列构造` 目录新增索引与报告。'
    ])
    with open(report_path, 'w', encoding='utf-8') as file:
        file.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
