import json
import os
import zipfile
from collections import OrderedDict

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ZIP_PATH = r'E:\ChatGPT\ChatGPT_Project\深圳金融比赛赛题五\赛题五\赛题五数据.zip'
OUTPUT_DIR = r'E:\ChatGPT\ChatGPT_Project\深圳金融比赛赛题五\因子数据'
CHUNK_SIZE = 200_000
EPS = 1e-8

RAW_X = [
    'ts_code', 'trade_date', 'open', 'high', 'low', 'close',
    'vol', 'amount', 'flag_limit_up', 'flag_limit_down'
]
MARKET_FIELDS = ['open', 'high', 'low', 'close', 'vol', 'amount']
FACTOR_COLS = [
    'ret_1d', 'mom_3d', 'mom_5d', 'mom_10d', 'mom_20d', 'mom_20d_skip1',
    'volatility_5d', 'volatility_20d', 'range_1d', 'gap_1d', 'true_range_1d',
    'intraday_ret', 'close_position', 'body_direction', 'body_ratio',
    'upper_shadow', 'lower_shadow', 'log_amount', 'log_volume',
    'delta_log_amount', 'delta_log_volume', 'amount_shock_5d',
    'amount_shock_20d', 'limit_up', 'limit_down', 'missing_price_volume',
    'history_available_count_20d'
]


def iter_stock_blocks(file_obj):
    carry = None
    last_code = None
    for chunk in pd.read_csv(file_obj, usecols=RAW_X + (['y_ret_1d'] if file_obj.name.endswith('训练集.csv') else []), chunksize=CHUNK_SIZE):
        chunk['ts_code'] = chunk['ts_code'].astype(str)
        chunk['trade_date'] = chunk['trade_date'].astype('int32')
        if carry is not None:
            chunk = pd.concat([carry, chunk], ignore_index=True)
        if chunk['ts_code'].is_monotonic_increasing is False:
            raise ValueError('原始文件未按 ts_code 升序排列，无法使用逐股票流式计算。')
        current_last = chunk['ts_code'].iloc[-1]
        complete = chunk[chunk['ts_code'] != current_last]
        for code, group in complete.groupby('ts_code', sort=False):
            if last_code is not None and code <= last_code:
                raise ValueError('原始文件的 ts_code 顺序不满足逐股票计算要求。')
            last_code = code
            yield code, group.reset_index(drop=True)
        carry = chunk[chunk['ts_code'] == current_last].reset_index(drop=True)
    if carry is not None and len(carry):
        code = carry['ts_code'].iloc[0]
        if last_code is not None and code <= last_code:
            raise ValueError('原始文件最后股票顺序异常。')
        yield code, carry.reset_index(drop=True)


def calculate_factors(raw):
    x = raw.copy()
    complete = x[MARKET_FIELDS].notna().all(axis=1)
    prev_close = x['close'].shift(1)
    close = x['close']
    log_amount = np.log1p(x['amount'])
    log_volume = np.log1p(x['vol'])

    out = pd.DataFrame(index=x.index)
    out['ret_1d'] = close / close.shift(1) - 1.0
    out['mom_3d'] = close / close.shift(3) - 1.0
    out['mom_5d'] = close / close.shift(5) - 1.0
    out['mom_10d'] = close / close.shift(10) - 1.0
    out['mom_20d'] = close / close.shift(20) - 1.0
    out['mom_20d_skip1'] = close.shift(1) / close.shift(20) - 1.0
    out['volatility_5d'] = out['ret_1d'].rolling(5, min_periods=5).std()
    out['volatility_20d'] = out['ret_1d'].rolling(20, min_periods=20).std()
    out['range_1d'] = (x['high'] - x['low']) / (close + EPS)
    out['gap_1d'] = (x['open'] - prev_close) / (prev_close + EPS)

    tr_values = pd.concat([
        x['high'] - x['low'],
        (x['high'] - prev_close).abs(),
        (x['low'] - prev_close).abs()
    ], axis=1)
    tr_valid = x[['high', 'low']].notna().all(axis=1) & prev_close.notna()
    out['true_range_1d'] = tr_values.max(axis=1, skipna=False) / (prev_close + EPS)
    out.loc[~tr_valid, 'true_range_1d'] = np.nan

    out['intraday_ret'] = close / (x['open'] + EPS) - 1.0
    out['close_position'] = (close - x['low']) / (x['high'] - x['low'] + EPS)
    out['body_direction'] = np.sign(close - x['open'])
    out['body_ratio'] = (close - x['open']).abs() / (close + EPS)
    out['upper_shadow'] = (x['high'] - pd.concat([x['open'], close], axis=1).max(axis=1, skipna=False)) / (close + EPS)
    out['lower_shadow'] = (pd.concat([x['open'], close], axis=1).min(axis=1, skipna=False) - x['low']) / (close + EPS)
    out['log_amount'] = log_amount
    out['log_volume'] = log_volume
    out['delta_log_amount'] = log_amount - log_amount.shift(1)
    out['delta_log_volume'] = log_volume - log_volume.shift(1)
    out['amount_shock_5d'] = log_amount - log_amount.shift(1).rolling(4, min_periods=4).mean()
    out['amount_shock_20d'] = log_amount - log_amount.shift(1).rolling(20, min_periods=20).mean()
    out['limit_up'] = x['flag_limit_up']
    out['limit_down'] = x['flag_limit_down']
    out['missing_price_volume'] = (~complete).astype('float32')
    out['history_available_count_20d'] = complete.astype('float32').rolling(20, min_periods=1).sum()
    return out[FACTOR_COLS]


def cast_output(raw, factors, include_label):
    result = pd.DataFrame({
        'ts_code': raw['ts_code'].astype(str),
        'trade_date': raw['trade_date'].astype('int32')
    })
    for col in FACTOR_COLS:
        result[col] = factors[col].astype('float32')
    if include_label:
        result['y_ret_1d'] = raw['y_ret_1d'].astype('float32')
    return result


def write_part(part, parquet_writer, csv_path, write_header):
    table = pa.Table.from_pandas(part, preserve_index=False)
    if parquet_writer[0] is None:
        parquet_writer[0] = pq.ParquetWriter(csv_path.replace('.csv', '.parquet'), table.schema, compression='zstd')
    parquet_writer[0].write_table(table)
    part.to_csv(csv_path, mode='w' if write_header else 'a', header=write_header, index=False)


def process_dataset(zf, member, include_label, train_tails=None):
    base = '训练集_第一次LST-Transformer因子' if include_label else '测试集_X_第一次LST-Transformer因子'
    csv_path = os.path.join(OUTPUT_DIR, base + '.csv')
    parquet_path = os.path.join(OUTPUT_DIR, base + '.parquet')
    if os.path.exists(csv_path):
        os.remove(csv_path)
    if os.path.exists(parquet_path):
        os.remove(parquet_path)
    audit = OrderedDict([
        ('rows', 0), ('stocks', 0), ('date_min', None), ('date_max', None),
        ('factor_missing', OrderedDict((c, 0) for c in FACTOR_COLS)),
        ('factor_inf', OrderedDict((c, 0) for c in FACTOR_COLS)),
        ('label_missing', 0 if include_label else None), ('duplicate_keys', 0)
    ])
    seen_stocks = set()
    tails = {} if train_tails is None else train_tails
    writer = [None]
    write_header = True
    with zf.open(member) as f:
        f.name = member
        for code, group in iter_stock_blocks(f):
            if code in seen_stocks:
                raise ValueError(f'股票 {code} 在文件中出现非连续记录。')
            seen_stocks.add(code)
            history = tails.get(code)
            work = pd.concat([history, group], ignore_index=True) if history is not None and len(history) else group.copy()
            factors = calculate_factors(work)
            current_factors = factors.iloc[-len(group):].reset_index(drop=True)
            part = cast_output(group.reset_index(drop=True), current_factors, include_label)
            write_part(part, writer, csv_path, write_header)
            write_header = False
            tails[code] = work.tail(20)[RAW_X].reset_index(drop=True)
            audit['rows'] += len(part)
            audit['stocks'] += 1
            dmin = int(part['trade_date'].min())
            dmax = int(part['trade_date'].max())
            audit['date_min'] = dmin if audit['date_min'] is None else min(audit['date_min'], dmin)
            audit['date_max'] = dmax if audit['date_max'] is None else max(audit['date_max'], dmax)
            for col in FACTOR_COLS:
                values = part[col].to_numpy(dtype='float64', na_value=np.nan)
                audit['factor_missing'][col] += int(np.isnan(values).sum())
                audit['factor_inf'][col] += int(np.isinf(values).sum())
            audit['duplicate_keys'] += int(part.duplicated(['ts_code', 'trade_date']).sum())
            if include_label:
                audit['label_missing'] += int(part['y_ret_1d'].isna().sum())
    if writer[0] is not None:
        writer[0].close()
    return audit, tails


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH) as zf:
        train_audit, tails = process_dataset(zf, '训练集.csv', True)
        test_audit, _ = process_dataset(zf, '测试集_X.csv', False, tails)
    audit = OrderedDict([
        ('input_zip', ZIP_PATH),
        ('factor_columns', FACTOR_COLS),
        ('train', train_audit),
        ('test', test_audit),
        ('rules', [
            '原始行情和因子计算阶段不进行缺失值填补',
            '训练集和测试集 X 统一按股票历史顺序计算',
            '测试集只使用当前日期及更早日期的 X 字段',
            '模型专用填充、标准化和序列化不在本阶段执行'
        ])
    ])
    with open(os.path.join(OUTPUT_DIR, '因子生成审计.json'), 'w', encoding='utf-8') as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    print(json.dumps({
        'train_rows': train_audit['rows'],
        'test_rows': test_audit['rows'],
        'train_stocks': train_audit['stocks'],
        'test_stocks': test_audit['stocks'],
        'output_dir': OUTPUT_DIR
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
