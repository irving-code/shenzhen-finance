import json
import os
import sys
import zipfile

import pandas as pd

BASE = r'E:\ChatGPT\ChatGPT_Project\深圳金融比赛赛题五'
MODEL_DIR = os.path.join(BASE, '模型训练')
RUN_DIR = os.path.join(MODEL_DIR, '第二次综合分验证实验')
CONFIG_PATH = os.path.join(MODEL_DIR, '第一次LST-Transformer实验配置.json')
CHECKPOINT_PATH = os.path.join(RUN_DIR, '最佳模型_按综合分.pt')
DATA_ZIP = r'C:\Users\Lenovo\Desktop\2026深圳国际金融科技大赛\赛题五\赛题五数据.zip'
STEP6_DIR = os.path.join(BASE, '模型预处理', '步骤6_序列构造')
sys.path.insert(0, MODEL_DIR)
from run_second_score_experiment import predict_checkpoint, smooth_predictions


def main():
    with open(CONFIG_PATH, 'r', encoding='utf-8') as file:
        config = json.load(file)
    with open(os.path.join(RUN_DIR, '本地留出集比赛指标评估.json'), 'r', encoding='utf-8') as file:
        result = json.load(file)
    alpha = float(result['best_alpha'])
    prediction = predict_checkpoint(
        CHECKPOINT_PATH, 'competition_test', config, 'cpu', 4096
    )
    smooth = smooth_predictions(prediction, alpha)
    eligible_path = os.path.join(RUN_DIR, 'competition_test_predictions.parquet')
    smooth[['ts_code', 'trade_date', 'smooth_pred']].rename(
        columns={'smooth_pred': 'pred'}
    ).to_parquet(eligible_path, index=False, compression='zstd')

    index = pd.read_parquet(
        os.path.join(STEP6_DIR, '序列索引_competition_test.parquet'),
        columns=['ts_code', 'target_date', 'prediction_eligible']
    )
    eligible_index = index.loc[index['prediction_eligible'].eq(1), ['ts_code', 'target_date']].copy()
    eligible_index = eligible_index.rename(columns={'target_date': 'trade_date'})
    with zipfile.ZipFile(DATA_ZIP) as archive:
        names = [name for name in archive.namelist() if name.endswith('_X.csv')]
        if len(names) != 1:
            raise ValueError('比赛数据压缩包中的测试集_X.csv 文件数量异常')
        with archive.open(names[0]) as file:
            test_x = pd.read_csv(file, usecols=['ts_code', 'trade_date'])
    test_x['ts_code'] = test_x['ts_code'].astype(str)
    eligible_index['ts_code'] = eligible_index['ts_code'].astype(str)
    prediction_keys = smooth[['ts_code', 'trade_date']]
    missing_keys = test_x.merge(prediction_keys, on=['ts_code', 'trade_date'], how='left', indicator=True)
    extra_keys = prediction_keys.merge(test_x, on=['ts_code', 'trade_date'], how='left', indicator=True)
    audit = {
        'test_rows': int(len(test_x)),
        'eligible_sequence_rows': int(len(eligible_index)),
        'prediction_rows': int(len(prediction_keys)),
        'test_duplicate_keys': int(test_x.duplicated(['ts_code', 'trade_date']).sum()),
        'prediction_duplicate_keys': int(prediction_keys.duplicated(['ts_code', 'trade_date']).sum()),
        'missing_test_keys': int((missing_keys['_merge'] == 'left_only').sum()),
        'extra_prediction_keys': int((extra_keys['_merge'] == 'left_only').sum()),
        'submission_created': False
    }
    with open(os.path.join(RUN_DIR, '官方测试集覆盖检查.json'), 'w', encoding='utf-8') as file:
        json.dump(audit, file, ensure_ascii=False, indent=2)
    if (
        audit['test_rows'] != audit['prediction_rows']
        or audit['test_duplicate_keys'] != 0
        or audit['prediction_duplicate_keys'] != 0
        or audit['missing_test_keys'] != 0
        or audit['extra_prediction_keys'] != 0
    ):
        raise ValueError('官方测试集预测未覆盖全部 ts_code + trade_date 记录，已停止生成 submission.csv')
    smooth[['ts_code', 'trade_date', 'smooth_pred']].rename(
        columns={'smooth_pred': 'pred'}
    ).to_csv(os.path.join(RUN_DIR, '最终submission.csv'), index=False, encoding='utf-8-sig')
    audit['submission_created'] = True
    with open(os.path.join(RUN_DIR, '官方测试集覆盖检查.json'), 'w', encoding='utf-8') as file:
        json.dump(audit, file, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
