# LightGBM 实验计划

## 1. 实验目标

使用统一未填补因子主表训练 LightGBM 回归模型，预测目标日的下一交易日收益率 `y_ret_1d`，并与第一次 LST-Transformer 实验使用相同的时间划分和评价指标进行比较。

本实验是单日横截面表格模型。每条记录对应一只股票在一个交易日的预测样本，不构造 20 个交易日序列。

## 2. 数据文件

训练和验证数据读取：

```text
因子数据/训练集_第一次LST-Transformer因子.parquet
```

官方测试数据读取：

```text
因子数据/测试集_X_第一次LST-Transformer因子.parquet
```

这两个文件是项目定义的统一未填补因子主表。保留其中的空值，由 LightGBM 原生缺失值分支处理。实验不读取第五步标准化文件，也不读取第六步序列索引和第七步序列输入接口。

## 3. 特征和标签

### 3.1 输入特征

使用因子主表中的 27 个因子：

```text
ret_1d
mom_3d
mom_5d
mom_10d
mom_20d
mom_20d_skip1
volatility_5d
volatility_20d
range_1d
gap_1d
true_range_1d
intraday_ret
close_position
body_direction
body_ratio
upper_shadow
lower_shadow
log_amount
log_volume
delta_log_amount
delta_log_volume
amount_shock_5d
amount_shock_20d
limit_up
limit_down
missing_price_volume
history_available_count_20d
```

`ts_code` 和 `trade_date` 只用于分组、排序和生成预测文件，不作为普通数值特征。训练集中的 `y_ret_1d` 只作为标签。

### 3.2 缺失值规则

1. 不对因子主表执行前向填充、后向填充、中位数填充或标准化。
2. 保留 `missing_price_volume` 和 `history_available_count_20d`，让模型同时使用因子值和数据完整性信息。
3. 训练、验证和本地留出集使用项目第二步确定的合格样本掩码，排除目标标签为空、股票有效区间外以及最近 20 个交易日有效历史不足的记录。
4. 官方测试集保留全部提交记录，模型对每个 `ts_code + trade_date` 生成一条预测。LightGBM 对测试因子中的空值使用与训练阶段相同的原生缺失值分支。

合格样本掩码通过 `ts_code + trade_date` 与步骤三输出的资格信息关联，特征数值仍来自统一未填补因子主表。

## 4. 时间划分

| 区间 | 日期范围 | 用途 |
|---|---|---|
| `train_fit` | 2018-01-02 至 2022-12-30 | 拟合模型和选择树数量 |
| `validation` | 2023-01-03 至 2023-12-29 | 选择超参数和最佳迭代轮数 |
| `local_holdout` | 2024-01-02 至 2024-12-31 | 确定模型后进行一次最终评价 |
| `competition_test` | 2025-01-02 至 2026-06-08 | 生成比赛预测文件 |

按照日期升序使用样本，不进行随机切分。训练、验证和本地留出之间不共享目标日记录。

## 5. 模型设置

### 5.1 训练目标

```text
objective = regression
metric = l2
label = y_ret_1d
```

目标值保持原始收益率尺度，不对标签做标准化或截断。

### 5.2 候选参数

固定随机种子为 `42`，候选配置使用以下范围：

```text
learning_rate: 0.03, 0.05
num_leaves: 31, 63, 127
min_data_in_leaf: 100, 300, 1000
feature_fraction: 0.8, 1.0
bagging_fraction: 0.8, 1.0
bagging_freq: 1
max_depth: -1
lambda_l1: 0
lambda_l2: 1, 10
max_bin: 255
```

每个候选配置最多训练 2000 轮。使用验证集损失监控训练是否继续，并每隔固定迭代轮数计算验证集每日 Rank IC，记录验证集 Rank IC 最优的迭代轮数。

### 5.3 模型选择

1. 第一选择指标为验证集每日 Rank IC 均值。
2. Rank IC 均值相同或差异小于 `1e-6` 时，选择验证集 `l2` 更低的配置。
3. 最佳配置和迭代轮数确定后，读取对应模型，在本地留出集只评估一次。
4. 本地留出集结果不回用于参数调整。

## 6. 评价指标

验证集和本地留出集均按交易日计算指标：

1. 每日 Rank IC：预测值与 `y_ret_1d` 的 Spearman 相关系数；
2. Rank IC 均值、标准差、ICIR 和正 IC 占比；
3. 剔除涨停股后，预测排名前 10% 组合相对当日市场平均收益的年化超额收益；
4. 相邻交易日预测前 10% 股票集合的 Jaccard 换手率；
5. 比赛综合分数：

```text
综合分数 = Rank IC 均值 × 0.4
         + Top 10% 年化超额收益 × 0.3
         + (1 - 换手率) × 0.3
```

评估实现与 [`赛题五/evaluate.py`](<E:\ChatGPT\ChatGPT_Project\深圳金融比赛赛题五\赛题五\evaluate.py>) 保持一致。

## 7. 信息泄漏控制

1. 因子主表中的因子只使用当前交易日及之前的数据，沿用因子生成阶段的时间规则。
2. 训练样本只使用 `train_fit` 日期范围和合格样本掩码。
3. 验证集只用于超参数和迭代轮数选择，不参与特征处理或标签处理。
4. 本地留出集不参与模型选择。
5. 官方测试集不参与训练、早停、参数选择或阈值估计。
6. `y_ret_1d`、未来日期字段和任何测试标签不进入输入特征。
7. 所有预测结果保留 `ts_code`、`trade_date`，并检查键唯一性和原始测试行数一致。

## 8. 输出文件

新建目录：

```text
模型训练/LightGBM实验结果/
```

输出文件：

```text
配置.json
候选配置结果.csv
训练日志.csv
最佳模型.txt
验证集预测.parquet
本地留出集预测.parquet
测试集预测.parquet
submission.csv
实验报告.md
```

`submission.csv` 包含：

```text
ts_code,trade_date,pred
```

## 9. 验收检查

1. 训练、验证、本地留出和测试预测的键字段无重复。
2. 训练和验证实际使用的样本数与合格样本掩码一致。
3. 预测值不存在正无穷或负无穷。
4. 测试预测行数与 `测试集_X_第一次LST-Transformer因子.parquet` 行数一致。
5. 测试预测覆盖全部 `ts_code + trade_date` 组合。
6. 训练配置、最佳迭代轮数、验证指标和本地留出指标能够从报告与结果文件复核。
7. LightGBM 结果与第一次 LST-Transformer 结果使用相同指标和日期范围进行并列比较。

