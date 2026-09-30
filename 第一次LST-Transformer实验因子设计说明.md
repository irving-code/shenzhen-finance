# 第一次 LST-Transformer 实验因子设计说明

## 1. 实验目标

本实验用于建立第一版 LST-Transformer 基线，检验单只股票自身的历史价格、波动、日内形态和成交活跃度对下一交易日收益率预测的作用。

本实验使用《因子设计说明.md》中的原始因子名称，并增加一个历史长度因子 `history_available_count_20d`。不在因子生成阶段对原始缺失行情进行填补。

## 2. 预测目标

对股票 `ts_code` 在交易日 `trade_date=t` 的记录，预测下一交易日收益率：

```text
y_ret_1d = close(t+1) / close(t) - 1
```

因子最多使用 t 日收盘时已经能够获得的信息。

## 3. 第一次实验因子集合

### 3.1 收益与动量因子

```text
ret_1d
mom_3d
mom_5d
mom_10d
mom_20d
mom_20d_skip1
```

### 3.2 波动与价格区间因子

```text
volatility_5d
volatility_20d
range_1d
gap_1d
true_range_1d
```

### 3.3 日内价格形态因子

```text
intraday_ret
close_position
body_direction
body_ratio
upper_shadow
lower_shadow
```

### 3.4 成交活跃度因子

```text
log_amount
log_volume
delta_log_amount
delta_log_volume
amount_shock_5d
amount_shock_20d
```

### 3.5 交易状态因子

```text
limit_up
limit_down
```

### 3.6 数据状态因子

```text
missing_price_volume
history_available_count_20d
```

其中：

- `missing_price_volume`：六个量价字段中任一字段缺失时取 1，否则取 0；
- `history_available_count_20d`：最近 20 个交易日中六个量价字段均完整的天数。

## 4. 本次实验暂不使用的候选因子

以下因子保留在完整因子表中，但第一次基线实验暂不输入模型：

```text
rev_1d
rev_3d
price_volume_interaction
volume_up_strength
volume_down_strength
csrank_mom_5d
csrank_mom_20d
csrank_volatility_5d
csrank_amount_shock_5d
csrank_intraday_ret
excess_ret_1d
excess_mom_5d
market_mean_ret_1d
market_breadth_1d
market_dispersion_1d
limit_up_count_5d
limit_down_count_5d
missing_count_5d
missing_count_20d
```

暂不使用这些因子是为了先得到一个结构清晰的时间序列基线，避免交互因子、横截面因子和缺失次数因子同时进入模型后难以判断增益来源。

## 5. 缺失值和序列输入处理

### 5.1 因子生成阶段

原始行情字段不做前向填充、后向填充或均值填充。原始值缺失时，依赖该字段的因子保留为空值。

滚动因子在历史窗口不足或有效行情数量不足时保留为空值，并由 `history_available_count_20d` 记录近期历史的可用程度。

### 5.2 LST-Transformer 输入阶段

LST-Transformer 的数值输入不能直接包含 NaN。填充参数只能使用训练时间段估计，建议：

1. 对每个因子使用训练集统计量进行填充；
2. 保留 `missing_price_volume` 作为缺失状态输入；
3. 对序列长度不足的样本使用统一的序列掩码或排除样本；
4. 不使用测试集统计量估计填充值和标准化参数。

第一次实验使用连续 20 个交易日构造样本，要求窗口中至少 16 个交易日的六个量价字段完整。有效行情区间内部的缺失在模型输入阶段填充，首个有效行情日前和最后有效行情日后的记录不进入训练样本。

有效行情区间和样本起始时间按 `ts_code` 独立确定。一只股票在较早年份没有有效记录时，只排除该股票对应日期的样本，不影响其他股票使用更早年份的有效记录。

## 6. 因子计算和数据切分规则

1. 将训练集和测试集的 X 字段合并后，按 `ts_code`、`trade_date` 升序排序，统一计算滚动因子；
2. `y_ret_1d` 只保留在训练数据中，不参与测试因子计算；
3. 完成因子计算后，再按日期拆分训练因子和测试因子；
4. 训练、验证和测试按照交易日期切分，不随机打乱日期；
5. 所有标准化、填充和异常值阈值只使用训练时间段估计；
6. 样本有效区间、首个可用目标日和最后可用目标日均按 `ts_code` 独立判断；
7. 横截面排名和市场状态因子暂不作为第一次实验输入，但后续实验可以在同一因子表上追加。

## 7. 第一次实验因子总表

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
