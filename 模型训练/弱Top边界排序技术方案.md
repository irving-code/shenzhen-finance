# 弱 Top 边界排序技术方案

## 1. 技术范围

本方案在现有 37 因子 LSTM-Transformer 训练和评价程序上增加两个弱 Top 边界权重实验：`lambda_top=0.05` 和 `lambda_top=0.10`。输入数据、预处理产物、时间划分、主办方评价程序、推理排序方式和历史留出评价方式沿用现有实现。

## 2. 程序复用关系

| 功能 | 程序 | 使用方式 |
|---|---|---|
| 读取配置和输入摘要 | `cross_section_experiment.py` | 复用 |
| 读取横截面输入 | `cross_section_input.py` | 复用 |
| 全局收益和排序损失 | `ranking_loss.py` | 复用 |
| Top 边界损失 | `top_boundary_rank_loss.py` | 复用 |
| 模型训练 | `train_top_boundary_rank.py` | 通过 `--top-weight` 传入 0.05 或 0.10 |
| 2023 年预测和官方评分 | `evaluate_top_boundary_rank.py` | 复用 |
| 统一评分和留出评价 | `evaluate_cross_section_rank.py` | 复用 |
| 候选选择 | `select_top_boundary_rank.py` | 为每个独立运行目录生成选择结果 |
| 实验核验 | `verify_top_boundary_experiment.py` | 增加 B2、B3 完整性检查 |
| 结果报告 | `generate_top_boundary_report.py` | 增加 B2、B3 训练和评分比较 |

本轮只通过新增实验配置和独立输出目录区分权重。已有配置、代码、检查点和报告保持原样。

## 3. 配置身份

每个运行目录保存 `config_sha256`、`input_sha256`、`feature_mode=extended37`、`top_boundary_weight`、`global_rank_weight=0.3` 和 `seed=42`。训练和评价开始前检查配置摘要、输入摘要、37 维输入、关闭的因子依赖训练惩罚以及运行目录快照的一致性。

## 4. 训练数据和标签实现

训练索引读取 `ts_code`、`target_date`、`panel_row_id`、`training_eligible`、`y_ret_1d` 和 `flag_limit_up`。

标签构造步骤：

1. 保留 `training_eligible == 1` 的训练记录；
2. 按 `target_date` 分组；
3. 保留 `flag_limit_up == 0` 且 `y_ret_1d` 非缺失的记录；
4. 按 `y_ret_1d` 降序排序；
5. 将前 `max(len(valid)//10, 1)` 条记录设置 `top_flag=True`；
6. 将所有有效记录设置 `boundary_valid=True`；
7. 缺少标签或涨停记录不参与 Top 边界损失。

训练标签只在内存中与同批次股票日期键对齐，不修改原始索引文件和预处理文件。

## 5. 批次和模型计算

继续使用同日批次采样器，一个批次只能包含同一个 `target_date`，最大批量为 512。每个批次先计算模型预测 `pred`，再计算现有全局收益和排序损失，以及 Top 组和普通组的边界损失。

最终损失为：

```text
loss_total = global_losses.loss_total + top_weight * top_losses.loss_top_boundary
```

B2 和 B3 分别传入 `top_weight=0.05`、`top_weight=0.10`。边界损失使用 `pred[top] - pred[rest]` 的成对差值和 `tau * softplus(-delta/tau)`，只改变 Top 组相对于普通组的分数边界，不改变模型输出维度，也不加入分类头。

## 6. 训练记录

每个周期记录收益损失、全局排序损失、Top 边界损失、两项加权损失、总损失、有效配对数、Top 配对数、Top 行数、普通行数、梯度范数、样本数、批次数、耗时和采样摘要。

重点检查 `weighted_top_boundary`：B2 中应约为原始 Top 损失的 0.05 倍，B3 中应约为 0.10 倍，并且不再与全局排序项处于同一量级。

## 7. 独立输出目录

```text
模型训练/弱Top边界排序实验/
└── 20261010_weak_top_boundary_rank/
    ├── B2_lambda_top_0.05/
    │   ├── 配置快照.json
    │   ├── 调度状态.json
    │   ├── runs/
    │   ├── 2023官方临时评分目录/
    │   └── evaluation/
    └── B3_lambda_top_0.10/
        ├── 配置快照.json
        ├── 调度状态.json
        ├── runs/
        ├── 2023官方临时评分目录/
        └── evaluation/
```

两个新实验目录之间不共享预测文件、候选文件和选择文件。临时目录只保存 `测试集_X.csv`、`测试集_Y.csv` 和候选预测文件，不保存正式提交文件。

## 8. 2023 年评分实现

对每个周期和 alpha：

1. 从对应检查点加载模型；
2. 预测 2023 年 `validation` 资格记录；
3. 按现有定义计算时间平滑预测；
4. 只保留官方选择资格记录；
5. 写出 `候选预测.csv`；
6. 调用 `赛题五/evaluate.py` 的 `evaluate()`；
7. 保存官方评分原始结果和统一比较表。

本地评分还要检查股票日期键完全一致、两个临时输入文件使用相同键、官方评分与本地指标在允许误差内一致，以及 241 个交易日均有有效评价记录。

## 9. 候选选择和稳定性

合并 B0、B2、B3 的官方候选表后，按官方综合分降序、Rank IC 降序、年化超额收益降序、训练周期升序、alpha 降序和 Top 边界权重升序选择候选。

最高候选执行推理批量 4096 和 2048 的稳定性检查，检查原始预测和平滑预测的每日 Spearman 排序相关性、Top10% 组合重叠率、交易收益组合一致性和换手率组合一致性。稳定性不通过时，候选不能进入 2024 年历史留出评价。

## 10. 2024 年历史留出实现

将通过 2023 年官方评分和稳定性核验的唯一候选固定下来，调用现有 `evaluate_cross_section_rank.py --operation holdout` 评价 2024 年 `local_holdout`，记录平滑指标、预测质量、因子依赖诊断和综合比较结果。如果最高候选是 B0，则复用既有 B0 的 2024 年历史留出结果，并记录复用来源。

## 11. 报告内容

最终报告包含 B0、B2、B3 的实验身份和数据范围，B2、B3 每个周期的损失和加权贡献，2023 年每个周期和 alpha 的官方细分分数，三项评分贡献，B0、B2、B3 最优候选比较，2024 年唯一固定候选的历史留出结果，与既有 27 因子模型、37 因子依赖惩罚模型和 37 因子无依赖惩罚模型的比较，稳定性、质量核验、训练耗时以及测试预测状态。

## 12. 完成核验

完成标志为 B2、B3 均完成 8 个周期训练，每组有 32 个 2023 年官方候选，选择排序可复现，最终候选稳定性通过，2024 年评价完成或明确复用 B0，报告包含全部比较表，新目录不存在 `submission.csv`，原有实验目录和预处理产物摘要未被改写。
