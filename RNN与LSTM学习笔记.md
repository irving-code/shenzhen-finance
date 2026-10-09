# RNN 与 LSTM 学习笔记

## 学习目标

- 理解循环神经网络处理序列数据的方式。
- 掌握隐藏状态、时间反向传播和梯度问题。
- 理解 LSTM 的记忆单元、遗忘门、输入门和输出门。
- 使用 PyTorch 完成 RNN、LSTM 和 GRU 的基础实验。
- 能够判断循环网络适合哪些任务，并与 Transformer 进行比较。

## 一、循环神经网络 RNN

### 1. 基本概念

RNN（循环神经网络）按照时间顺序处理输入数据，并通过隐藏状态保存前面时间步的信息。

生活类比：阅读一句话时，读者会保留前面词语形成的上下文，再用它理解当前词语。

### 2. 基本结构

在时间步 `t`，RNN 接收当前输入 `x_t` 和上一个隐藏状态 `h_{t-1}`，得到新的隐藏状态 `h_t`：

```text
h_t = tanh(W_xh x_t + W_hh h_{t-1} + b_h)
y_t = W_hy h_t + b_y
```

其中：

- `x_t`：当前时间步的输入。
- `h_t`：当前隐藏状态，也可以理解为当前的短期记忆。
- `y_t`：当前输出。
- `W` 和 `b`：需要通过训练学习的参数。

### 3. 时间反向传播 BPTT

BPTT（通过时间反向传播）会沿着时间顺序展开 RNN，再计算每个时间步对损失函数的影响。

### 4. 梯度问题

- 梯度消失：早期时间步的信息在反向传播中逐渐变弱，模型难以学习长期依赖。
- 梯度爆炸：梯度数值变得很大，训练过程出现不稳定。
- 常见处理方法包括梯度裁剪、合理初始化，以及使用 LSTM 或 GRU。

### 5. 我的理解

<!-- 在这里记录自己对 RNN 的解释、公式推导和疑问。 -->

## 二、LSTM

### 1. 核心思想

LSTM（长短期记忆网络）在 RNN 的基础上增加了细胞状态和门控机制，使模型可以选择保留、写入或输出哪些信息。

生活类比：LSTM 像一位整理长篇资料的读者。遗忘门删除无关内容，输入门写入新信息，输出门决定当前需要使用哪些记忆。

### 2. 四个主要部分

- 细胞状态 `c_t`：贯穿序列的长期记忆通道。
- 遗忘门 `f_t`：决定旧记忆保留多少。
- 输入门 `i_t`：决定当前信息写入多少。
- 输出门 `o_t`：决定当前输出使用多少记忆。

### 3. 关键公式

```text
f_t = sigmoid(W_f [h_{t-1}, x_t] + b_f)
i_t = sigmoid(W_i [h_{t-1}, x_t] + b_i)
g_t = tanh(W_g [h_{t-1}, x_t] + b_g)
o_t = sigmoid(W_o [h_{t-1}, x_t] + b_o)

c_t = f_t * c_{t-1} + i_t * g_t
h_t = o_t * tanh(c_t)
```

### 4. 优点与局限

| 方面 | 记录 |
|---|---|
| 主要优点 | 能够保存较长时间范围内的重要信息，适合有顺序的数据 |
| 主要局限 | 计算需要按时间顺序进行，长序列训练速度可能较慢 |
| 常见任务 | 文本分类、语音识别、时间序列预测、序列生成 |
| 常见替代模型 | GRU、Transformer |

### 5. 我的理解

<!-- 在这里记录自己对 LSTM 门控机制和细胞状态的理解。 -->

## 三、GRU 对比

### 学习要点

- GRU（门控循环单元）使用较少的门控结构。
- GRU 通常比 LSTM 更简单，参数数量较少。
- LSTM 和 GRU 的实际效果需要结合数据集和任务进行比较。

### 实验记录

| 模型 | 数据集 | 任务 | 训练轮数 | 验证集指标 | 备注 |
|---|---|---|---:|---:|---|
| RNN |  |  |  |  |  |
| LSTM |  |  |  |  |  |
| GRU |  |  |  |  |  |

## 四、学习资源

### 教材与教程

- [动手学深度学习：基础循环神经网络](https://d2l.ai/chapter_recurrent-neural-networks/index.html)
- [动手学深度学习：现代循环神经网络](https://d2l.ai/chapter_recurrent-modern/index.html)
- [PyTorch：从零开始 NLP](https://docs.pytorch.org/tutorials/intermediate/nlp_from_scratch_index.html)

### 代码仓库

- [D2L 官方 GitHub](https://github.com/d2l-ai/d2l-en)
- [PyTorch 官方语言模型示例](https://github.com/pytorch/examples/tree/main/word_language_model)
- [Karpathy char-rnn](https://github.com/karpathy/char-rnn)

### 论文

- [Elman, 1990：Finding Structure in Time](https://doi.org/10.1207/S15516709COG1402_1)
- [Bengio et al., 1994：Learning Long-Term Dependencies with Gradient Descent is Difficult](https://doi.org/10.1109/72.279181)
- [Hochreiter and Schmidhuber, 1997：Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735)

### 视频

- [Stanford CS224N Lecture 5：Recurrent Neural Networks](https://www.youtube.com/watch?v=fyc0Jzr74y4)
- [Stanford CS224N Lecture 6：Sequence to Sequence Models](https://www.youtube.com/watch?v=Ba6Fn1-Jsfw)
- [Bilibili：李宏毅机器学习课程](https://www.bilibili.com/video/BV1YpcQzpENc/)
- [Bilibili：PyTorch 循环神经网络实战](https://www.bilibili.com/video/BV1fb4y1h7Cv/)

## 五、学习进度

- [ ] 读完基础 RNN 章节
- [ ] 手写 RNN 前向传播
- [ ] 理解 BPTT 和梯度问题
- [ ] 读完 LSTM 和 GRU 章节
- [ ] 使用 PyTorch 完成字符级 RNN
- [ ] 使用 PyTorch 完成文本分类
- [ ] 比较 RNN、LSTM 和 GRU
- [ ] 阅读三篇经典论文
- [ ] 学习序列到序列模型和注意力机制
- [ ] 了解 Transformer 与循环网络的差异

## 六、实验记录

### 实验名称

<!-- 例如：使用 LSTM 完成文本情感分类。 -->

### 实验日期

<!-- YYYY-MM-DD -->

### 数据集

<!-- 记录数据来源、样本数量和训练集划分。 -->

### 模型设置

<!-- 记录输入维度、隐藏层维度、层数、批次大小、学习率和训练轮数。 -->

### 实验结果

<!-- 记录训练损失、验证集指标和测试集指标。 -->

### 问题与改进

<!-- 记录训练中遇到的问题、原因分析和下一步修改。 -->

## 七、问题清单

1. 为什么普通 RNN 容易出现梯度消失？
2. 细胞状态和隐藏状态分别承担什么作用？
3. LSTM 的三个门如何共同更新记忆？
4. GRU 为什么可以使用更简单的结构？
5. 序列长度、批次大小和截断长度如何影响训练？
6. 什么情况下应该选择 LSTM，什么情况下应该选择 Transformer？

## 八、每次学习后的记录

### 日期：

#### 今天学到的内容

#### 仍然不清楚的内容

#### 一个生活类比

#### 一个公式或代码片段

#### 下一步学习内容

