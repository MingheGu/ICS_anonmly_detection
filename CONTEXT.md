# ICS Anomaly Detection - Rolling LSTM 项目上下文

## 项目概述

用 LSTM 做 ICS（工业控制系统）Modbus 网络流量的异常检测。数据来自 GRFICSv3 仿真环境（化学反应器），攻击类型包括端口扫描（scan）和 Modbus 写攻击（write）。

## 当前架构

- **模型**：Packet-level LSTM，逐包预测下一个 token，用 cross-entropy 作为 anomaly score
- **特征**：只用 Modbus function code + address，组合成 token（如 `fc=1|addr=40`）
- **词汇表**：6 个 token（`fc=-1|addr=-1`, `fc=1|addr=40`, `fc=4|addr=100`, `fc=5|addr=0`, `fc=5|addr=1`, `fc=6|addr=100`）
- **训练方式**：只用 normal 数据训练，anomaly = 偏离正常模式
- **代码位置**：`/Users/guminghe/ics-anomaly-study/roll script/`
  - `pre_process_rolling_fc_address.py` — 预处理
  - `train_roll_packet_lstm_v2.py` — 训练和评估（主文件）

## 数据

所有 pcap 来自同一次 GRFICSv3 环境：

| 文件 | 路径 | 用途 |
|------|------|------|
| `normal_long_00.pcap` | `rolling-data/normal/` | Train（60min 纯正常流量） |
| `normal_long_03.pcap` | `rolling-data/normal/` | Validation（20min 纯正常流量） |
| `mixed_long_03.pcap` | `rolling-data/mixed/` | Test - scan 攻击（含 3 次不同速度 nmap 扫描） |
| `mixed_long_04.pcap` | `rolling-data/mixed/` | Test - write 攻击（5min 快速写 + 20min 慢速写） |

## 当前最佳结果（v2 + smoothing）

| | Precision | Recall | F1 | TP | FP |
|---|-----------|--------|----|----|-----|
| **Scan** | 0.846 | 0.998 | 0.916 | 2001 | 364 |
| **Write** | 0.195 | 0.996 | 0.326 | 258 | 1066 |
| **Overall** | 0.587 | 0.998 | 0.739 | 2260 | 1430 |
| AUC | 0.992 | | | | |

## 已发现的关键问题和实验结论

### 问题 1：Threshold 校准失败

正常 Modbus 流量只有 6 个 token，高度周期性，LSTM 预测几乎完美 → validation scores 全接近 0 → p99 threshold = 0.004 → 太低 → 大量 FP。

**证据**：手动设 threshold=3.0 时，precision=0.916, recall=0.996。模型区分能力很强（AUC=0.992），纯粹是 threshold 太低。

### 问题 2：Write 的 Context 污染

Write 攻击使用 fc=5/fc=6，训练时从未见过。攻击结束后，后续正常包的 context window（30 个包）里残留 fc=5/fc=6 token → 模型预测不准 → score 偏高 → FP。

同时攻击期间，正常包（HMI 的 fc=1/fc=4）和攻击包交替出现，正常包的 context 被攻击包包围 → score 也偏高 → FP。

**关键数据**：Write 的 1066 个 FP 几乎全部集中在攻击发生的 6-7min 和 20-22min，正常时间段 FP 接近 0。

### 问题 3：LSTM 对此场景过于强大

教授反馈：数据太 "flat"（太规律），LSTM 是为复杂时序设计的，对简单周期性 Modbus 流量来说杀鸡用牛刀。建议尝试 SVM 等更简单的模型作为对比基线。

## 已尝试的方法和结果

| 方法 | Precision | Recall | 结论 |
|------|-----------|--------|------|
| 原版（raw score + p99 threshold） | 0.503 | 0.997 | Threshold 太低 |
| Score smoothing（滑动平均 W=30） | 0.587 | 0.998 | 有改善，消除孤立噪声 |
| Smoothing + Anomaly Masking（低阈值） | 0.110 | 1.000 | 崩溃，context 被替换偏移 |
| Smoothing + High-confidence Masking | 0.569 | 0.998 | 不如单独 smoothing |
| 手动 threshold=3.0（raw scores） | 0.916 | 0.996 | 最佳，但无法自动校准 |

## 教授的要求

1. **Packet-level（逐包）检测**，不是 window-level 聚合
2. **Expanding window** 做 time-series validation，证明模型跨时间段稳定
3. **标签只用于测试评估**，训练不用标签
4. **尝试其他模型**（SVM、Isolation Forest 等）做对比，因为当前数据太周期性，LSTM 不一定最优
5. 特征主实验用 function code + address，扩展实验可以加更多特征做 ablation

## 下一步需要做的事情

### A. 加 One-Class SVM / Isolation Forest 做对比基线（教授要求）

思路：不做逐包预测，而是对每个时间窗口（比如 1 秒）提取统计特征，用 OC-SVM 判断窗口是否正常。

窗口特征（基于 fc+address）：
- 各 token 的出现次数/比例
- 包总数
- fc=5/fc=6 是否出现（write 攻击核心指标）
- 出现过的 unique token 数

优势：不存在 context 污染问题，fc=5 出现在窗口里 → 该窗口直接异常。

### B. Expanding Window 实验

```
Pass 1: Train [normal_00 的 0-10min]  Val [10-15min]  Test [mixed_03 + mixed_04]
Pass 2: Train [normal_00 的 0-20min]  Val [20-25min]  Test [mixed_03 + mixed_04]
Pass 3: Train [normal_00 的 0-30min]  Val [30-35min]  Test [mixed_03 + mixed_04]
Pass 4: Train [normal_00 的 0-40min]  Val [40-45min]  Test [mixed_03 + mixed_04]
Pass 5: Train [normal_00 的 0-50min]  Val [50-55min]  Test [mixed_03 + mixed_04]
```

### C. Threshold 校准改进

当前 `p99(val_scores)` 对这种太规律的数据不适用。备选方案：
- `max(val_scores)` — "超过正常最高分才算异常"
- `mean + k*std` — 统计方法（但 score 分布不是正态）
- Score Winsorization（截断极端值）+ smoothing + max

### D. 扩展特征实验（ablation study）

在 fc+address 基础上加入 port/protocol/flags，对比检测效果。预期 scan 检测会显著提升（正常和 scan 的 token 变得不同），write 的 context 污染问题可能仍然存在。

## 重要术语

- **Score Smoothing / Score Aggregation**：对逐包 anomaly score 做滑动平均，减少孤立噪声
- **Anomaly Masking / Context Cleansing**：检测到异常包后用预测值替换，防止污染后续 context（实验证明在此场景下不可行）
- **Score Winsorization / Score Clipping**：对极端 score 设上限，减少 smoothing 时的溢出效应
- **Expanding Window**：time-series cross-validation，训练集逐步扩大
- **Context Pollution / Contamination**：attack token 残留在后续包的 context window 中，导致正常包被误判
