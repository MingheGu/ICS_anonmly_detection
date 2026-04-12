# Sliding Window LSTM 实施计划

## Context

导师要求用 **rolling forward（滑动窗口）** 方式训练 LSTM。用户已抓取连续 pcap：`rolling-data/continue_data/mixed_long_conti.pcap`（8MB，约2小时）。结构：~60min normal → ~30min scan → 5min recovery → ~22min write → ~3min recovery。

核心思路（如用户分享的图）：
```
窗口1: |==== Train (10min) ====|= Test (5min) =|
窗口2:   |==== Train (10min) ====|= Test (5min) =|    ← 每次前移 step_s
窗口3:     |==== Train (10min) ====|= Test (5min) =|
```

---

## Step 0: 分析 pcap 找出精确攻击时间段

**新建脚本**: `roll-script/analyze_pcap_segments.py`

用户按计划抓了数据但未记录精确秒数。通过分析 pcap 内容自动检测：
- **Scan 攻击**：ICMP 包、SYN 探测到非 502 端口、来自 attacker_ip 的异常流量
- **Write 攻击**：来自 attacker_ip 的 Modbus fc=5 (Write Single Coil) / fc=6 (Write Register) 包

脚本输出每种攻击的 start_offset_s 和 end_offset_s，直接用于填入 rolling_labels.json。

---

## Step 1: 预处理配置（修改2个文件）

### 1a. `roll-script/rolling_labels.json` — 添加新 pcap 条目

用 Step 0 的分析结果填入精确时间段：

```json
{
  "name": "mixed_long_conti",
  "path": "rolling-data/continue_data/mixed_long_conti.pcap",
  "segments": [
    {"start": 0.0,     "end": <scan_start>,  "label": "normal"},
    {"start": <scan_start>,  "end": <scan_end>,  "label": "attack_scan"},
    {"start": <scan_end>,    "end": <write_start>, "label": "normal"},
    {"start": <write_start>, "end": <write_end>,   "label": "attack_write"},
    {"start": <write_end>,   "end": null,           "label": "normal"}
  ]
}
```

### 1b. `roll-script/pre_process_rolling_fc_address.py` — 小改动

在 `assign_packet_split()` 里加一行：
```python
if row["pcap_name"] == "mixed_long_conti":
    return "sliding"  # sliding window 脚本自行按时间切分
```

然后运行预处理生成 CSV。

---

## Step 2: 新建滑动窗口训练脚本

**新建文件**: `roll-script/train_sliding_window_packet_lstm.py`

### 2a. 从 v2 复用（直接 import）

```python
from train_roll_packet_lstm_v2 import (
    PacketRollLSTM,        # 模型不改
    build_token_mapping,   # token 映射
    build_packet_samples,  # 构建 context window
    cross_entropy_loss,    # validation loss
    anomaly_scores,        # 计算异常分数
    smooth_scores,         # 平滑
    set_seed,
)
```

### 2b. 新增参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--pcap-name` | `mixed_long_conti` | 过滤哪个 pcap |
| `--train-duration-s` | 600 (10min) | 训练窗口时长 |
| `--test-duration-s` | 300 (5min) | 测试窗口时长 |
| `--step-s` | 120 (2min) | 窗口每步前移距离 |
| `--val-fraction` | 0.2 | 训练窗口末尾切出多少做 validation |
| `--self-clean-rounds` | 0 | self-cleaning 轮数（0=关闭）|
| 其余参数 | 同 v2 | context-length=30, epochs=50, patience=10 等 |

### 2c. 主流程

```python
def main():
    df = load_csv, filter to pcap_name, sort by timestamp
    token_to_idx = build_token_mapping(df)  # 全局词表

    # 生成窗口时间表
    windows = []
    t = t_min
    while t + train_duration + test_duration <= t_max:
        windows.append(train=[t, t+train_dur], test=[t+train_dur, t+train_dur+test_dur])
        t += step_s

    # 逐窗口训练评估
    for win in windows:
        metrics, scores = run_one_window(df, win, token_to_idx, args)

    # 汇总输出
    save_metrics, save_scores, save_plots
```

### 2d. `run_one_window()` — 核心函数

```
输入: 全量 df, 窗口时间范围, token_to_idx, args
输出: 该窗口的 metrics + test scores

1. 按 time_offset_s 过滤数据
   train_df = df[train_start <= t < train_end]
   val_cutoff = train_start + (train_end - train_start) * 0.8
   train_proper_df = train_df[t < val_cutoff]
   val_df = train_df[t >= val_cutoff]

2. test 数据加 context 重叠（关键！）
   第一个 test 包需要前 context_length 个包作为历史
   → 从 test_start 前取 context_length 个包拼到 test_df 前面
   → build_packet_samples 后，只保留 time_offset_s >= test_start 的结果

3. 训练（复用 v2 的 epoch loop + early stopping）
   build_packet_samples(train_proper_df) → train
   build_packet_samples(val_df) → validation
   训练 LSTM，early stopping on val loss

4. Threshold 校准
   val_scores = anomaly_scores(model, val)
   threshold = quantile(val_scores, 0.99) 或 max(val_scores)

5. 评估
   test_scores = anomaly_scores(model, test)
   test_pred = test_scores >= threshold
   计算 precision, recall, f1, AUC
```

### 2e. Context 重叠处理（关键细节）

```python
# test 的第一个包需要前 context_length 个包做 context
# 找到 test_start 在 df 中的位置
first_test_pos = df[df["time_offset_s"] >= test_start].index[0]
buffer_start_pos = max(0, first_test_pos - context_length)

# test_with_buffer 包含 buffer + 实际 test 数据
test_with_buffer_df = df.iloc[buffer_start_pos : last_test_pos + 1]

# build_packet_samples 后，过滤掉 buffer 部分
contexts, targets, meta = build_packet_samples(test_with_buffer_df, ...)
meta_df = pd.DataFrame(meta)
actual_test = meta_df["time_offset_s"] >= test_start
# 只评估 actual_test 部分
```

### 2f. Self-Cleaning（`--self-clean-rounds > 0` 时启用）

```
for round in range(self_clean_rounds):
    model = train(clean_train_df)
    scores = anomaly_scores(model, clean_train_df)
    cutoff = percentile(scores, 95)  # 剔除 top 5%
    clean_train_df = clean_train_df[scores < cutoff]
# 最终用 clean_train_df 训练
```

---

## Step 3: 输出

### `sliding_window_step_metrics.csv`（每个窗口一行）

window_step, train_start_s, train_end_s, test_start_s, test_end_s,
train_packets, val_packets, test_packets, train_attack_frac, test_attack_frac,
threshold, precision, recall, f1_score, tp, fp, tn, fn, roc_auc,
epochs_run, best_epoch, best_val_loss

### `sliding_window_all_scores.csv`（每个 test packet 一行）

window_step, time_offset_s, timestamp, label, is_attack, pair_token,
raw_anomaly_score, anomaly_score, pred_is_anomaly, threshold

### 可视化

1. **Score Timeline**: 横轴=time_offset_s，纵轴=anomaly score，标注攻击时段
2. **Metrics Over Time**: 横轴=window_step，纵轴=precision/recall/f1
3. **Threshold Over Time**: 横轴=window_step，纵轴=threshold

---

## 文件变更总结

| 文件 | 动作 | 工作量 |
|---|---|---|
| `roll-script/analyze_pcap_segments.py` | **新建** | 分析 pcap 找攻击时间段 |
| `roll-script/rolling_labels.json` | 修改 | 加一个 pcap 条目 |
| `roll-script/pre_process_rolling_fc_address.py` | 修改 | 加 1 行 split 逻辑 |
| `roll-script/train_sliding_window_packet_lstm.py` | **新建** | 主要工作 |

---

## 验证方法

1. 先用小窗口 `--train-duration-s 120 --test-duration-s 60 --step-s 60` 快速跑通
2. 检查第一个 test 包的 time_offset_s 是否等于 test_start（context 重叠正确）
3. 检查全 normal 区域的 precision 是否接近 1（无误报）
4. 检查 attack 区域 recall 是否 > 0.9
5. 用默认参数完整运行，比较与 v2 的差异
