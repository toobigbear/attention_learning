# -*- coding: utf-8 -*-
"""
================================================================================
04_attention_family.py
================================================================================
【对应图片】图7 (SRDN7754.JPG)：《Attention 技术一览图》

【这张图讲了啥】
一张图梳理了 7 种主流注意力机制，它们都在解决同一个核心问题：
  上下文越来越长 → 全量 Attention 的计算量 O(n²) 爆炸、
  KV Cache（缓存的历史 K/V）越来越大、解码时显存带宽压力大。

7 种机制 = 4 个不同的解题角度：
  ① 减少 KV Head 数量    → MHA → MQA → GQA（让多头"拼车"用 K/V）
  ② 压缩 KV 表示         → MLA（把 K/V 压成小 latent，用的时候再展开）
  ③ 只关注重要的 Token   → DSA（动态挑 Top-K 个相关词，不全看）
  ④ 压缩 + 稀疏结合      → CSA（先按块压缩，再稀疏挑块）
  ⑤ 线性/递推形式        → KDA（状态更新，O(n) 线性复杂度）

演进路线（图片右下角）：
  MHA → MQA → GQA/CQA → MLA → DSA → CSA → KDA

【大白话理解】
把注意力想象成"全班同学互相抄笔记"：
- MHA：每人都有自己的笔记（K/V），信息最全，但笔记占地方最大
- MQA：全班只共用一份笔记（K/V），地方省了，但可能漏掉个人特色
- GQA：分成几个小组，组内共用一份笔记 —— 折中方案，现在主流模型都用它
- MLA：笔记压缩成"关键词提纲"，要用时再展开成完整笔记 —— 更省地方
- DSA：抄笔记前先瞄一眼，只找最相关的几个同学抄，不用抄全班
- KDA：不抄笔记了，用"滚动更新"的方式随时在心里记最新状态，抄再多也不累

【本代码做什么】
1. 用公式算出每种机制 KV Cache 实际占多少内存（数值化对比，超直观）
2. 用 numpy 写出 MHA / MQA / GQA 三种实现，让你亲眼看到"共享 K/V"在代码上怎么写
3. 模拟 DSA 的 Top-K 选择思想
================================================================================
"""
import numpy as np

np.set_printoptions(precision=3, suppress=True)

# ==============================================================================
# Part 1：KV Cache 大小数值对比（图片表格第一列的量化版）
# ==============================================================================
print("=" * 80)
print("Part 1：KV Cache 内存占用对比（把图里的'最大/较小/很小'变成具体数字）")
print("=" * 80)

# 假设一个中等规模模型（类比 qwen2.5:7b 的规模）
n_layers = 32          # Transformer 层数
n_heads = 8            # 查询头数
d_head = 128           # 每个头维度
seq_len = 4096         # 上下文长度（4K）
bytes_per_num = 2      # fp16 半精度，每个数占 2 字节

print(f"\n参数假设：{n_layers} 层 × {n_heads} 个查询头 × 每头 {d_head} 维 × 上下文 {seq_len} tokens × fp16")
print(f"（这就是一个 7B 级别模型跑 4K 上下文时的典型配置）\n")


def kv_cache_size(n_kv_heads, desc, note=""):
    """计算 KV Cache 大小：K和V两份 × 层数 × KV头数 × 序列长度 × 维度 × 字节数"""
    size = 2 * n_layers * n_kv_heads * seq_len * d_head * bytes_per_num
    size_mb = size / 1024 / 1024
    print(f"  {desc:<28} KV头数={n_kv_heads:<3} → {size_mb:>8.1f} MB  {note}")
    return size_mb


print("【7种机制的 KV Cache】")
mha = kv_cache_size(n_heads, "MHA (每Q头独立K/V)", "← 最大，全量缓存")
mqa = kv_cache_size(1, "MQA (所有Q头共享1套K/V)", "← 只剩 1/8")
gqa = kv_cache_size(2, "GQA (8头分2组，组内共享)", "← 只剩 1/4")
# MLA：把 K/V 压成低维 latent（比如 64 维），不再按头缓存
mla = kv_cache_size(1, "MLA (压成 64 维 latent)", "← 表示维度也变小")
mla = 2 * n_layers * seq_len * 64 * bytes_per_num / 1024 / 1024
print(f"  {'MLA (latent=64维)':<28} 按64维压缩       → {mla:>8.1f} MB  ← 最省")
# DSA：稀疏，只看 k 个 token（假设 k = 256）
dsa = 2 * n_layers * seq_len * 256 * d_head * bytes_per_num / 1024 / 1024 / 1  # 简化：每层只缓存选中的k个
print(f"  {'DSA (只缓存Top-256 token)':<28} 稀疏         → {dsa/1000:>8.1f} GB  ← 随k线性增长")

print(f"\n结论：MHA 是 MQA 的 {mha/mqa:.0f} 倍！长上下文场景下，KV Cache 直接决定你能跑多长。")

# ==============================================================================
# Part 2：代码级对比 MHA vs MQA vs GQA（共享 K/V 怎么写）
# ==============================================================================
print("\n\n" + "=" * 80)
print("Part 2：MHA / MQA / GQA 三种实现对比（观察 K/V 怎么共享）")
print("=" * 80)

np.random.seed(42)
seq, d_model = 3, 4   # 3个词，4维
X = np.random.randn(seq, d_model)

print(f"\n输入 X ({seq}×{d_model})，模拟 4 个查询头，每头 1 维（d_head=1）")


def mha_forward(X, n_heads=4):
    """MHA：每个 Q 头都有自己的 K、V（不共享）"""
    d_head = d_model // n_heads
    # 每个头独立投影：Wk 是 (n_heads, d_model, d_head)
    Wk = np.random.randn(n_heads, d_model, d_head) * 0.5
    Wv = np.random.randn(n_heads, d_model, d_head) * 0.5
    K = X @ Wk   # (seq, n_heads, d_head)
    V = X @ Wv
    print(f"  MHA: K 形状={K.shape}（每个头都有自己的 K，共 {n_heads} 份 K/V）")
    return K, V


def mqa_forward(X, n_heads=4):
    """MQA：所有 Q 头共享同一套 K、V"""
    d_head = d_model // n_heads
    Wk = np.random.randn(d_model, d_head) * 0.5   # 只有一份！
    Wv = np.random.randn(d_model, d_head) * 0.5   # 只有一份！
    K = X @ Wk   # (seq, d_head)
    V = X @ Wv
    print(f"  MQA: K 形状={K.shape}（只有 1 份 K/V，所有头共用）")
    print(f"       头在计算时用 K[None, :, :] 广播给每个头 → 效果等同 {seq} 个头都看同一份")
    return K, V


def gqa_forward(X, n_heads=4, n_groups=2):
    """GQA：Q 头分成 n_groups 组，组内共享 K/V"""
    d_head = d_model // n_heads
    # 只有 n_groups 份 K/V（2组 → 2份）
    Wk = np.random.randn(n_groups, d_model, d_head) * 0.5
    Wv = np.random.randn(n_groups, d_model, d_head) * 0.5
    K = X @ Wk   # (seq, n_groups, d_head)
    V = X @ Wv
    # 头到组的映射：头0,1 → 组0；头2,3 → 组1
    group_map = [h // (n_heads // n_groups) for h in range(n_heads)]
    print(f"  GQA: K 形状={K.shape}（{n_groups} 份 K/V）")
    print(f"       头分组映射: 头{list(range(n_heads))} → 组{group_map}")
    return K, V


print("\n" + "-" * 40)
K_mha, V_mha = mha_forward(X)
print("-" * 40)
K_mqa, V_mqa = mqa_forward(X)
print("-" * 40)
K_gqa, V_gqa = gqa_forward(X)

print("\n💡 一句话看懂代码差异：")
print("   MHA: Wk 是 (头数, 模型维, 头维) —— 每头一份 → 内存 × 头数")
print("   MQA: Wk 是 (模型维, 头维)       —— 只有一份 → 最省")
print("   GQA: Wk 是 (组数, 模型维, 头维) —— 每组一份 → 折中")

# ==============================================================================
# Part 3：DSA 动态稀疏注意力（图里第⑤种）的思想模拟
# ==============================================================================
print("\n\n" + "=" * 80)
print("Part 3：DSA（动态稀疏注意力）—— 只看 Top-K 个相关 token")
print("=" * 80)

# 模拟：当前要预测第 100 个词，历史有 99 个词，全算要 99 次点积
# DSA 的做法：先用轻量打分挑出最相关的 k 个，再只对它们做注意力
history_len = 99
k = 5  # 只挑 5 个

np.random.seed(7)
q = np.random.randn(8)                        # 当前词的 Query
history_keys = np.random.randn(history_len, 8)  # 99 个历史词的 Key

# 轻量打分（比如用一个小投影，图上写"通过轻量检索/路由/打分"）
scores = history_keys @ q                      # 99 个分数
top_k_idx = np.argsort(scores)[-k:][::-1]      # 取分数最高的 k 个

print(f"历史有 {history_len} 个词，DSA 只挑 Top-{k} 个关注")
print(f"选中位置: {sorted(top_k_idx)}")
print(f"分数最高的前3: {np.sort(scores)[-3:][::-1]}")
print(f"计算量对比：全量 {history_len} 次点积 → 稀疏只算 {k} 次（省 {(1-k/history_len)*100:.0f}%）")

# ==============================================================================
# Part 4：总结表（图片右下角的表格，代码化打印）
# ==============================================================================
print("\n\n" + "=" * 80)
print("Part 4：7 种注意力机制总览（对应图片右下角对比表）")
print("=" * 80)
print(f"""
  {'方法':<5} {'KV Cache':<16} {'计算复杂度':<10} {'效果':<8} {'代表'}
  {'─'*60}
  MHA   最大 (全量)          O(n²)        最强      早期Transformer / BERT
  MQA   小 (1/n_heads)       O(n²)        略损失    PaLM变体 / 早期LLaMA实验
  GQA   较小 (1/组数)        O(n²)        接近MHA   Llama3 / Qwen2 / Mistral / Gemma2
  MLA   很小 (压缩为latent)  O(n)         接近更好  DeepSeek-V3 / GLM-5
  DSA   小 (只缓存Top-K)     O(k·n)       较好      超长上下文 1M+ (GLM-5)
  CSA   很小 (压缩块)        O(n)         较好      超长上下文 1M+ (DeepSeek-V4)
  KDA   极小 (状态递推)      O(n)         较好      Kimi K3 (混合 Gated MLA)

演进逻辑：先解决 KV Cache 太大（MHA→GQA→MLA），
         再解决"不用看所有 token"（DSA/CSA），
         最后走向线性复杂度（KDA）。

没有最好的 Attention，只有最合适的组合 —— 效果、效率、成本之间找平衡。
""")

"""
【运行结果解读】
Part 1：你会看到 MHA 的 KV Cache 是 MQA 的 8 倍 —— 这就是为什么长上下文
        模型必须换掉 MHA。
Part 2：三份代码的差异就一行：Wk 是"每头一份 / 一份 / 每组一份"。
        共享得越多，KV Cache 越小，但表达力越受限。
Part 3：DSA 的核心 = 先打分，后只看 Top-K，省掉大部分计算。
"""
