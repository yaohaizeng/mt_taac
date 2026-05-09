# 论文笔记：HyFormer

> **阅读日期**：2026-05-08
> **论文链接**：https://arxiv.org/abs/2601.12681
> **来源**：ACM 会议论文（Conference acronym 'XX, 2025），arXiv v2: 2026-01-23
> **作者**：Yunwen Huang*, Shiyong Hong*, Xijun Xiao*, Jinqiu Jin* et al.（ByteDance AML & Search）
> **代码**：未开源

---

## 1. 一句话总结

> HyFormer 提出了一种统一的混合 Transformer 架构，通过 Global Tokens + 交替执行的 Query Decoding（跨注意力解码长序列）和 Query Boosting（MLP-Mixer 风格特征交互）两个模块，打破了工业推荐系统中序列建模与特征交互两阶段解耦的范式，在参数量和 FLOPs 相当甚至更低的情况下取得 SOTA AUC，并在抖音搜索全量上线。

---

## 2. 研究背景与动机

**要解决的核心问题：**
工业大规模推荐模型（LRM）需要在严格延迟约束下同时建模超长用户行为序列（千级别）与高维异构非序列特征（用户画像、上下文、交叉特征等），二者的联合建模是核心挑战。

**现有方法的局限性：**
- **局限 1 — 查询表达能力弱**：LONGER 等序列压缩器使用过于简单的 query token（仅来自候选 item 相关特征），上下文信息利用不充分；直接增加 query 数量又会在 KV-Cache / M-Falcon 机制下严重损害在线服务效率。
- **局限 2 — 特征交互来得太晚**：序列压缩后才与异构特征交互（Late Fusion），早期层表示无法受益于跨域上下文信息，细粒度依赖难以捕获。
- **局限 3 — Scaling 效率低**：交互模块仅作用于压缩后的序列表示，增加深度/参数量时性能收益递减，计算资源无法有效转化为更丰富的联合表示。

**本文切入点：**
将序列建模与特征交互从两个松耦合的阶段统一到单一 Backbone，通过 Global Tokens 作为共享语义接口，实现序列信息与异构特征的双向、逐层交互。

---

## 3. 核心贡献（Contributions）

1. **指出解耦范式的三大内在局限**：系统性分析了"先序列建模再特征交互"流水线的表达能力瓶颈和 Scaling 限制。
2. **提出 HyFormer 统一架构**：通过 Query Decoding（序列解码）和 Query Boosting（特征交互）的交替叠加，实现序列信息与异构特征的双向、逐层融合，已在抖音搜索系统全量上线，服务十亿级用户。
3. **验证超强 Scaling 特性**：在 200M–1B+ 参数和 3T–10T FLOPs 范围内，HyFormer 的性能增益斜率始终高于 LONGER+RankMixer 基线，具备更优的算力转化效率。

---

## 4. 方法详解

### 4.1 整体框架

模型输入包含多条行为序列（长期序列 ≤3000、搜索序列 Top-50、Feed 序列 Top-50）以及异构非序列（NS）特征（用户画像、上下文、交叉特征等）。NS 特征首先经 Query Generation 生成少量 Global Tokens，随后多个 HyFormer 层交替执行 Query Decoding 和 Query Boosting，最终将顶层输出送入 MLP 输出 CTR 预测概率。

```
输入：长行为序列 S（多条） + 异构 NS 特征 F1...FM
  ↓
Query Generation（首层）：NS 特征 + 序列 MeanPool → N 个 Global Query Tokens
  ↓  ↓  ↓  (每层循环)
Query Decoding：Global Tokens cross-attn 长序列 KV → 序列感知 Query
  ↓
Query Boosting：[Decoded Query ‖ NS Tokens] MLP-Mixer token mixing → 增强 Query
  ↓（重复 L 层）
顶层 Global Tokens → MLP → CTR 预测
```

**一个端到端的例子（用户搜索 "iPhone 17 Pro 评测"，候选视频为某科技博主评测）**：

```
第 0 层（初始化）
  Query Generation：
    Global Info = [用户画像, 搜索词, 候选视频特征, 交叉特征, 序列MeanPool]
    Q_1, Q_2, Q_3 ← 分别由 3 个 FFN 从 Global Info 投影而来
    此时 Q 仅含 NS 特征信息，尚未接触序列

第 1 层
  Query Decoding（Q_1 解码长期序列）：
    Q_1 关注历史中的 iPhone 14/16 开箱、MacBook 评测等条目
    Q_1 更新为：「对苹果生态感兴趣的用户的长期偏好向量」
  Query Boosting：
    16 个 Token 做 MLP-Mixer 混合
    Q_1 吸收了用户画像（高消费力）、搜索词（强意图）信息
    Q_1 升级为：「高消费力用户搜索苹果旗舰机评测的综合语义向量」

第 2 层
  Query Decoding（Q_1 再次解码长期序列，但现在 Q_1 更富语义）：
    这次 Q_1 能更精准地关注"高端手机评测"类内容，弱化早期的泛娱乐行为
    Q_1 进一步更新为更精炼的用户兴趣表示
  Query Boosting：
    进一步融合候选视频特征（该博主的风格、完播率历史）
    最终 Q_1 包含：用户兴趣 × 搜索意图 × 序列行为 × 候选视频的综合语义

输出：
  [Q_1_final, Q_2_final, Q_3_final] → MLP → P(点击) = 0.87（示意）
```

与旧范式对比：LONGER+RankMixer 在 LONGER 阶段把 3000 条序列压缩为 1 个固定向量，再拼入 RankMixer 做一次性交互，**Query 无法随着特征交互的深入而演化**；HyFormer 则让 Query 在每一层都被序列信息和 NS 特征联合打磨，越深层的 Query 越精准。

### 4.2 核心模块

#### 模块 1：Query Generation（查询生成）

**作用**：将异构 NS 特征压缩为少量（N 个）Global Query Tokens，作为后续逐层解码长序列的语义载体。

**数学形式**：

$$
\text{Global Info} = \text{Concat}(F_1, \dots, F_M, \text{MeanPool}(S))
$$

$$
Q = [\text{FFN}_1(\text{Global Info}), \dots, \text{FFN}_N(\text{Global Info})] \in \mathbb{R}^{N \times D}
$$

**实现要点**：
- N 远小于 NS Token 总数（MTGR/OneTrans 的做法），保持在线服务效率，仅 3 个 Global Token（每条序列对应一个）。
- 深层 HyFormer 中不再重新生成 Query，而是复用上一层的 boosted query 作为当前层输入。
- 额外引入跨序列的 MeanPool token 作为 Query 的一部分，消融实验证明去掉它 AUC 下降 0.05%。

**直观例子**：

假设当前用户正在抖音搜索"iPhone 17 Pro 评测"，候选视频是某科技博主发布的手机评测视频。NS 特征包含：

```
F_user    = [25岁, 男, 北京, 高消费力]           # 用户画像
F_query   = ["iPhone 17 Pro 评测" 的文本向量]     # 搜索词
F_item    = [手机评测, 科技博主, 时长8分钟]        # 候选视频特征
F_cross   = [用户×科技类交叉特征]
MeanPool(S_long) = 用户过去3000条行为的平均向量   # 序列摘要
```

Query Generation 把以上所有信息 Concat 后，通过 3 个独立 FFN 各自投影出 3 个 Global Query Token：

```
Q_1（for 长期序列）= FFN_1([F_user, F_query, F_item, F_cross, MeanPool(S)])
Q_2（for 搜索序列）= FFN_2([F_user, F_query, F_item, F_cross, MeanPool(S)])
Q_3（for Feed 序列）= FFN_3([F_user, F_query, F_item, F_cross, MeanPool(S)])
```

这 3 个 token 的语义大致是「一个对手机感兴趣、有消费力的北京男性，正在搜索 iPhone 评测，想从历史行为中提取相关信号」——它们将作为"探针"去查询长序列 KV，比旧方法只用候选 item 特征做 query 携带了更多上下文信息。

#### 模块 2：Query Decoding（查询解码）

**作用**：让 Global Tokens 通过跨注意力直接从长行为序列的逐层 KV 表示中提取目标相关信息，实现序列信息向 Query 的双向流动。

**数学形式**（以第 l 层为例）：

$$
\tilde{Q}^{(l)} = \text{CrossAttn}(Q^{(l)}, K^{(l)}, V^{(l)})
$$

其中 KV 来自序列编码器，支持三种策略：

| 编码策略 | 公式 | 复杂度 | 适用场景 |
|---------|------|--------|---------|
| Full Transformer | $H_l = \text{TransformerEnc}_l(S)$ | $O(L_S^2)$ | 最高精度 |
| LONGER-style（推荐） | $H_l = \text{CrossAttn}(S_{\text{short}}, S, S)$ | $O(L_H L_S)$ | 工业在线服务 |
| Decoder-style | $H_l = \text{SwiGLU}_l(S)$ | $O(L_S)$ | 极低时延场景 |

**实现要点**：
- 长序列 KV 在每层重新计算，而非共享，使序列表示可随 Decoder 深度演化。
- 多序列场景下，每条序列使用独立的 Query Token 组做 Decoding，不强制合并序列。

**直观例子**：

延续上面的场景。用户的长期序列有 3000 条历史行为，包含：

```
位置   1: [篮球教学视频, 3个月前]
位置  88: [iPhone 14 开箱视频, 2个月前]
位置 301: [MacBook 评测, 1个月前]
位置 890: [iPhone 16 评测, 2周前]
...
位置3000: [今天刚看的手机壳推荐]
```

序列编码器把这 3000 条行为映射为 KV 矩阵（$K, V \in \mathbb{R}^{3000 \times D}$）。

Query Decoding 让 Q_1（携带"用户搜索 iPhone 评测"语义的 Global Token）对这 3000 个 KV 做 Cross-Attention：

```
Q_1 的注意力权重（示意）：
  位置 88  (iPhone 14 开箱)  → 权重 0.18  ← 高度相关
  位置 301 (MacBook 评测)   → 权重 0.12  ← 相关（同品牌）
  位置 890 (iPhone 16 评测) → 权重 0.31  ← 最相关
  位置   1 (篮球教学)        → 权重 0.001 ← 不相关，被忽略
```

Attention 加权后，Q_1 被更新为「融合了用户历史苹果系产品兴趣的序列感知向量」，旧方法只把序列压缩成一个固定向量再拼接，无法像这样让 Query 主动选择关注哪些历史片段。

#### 模块 3：Query Boosting（查询增强）

**作用**：在 Decoding 后，对 Decoded Query 和 NS Tokens 进行跨 Token 混合，强化 cross-query、cross-sequence 的异构特征交互。

**数学形式**：

$$
Q = [\tilde{Q}^{(l)}, F_1, \dots, F_M] \in \mathbb{R}^{T \times D}, \quad T = N + M
$$

每个 token 拆分为 T 个子空间，MLP-Mixer 对各子空间做跨位置聚合：

$$
\tilde{q}_h = \text{Concat}(q_1^{(h)}, q_2^{(h)}, \dots, q_T^{(h)}) \in \mathbb{R}^D
$$

经 PerToken-FFN 精炼后残差连接：

$$
Q_{\text{boost}} = Q + \text{PerToken-FFN}(\hat{Q})
$$

**实现要点**：
- Token 数固定为 16（13 个 NS Token + 3 个 Global Token），与 RankMixer 输入对齐，便于部署替换。
- MLP-Mixer 计算复杂度为线性，不引入额外 Self-Attention，规避 MTGR 的效率问题。
- Residual 连接稳定优化，保留 Decoded Query 的原始语义。

**直观例子**：

经过 Query Decoding，3 个 Global Token 分别从长期序列、搜索序列、Feed 序列中提取了序列感知信息：

```
解码后的 16 个 Token（T = 3 Global + 13 NS）：
  Token_1 = Q̃_长期  [融合了iPhone历史兴趣]
  Token_2 = Q̃_搜索  [融合了近期搜索行为]
  Token_3 = Q̃_Feed  [融合了推荐流浏览偏好]
  Token_4 = F_user   [用户画像: 25岁男性, 北京, 高消费]
  Token_5 = F_query  [搜索词: iPhone 17 Pro 评测]
  Token_6 = F_item   [候选视频: 科技博主, 手机评测]
  ...
  Token_16 = F_cross [用户×科技类交叉特征]
```

此时 Q̃_长期 虽然知道"用户历史喜欢苹果产品"，但还不知道"用户是高消费人群"（这在 F_user 里）。

MLP-Mixer 对这 16 个 token 做**跨位置混合**：把每个 token 的第 h 个子空间中的所有位置信息拼起来过 MLP：

```
子空间 h=5 的混合过程（示意）：
  输入：[q_1^(5), q_2^(5), ..., q_16^(5)]  ← 16 个 token 在第 5 个子空间的切片
  拼接后过 MLP → 新的 16 维向量
  每个位置都能"看到"其他所有 token 在该子空间的信息
```

混合后 Q̃_长期 的表示里就同时融合了"用户历史喜欢苹果产品"**+**"用户是高消费力的北京男性"，下一层 Decoding 时它能更精准地从序列中提取与高消费用户更匹配的内容（如高端机型、专业评测），而不只是泛泛的"手机相关内容"。

#### 模块 4：Multi-Sequence Modeling

**作用**：工业场景中存在多条异构序列（长期序列、搜索序列、Feed 序列），语义空间不同，强制合并会丢失各序列的个性化信息。

**设计**：每条序列独立分配专属 Global Query Token，各自做 Query Decoding；跨序列交互在 Query Boosting 的 token mixing 中隐式完成。

> 消融：强制合并序列（sequence merge）导致 AUC 下降 0.06%，验证了独立建模的必要性。

**直观例子**：

同一用户有三条行为序列，语义完全不同：

```
长期序列（3000条，跨越数月）：
  [篮球视频, 手机评测, 美食探店, iPhone开箱, 旅行vlog, ...]
  → 记录用户长期兴趣，条目跨越多个领域，时间跨度大

搜索序列（Top-50，近期搜索点击）：
  ["iPhone 17 Pro 评测", "索尼相机对比", "MacBook Air M4", ...]
  → 主动搜索意图明确，代表当前强需求

Feed 序列（Top-50，推荐流浏览）：
  [搞笑视频, 宠物视频, 美食视频, 科技新闻, ...]
  → 被动浏览，兴趣更泛化，时效性强
```

**合并方式（MTGR/OneTrans 的做法）**：把三条序列硬拼成 3100 条，用统一的 KV 做 Decoding：

```
合并后 KV：[长期序列 3000条 ‖ 搜索序列 50条 ‖ Feed 序列 50条]
问题：搜索序列里"iPhone 17 Pro"的 item embedding 和长期序列里的同一个视频
embedding 来自不同特征空间（一个是搜索语义，一个是推荐语义），
强行拼接后 Attention 会产生跨域的噪声对齐。
```

**HyFormer 的独立方式**：三条序列各自有专属 Global Token，独立 Decoding：

```
Q_1 专门解码长期序列 → 提取"用户对苹果产品的长期兴趣"
Q_2 专门解码搜索序列 → 提取"用户近期主动搜索 iPhone 的强意图"
Q_3 专门解码 Feed 序列 → 提取"用户在推荐流上对科技内容的浏览偏好"

三者的语义被单独保留，在 Query Boosting 的 token mixing 阶段
才进行跨序列的信息融合（此时已是抽象的语义 token，不再有特征空间冲突）。
```

### 4.3 训练设置

| 项目 | 内容 |
|------|------|
| 损失函数 | 标准 Binary Cross-Entropy（BCE） |
| Batch Size | 2048 |
| 训练集群 | 64 GPU 集群 |
| 数据规模 | 30 亿样本，70 天用户交互日志 |
| 关键工程优化 | GPU Pooling（序列特征去重）+ 异步 AllReduce |

---

## 5. 实验分析

### 5.1 数据集与评估指标

| 数据集 | 规模 | 任务 | 指标 |
|--------|------|------|------|
| 抖音搜索系统内部数据集 | 30 亿样本，70 天 | CTR 预测 | Query-level AUC |

序列组成：长期序列（≤3000 items）、搜索序列（Top-50）、Feed 序列（Top-50）

### 5.2 Baseline 对比（Table 1）

| 序列建模 | 特征交互 | AUC | ΔAUC | Params (M) | FLOPs (T) |
|---------|---------|-----|------|-----------|----------|
| LONGER | RankMixer（生产基线） | 0.6478 | – | 386 | 3.5 |
| LONGER | Full Transformer | 0.6472 | -0.09% | 416 | 6.2 |
| LONGER | Wukong | 0.6465 | -0.20% | 385 | 5.2 |
| Full Transformer | RankMixer | 0.6481 | +0.05% | 388 | 6.6 |
| Full Transformer | Full Transformer | 0.6474 | -0.06% | 418 | 9.3 |
| Full Transformer | Wukong | 0.6468 | -0.15% | 387 | 8.3 |
| MTGR/OneTrans（LONGER） | — | 0.6480 | +0.03% | 406 | 6.6 |
| MTGR/OneTrans（Full Trans） | — | 0.6483 | +0.08% | 450 | 21.9 |
| **HyFormer（本文）** | — | **0.6489** | **+0.17%** | **418** | **3.9** |

**核心结论**：HyFormer 在 AUC 最高的同时 FLOPs 仅 3.9T，是所有高性能方法中计算量最低的，比 MTGR/OneTrans（Full Trans）低 5.6×，比 Full Transformer + Full Transformer 低 2.4×。

### 5.3 消融实验（Table 2）

| 配置 | AUC | ΔAUC | 结论 |
|------|-----|------|------|
| HyFormer（完整） | 0.6489 | – | 完整模型 |
| Query 去掉序列 Pooling Token | 0.6486 | -0.05% | 跨序列汇聚信息有贡献 |
| Query 仅保留 target 特征 | 0.6484 | -0.08% | Global Context 不可或缺 |
| HyFormer 去掉 Global Tokens | 0.6484 | -0.08% | Query Boosting 模块贡献 0.08% |
| BaseArch + Global Tokens | 0.6480 | -0.14% | 仅加 Global Token 不够，统一架构才是关键 |
| BaseArch（LONGER+RankMixer） | 0.6478 | -0.17% | 生产基线 |
| HyFormer + 强制合并多序列 | 0.6485 | -0.06% | 独立多序列建模优于合并 |

**最关键组件**：Query Boosting（统一架构设计）是最核心的贡献，仅靠丰富 Query 信息而不改变架构，增益从 0.17% 缩水至 0.03%。

### 5.4 Scaling 分析（Table 3 & Figure 3）

**参数 Scaling（200M → 1B+）**：HyFormer 的 AUC-Params 曲线斜率始终高于 BaseArch，说明每个额外参数在 HyFormer 中带来更大的性能增益。

**FLOPs Scaling（3T → 10T）**：遵循幂律趋势，双向信息流设计使计算资源被更有效地利用。

**稀疏维度 Scaling**（序列 side info 从 64 维扩展到 224 维）：

| 序列长度 | 架构 | Δ(sparse dim) AUC gain |
|---------|------|----------------------|
| 1k | BaseArch | +0.09% |
| 1k | HyFormer | +0.12%（多 0.03%）|
| 3k | BaseArch | +0.06% |
| 3k | HyFormer | +0.12%（多 0.06%）|

序列越长、side info 越丰富，HyFormer 的相对优势越显著。

### 5.5 在线 A/B 测试（Table 4）

测试平台：抖音搜索，对照组为部署中的 RankMixer 强基线

| 指标 | 增益 |
|------|------|
| 人均观看时长 | +0.293% |
| 人均完播视频数 | +1.111% |
| 查询改写率（负向指标） | -0.236% |

---

## 6. 工程落地视角

### 6.1 效率分析

| 维度 | 描述 |
|------|------|
| 参数量 | 418M，与 Full Transformer + Full Transformer（418M）持平 |
| 训练 FLOPs | 3.9T，是同等精度竞争者中最低（MTGR Full Trans 的 18%） |
| 在线依赖 | KV-Cache 兼容（LONGER-style 编码），不增加 query 数量，维持服务效率 |
| 部署状态 | 已在抖音搜索全量上线，服务十亿级用户 |

### 6.2 工程优化细节

**GPU Pooling（序列去重）**：
- 长序列中唯一 Feature ID 仅占约 25%，通过去重 + GPU 端重建大幅降低 Host→Device 传输量和 Host 内存压力。
- 前向算子在 GPU 端重建序列特征，反向算子将梯度聚合回 Embedding Table，对稀疏参数透明。

**异步 AllReduce**：
- 步骤 k 的梯度同步与步骤 k+1 的前向/反向并行执行，消除通信气泡，最大化 GPU 利用率。
- 副作用：Dense 参数存在一步 staleness（`W_k = W_{k-1} + g_{k-1}`），Sparse 参数可即时更新（`W_k = W_{k-1} + g_k`），实验表明此不一致不影响收敛。

### 6.3 可复现要点

- **Token 数量对齐**：MLPMixer 输入 Token 数固定为 16（13 NS + 3 Global），与 RankMixer 对齐，替换时只需改交互模块。
- **序列编码灵活替换**：三种序列编码策略（Full Trans / LONGER / SwiGLU）可按时延预算灵活选择，推荐工业默认用 LONGER-style。
- **多序列不合并**：不同语义的序列（搜索/Feed/长期行为）需分配独立 Query Token，强制合并会损失 0.06% AUC。
- **Query Token 数量**：每条序列 1 个 Global Token（本文），可按重要性自适应分配更多 Token 给关键序列（论文提及作为扩展方向）。

### 6.4 改进/扩展方向

- 自适应 Global Token 分配：对重要序列分配更多 Global Token（论文提及但未实验）。
- 序列侧 KV 异步预计算：将长序列 KV 离线缓存（类 M-Falcon），进一步降低在线时延。
- 与 MoE 结合：在 Query Boosting 的 PerToken-FFN 中引入专家混合，提升参数利用效率。

---

## 7. 相关工作定位

| 类别 | 代表方法 | HyFormer 的改进点 |
|------|---------|-----------------|
| 长序列建模 | SIM, ETA, TWIN, TransAct, LONGER | 复用 LONGER 的高效编码，但将其解码器从单一 query 升级为多 Global Query，并嵌入统一框架 |
| 特征交互 | DeepFM, xDeepFM, DCNv2, Wukong, RankMixer | 保留 MLP-Mixer 的高效 token mixing，但将其从序列建模下游提升为逐层双向交互环节 |
| 统一架构 | HSTU, InterFormer, MTGR, OneTrans | 避免了 MTGR/OneTrans 的两大缺陷：①query 数量与 NS token 数量正比（效率差）；②使用 Self-Attention 做特征交互（精度反而不如 RankMixer） |

---

## 8. 个人评价与疑问

**亮点**：
- 工程与算法联合设计的典范：Global Token 数量被精心控制为 3（维持 KV-Cache 效率），token mixing 输入固定为 16（复用 RankMixer 架构），改动最小、增益最大。
- 消融设计严谨，清晰区分了"丰富 Query 信息"和"改变架构为双向"两个维度各自的贡献（+0.03% vs +0.14%）。
- Scaling Law 实验完整，覆盖参数量、FLOPs、序列长度、稀疏维度四个维度，工程说服力强。

**疑问 / 待深入**：
- > ⚠️ Query Boosting 中 MLP-Mixer 的 token mixing 对 NS Token 也做了 mixing（非仅 Global Token），这意味着 NS Token 的表示也会在每层被更新？论文中不够明确，需确认 NS Token 是否参与 residual 更新。
- > ⚠️ 三条序列各用 1 个 Global Token，但序列重要性不同（长期序列 ≤3000 vs 搜索/Feed Top-50），1:1:1 分配是否次优？论文提到可自适应分配但未给出结果。
- > ⚠️ 在线部署中，LONGER-style 的 KV 是否做了离线预计算（M-Falcon）？论文提及 KV-Cache 但未明确在线架构细节。

**与当前工作的关联**：
- 若当前项目涉及长视频/短视频行为序列 + 多模态特征的 CTR/排序任务，HyFormer 的"少量 Global Token 作为序列解码器 + 逐层双向交互"思路值得直接借鉴，可替换现有两阶段序列编码 + 特征交互的流水线。
- GPU Pooling（序列特征去重）和异步 AllReduce 两个工程优化技巧可独立应用于任何包含长序列特征的训练任务。

---

## 9. 关键词 & 标签

`CTR预测` `推荐系统` `长序列建模` `特征交互` `Transformer` `MLP-Mixer` `工业大模型` `KV-Cache` `Scaling Law` `ByteDance` `抖音搜索`
