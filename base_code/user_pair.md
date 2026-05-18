# User Pair v3 方案说明

本文档说明如何在原始 baseline 的基础上加入 `user_int_feats_62` 到 `user_int_feats_66` 的 user pair 建模，并形成 v3 方案：

```text
profile_pair_token + item_aware_pair_token
```

文档重点包括：

```text
1. 原 baseline 的问题
2. 为什么要单独建模 62-66
3. 如何从 schema 定位这些特征
4. UserPairEncoder 的设计
5. 两个 token 的构造方式
6. 如何接入 PCVRHyFormer 主模型
7. 训练端和推理端如何适配
8. v4 的简要补充思路
```

---

## 1. 原 baseline 的问题

原始 baseline 的模型输入主要由以下部分组成：

```python
ModelInput(
    user_int_feats,
    item_int_feats,
    user_dense_feats,
    item_dense_feats,
    seq_data,
    seq_lens,
    seq_time_buckets,
    ...
)
```

其中：

```text
user_int_feats:
  用户侧离散特征，通常进入 embedding/tokenizer。

user_dense_feats:
  用户侧连续特征，通常进入 dense projection。

item_int_feats:
  广告/物品侧离散特征，通常进入 item tokenizer。

seq_data:
  用户历史行为序列特征。
```

原 baseline 对 `user_int_feats` 和 `user_dense_feats` 是分开处理的：

```text
user_int_feats -> embedding/tokenizer -> user_ns
user_dense_feats -> dense projection -> user_dense_token
```

这种方式对普通特征是合理的，但对 `user_int_feats_62` 到 `user_int_feats_66` 这几个特征不够充分。

原因是这几个特征具有明显的 pair 结构：

```text
user_int_feats_65   = [兴趣id1, 兴趣id2, 兴趣id3, ...]
user_dense_feats_65 = [权重1,   权重2,   权重3,   ...]
```

它们不是两组互相独立的特征，而是同位置一一对应：

```text
(兴趣id1, 权重1)
(兴趣id2, 权重2)
(兴趣id3, 权重3)
```

如果仍然按照原 baseline 的方式分开建模，模型虽然能看到 id 和 dense 数值，但不一定能明确知道：

```text
哪个 id 对应哪个权重
哪个兴趣更强
哪个兴趣和当前广告更相关
```

v3 的目标就是把这种 `(id, weight)` 的对齐关系显式建模出来。

---

## 2. v3 的核心思路

v3 的核心思想是：

```text
把 user_int_feats_62 到 user_int_feats_66
和对应的 user_dense_feats_62 到 user_dense_feats_66
作为 aligned pair 特征处理。
```

所谓 aligned pair，就是：

```text
同一个 fid 下，
user_int_feats 的第 i 个位置
与 user_dense_feats 的第 i 个位置
语义对应。
```

例如：

```text
user_int_feats_65[j]   表示某个兴趣/标签/类别 id
user_dense_feats_65[j] 表示这个 id 对应的强度、权重、分数或频次
```

v3 不再只让模型分别理解 id 和 dense 值，而是直接建模：

```text
某个用户兴趣 id 的强度是多少。
```

进一步地，v3 会构造两个额外的 non-sequence token：

```text
profile_pair_token
item_aware_pair_token
```

它们的作用分别是：

```text
profile_pair_token:
  表示用户整体兴趣画像。

item_aware_pair_token:
  表示当前广告/item 感知的用户兴趣匹配。
```

这两个 token 会被拼入 HyFormer 的 `ns_tokens`，和其他用户、广告、时间、序列信息一起参与后续建模。

---

## 3. 使用哪些原始特征

v3 默认使用以下用户侧 fid：

```text
62, 63, 64, 65, 66
```

这些 fid 同时存在于：

```text
user_int_feats
user_dense_feats
```

其中 `user_int_feats` 提供离散 id，`user_dense_feats` 提供对应权重。

运行参数可以设计为：

```text
--use_user_pair
--user_pair_fids 62,63,64,65,66
--user_pair_emb_dim 32
--user_pair_use_rank
--user_pair_item_aware
```

推荐默认配置：

```text
use_user_pair = True
user_pair_fids = 62,63,64,65,66
user_pair_emb_dim = 32
user_pair_use_rank = True
user_pair_item_aware = True
```

---

## 4. 如何从 schema 定位 62-66

训练代码不能手写这些 fid 在 tensor 中的位置，因为 demo 数据、全量数据或后续数据版本的展开维度可能不同。

正确做法是从 `schema.json` 中读取每个 fid 的：

```text
int_offset
dense_offset
length
vocab_size
```

训练端新增函数：

```python
build_user_pair_specs(
    schema,
    dense_schema,
    per_position_vocab_sizes,
    fids,
)
```

它为每个 fid 生成：

```python
{
    "fid": fid,
    "vocab_size": vocab_size,
    "int_offset": int_offset,
    "dense_offset": dense_offset,
    "length": int_len,
}
```

字段含义：

```text
fid:
  特征编号，例如 62。

vocab_size:
  当前 fid 的离散 id 词表大小，用来创建 embedding。

int_offset:
  当前 fid 在 user_int_feats 展开向量中的起始位置。

dense_offset:
  当前 fid 在 user_dense_feats 展开向量中的起始位置。

length:
  当前 fid 的多值长度。
```

以全量训练日志里的 schema 为例：

```text
fid 62: int_offset=43,  dense_offset=256, length=6
fid 63: int_offset=49,  dense_offset=262, length=19
fid 64: int_offset=68,  dense_offset=281, length=26
fid 65: int_offset=94,  dense_offset=307, length=111
fid 66: int_offset=205, dense_offset=418, length=150
```

那么 `fid=65` 对应：

```text
ids:
  user_int_feats[:, 94:205]

weights:
  user_dense_feats[:, 307:418]
```

两者长度都是 111，所以可以一一配对。

代码中必须检查：

```python
if int_len != dense_len:
    raise ValueError(...)
```

因为 user pair 建模的前提就是：

```text
离散 id 数量 == dense 权重数量
```

如果长度不同，就不能可靠地建立 `(id, weight)` 对应关系。

---

## 5. UserPairEncoder 的输入输出

v3 的核心模块是：

```python
class UserPairEncoder(nn.Module):
    ...
```

它的输入是：

```text
user_int_feats:   [B, user_int_total_dim]
user_dense_feats: [B, user_dense_total_dim]
item_ns:          [B, num_item_ns, d_model]
```

它的输出是：

```text
pair_tokens: [B, 2, d_model]
```

其中：

```text
pair_tokens[:, 0, :] = profile_pair_token
pair_tokens[:, 1, :] = item_aware_pair_token
```

常用维度：

```text
B = batch size
pair_emb_dim = 32
d_model = 64
num_item_ns = 2
```

UserPairEncoder 内部先用 32 维空间处理每个 fid 的 pair 信息，再投影到主模型的 `d_model` 维空间。

---

## 6. 每个 fid 内部如何编码

对每个 fid，例如 `fid=65`，先根据 `user_pair_specs` 从输入中切片：

```python
ids = user_int_feats[:, int_offset:int_offset + length].long()
dense_weights = user_dense_feats[:, dense_offset:dense_offset + length].float()
```

形状：

```text
ids:           [B, L]
dense_weights: [B, L]
```

其中：

```text
L = 当前 fid 的多值长度
```

### 6.1 padding mask

离散 id 中，0 表示 padding。

```python
mask = ids > 0
```

形状：

```text
mask: [B, L]
```

后续计算中，padding 位置不会参与 pooling。

### 6.2 dense 权重变换

对 dense 权重做：

```python
log_weights = torch.log1p(torch.clamp(dense_weights, min=0.0))
```

含义：

```text
先把负数截断为 0
再用 log1p 压缩大权重
```

这样可以：

```text
1. 避免负值干扰注意力分布
2. 避免极大权重完全支配结果
3. 保留权重大小关系
```

### 6.3 fid 内部 softmax

padding 位置设为很小的数：

```python
scores = log_weights.masked_fill(~mask, -1.0e4)
```

再做 softmax：

```python
attn = torch.softmax(scores, dim=1) * mask.float()
```

形状：

```text
attn: [B, L]
```

`attn` 表示当前 fid 内每个 id 的归一化重要性。

### 6.4 id embedding

每个 fid 单独创建一个 embedding 表：

```python
nn.Embedding(vocab_size + 1, pair_emb_dim, padding_idx=0)
```

查表得到：

```python
slot_emb = emb(ids)
```

形状：

```text
slot_emb: [B, L, pair_emb_dim]
```

默认：

```text
slot_emb: [B, L, 32]
```

### 6.5 rank embedding

多值特征的顺序可能隐含重要性，例如越靠前越重要。

v3 加入 rank embedding：

```python
ranks = torch.arange(1, length + 1, device=ids.device).unsqueeze(0)
ranks = ranks.expand_as(ids) * mask.long()
slot_emb = slot_emb + rank_emb(ranks)
```

这样模型可以区分：

```text
同一个 id 出现在第 1 位
和同一个 id 出现在第 50 位
```

### 6.6 weighted pooling 得到 fid_vec

对当前 fid 内部的所有 slot 加权求和：

```python
fid_vec = (slot_emb * attn.unsqueeze(-1)).sum(dim=1)
```

形状：

```text
fid_vec: [B, 32]
```

对 `62-66` 五个 fid 都做同样操作，得到：

```text
fid62_vec: [B, 32]
fid63_vec: [B, 32]
fid64_vec: [B, 32]
fid65_vec: [B, 32]
fid66_vec: [B, 32]
```

堆叠：

```python
fid_stack = torch.stack(fid_vecs, dim=1)
```

得到：

```text
fid_stack: [B, 5, 32]
```

---

## 7. 构造 profile_pair_token

`profile_pair_token` 表示用户整体兴趣画像。

它不依赖当前广告，只总结用户在 `fid62-fid66` 这五组特征上的整体兴趣。

实现方式：

```python
profile_pair_token = self.profile_proj(fid_stack.flatten(1)).unsqueeze(1)
```

维度变化：

```text
fid_stack: [B, 5, 32]
flatten:   [B, 160]
```

`profile_proj` 结构：

```text
Linear(5 * pair_emb_dim -> d_model)
LayerNorm(d_model)
SiLU
Dropout
```

默认：

```text
Linear(160 -> 64)
```

输出：

```text
profile_pair_token: [B, 1, d_model]
```

如果 `d_model=64`：

```text
profile_pair_token: [B, 1, 64]
```

业务含义：

```text
这是一个浓缩后的用户兴趣画像 token。
它告诉主模型：这个用户整体偏好什么。
```

---

## 8. 构造 item_aware_pair_token

`item_aware_pair_token` 是 v3 的关键。

用户兴趣通常是多面的。例如一个用户可能同时对：

```text
游戏
汽车
美妆
母婴
理财
```

都有兴趣。

但当前广告只会命中其中一部分兴趣。

所以模型不应该只问：

```text
用户喜欢什么？
```

还应该问：

```text
用户的哪部分兴趣和当前广告最相关？
```

### 8.1 得到 item_pool

原模型会把 `item_int_feats` 编码成 item 侧 NS token：

```python
item_ns = self.item_ns_tokenizer(inputs.item_int_feats)
```

形状：

```text
item_ns: [B, num_item_ns, d_model]
```

常用配置：

```text
item_ns: [B, 2, 64]
```

对 item token 做平均池化：

```python
item_pool = item_ns.mean(dim=1)
```

得到：

```text
item_pool: [B, d_model]
```

即：

```text
item_pool: [B, 64]
```

它可以理解为当前广告/item 的整体向量表示。

### 8.2 用 item_pool 生成 query

把 item_pool 投影到 pair 内部空间：

```python
query = self.item_query_proj(item_pool)
```

形状：

```text
query: [B, pair_emb_dim]
```

默认：

```text
query: [B, 32]
```

这个 query 表示：

```text
当前广告想从用户兴趣中查询什么。
```

### 8.3 对 5 个 fid 做 item-aware attention

已有：

```text
fid_stack: [B, 5, 32]
query:     [B, 32]
```

用 query 和每个 fid_vec 做点积：

```python
attn_scores = (fid_stack * query.unsqueeze(1)).sum(dim=-1)
attn_scores = attn_scores / math.sqrt(pair_emb_dim)
attn = torch.softmax(attn_scores, dim=1)
```

形状：

```text
attn_scores: [B, 5]
attn:        [B, 5]
```

含义：

```text
当前广告对 fid62-fid66 这五组用户兴趣的关注程度。
```

例如：

```text
fid62: 0.05
fid63: 0.08
fid64: 0.17
fid65: 0.60
fid66: 0.10
```

说明当前广告最关注 fid65 表达的用户兴趣。

### 8.4 得到 item_pair_vec

用 attention 对 5 个 fid_vec 加权求和：

```python
item_pair_vec = (fid_stack * attn.unsqueeze(-1)).sum(dim=1)
```

形状：

```text
item_pair_vec: [B, 32]
```

它表示：

```text
针对当前广告动态提取出来的用户兴趣。
```

### 8.5 显式建模兴趣和 item 的匹配

将 item_pool 也投影到 32 维：

```python
item_ctx = self.item_context_proj(item_pool)
```

形状：

```text
item_ctx: [B, 32]
```

然后拼接：

```python
item_token_input = torch.cat([
    item_pair_vec,
    item_ctx,
    item_pair_vec * item_ctx,
], dim=-1)
```

形状：

```text
item_token_input: [B, 96]
```

其中：

```text
item_pair_vec * item_ctx
```

是显式交互项，用来表达：

```text
用户当前被激活的兴趣
和当前广告表示
在哪些维度上匹配。
```

最后经过：

```text
Linear(96 -> d_model)
LayerNorm(d_model)
SiLU
Dropout
```

得到：

```text
item_aware_pair_token: [B, 1, d_model]
```

默认：

```text
item_aware_pair_token: [B, 1, 64]
```

---

## 9. UserPairEncoder 最终输出

最终把两个 token 拼接：

```python
pair_tokens = torch.cat([
    profile_pair_token,
    item_aware_pair_token,
], dim=1)
```

形状：

```text
pair_tokens: [B, 2, d_model]
```

默认：

```text
pair_tokens: [B, 2, 64]
```

两个 token 分别表示：

```text
pair_tokens[:, 0, :]:
  用户整体兴趣画像。

pair_tokens[:, 1, :]:
  当前广告感知的用户兴趣匹配。
```

---

## 10. 如何接入 PCVRHyFormer

原模型会构造 non-sequence tokens，即 `ns_tokens`。

v3 的做法是把：

```text
profile_pair_token
item_aware_pair_token
```

也作为 NS token 拼进去。

代码结构如下：

```python
user_ns = self.user_ns_tokenizer(inputs.user_int_feats)
item_ns = self.item_ns_tokenizer(inputs.item_int_feats)

ns_parts = [user_ns]

if self.has_user_dense:
    user_dense_tok = F.silu(
        self.user_dense_proj(inputs.user_dense_feats)
    ).unsqueeze(1)
    ns_parts.append(user_dense_tok)

sample_time_tok = self._make_sample_time_token(
    inputs.sample_time_dense,
    inputs.sample_time_buckets,
)
if sample_time_tok is not None:
    ns_parts.append(sample_time_tok)

if self.has_user_pair:
    pair_tokens = self.user_pair_encoder(
        inputs.user_int_feats,
        inputs.user_dense_feats,
        item_ns=item_ns,
    )
    ns_parts.append(pair_tokens)

ns_parts.append(item_ns)

if self.has_item_dense:
    item_dense_tok = F.silu(
        self.item_dense_proj(inputs.item_dense_feats)
    ).unsqueeze(1)
    ns_parts.append(item_dense_tok)

ns_tokens = torch.cat(ns_parts, dim=1)
```

这样 user pair 信息会进入：

```text
MultiSeqQueryGenerator
MultiSeqHyFormerBlock
output projection
classifier
```

它不是简单加在最后一层 MLP 前，而是作为 token 级信息参与主干建模。

---

## 11. 对 num_ns 和 T 的影响

假设常用配置为：

```text
user_ns_tokens = 5
item_ns_tokens = 2
user_dense_token = 1
sample_time_token = 1
item_dense_token = 0
```

原模型大致有：

```text
num_ns = 5 + 1 + 1 + 2
       = 9
```

v3 多了两个 token：

```text
profile_pair_token:    1
item_aware_pair_token: 1
```

因此：

```text
num_ns = 5 + 1 + 1 + 2 + 2
       = 11
```

如果有 4 个序列域：

```text
seq_a, seq_b, seq_c, seq_d
```

且：

```text
num_queries = 2
```

则：

```text
T = num_queries * num_sequences + num_ns
T = 2 * 4 + 11
T = 19
```

训练日志中通常会看到：

```text
PCVRHyFormer model created: num_ns=11, T=19
```

---

## 12. 训练端实现方式

训练端主要修改：

```text
train.py
model.py
```

### 12.1 train.py

需要做四件事。

第一，新增 user pair 参数：

```python
parser.add_argument('--use_user_pair', action='store_true', default=True)
parser.add_argument('--no_user_pair', dest='use_user_pair', action='store_false')
parser.add_argument('--user_pair_fids', type=str, default='62,63,64,65,66')
parser.add_argument('--user_pair_emb_dim', type=int, default=32)
parser.add_argument('--user_pair_use_rank', action='store_true', default=True)
parser.add_argument('--no_user_pair_rank', dest='user_pair_use_rank', action='store_false')
parser.add_argument('--user_pair_item_aware', action='store_true', default=True)
parser.add_argument('--no_user_pair_item_aware', dest='user_pair_item_aware', action='store_false')
```

第二，解析 fid 列表：

```python
user_pair_fids = _parse_int_list(args.user_pair_fids)
```

第三，根据 schema 构造 `user_pair_specs`：

```python
user_pair_specs = (
    build_user_pair_specs(
        pcvr_dataset.user_int_schema,
        pcvr_dataset.user_dense_schema,
        pcvr_dataset.user_int_vocab_sizes,
        user_pair_fids,
    )
    if args.use_user_pair and user_pair_fids
    else []
)
```

第四，把配置传入模型：

```python
model_args.update({
    "user_pair_specs": user_pair_specs,
    "user_pair_emb_dim": args.user_pair_emb_dim,
    "user_pair_use_rank": args.user_pair_use_rank,
    "user_pair_item_aware": args.user_pair_item_aware,
})
```

### 12.2 model.py

需要做五件事。

第一，新增：

```python
UserPairEncoder
```

第二，在 `PCVRHyFormer.__init__` 中接收：

```python
user_pair_specs=None
user_pair_emb_dim=32
user_pair_use_rank=True
user_pair_item_aware=True
```

第三，创建 encoder：

```python
self.has_user_pair = len(self.user_pair_specs) > 0
if self.has_user_pair:
    self.user_pair_encoder = UserPairEncoder(
        specs=self.user_pair_specs,
        d_model=d_model,
        pair_emb_dim=user_pair_emb_dim,
        use_rank=user_pair_use_rank,
        item_aware=user_pair_item_aware,
        dropout=dropout_rate,
    )
```

第四，把 user pair token 数加入 `num_ns`：

```python
num_user_pair_ns = self.user_pair_encoder.num_output_tokens
self.num_ns = original_num_ns + num_user_pair_ns
```

第五，在 `forward()` 和 `predict()` 中把 `pair_tokens` 拼进 `ns_tokens`。

---

## 13. 推理端实现方式

推理端必须和训练端结构一致，否则 checkpoint 无法加载。

主要修改：

```text
infer/model.py
infer/infer.py
```

`infer/model.py` 中要包含和训练端一致的：

```text
UserPairEncoder
PCVRHyFormer 中的 user pair 参数
forward/predict 中拼接 pair_tokens 的逻辑
```

`infer/infer.py` 需要从 checkpoint 中读取：

```text
train_config.json
schema.json
model.pt
```

其中 `train_config.json` 应记录：

```text
use_user_pair
user_pair_fids
user_pair_emb_dim
user_pair_use_rank
user_pair_item_aware
```

推理时根据这些配置重新构造：

```python
user_pair_specs
```

然后创建与训练时完全一致的模型结构。

这样才能避免：

```text
missing keys
unexpected keys
size mismatch
```

---

## 14. 推荐运行方式

如果当前代码只包含原 baseline + v3 方案，可以直接运行：

```bash
bash run.sh
```

也可以显式指定：

```bash
bash run.sh \
  --use_user_pair \
  --user_pair_fids 62,63,64,65,66 \
  --user_pair_emb_dim 32 \
  --user_pair_use_rank \
  --user_pair_item_aware
```

如果当前代码中还包含请求时间感知等扩展，而你只想跑纯 v3，需要关闭额外扩展：

```bash
bash run.sh \
  --use_user_pair \
  --user_pair_fids 62,63,64,65,66 \
  --user_pair_emb_dim 32 \
  --user_pair_use_rank \
  --user_pair_item_aware \
  --no_user_pair_time_aware
```

---

## 15. v3 总结

v3 的完整链路可以概括为：

```text
1. 从 schema 中找到 fid 62-66 在 user_int_feats/user_dense_feats 中的位置。
2. 对每个 fid 取出 aligned 的 ids 和 dense weights。
3. 对 dense weights 做 log1p + softmax，作为每个 id 的重要性。
4. 对 ids 查 embedding，并加入 rank embedding。
5. 对每个 fid 内部做 weighted pooling，得到 fid_vec。
6. 把 5 个 fid_vec 组成 fid_stack。
7. 用 fid_stack 构造 profile_pair_token，表达用户整体画像。
8. 用当前 item_ns 生成 query，对 fid_stack 做 attention。
9. 构造 item_aware_pair_token，表达当前广告相关的用户兴趣匹配。
10. 把两个 token 拼入 ns_tokens，进入 HyFormer 主干训练。
```

最终新增的模型信息是：

```text
profile_pair_token:    [B, 1, d_model]
item_aware_pair_token: [B, 1, d_model]
```

拼接后：

```text
pair_tokens: [B, 2, d_model]
```

这个设计的核心价值是：

```text
既保留 user_int 与 user_dense 的一一对应关系，
又让当前广告动态选择最相关的用户兴趣，
从而增强 PCVR 模型对“用户兴趣 - 广告匹配”的表达能力。
```

---

## 16. v4 思路补充

v4 可以理解为在 v3 的 `item_aware_pair_token` 基础上，进一步加入“请求时间感知”。

v3 中，item-aware query 主要来自当前广告/item：

```text
item_aware_query = f(item_pool)
```

也就是说，模型会根据当前广告，从 `fid62-fid66` 的用户兴趣中选择最相关的部分。

v4 的想法是：同一个用户、同一个广告，在不同请求时间下，兴趣匹配强度可能不同。

例如：

```text
工作日上午
午休时间
周末晚上
节假日前
凌晨
```

这些时间上下文可能会影响用户对广告的转化倾向。

因此 v4 把每条样本的请求时间 token 也加入 item-aware query：

```text
item_aware_query = f(item_pool, sample_time_token)
```

实现上，就是在 `UserPairEncoder` 的 item-aware 分支里，当 `user_pair_time_aware=True` 时，把：

```text
item_pool
sample_time_token
```

拼接后一起投影成 query。

这样得到的 `item_aware_pair_token` 不仅知道：

```text
当前广告是什么
```

也知道：

```text
当前请求发生在什么时间上下文
```

所以它表达的是：

```text
时间感知的广告相关用户兴趣匹配。
```

如果 v4 相比 v3 涨点更明显，说明这份数据里：

```text
用户兴趣 - 广告匹配
```

和：

```text
请求时间上下文
```

之间确实存在有效交互。

