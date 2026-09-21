# FreeToken 模型 TP/EP 多卡适配规范

本文是 FreeToken 所有新模型和已有模型接入单机多卡 TP/EP 的统一工程规范，不是某个模型的专项设计。通用并行框架接受任意 `P >= 2`，不把 world size 限定为二次幂；具体 checkpoint 能否使用某个 `P`，由 attention heads、output groups、expert 数、量化 block、视觉 heads 和 kernel shape 等全部几何约束的交集决定。以后适配任何 dense、MoE、混合注意力或多模态模型，都应先按本文完成能力盘点、shape 契约、权重分片、kernel/KV 适配和测试验收，再把模型加入支持列表。

Qwen3.8-Flash-Next 是当前参考实现；DeepSeek-V4-Flash、DeepSeek-V4-Pro 和 DeepSeek-V4.1-Flash 章节用于展示如何实施和审查本规范。具体 checkpoint 只有在第 9 节要求的真实 GPU 验收记录完整后，才能按平台支持范围标记为“实验性/本地 GPU 已验证”或“正式范围内真实 GPU 已验证”；只有静态或 synthetic 结果时，必须明确标记为“静态/synthetic 已覆盖”。文中的具体模型章节不能代替前面的通用检查清单。

本规范补充多卡模型的技术要求；贡献流程、测试目录和性能报告仍以 [CONTRIBUTING.md](../CONTRIBUTING.md) 与 [tests/README.md](../tests/README.md) 为准。开始实现前应先搜索已有 issue/PR，并遵守 Roadmap、一个 PR 一个改动和真实测试记录等要求。

## 0. 文档适用范围和完成标准

### 0.1 适用范围

本规范覆盖以下模型组件：

- decoder-only 和带 encoder 的生成模型；
- dense FFN、resident MoE、offload MoE 和 owner-local EP；
- MHA、GQA、MLA、线性注意力、稀疏索引和跨层共享 KV；
- BF16、FP8、MXFP8、FP4、NVFP4 等已有量化方法；
- 词表 embedding、LM head、视觉/音频 encoder、projector/aligner；
- paged KV、recurrent state、超大 host table 和 CUDA graph；
- raw safetensors、并行 reader，以及明确声明支持时的 FTW。

### 0.2 “已适配”的定义

支持矩阵必须区分“静态/synthetic 已覆盖”“实验性/本地 GPU 已验证”和“正式范围内真实 GPU 已验证”。一个模型只有同时满足以下条件，才可以把对应拓扑报告为任一种真实 GPU 验证；平台是否属于正式支持范围决定使用后两种状态中的哪一种：

1. 所有列入支持矩阵的 `P >= 2` 都能构造 rank-local 模型参数 shape。
2. loader 为每个 rank 产出的 key、shape、dtype 与该 rank 的 `state_dict` 完全一致。
3. dense TP、expert ownership、量化 scale 和 KV/state 几何与实际 kernel 一致。
4. 非法拓扑在分配大块 GPU/host 内存前明确报错。
5. CPU/synthetic 测试覆盖多个合法 `P` 的分片、重组和 owner route；除了一个常见规模，还要在几何允许时加入非二次幂或更大的 `P`，例如 3、6、16。
6. 在声明已验证的真实多卡拓扑上完成 strict load、短/长 prefill、长时间 decode、CUDA graph 和长上下文验收；长 decode 必须跨越模型的 window、compression、candidate、cache miss/evict 等适用边界。
7. 多模态模型还必须通过真实图片/音频请求。
8. 文档明确列出 raw/FTW、GPU/CPU/hybrid 和量化格式的支持边界。

只完成注册、删除 `TP=1` guard、让模型对象能够实例化，或者只通过 TP2 shape 测试，都不能称为完成多卡适配。

### 0.3 适配流程总览

```text
官方配置与权重索引
        -> 内核能力和并行语义盘点
        -> 全局/本地 geometry 契约
        -> 模型 rank-local 参数声明
        -> dense 与 expert 权重 loader
        -> quant method / kernel / KV-state 适配
        -> Engine/registry/CLI 接线
        -> 多个合法 P 的 dense/owner EP synthetic 测试
        -> 声明支持拓扑的真实多卡验收
        -> 支持矩阵与限制文档
```

前一步没有闭环时不要跳到后一步。例如 loader 可以切出某个 shape，不代表 kernel 能读取；kernel 能运行某个 `P`，也不代表另一个 `P` 的 KV heads、量化 block 或 expert ownership 合法。

## 1. 当前并行语义

FreeToken 当前实现的是同一通信组中的 dense TP，以及可选的 owner-local EP，不是独立二维网格。

- `moe_ep_size == 1` 表示 TP-only；启用 owner EP（`moe_ep_size > 1`）时，要求 `tensor_parallel_size == moe_ep_size`。
- 多卡并行组大小 `P >= 2`，不设固定规模白名单；实际模型必须满足所有 dense TP 和 owner EP 分片约束。TP1 仍是合法基线拓扑。
- dense attention、dense/shared MLP、词表和输出头按 TP 分片。
- routed experts 按 expert ID 连续分配给各 rank，每个 expert 在所属 rank 上保持完整，不再切 intermediate 维。
- router 在各 rank 复制并产生全局 expert ID。
- 每个 rank 只计算本地 expert 对应的 route，远端 route 权重置零。
- shared expert 的 TP partial 和 routed expert 的 EP partial 合并后做一次 SUM all-reduce，得到完整 MoE 输出。

设全局 expert 数为 `E`，并行组大小为 `P`：

```text
local_E = E / P
rank r owns [r * local_E, (r + 1) * local_E)
```

模型必须满足 `E % P == 0`。owner-local cache、source bank 和 slot map 都使用本地 expert ID；router 始终使用全局 expert ID，二者不能混用。

启动形式为：

```bash
ft serve --model <checkpoint> --tensor-parallel-size P --moe-ep-size P
```

owner EP 下默认的 `--moe-strategy auto` 会解析为 `offload`；未指定 cache size/auto 时，会按 owner-local expert 几何自动计算 cache。也可以显式写出 `--moe-strategy offload --moe-cache-auto`。owner EP 明确拒绝按全局 expert 数解释的 `--moe-cache-rate`，只接受 `--moe-cache-size` 或 `--moe-cache-auto`。当前不支持 owner EP 与 CPU/hybrid expert decode 组合；FTW 的更广限制见第 10 节。

## 2. 为什么不能只删除 TP2 限制

通信原语本身通常可处理任意 world size，但模型 shape、checkpoint 分片和量化 block 必须同时成立。典型的 TP2 隐式假设包括：

- KV heads 数较少时，`P` 大于 KV head 数就需要复制到连续 rank 组，而不是继续整数切分。
- fused QKV 层按总行数平均切分，破坏 Q/K/V 各自的 head 边界。
- FP8 scale 仍按全量 weight shape 声明，或在错误的 block 轴上切分。
- routed experts 同时按 expert ID 和 intermediate 维切分，导致 owner EP 的 expert 不完整。
- row-parallel bias 在每个 rank 都加载，all-reduce 后被累加 `P` 次。
- 测试只验证 TP2 拼接，没有覆盖 KV head 复制和全远端 route。

因此每个模型都必须完成以下四个闭环：

```text
模型声明的 rank-local shape
        == checkpoint loader 输出 shape
        == 量化方法声明的 bank/scale shape
        == kernel 实际读取 shape
```

## 3. 通用内核和层能力

### 3.1 列并行和行并行

- Column parallel：weight 沿输出维 `dim 0` 分片，bias 同轴分片，不需要通信。
- Row parallel：weight 沿输入维 `dim 1` 分片，输出做 SUM all-reduce。
- Row-parallel bias 必须只加一次。若 kernel 在 all-reduce 前加 bias，则 rank 0 加原 bias，其他 rank 加零。
- fused projection 必须先按语义拆分，再分别分片，最后按模型要求重新融合。

### 3.2 Attention heads

Q heads 必须能被 TP size 整除。KV heads 有两种情况：

- `num_kv_heads >= P`：要求 `num_kv_heads % P == 0`，每个 rank 持有 `num_kv_heads / P` 个 head。
- `num_kv_heads < P`：要求 `P % num_kv_heads == 0`，一个 KV head 复制到连续的一组 rank。

例如全局 2 个 KV heads：

| TP | 每 rank KV heads | rank 到全局 KV head 的映射 |
|---:|---:|---|
| 2 | 1 | `0, 1` |
| 4 | 1 | `0, 0, 1, 1` |
| 8 | 1 | `0, 0, 0, 0, 1, 1, 1, 1` |

模型层和 loader 必须共用同一套映射。不能让 loader 复制 KV head，而通用 fused linear 仍按 `global_rows / P` 声明参数。

### 3.3 量化 block

对 block-quantized weight，weight 和 scale 必须沿相同语义轴切分，但 scale shape 由具体量化格式定义，不能统一假设边缘 block 支持向上取整。对于当前要求完整二维 block 的 FP8 格式：

```text
weight [N, K]
scale  [N / block_n, K / block_k]
```

此时要求 `N % block_n == 0` 且 `K % block_k == 0`。列并行切 `N` 及 scale 的第 0 维；行并行切 `K` 及 scale 的第 1 维。rank-local 分片边界必须落在完整 block 上。只有当量化格式、checkpoint 存储和 kernel 都明确支持 padded edge block 时，才能使用 `ceil`；NVFP4 等采用不同 scale 布局的格式必须按其 scheme 单独描述。

当前相关格式：

- DeepSeek-V4 Flash/Pro dense：FP8 128x128，scale 为 E8M0。
- DeepSeek-V4.1 Flash dense：FP8 32x32，scale 为 E8M0。
- DeepSeek routed experts：MXFP4，weight K block 为 32，scale 为 E8M0。
- Qwen3.8 routed experts：NVFP4，weight K block 为 16，另有全局 scale。

### 3.4 Collective

当前模型 forward 数据路径的跨 rank tensor 合并主要使用 SUM all-reduce 和词表 all-gather，NCCL/PyNCCL 不应写死卡数或二次幂 world size。控制面仍可能使用 broadcast/barrier，运行时兼容性也不能仅凭 collective API 正确性推断。测试应覆盖多个合法 world size，并在模型几何允许时加入 3、6、16 等非二次幂或更大组。尤其检查：

- 每个 row-parallel 子层恰好 reduce 一次。
- owner EP 的 shared+routed MoE 只在合并后 reduce 一次。
- rank 无本地 route 时输出为有限的全零 partial，不能出现非法 expert/slot ID 或 `0 * NaN`。

## 4. 仓库子系统适配清单

新模型适配不能只修改 `python/freetoken/models/<family>/`。下面按仓库子系统列出需要检查的位置；不适用的项目也要在适配记录中写明“不适用及原因”。

| 子系统 | 所有模型必查内容 | 常见产物 |
|---|---|---|
| `models/register.py` | architecture、checkpoint root、packed mapping、unquantized module、encoder 注册 | `ModelSpec` / `EncoderSpec` |
| `models/<family>/config.py` | 全局几何、量化格式、attention groups、本地几何合法性 | `ModelConfig` 和 family args |
| `models/<family>/model.py` | rank-local 参数声明、forward reshape、collective 边界 | TP-aware model/layer |
| `models/<family>/weight.py` | raw key 映射、dense 分片、scale 分片、expert ownership | rank-local iterator |
| `layers/linear.py` | column/row/fused/QKV 是否表达该模型的本地 shape | 通用层或 family 专用层 |
| `layers/quantization/` | local shape、block alignment、bank layout、backend 选择 | scheme/method/kernel 接线 |
| `moe/` | 全局/本地/slot ID 边界、owner cache、prefill/decode | ownership 和 cache adapter |
| `attention/`、`kvcache/` | KV 行布局、source layer、page addressing、graph metadata | backend/pool/cost model |
| `kernel/` | world size 假设、local head/intermediate、AOT/JIT shape | kernel guard/AOT spec |
| `mm/` 和 encoder | placeholder、processor、encoder TP、projector、host streaming | processor/vision/audio 模块 |
| `engine/`、`server/args.py` | 拓扑校验、配置透传、cache budget、进程启动 | fail-fast 和 CLI |
| `tests/` | 多个合法 P 的 shape/数值/ownership、非二次幂回归、真实硬件门禁；按 [tests/README.md](../tests/README.md) 放到所保护的 subsystem，优先扩展已有文件 | subsystem-local + shared regression tests |
| `docs/models.md`、`docs/cli.md` | 支持矩阵、启动参数、限制和实测硬件 | 用户可见说明 |

### 4.1 新模型适配记录模板

每次新增模型时，在实现说明或模型文档中填写以下模板：

```text
模型/Checkpoint:
Architecture / model_type:
支持格式: BF16 / FP8 / FP4 / ...
支持拓扑: TP1；支持的 `P >= 2` 集合或整除/对齐约束；是否 owner EP
验证矩阵: 每行填写 checkpoint / 格式 / raw或FTW / platform或environment ID / TP / EP / collective backend / 状态
状态取值: 静态/synthetic 已覆盖；实验性/本地 GPU 已验证；正式范围内真实 GPU 已验证

全局几何:
  hidden / layers / q_heads / kv_heads / head_dim
  experts / top_k / expert_intermediate
  vision/audio/linear-state/sparse-index 几何

rank-local 规则:
  vocab:
  Q/K/V:
  attention output:
  dense/shared MLP:
  routed experts:
  quant scale:
  encoder/projector:

复制参数:
跨 rank collective:
KV/state/cache 变化:
loader/FTW 状态:
kernel/AOT 状态:

FreeToken commit:
OS / CPU 型号与架构 / system RAM / NUMA:
GPU / driver / PCIe或NVLink topology:
PyTorch / CUDA:
collective backend: PyNCCL；或 torch.distributed（`--disable-pynccl`）
实际加载的 libnccl.so.2 路径与 runtime version:
NCCL_LIB，以及所有非默认 NCCL_* / CUDA / 通信相关环境变量:
CPU/synthetic 命令、独立 oracle 与结果:
真实 GPU 服务端启动命令与结果:
客户端 endpoint、请求/fixture、prompt、sampling、max_tokens、timeout、重复次数:
跨越 window/compression/candidate/cache 边界的证据:
性能改动的 main/branch A/B 命令与结果:
仍不支持:
```

这个记录用于代码审查和后续模型复用。不能只写“参考某模型”，因为形状、量化 block、KV sharing 和 router 语义经常不同。测试期望必须来自独立 oracle，例如原 tensor round-trip、PyTorch/CPU reference、真实 checkpoint metadata 或手工推导，不能由被测的 production helper 同时生成结果和期望值。

## 5. 模型内部适配工作清单

### 5.1 配置和拓扑校验

1. 从 checkpoint 配置读取全局几何，不修改全局 head/expert 数。
2. 提供 rank-local geometry helper，显式计算 Q/KV/GDN/indexer/vision heads。
3. 校验 TP size 至少为 1；声明多卡或启用 owner EP 时 `P >= 2`，并校验模型所有分片轴合法。通用拓扑层不得额外写死二次幂白名单。
4. 启用 owner EP（`moe_ep_size > 1`）时，要求 `moe_ep_size == tensor_parallel_size`。
5. 启用 owner EP 时校验 `num_experts % moe_ep_size == 0`。
6. 在分配模型、KV cache 或 expert banks 之前 fail fast。

### 5.2 模型参数 shape

1. 将 Q、KV、output group 等全局几何转换为 rank-local shape。
2. replicated 模块保持全量 shape，例如 router、norm、MLA latent、compressor 状态。
3. column/row-parallel 模块使用 local shape。
4. KV head 复制必须在层声明阶段体现，不能只在 loader 中处理。
5. 模型 forward 的 reshape/split 使用 local head/group 数。

### 5.3 Dense checkpoint loader

1. 先重命名 checkpoint key。
2. 在 fused projection 形成之前切 raw tensor。
3. Q/K/V、gate/up 等 fused 组分别切分。
4. vocab 使用 `ceil(vocab / P)`，最后一个 rank 补零到模型声明的行数。
5. weight scale 与 weight 同轴、按完整量化 block 切分。
6. norm、router、HC、compressor等 replicated tensor 在各 rank 完整加载。
7. row-parallel bias 只保留一份。
8. loader 输出的每个 key 必须与当前 rank 的 `state_dict` shape 一致。

### 5.4 Routed expert loader

owner EP 下不切 expert intermediate 维，步骤如下：

1. 用全局 expert ID 解析 checkpoint key。
2. 跳过不属于当前 rank 的 expert。
3. 将 `[global_start, global_end)` 重编号到 `[0, local_E)`。
4. 一个本地 expert 的 gate/up/down weight 和全部 scale 必须完整。
5. serial reader 和 parallel reader 使用同一 ownership 过滤。
6. bank builder 的首维为 `local_E`，cache floor 也按 `local_E` 计算。

如果只启用 TP、不启用 owner EP，才允许由明确支持 TP 的 expert kernel 沿 intermediate 维切 expert；两种模式不能叠加。

### 5.5 MoE forward

推荐顺序：

```text
router_logits / global route
shared_partial = shared_expert(..., reduce=False)
routed_partial = owner_experts(..., reduce=False)
output = all_reduce(shared_partial + routed_partial)
```

router 和 shared gate 必须在可能原地修改 hidden states 的 routed kernel 之前计算。

### 5.6 KV cache、状态池和稀疏索引

1. MHA/GQA 的 Q heads 按 TP 分片，KV heads 按第 3.2 节分片或复制；共享 MLA latent KV 通常复制，query/output heads 或 groups 仍按模型几何切分。
2. paged KV 的每行 shape 必须以 kernel 实际存储的数据为准。
3. recurrent/compressor state 若与 head 数相关，应按 rank-local head 数分片；与共享 latent 相关则复制。
4. 跨层共享 KV/indexer 的模型必须显式记录 source layer，消费者不能错误读取自己的 layer pool。
5. cost model、pool allocator 和 runtime addressing 必须使用同一 source-layer/ratio 描述。
6. CUDA graph replay 使用的 metadata buffer 地址和 shape 必须固定。

### 5.7 多模态 encoder

视觉塔也按 dense TP 规则适配：

- fused QKV 按 Q/K/V head 分组切分。
- MLP `fc1/w1` 列并行，`fc2/w2` 行并行。
- attention output projection 行并行。
- patch embedding、position/rope、norm 复制。
- aligner/projector 按实际计算图决定列/行并行。
- encoder-only 工具读取器若没有 TP 参数，必须保留 TP fail-fast，不能返回全量 tensor 去加载 local shape。

### 5.8 Engine、注册和加载接线

1. 在 `ModelSpec` 注册 architecture、checkpoint roots、packed projection 和不量化模块。
2. `EngineConfig.model_config` 必须在解析模型前设置当前 TP rank/size，使 parser 和 quant config 看到正确拓扑。
3. family reader若自行按 `get_tp_info()` 分片，不接收 `tp_shard`；显式声明 `tp_shard` 的 reader 由公共 loader 透传。两种约定只能选一种。
4. owner EP 的模型和 expert 格式必须通过能力检查后才允许创建 owner cache，不能仅按 model name 静默放行。
5. cache auto sizing 使用 `local_E`，而 router/config仍保留全局 `E`。
6. `model_config.moe_ep_size` 必须在模型构造前写入，以便 expert quant method 看到 owner 模式下 `expert_tp_size=1`。
7. 当前 FTW 存储全局 dense tensor 且没有 rank-layout metadata，因此所有 `TP > 1` 加载都必须拒绝；即使未来补齐 dense TP，owner EP 仍需额外实现 expert ownership 过滤，Engram/PLE 等 side-table 也要由各 family 单独声明支持状态。
8. 多模态 architecture 只有在 encoder 注册、processor、weight reader 和 model hook 全部存在时才能声明相应 modality。

### 5.9 Kernel 和 AOT

1. 阅读 kernel 的实际索引方式，不以 Python wrapper 的 shape 注释代替验证。
2. 检查 kernel 是否把 head 数、top-k、block size、split-k 或 world size 编译为常量。
3. kernel 接收 local intermediate 时，layout、pack、workspace 和 slot limit 都必须使用 local geometry。
4. owner EP 的 expert kernel接收完整 expert，因此其 kernel TP size 为 1；通信发生在 expert 输出之后。
5. 新增的 `(N, K)` 或 bank row bytes 要加入 AOT shape 表；没有预编译覆盖时文档需说明会 JIT。
6. AOT 表中的 checkpoint 别名只能用于几何和格式完全相同的模型；Pro 等几何不同的模型要单独登记。
7. 对每个声明支持的 `P` 列出 local shape；几何允许时要列出非二次幂或更大的 `P`，防止 AOT 只覆盖少数固定档位。

### 5.10 可观测性和诊断工具

1. route trace/replay 工具必须接受任意 `EP size >= 2`，并要求 expert 数可整除，不能只把 experts 对半切或写死少数规模。
2. 日志显示全局 experts、本地 expert 范围、TP/EP size、cache slots、量化 backend，以及 OS/CPU、GPU/driver/topology、PyTorch、CUDA、collective backend 和 FreeToken commit 等运行时指纹。NCCL 必须记录实际加载的 `libnccl.so.2` 路径和 runtime version，不能只记录 `torch.cuda.nccl.version()`；同时记录 `NCCL_LIB` 以及所有非默认 `NCCL_*`、CUDA 和通信相关环境变量。
3. shape mismatch 报错应包含 key、global shape、rank、world size、切分轴和期望 local shape。
4. 对 unsupported topology/format 采用启动期 fail-fast，不能运行到 kernel OOB 或 strict load 才失败。
5. 性能报告按每个声明支持并实测的 `P` 分开记录 prefill、decode、通信占比和 MoE miss rate；性能改动还必须按 [CONTRIBUTING.md](../CONTRIBUTING.md) 使用相同模型、prompt 和设置提供 `main`/分支 A/B、tokens/s、受影响时的 TTFT 和双方完整命令。

## 6. 参考案例：Qwen3.8-Flash-Next

Qwen3.8 同时包含 QSA、GDN、PLE、视觉塔和 owner-local NVFP4 experts，是适配其他模型时的基准。当前 owner EP 只对 NVFP4 expert checkpoint 开放；官方 FP8 checkpoint 的 routed experts 为 `fp8_block`，不在当前 owner EP capability 列表中。

发布 checkpoint 的 Q heads=24、GDN key heads=16、KV heads=2、vision heads=16。该 checkpoint 的可用 `P` 必须同时满足这些 head/group 维度和 512 experts 的整除条件，再叠加 dense projection、视觉投影及量化 block 的对齐条件。这里得到的具体 `P` 只是该 checkpoint 的结果，通用并行层不得把它写成全局白名单；若要让同一 checkpoint 接受不满足这些整除条件的卡数，需要额外实现 padding 或不均匀分片。

### 6.1 Dense TP

- QSA Q projection 每个 head 包含 gate+value 两组行，因此 Q 按 `2 * head_dim` 为单位切分。
- 当 `P` 大于 KV head 数时，QSA KV heads 复制到连续 rank 组。
- GDN q/k/v/z/b/a 按各自 key/value head 分组切分。
- QSA/GDN output projection 沿输入维切分并 all-reduce。
- shared expert gate/up 列并行，down 行并行。
- vocab embedding/lm_head 分片并为尾 rank 补零。
- PLE table、hash 常量和 HC 参数复制。

### 6.2 Owner EP（当前仅 NVFP4 checkpoint）

- 全局 512 experts 按 `local_E = 512 / P` 分配；任意满足 `512 % P == 0` 的组大小都使用同一 ownership 规则，但最终支持集合还必须与 QSA/GDN heads、视觉 heads、dense/shared projection 和量化 block 的合法 `P` 取交集。
- checkpoint reader 只读取本 rank 的 NVFP4 rows，并将全局 ID 重编号为本地 ID。
- owner cache 只为本地 expert 分配 source rows 和 slots。
- shared partial 与 routed partial 合并后一次 all-reduce。

### 6.3 必须避免的 TP2 假设

通用 `LinearColParallelMerged` 会把每个 fused segment 直接除以 TP size，不适合 TP 大于 KV head 数时的 KV 复制。Qwen QSA 必须使用支持“Q 分片、KV 复制”的专用 layer geometry，并与 loader 的 head partition helper 保持一致。

## 7. 实施案例：DeepSeek-V4 Flash 和 Pro

两个 checkpoint 使用同一 `DeepseekV4ForCausalLM` 架构、同一 key 布局和量化格式，Pro 只是几何更大。

| 字段 | V4 Flash | V4 Pro |
|---|---:|---:|
| hidden size | 4096 | 7168 |
| MoE intermediate | 2048 | 3072 |
| layers | 43 | 61 |
| attention heads | 64 | 128 |
| output groups | 8 | 16 |
| q LoRA rank | 1024 | 1536 |
| routed experts | 256 | 384 |
| top-k | 6 | 6 |

适配规则：

- `wq_a`、`wkv`、compressor 和 MLA latent 复制。
- `wq_b` 按 query heads 列并行。
- `attn_sink` 按 query heads 分片。
- `wo_a` 按完整 output group 切分。
- `wo_b` 沿输入 group/rank 维行并行。
- indexer/compressor 当前保持复制，避免改变共享索引语义。
- shared expert `w1/w3` 列并行，`w2` 行并行。
- routed MXFP4 experts按 expert ID owner-local 分配，不切 intermediate。
- V4/Pro 的 dense FP8 128 block 在任意 `P` 下都必须保持 block 对齐。

具体 `P` 不能只看 expert 数或 attention heads，而应取 `n_heads`、`o_groups`、shared expert intermediate、LoRA/group 维和 FP8 block 对齐条件的交集。V4 Flash 的 `o_groups=8`，所以不满足 `8 % P == 0` 的规模会在模型构造或 loader 阶段被拒绝。V4 Pro 虽然有 16 个 output groups，但 shared expert intermediate 为 3072；在 FP8 128 block 下，还必须满足 `3072 % (128 * P) == 0`。这些是 checkpoint 几何约束；如果产品要求同一 checkpoint 覆盖更多 `P`，需要补齐 group/padding 或实现不均匀分片，而不是删除校验。

## 8. 实施案例：DeepSeek-V4.1 Flash

V4.1 是新的 `DeepseekV41ForCausalLM`，不能作为 V4 Flash 的配置别名。除常规 TP/EP 外，还必须适配：

- dense FP8 32x32 block。
- 40 层 backbone，384 experts，top-6。
- ratio 2/1 的压缩 KV。
- `kv_source_layers` 指定跨层共享 compressed KV。
- `index_source_layers` 指定跨层共享 index top-k。
- candidate source 的两级 block 筛选。
- 两个超大 Engram n-gram table，按行 TP 分片并在 lookup 后 all-reduce。
- 32 层视觉塔和 aligner。
- image token 专用 router bias。
- Single-Pass mHC 的跨层 pre-mix 语义。
- checkpoint 中的 DSpark/MTP draft layers。FreeToken 未启用 speculative decode 时可以跳过 draft weights，但必须显式记录为不参与普通 forward，不能误加载进 backbone。

V4.1 的 KV pool/cost model 必须按 source layers 计费和分配，不能沿用 V4“每个 ratio>0 layer 都拥有独立 compressed/index pool”的假设。

V4.1 Flash 的 `o_groups=8`，因此必须满足 `8 % P == 0`，同时满足 Q heads、index heads、视觉 heads、expert 数和 FP8 block 对齐条件。这不是通用 owner EP 限制，而是该 checkpoint 的 dense attention/output-group 几何。若未来 checkpoint 采用不同 group/head 几何，应重新计算合法 `P`；若要支持当前 checkpoint 的非整除 `P`，需要实现 padding 或不均匀 group 分片。

两个 Engram 表合计约 188.8 GiB（FP8 weight 加每 32 列一个 E8M0 scale），运行时按行在 TP ranks 间分片，lookup 后 SUM all-reduce；因此每 rank 的 host bank 约为总量的 `1 / P`，另有向上取整的尾行。当前 Engram reader 面向原始 safetensors，V4.1 FTW side-table 尚未接线，不能把 raw 与 FTW 混写为同一支持状态。

## 9. 通用测试与验收矩阵

每个新模型至少包含以下 CPU/synthetic 测试；预期结果必须来自独立 oracle，不能复用被测的 geometry/sharding helper 同时计算 expected：

1. 对每个声明支持的 `P`，rank-local model state shape 与 loader 输出 shape 完全一致；至少覆盖多个 `P`，且几何允许时包含非二次幂或更大组。
2. column shards 沿 dim 0 重组为原 tensor。
3. row shards 沿 dim 1 重组为原 tensor。
4. fused Q/K/V 按各组重组；KV 复制按 rank group 核对。
5. FP8/FP4 scale 与 weight 的分片边界一致。
6. vocab 非整除时尾 rank padding 正确。
7. owner EP 对任意合法 `P >= 2` 都保证每个全局 expert 恰好属于一个 rank；模型几何允许时加入 EP3 或 EP6 的非二次幂回归，否则由通用 ownership 测试覆盖这些规模。
8. 所有 rank 的 route weights 求和等于原始全局 route。
9. 全远端 route 安全地产生零 partial。
10. shared+routed partial SUM 等于单卡参考，且只发生一次 MoE all-reduce。
11. attention partial SUM 等于单卡参考。
12. 多模态视觉 QKV/MLP/aligner TP 重组及 bias exactly-once。
13. config 对非法 TP、EP 不相等、expert 不可整除和量化 block 不对齐 fail fast。

GPU 验收需要在每个声明“实验性/本地 GPU 已验证”或“正式范围内真实 GPU 已验证”的 `P` 上运行；若只能实测其中一部分，支持矩阵必须把其他规模标记为“静态/synthetic 已覆盖”，不能笼统写成已验证。支持状态必须按 checkpoint、格式、raw/FTW、platform/environment、`P` 和 collective backend 逐行记录，不能用一个状态覆盖全部组合。

每次验收都要记录 FreeToken commit、OS、CPU 型号与架构、system RAM/NUMA、GPU/driver/PCIe或NVLink topology、PyTorch/CUDA、collective backend、实际加载的 `libnccl.so.2` 路径和 runtime version、`NCCL_LIB`、所有非默认 `NCCL_*`/CUDA/通信相关环境变量，以及服务端完整启动命令：

- 启动及 strict weight load。
- 文本 prefill/decode。
- 至少一个短 prompt 和一个跨 chunk 长 prompt。
- 长时间生成：记录客户端 endpoint 和完整请求或固定 fixture，包括 prompt/input、sampling 参数、`max_tokens`、客户端超时和重复次数。输出 token budget 必须跨越适用的 window、compression、candidate 和 cache 边界，并记录实际跨越这些边界的日志或计数证据；确认所有 rank 持续推进且请求能够正常结束。
- CUDA graph decode，包括连续多次 replay 的长生成；必要时同时保留 eager A/B 以区分 graph 与非 graph 问题。
- MoE cache miss/evict 路径。
- 图像请求（多模态模型）。
- TP1 与对应 `TP=P` 多卡拓扑的独立 logits/reference parity，明确输入、比较步数和数值容差；greedy 输出只能作为 smoke test，不能替代 logits parity。TP1 放不下时可使用 CPU/HF/layer reference。

至少运行并记录仓库通用测试和本次改动的定向测试命令；命令格式和 `needs_weights` 环境变量见 [tests/README.md](../tests/README.md)。若改动影响性能，还要按 [CONTRIBUTING.md](../CONTRIBUTING.md) 提供相同模型、prompt 和设置下 `main` 与分支的端到端 A/B。

```bash
uv run pytest tests/ -m "not slow"
```

未在真实硬件运行的项目只能报告为“静态/synthetic 已覆盖”，不能报告为任一种真实 GPU 验证。

若测试平台不在 [安装要求](install.md) 当前声明的正式支持范围内，例如 Linux aarch64，则状态只能标记为“实验性/本地 GPU 已验证”；除非同时补齐安装依赖、发布产物和用户文档，否则不能据此扩大项目的正式平台支持声明。

## 10. 当前并行框架限制

- 只支持单机 TP，以及同组 TP+owner EP；不支持独立 `TP x EP` 二维进程网格。
- owner EP 拓扑校验接受任意 `P >= 2`，但具体模型仍可能因 attention heads、量化 block、视觉 heads 或 kernel shape 不能被 `P` 合法分片而拒绝启动；这属于模型几何限制，不是 owner EP 的固定规模白名单。
- owner EP 当前只支持 GPU offload cache；CPU/hybrid expert decode 尚未实现，并且拒绝 `--moe-cache-rate`。
- FTW 当前整体只支持 TP1：dense tensor 是全局布局且没有 rank-layout metadata，所有 `TP > 1` 加载都会拒绝。即使未来支持 FTW dense TP，owner EP 的全局 expert rows 仍需增加 owner-local 过滤。V4.1 Engram side-table 尚未接线；Qwen PLE 已通过 FTW sidecar 接线，但仍受当前 FTW 仅 TP1 的全局限制；其他 side-table 由各 family 分别声明支持状态。
- AOT kernel cache 默认记录的是特定 shape；任何尚未预编译的 `P` 或新模型 shape 都会回退 JIT，需要可用的 CUDA toolkit。
