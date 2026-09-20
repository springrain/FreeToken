# Tensor Parallelism Implementation Plan

Goal: make `--tensor-parallel-size N` (N > 1) actually serve models, with
three configurations explicitly in scope, in priority order:

1. **deepseek-v4-flash MXFP4 checkpoints** (the motivating example).
2. **gpt-oss MXFP4 variant with bias tensors** (TP-complete today; add tests).
3. **bf16/fp16 unquantized dense models** (correct-by-construction today;
   one loader defect to fix).

References used while designing this plan: sglang's TP linear/moe layers
(`python/sglang/srt/layers/linear.py`, `moe/fused_moe_triton`) and
ktransformers (`E:/git/github/ktransformers`). The FreeToken runtime
scaffolding (per-rank process spawn, distributed communicator, TP-aware
linear/embedding/kernel classes) already exists; the gaps are in the
**weight loading** paths and a few kernel gates.

## TP invariants (derived from sglang)

- **SPMD**: one process per rank; every rank runs identical scheduling;
  rank 0 is the only reply/IO face.
- **Column-parallel** (out-dim sharded, e.g. q/k/v, gate/up, wq_b):
  no collective needed.
- **Row-parallel** (in-dim sharded, e.g. o_proj, down, wo_b, shared w2):
  exactly ONE all-reduce **after** the partial matmul. Bias must be added
  on **rank 0 only** (zeros elsewhere) because apply-then-all-reduce
  would count any replicated bias N times.
- **Replicated**: routers/gates, norm weights, MLA latents and their
  projections (wq_a, wkv, q_norm, kv_norm, compressor, indexer) are kept
  full on every rank so latents stay identical.
- **KV replication**: when tp > num_kv_heads, KV shards replicate
  (div_even allow_replicate); never an error.
- **Vocab**: ceil-partition `div_ceil(V, tp)`; rank-local embedding lookup
  + all-reduce; LM head all-gathers logits and truncates to V.

## Decision: Option A — loader-side slicing

Slice raw checkpoint tensors per rank **inside each model's weight
iterator**, before the existing kernel/pack code runs. Kernels and pack
contracts stay TP-unaware (they already size everything from
`MoEConfig.local_intermediate` / local `LinearConfig`); only the shard
boundaries are new. This matches how gpt-oss already does it
(`shard_gpt_oss_tensor`, `local_mxfp4_intermediate_range`).

## Current state (verified)

- `freetoken.layers.linear`: `LinearColParallelMerged`, `LinearQKVMerged`,
  `LinearOProj`, `LinearRowParallel` — all TP-ready; row-parallel applies
  all-reduce after `quant_method.apply`.
- `freetoken.layers.embedding`: `VocabParallelEmbedding` (div_ceil rows on
  every rank, all-reduce) and `ParallelLMHead` (all-gather + truncate).
- `freetoken.layers.quantization.moe.base`: `MoEConfig` carries tp_rank /
  tp_size + `local_intermediate = intermediate // tp_size`; `fused_piece`
  concatenates gate/up along dim=1 of `[1, rows, cols]` pieces.
- `freetoken.moe.expert_pieces.per_expert_pieces`: yields `[1, ...]`
  per-expert pieces (unsqueeze(0)) — the dsv4 slicing wrapper goes before it.
- `freetoken.moe.expert_banks.build_expert_banks._fill` validates range/dup
  and calls `method.pack(piece, out)` — unchanged.
- gpt-oss: `gpt_oss/weight.py` fully shards dense + MXFP4 experts
  (rank0-only/zeros biases, 32-block ceil windows); kernel
  `TritonGptossMxfp4MoEKernel` opts in only when
  `intermediate % (32*tp) == 0` — the sharder zero-pads ranks to
  `ceil(blocks/tp)*32` while the kernel banks size from
  `floor(intermediate/tp)`, so without divisibility the windows disagree
  and weight load crashes on the shape mismatch (review finding; gpt-oss
  2880 works at tp 2/3/5/6 and rejects 4/8 with the message).
- dsv4 (`models/deepseek_v4/weight.py`): dense yields and MXFP4 expert
  pieces are sliced per rank (the old `raise NotImplementedError` TP gate is
  removed); `TritonMxfp4MoEKernel` opts in when
  `intermediate % (32 * tp) == 0` and sizes banks from the local
  intermediate.
- Runtime: `server/launch.py` spawns one process per rank with
  `DistributedInfo(i, world_size)` and `set_assigned_gpu(targets[i])`;
  engine keeps seed deterministic per rank; answer on rank 0 only.

## Work items (phases)

### Phase 0 — hygiene / infrastructure

| # | File | Change |
|---|------|--------|
| 0.1 | `python/freetoken/models/loader.py` | `shard_tensor`: in the SPLIT_DIM_1 branch, handle `value.ndim == 1` (o_proj-style bias): rank 0 keeps a clone, other ranks get zeros. Today a 1-D tensor under a split name hits `chunk(dim=1)` -> IndexError; where it would not crash, bias would be double-counted by the post-apply all-reduce. |
| 0.2 | `python/freetoken/distributed/info.py` | Add `reset_tp_info()` (sets `_TP_INFO = None`) as a tests-only seam; export it. |
| 0.3 | `python/freetoken/distributed/impl.py` | `destroy_distributed` must restore `DistributedCommunicator.plugins = [TorchDistributedImpl()]` instead of `[]` (plugins[-1] dispatch breaks after teardown). |
| 0.4 | `python/freetoken/utils/misc.py` | `div_even`: replace bare asserts with explicit raises naming a, b and the TP context. |
| 0.5 | `python/freetoken/engine/config.py` | `distributed_addr`: read `FREETOKEN_DIST_ADDR` env, default `tcp://127.0.0.1:2333` (deployment collision override). |
| 0.6 | `python/freetoken/engine/engine.py` | Init free-VRAM baseline: use `free_min` when `tp_info.size > 1` (otherwise aggressive ranks race the smallest rank); rebuild baseline is already MIN. |
| 0.7 | `python/freetoken/server/args.py` | Drop the stale "tensor parallelism is not supported yet" error text; refresh `--tensor-parallel-size` help (comma-per-rank `--gpu` semantics stay). Keep the gpu-count check. |
| 0.8 | `python/freetoken/kernel/pynccl.py` | Python wrapper only: validate dtype before all_reduce/all_gather with a clear error; on init failure suggest `--disable-pynccl`. The `.cu` map change (fp32) is intentionally NOT done — cannot compile/verify CUDA on this machine; documented as a follow-up. |
| 0.9 | `tests/conftest.py` | Add autouse fixture resetting tp info after each test (uses 0.2). |

### Phase 1 — gpt-oss validation (code already TP-complete)

CPU-runnable unit tests in `tests/moe/` (new file
`test_gpt_oss_tp_sharding.py`, mirroring `gpt_oss/weight.py`):

- Dense reassembly: q/k/v weight+bias shards (heads split, kv replicate),
  concat over ranks == original; o_proj weight dim1 + bias (rank0 real,
  rank1 zeros); sinks per-head.
- `local_mxfp4_intermediate_range`: tiling edge table (tp 1/2/4;
  intermediate divisible vs 32-block ceil remainder; last rank shorter).
- `_source_slices`: block-aligned slices for blocks/scales/bias on both
  gate_up (rows) and down (cols).
- Expert pieces: per-layer shard for tp=2 (and 4 where uniform) reassembles
  to the full tensor for each role.

### Phase 2 — deepseek-v4-flash MXFP4 (the example)

dsv4 geometry (4096 dim, 64 heads, 8 o-groups, moe inter 2048, 256
experts, vocab 129280) passes all tp=2 divisibility constraints:
`2048 % (32*2) == 0`, `64 % 2`, `8 % 2`, `129280 % 2`.

1. **`models/deepseek_v4/attention.py`** — local geometry:
   `get_tp_info()`; explicit raises `n_heads % tp == 0`,
   `o_groups % tp == 0`; `n_heads_local`, `n_groups_local`; attn_sink
   sized `n_heads_local`; `wo_a` rows = `n_groups_local * o_lora_rank`;
   `_wo` reshape and both `unflatten` sites use the local counts. wq_b /
   wo_b keep global sizes (Inner classes divide internally); wq_a / wkv /
   q_norm / kv_norm / compressor / indexer stay replicated.
2. **`models/deepseek_v4/weight.py` `iter_weights`** — slice dense yields:
   - embed/head: div_ceil vocab window (clamp hi defensively);
   - `wq_b.weight` dim0 rows + `weight_scale_inv` rows (/128 window,
     same axis);
   - `wo_b.weight` dim1 cols + scale cols (/128 window);
   - `shared_experts.w1/w3` (+scales) dim0; `shared_experts.w2` (+scale)
     dim1;
   - `attn_sink`: head chunk n_heads//tp;
   - `wo_a` (dequantized bf16): group-aligned row chunk
     `[rank*ngl*olr : (rank+1)*ngl*olr]`;
   - wq_a / wkv / q_norm / kv_norm / compressor / indexer / gate /
     norms / hc_*: full.
   fp8_block scale companion rule: block scales slice at 128-block
   granularity on the SAME axis as the weight (col-parallel -> rows;
   row-parallel -> cols); replicated fp8 linears keep full scales
   (`Fp8BlockLinearMethod.create_weights` asserts local segments %128==0).
3. **`models/deepseek_v4/weight.py` `iter_expert_pieces`** — delete the
   TP gate; wrap the raw (name, tensor) stream (BOTH the parallel
   `iter_expert_tensors_parallel` path and the serial fallback) BEFORE
   `per_expert_pieces` with per-role rank slicing. Guard
   `I % (32 * tp.size) == 0` with an explicit error; `lo = rank * (I//tp)`,
   `hi = lo + I//tp`.

   | role (w1=w gate, w3=up, w2=down) | checkpoint shape | slice |
   |---|---|---|
   | gate/up weight | `[I, H//2]` uint8 | rows `[lo:hi]` |
   | gate/up scale | `[I, H//32]` e8m0 (dim0 = I) | rows `[lo:hi]` |
   | down weight | `[H, I//2]` uint8 | cols `[lo//2 : hi//2]` |
   | down scale | `[H, I//32]` e8m0 | cols `[lo//32 : hi//32]` |

   Reasoning: `fused_piece` cats gate/up along dim=1 of `[1, rows, cols]`
   into a `(2I, H//2)` bank row span; dim0 of every gate/up tensor is I.
4. **`layers/quantization/moe/mxfp4.py`** — `TritonMxfp4MoEKernel`:
   `tp_ok = True`; add divisibility reject
   (`intermediate % (GROUP * tp_size)`) in `unusable_reason`; `layout()`
   uses `cfg.local_intermediate`; `pack` unchanged. (DSV4 has_bias=False
   is refused by `Mxfp4MoEMethod.create_weights`, so DSV4 remains
   offload-only through this kernel — unchanged behavior.)
5. Tests under `tests/dsv4/` (CPU-runnable):
   - expert slicing: per-role reassembly == full for tp=2/4; exact
     `[1, local, ...]` piece shapes through `per_expert_pieces`;
     divisibility guard error text; numeric identity
     sum(partial over shards) == full swiglu output (small fp weights);
   - dense slicing in `iter_weights` over a synthetic tiny checkpoint
     (monkeypatched): shard shapes match what the layer objects allocate,
     reassembly == full, replicated tensors identical;
   - attention local geometry with monkeypatched tp: attn_sink/wo_a
     shapes, divisibility raises;
   - exactly-once collectives: fake communicator counts one all-reduce
     after routed MoE and one after o_proj (code-path test where cheap).

### Phase 3 — optional kernel opt-ins (not required for the 3 configs)

- fp8_block MoE TP opt-in (needs `intermediate % (128*tp)`): tp_ok + local
  layout + alignment asserts. Reachable families' readers (glm5_next,
  qwen3_5_moe) reject TP=1-only loudly before load, so the opt-in cannot
  crash on unsharded pieces. marlin / b12x stay `tp_ok=False`. GGUF q4_0
  expert provider stays gated.
- nvfp4-triton opt-in: **reverted after adversarial review** — no NVFP4
  expert reader shards pieces by rank, and glm4_moe is reachable under TP,
  so the opt-in loaded full pieces into rank-local banks and crashed
  mid-load instead of the previous clean `KernelSelectionError`. The kernel
  stays `tp_ok=False`; re-enabling needs reader-side NVFP4 sharding first
  (follow-up).

### Phase 4 — runtime + docs

- Unit test `destroy_distributed` plugin restore (0.3), div_even raises
  (0.4), env override (0.5), loader bias rule (0.1), reset fixture (0.9).
- Optional gloo 2-process communicator test (file:// init to dodge the
  default port) — CPU-capable.
- `docs/cli.md` TP section: `--gpu` comma-per-rank +
  `--tensor-parallel-size`; note offline `llm.py` stays TP=1.
- 2-GPU e2e smoke (`tests/e2e/test_tp_two_gpu.py`): boots a real small
  checkpoint (`FREETOKEN_TP_TEST_MODEL`, falling back to
  `FREETOKEN_TEST_MODEL`) by spawning
  `python -m freetoken --tensor-parallel-size 2 --gpu 0,1` as a subprocess,
  waits for the "serving" state and asserts a deterministic completion. CUDA +
  2-device gated — NOT run on this machine (no GPU/NCCL).

## Test matrix

| Suite | Here (Win, no GPU) | Dev box (Linux + 2 GPU) |
|---|---|---|
| Phase 0 unit tests | run | run |
| Phase 1 gpt-oss sharding tests | run | run |
| Phase 2 dsv4 slicing/geometry tests | run | run |
| gloo 2-proc communicator | run (optional) | run |
| 2-GPU e2e smoke (dsv4 tiny + MXFP4 + bf16 + gpt-oss) | NOT run | run |

Per repo rules, nothing GPU/NCCL-dependent may be reported as run from
this machine.

## Non-goals / known limits

- Single node only (no inter-node TP).
- qwen3_vl vision encoder TP stays rejected.
- GGUF q4_0 expert provider TP stays gated.
- nvfp4 experts stay TP-gated on every kernel (triton included) until an
  expert reader shards pieces by rank — follow-up.
- Offline `freetoken.llm` stays TP=1 (documented).
- pynccl CUDA source fp32 map change: documented follow-up, not done here.
- bf16 fused-MoE under TP (follow-up): `FusedMoEKernel`
  (`layers/quantization/moe/unquantized.py`) has no TP gate while the dense
  expert readers (qwen3_moe, minimax_m2) chunk expert rows with ceil-sized
  windows. With `intermediate % tp != 0` the last rank's shard is shorter
  than the locally allocated bank and the load dies on a cryptic `copy_`
  shape error instead of a divisibility message. Follow-up: opt in with
  `intermediate % tp == 0` (or zero-pad the tail like the vocab fix).
