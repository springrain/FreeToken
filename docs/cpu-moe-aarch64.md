# CPU MoE 的 aarch64 支持(Ampere One)

本文档说明 FreeToken CPU MoE 执行器(`--moe-backend cpu`,对应扩展
`freetoken.kernel._cpu_moe`)对 aarch64 Linux 主机的支持,目标平台为
Ampere One 系列:ARMv8.6-A 自研核,128-bit NEON,支持 FEAT_DotProd(SDOT)
与 FEAT_BF16(BFDOT),不支持 SVE。

范围仅覆盖 CPU 专家 GEMV 内核。引擎其余部分(GPU 路径、JIT CUDA 内核、
安装与 wheel)不变,仍为 Linux x86_64;整引擎的 aarch64 移植不在本次范围内。

## 当前状态

aarch64 内核已写完并通过静态检查(预处理配平、符号唯一性),但尚未在
aarch64 硬件上编译或压测过;首次启用请按文末验收清单执行。清单通过之前,
应按未验证状态对待本移植。

## 改动内容

- `python/freetoken/kernel/csrc/cpu_moe/cpu_moe_ext.cpp` — 全部 aarch64
  工作都在这里,由 `CPU_MOE_AARCH64` 与编译期特性宏保护;x86 行为逐字节不变。
- `setup.py` — 仅注释更新;仍不设全局 `-march`,单个二进制保持可移植
  (可回退至标量)。
- `python/freetoken/moe/benchbw.py` — `_ISA_TIERS` 按架构分派
  (aarch64 下为 `scalar/neon/neonbfdot`),使 `ft bench bw` 在 ARM 主机上
  扫描正确的 tier 名称。

## 设计要点

### 运行时分发

aarch64 上没有 `__builtin_cpu_supports()`,因此用
`getauxval(AT_HWCAP)` / `getauxval(AT_HWCAP2)` 探测特性
(`arm_has_dotprod()` / `arm_has_bfdot()`,结果缓存)。向量内核带
per-function `__attribute__((target("arch=armv8.2-a+dotprod")))` /
`+bf16`,仅在工具链支持时编译(`CPU_MOE_HAS_NEON_DOTPROD`:gcc>=8 或
clang>=9;`CPU_MOE_HAS_NEON_BF16`:gcc>=10 或 clang>=12)。老工具链或非
Ampere 的 ARM 主机会静默回退到 NEON/标量,不会编译失败。

`IsaTier` 增加 `ISA_NEON` / `ISA_NEON_BFDOT`(cap-down 语义不变)。
aarch64 下的环境变量名:`FREETOKEN_CPU_MOE_ISA=scalar|neon|neonbfdot`。
`FREETOKEN_CPU_MOE_SCALAR` 与 W4A8 总开关 `FREETOKEN_CPU_MOE_NO_VNNI` 在
两种架构上行为一致;aarch64 上该开关作用于 `cpu_has_sdot()`,即 x86
AVX-VNNI 探测的对应物。

### 各权重格式的内核

| 权重格式 | aarch64 内核 | 说明 |
|---|---|---|
| BF16 (WF_BF16) | `dot_neon`、`dot_neon_bfdot` | BFDOT 需要 FEAT_BF16;普通 NEON 为回退。唯一的 16-bit 权重格式,不存在 fp16 权重格式。 |
| NVFP4 (WF_NVFP4) | `dot_nvfp4_neon`(W4A16)、`dot_nvfp4_i8_sdot`(W4A8) | e2m1 经 `vqtbl1q_s8` LUT(`kE2M1x2`)解码。SDOT 为有符号×有符号,加倍的有符号 LUT 无需符号技巧;0.5 折进 block scale(2 的幂,精确)。 |
| ds_fp4 (WF_DSFP4) | `dot_dsfp4_neon` | 16 字节恰为一个 32-K 块,e8m0 每 32-K scale,`acc += sc*bsum` 顺序与标量参考一致。FP8 语义由该路径承载(FP8-e4m3 激活 round-trip),外加 nvfp4 的 e4m3 block scale。 |
| MXFP4 (WF_MXFP4) | `mxfp4_gemv_neon` | 转置 split-K `blk[Kpairs, N2]`,K 外/N 内,每块 16 列。e8m0 scale 用 `<<23` 位构造(与 x86 相同,不走 LUT)。软件预取 PFD=8 覆盖 N2 大步长的 K 遍历。 |
| Q4_0 (WF_Q4_0) | `q4_0_dot_i8_sdot` | W4A8,激活为 Q8_0;`nibble - 8` 用 `vsubq_s8`,在 [-8, 7] 内精确。 |

用户所说的 INT4 即 NVFP4 / Q4_0 / ds_fp4;不存在单独的 INT4 权重格式。

### 数值注意点

- NEON fp4 内核使用加倍的 e2m1 int8 LUT(`kE2M1x2`);0.5 是精确的 2 的幂,
  折进每块 scale(nvfp4:e4m3 scale × 0.5;ds_fp4:e8m0 scale × 0.5;
  mxfp4:每 32-K e8m0 × 0.5)。标量 tail 用未加倍的 float LUT,与 x86 的
  tail 逐字节一致。
- SDOT 点积(`vdotq_s32`)是有符号×有符号,e2m1x2 LUT 与 Q4_0 的 [-8,7]
  权重可直接送入。x86 需要 `|w| * (sign(w)*a)` 重排,只是因为
  VPDPBUSD/VPMADDUBSW 是无符号×有符号。
- Q8_0 / int8 激活路径与 x86 共用(同样的缓冲区、同样的预量化契约);
  aarch64 只替换了点积内核。

### 刻意不做的事

- 不用 SVE/SVE2:Ampere One 没有;NEON 定宽 128-bit。
- `setup.py` 不加全局 `-march`:单一二进制必须能在任意主机上回退至标量运行。
- 不改 CUDA graph 的提交/同步协议与 flag 握手;内存序原本就是可移植的
  (acquire/release)。只把两处 `_mm_pause()` spin 点换成跨架构的
  `moe_pause()`(aarch64 上为 `yield`)。

## 启用验收清单(在 aarch64 Linux 主机上执行)

```bash
python setup.py build_ext --inplace
uv run pytest tests/moe -m "not slow" -x -q
uv run pytest tests/engine/test_moe_cpu_layers.py -q

# 每个格式做 tier A/B:不同 tier 的输出必须在容差内一致
for fmt in nvfp4 mxfp4 ds_fp4 bf16; do
  for isa in scalar neon neonbfdot; do
    FREETOKEN_CPU_MOE_ISA=$isa uv run python -m freetoken.moe.benchbw --dtype $fmt
  done
done

# W4A8 总开关:回退路径必须仍然正确
FREETOKEN_CPU_MOE_NO_VNNI=1 uv run pytest tests/moe -q
```

构造器标签(`isa_name()`)会显示分发结果,例如
`neon-bfdot+sdot(nvfp4-w4a8)`。
