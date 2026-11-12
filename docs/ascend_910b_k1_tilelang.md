# K1 TileLang Ascend 实现与性能、精度对比

实测日期：2026-11-10。Ascend 910B2；CANN 9.1.0、Python 3.10、torch 2.10.0、torch-npu 2.10.0.post4。

## 结论

- 已完成纯 TileLang DSL K1，编译后每次调用为 **1 个 kernel**，没有转调此前手写的 Ascend C K1。
- 固定采用 `blocks=20` 时，所测设备 kernel 比手动同步 Ascend C 快 **1.72～2.05 倍**。这不等于完整调用接口或整模型的加速倍数。
- **当前 TileLang Python/Cython 接口在 B=1 时比手写 torch-npu 绑定慢；B=4 时更快。** mixed B=1：kernel 0.0656 vs 0.1311 ms，但接口 wall 0.2605 vs 0.1431 ms。mixed B=4：kernel 0.2335 vs 0.4643 ms，接口 wall 0.2611 vs 0.4778 ms。
- 本轮 16 个真实 TileLang 配置、18 个补充/边界用例、非默认 stream 检查，对已有 NPU Tensor 的有限法线数值、mask、NaN/Inf 位置均严格一致。
- 当前已通过 custom op 接入固定桶 ACLGraph，并与 K2/K3 串联；默认 `model.infer` 路径不变，部署需显式启用。见 [当前设计](ascend_910b_design.md) 和 [图部署](ascend_910b_aclgraph.md)。下表保留 2026-11-10 的同进程单算子测量。

## 实现与 tiling

公式与 [当前设计](ascend_910b_design.md) 一致；[两版 Ascend C 源码](../operators/k1/) 保留为对比基线：中心点与上/左/下/右点形成四个有向三角形；每个叉积除以 `sqrt((x*x+y*y)+z*z)+1e-12`，乘三角形有效性，再按原顺序求和并归一化。无有效三角形的位置输出零。保留 `NaN * 0` 的参考语义，未以提前置零替代。

输入为连续 FP32 `[B,H,W,3]` 点图与 bool `[B,H,W]` mask。接口当前限定 `B,H>0、1≤W≤576`，不支持梯度。支持的上限 W=576 已补测；实际模型 W=518。

- 行 tile：`WP=ceil(W/64)*64`，模型实际 WP=576。
- 每轮只搬入中心、上、下三行点与 mask；左/右邻居由 UB 中的 Gather 生成。图外点及有效性置零，batch 边界不会跨图取邻居。
- 使用 `T.tile` 的 FP32 算术、Gather、Compare/Select 和显式 Cast，开启 TileLang 自动同步、V/S 同步及内存规划。Bisheng 使用派生的 O3 / CCE auto-sync off，另指定 `-ffp-contract=off`。
- `T.Kernel(blocks)` 使用当前后端的 `MIX_AIC_1_2` 模式，计算仅在 V 域执行；`blocks=20` 对应 20 对、40 个逻辑 AIV。`blocks=40` 是 80 个逻辑 AIV 的配置，不应与旧纯 AIV kernel 的 `blocks=40` 直接视为相同 launch 参数。
- Gather 索引在片上用向量指令生成，每核初始化一次，未在计时外做输入转置/预处理，也没有额外 GM 索引输入。
- W=518 的生成代码 UB 最大使用地址为 **172224 字节**；后端保留的 UB 区域为 196352 字节。单缓冲行循环，未引入双缓冲。
- N 输出行 stride 为 `align_up(3W,8)` 个 float，V 为 `align_up(W,32)` 字节，与旧版本同样采用行 padding；返回逻辑 shape 不变，通常不连续。计时不含额外 `.contiguous()`。

两版底层 tiling 并不相同：旧 C++ 是每行再分成 T=256 的 tile，新版为整行。因此性能结果代表这两种具体实现，不能归结成“TileLang 语言天然比 Ascend C 快”。

## 同一进程的性能比较

四轨道各取之前缓存的第 0 个 tuple；B=1 为第一个视角，B=4 为四视角合批。每种调用预热 5 次，10 组 × 每组 10 次，组末 NPU 同步；JIT 构建/加载在计时前完成。没有将编译耗时计入稳态延迟。

### 设备 kernel 时间，ms

Tensor 列为 205 个设备任务时长之和，其余各为 1 个 kernel。独立 profiler 采集 2 个 active step 后取均值。

| 输入 | B | H×W | Tensor | TQue | 手动同步 | TileLang blocks=20 | TileLang blocks=40 |
|---|---:|---|---:|---:|---:|---:|---:|
| erp | 1 | 252×518 | 1.7556 | 0.0926 | 0.0894 | 0.0520 | 0.0468 |
| mixed | 1 | 392×518 | 1.8752 | 0.1356 | 0.1311 | 0.0656 | 0.0695 |
| pinhole | 1 | 392×518 | 1.8677 | 0.1360 | 0.1307 | 0.0649 | 0.0696 |
| fisheye | 1 | 518×518 | 1.9862 | 0.1725 | 0.1650 | 0.0818 | 0.0899 |
| erp | 4 | 252×518 | 2.5777 | 0.3186 | 0.3053 | 0.1528 | 0.1563 |
| mixed | 4 | 392×518 | 2.9480 | 0.4867 | 0.4643 | 0.2335 | 0.2305 |
| pinhole | 4 | 392×518 | 2.9535 | 0.4865 | 0.4645 | 0.2332 | 0.2304 |
| fisheye | 4 | 518×518 | 3.3953 | 0.6461 | 0.6116 | 0.2986 | 0.3025 |

默认接口选 blocks=20，保留 blocks=40 用于比较；没有宣称全局最优。ERP B=1 的 blocks=40 设备时间更低，而多数配置差距较小。

### 无 profiler 的接口 wall time，ms

包括输入/输出 view、Python/Cython 调用、输出分配和同步摊销；与旧版本的 C++/PrivateUse1 入口比较。这是当前实际暴露接口的代价。

| 输入 | B | H×W | Tensor | TQue | 手动同步 | TileLang blocks=20 | TileLang blocks=40 |
|---|---:|---|---:|---:|---:|---:|---:|
| erp | 1 | 252×518 | 3.2419 | 0.1064 | 0.1026 | 0.2634 | 0.2603 |
| mixed | 1 | 392×518 | 3.2170 | 0.1499 | 0.1431 | 0.2605 | 0.2591 |
| pinhole | 1 | 392×518 | 3.2141 | 0.1502 | 0.1431 | 0.2580 | 0.2591 |
| fisheye | 1 | 518×518 | 3.0819 | 0.1883 | 0.1784 | 0.2991 | 0.2540 |
| erp | 4 | 252×518 | 3.1209 | 0.3339 | 0.3187 | 0.2575 | 0.2573 |
| mixed | 4 | 392×518 | 3.1741 | 0.5040 | 0.4778 | 0.2611 | 0.2568 |
| pinhole | 4 | 392×518 | 3.1846 | 0.5021 | 0.4767 | 0.2600 | 0.2562 |
| fisheye | 4 | 518×518 | 3.4006 | 0.6524 | 0.6211 | 0.3266 | 0.3274 |

向量化后设备计算明显缩短，但 B=1 的 TileLang wall 仍约 0.25～0.30 ms，明显高于自身 0.05～0.08 ms 的 kernel 时间。调用端和调度开销已盖过单 kernel 收益。fisheye B=1 blocks=20 的 wall 样本也存在 host 波动，不能用这一条 wall 差值判断两种核分配的设备优劣。JSON 保留逐组 wall/Event 样本；Event span 可能含 host 供给间隔，也不应替代设备 kernel duration。

TileLang 的直接 Cython launch 在本次 torch-npu trace 中没有可用的 host→device flow 连线。为避免错误归属，给两种配置生成不同的 kernel 符号，`summarize_npu_k1_tilelang.py` 按 `meow_k1_tilelang_b20_kernel` / `b40_kernel` 名称匹配，并检查每个 active step 恰好一个对应 kernel。通用汇总的 `unmatched` 中仍包含这些任务，`tilelang_by_symbol` 是对它们的细分，不能再叠加到总时长。

## 精度

| 对比项 | 本次结果 |
|---|---|
| TileLang vs NPU Tensor，有限法线 max abs | 0 |
| TileLang vs NPU Tensor，mask mismatch | 0 |
| NaN、正 Inf、负 Inf 位置 mismatch | 0 |
| 两版旧 Ascend C vs NPU Tensor | 本轮同样严格一致 |
| 真实点图 vs CPU 法线最大绝对误差 | 1.6488e-5，与 NPU Tensor 相同 |
| 大随机点图 vs CPU 最大绝对误差 | 0.0006983，与 NPU Tensor 相同 |

真实输入 8 组 × 5 条路线共 40 组检查。补充验证为 7 个 shape × 常规/非有限值共 14 组，再加平面、全退化零点、有效 mask 内 NaN、有效 mask 内 Inf 共 18 组；包含 1×1、1×65、65×1、2×3、17×33、65×518、3×576，覆盖 B=1/2。另有非默认 NPU stream 的输入生产/消费依赖检查。

脚本同时保存 `exact` 与 `atol=2e-4, rtol=1e-4` 的检查结果；本轮 NPU 侧 `exact` 全部为 true。CPU 大随机用例仍不通过上述容差，不能将其表述为 CPU 全等价。最终 K2/K3 mask 的历史差异也没有被本次 K1 验证消除。

## 实现中修复的问题

1. Ascend Vector 不接受 uint8 fill：用 half 零值转换初始化 mask 存储。
2. `uint8 → half` 需显式 `CAST_NONE`。该版本 `T.copy` 的默认转换模式曾导致 mask 错误，最终使用 `T.tile.cast` 明确指定。
3. 初版整数索引的 `T.Parallel` 被降为逐元素 SetValue 和大量 PIPE_ALL。最终用 CreateVecIndex、整数向量算术、Cast/ReinterpretCast 生成 Gather 索引。
4. 输出索引中的整数除 3 用浮点商取整实现；该版本代码生成器把 `1/3` 打印为 `3.333333e-01f`，偏小会取错整数商。仅在这一步索引生成加 `1e-3` 修正，在受限的 W≤576 范围内覆盖舍入误差且小于相邻商的间隔。未修改几何 epsilon 或法线值，W=576 已验证。

初版的通过结果保留在 `work/npu-k1/tilelang/results.json`，其源快照为 `kernel_scalar_indices.py`；最终向量索引版本在 `work/npu-k1/tilelang/vector/`。失败版本的数值和性能不计入最终结果。

## 代码、环境与复现

| 文件 | 用途 |
|---|---|
| `operators/tilelang_kernels.py` | 统一的 TileLang K1/K2/K3、数学公式、缓存和 Tensor 调用入口 |
| `operators/k1_tilelang/kernel.py` | K1 兼容导入入口 |
| `operators/k1_tilelang/requirements-build.txt` | 本轮新增的 Python 编译/运行依赖版本 |
| `scripts/build_npu_tilelang.sh` | 使用项目内 GCC 13 构建 TileLang/TVM，不替换系统 SDK |
| `scripts/run_npu_tilelang.sh` | 加载源构建、CANN、torch-npu 环境 |
| `scripts/npu_bench_k1_tilelang.py` | Tensor/两版 C++/两种 TileLang 配置的精度、稳态计时与 profiling |
| `scripts/summarize_npu_k1_tilelang.py` | 按独立 kernel 符号补充 TileLang profiler 归属 |

源码版本锁定为官方 [tile-ai/tilelang-ascend](https://github.com/tile-ai/tilelang-ascend/tree/4805777d691b0c586e647fc0f975fa4cc306c00e) 的 `4805777d691b0c586e647fc0f975fa4cc306c00e`。采用 [官方 elementwise 示例](https://github.com/tile-ai/tilelang-ascend/blob/4805777d691b0c586e647fc0f975fa4cc306c00e/examples/elementwise/elementwise_add.py) 的 kernel/UB/Vector 结构与该版本 tile API。当次验收环境使用 Python 3.10，因此固定使用上述源码构建；不据此推断当前发行版 wheel 的支持范围。

源目录在 `work/npu-k1/tilelang/source`；读取其 pinned TVM、catlass、shmem 子模块，并应用官方自带的 TVM patch 脚本。没有修改 TileLang 编译器源码来放宽数值检查。新增依赖装在仓库 `.venv`，复用既有 `work/npu-k1/toolchain` GCC 13。

现有远端环境可直接复现：

```bash
cd /root/workspace/MEOW-het
# 如需重编已配置的源树：
MEOW_BUILD_JOBS=64 bash scripts/build_npu_tilelang.sh

ASCEND_RT_VISIBLE_DEVICES=2 bash scripts/run_npu_tilelang.sh \
  scripts/npu_bench_k1_tilelang.py --blocks 20 40 --trace \
  --out work/npu-k1/tilelang/retest/results.json

.venv/bin/python scripts/summarize_npu_k1_tilelang.py \
  work/npu-k1/tilelang/retest/trace \
  --out work/npu-k1/tilelang/retest/summary.json
```

从一个通过 `run_npu_tilelang.sh` 启动的 Python 脚本调用：

```python
import torch
import torch_npu
from operators.k1_tilelang.kernel import pointmap_to_normals

with torch.inference_mode():
    normals, normal_mask = pointmap_to_normals(points, mask, blocks=20)
```

本地和远端均有最终 `results.json`、`summary.json`、`edgecases.json`、生成的 C++、kernel/operator CSV 与 trace JSON；原始 CANN profiling 及完整编译目录保留远端。

## 当前集成状态与剩余范围

K1 已注册 PyTorch custom op 和 FakeTensor，按四视图合批接入固定桶 ACLGraph，并与 K2/K3 串联；下游支持其 padded stride。当前仍使用 TileLang JIT 调用链，未新增手写 launch 绑定或训练反向，也不需要 GE converter。进一步降低调用开销需同口径实测，不能从旧单算子 wall time 推断当前图模式收益。

完整模型与后处理验收见 [当前设计](ascend_910b_design.md) 和 [最新融合结果](ascend_910b_tilelang_edges.md)。动态 shape、可选置信度/多视图一致性及相对原 NumPy 的 mask 差异仍不在已解决范围内。
