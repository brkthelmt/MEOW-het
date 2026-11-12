# TileLang K2/K3 几何后处理融合

## 实现与接入

统一 kernel 与数学公式：`operators/tilelang_kernels.py`（`operators/geometry_tilelang/kernels.py` 保留兼容导入）；PyTorch dispatcher 与 FakeTensor：`operators/geometry_tilelang/torch_op.py`。复用既有 K1 TileLang 实现。没有调用手写 Ascend C 外部 kernel；TileLang 自动同步、内存规划开启，禁用浮点乘加收缩。

实际拆分与早期候选设计不同：

| 阶段 | 输入 | 输出 | 融合内容 |
|---|---|---|---|
| K1，已有 | 世界坐标点图 P、基础 mask M | 法线 N、有效性 V | 邻域差分、叉积、归一化 |
| K2，本轮 | N、V、法线角阈值 | 未扩张的布尔边缘 E | 法线再归一化、3×3 点积、NaN/域判断、角阈值和归约 |
| K3，本轮 | E、径向深度 D、M、P、相机点图 C | 最终 F、P′、C′、D′、Z′ | E 的 3×3 扩张、深度范围、最终 mask、四个几何字段过滤 |

将深度判断放入 K3，避免单独保存深度边缘图。K2 只保存一字节布尔结果，避免完整角度图和 normalized-normal 中间图写回 HBM。

接口：`normal_candidates(N,V,degrees=5.0)` 与 `mask_and_apply(E,D,M,P,C,depth_tol=0.03)`。P/C/N 为 FP32 BHWC，D 为 FP32 BHW，mask 为 bool BHW。同一 NPU，推理模式；B/H 正数，`1<=W<=576`。P/C/D/M 要求连续，N/V/E 支持内核产生的行 padding。输出有明确的 padded row stride，不承诺 contiguous。

部署显式启用：

```python
runner = NpuGraphRunner(model, processed_views, fused_edges=True)
outputs = runner(*(view['img'] for view in processed_views))
```

`fused_edges=False` 保留原 K1 TileLang + Tensor K2/K3 路径，便于同机对比。图像预处理、元数据分桶、可选 confidence/多视图一致性分支的范围与此前 ACL Graph 接入相同。

## 公式与精度约束

仍使用 `U(n)=n/(sqrt((nx²+ny²)+nz²)+1e-12)`，按原加法顺序计算 dot。法线与有效性使用 replicate 边界，并保留参考实现中 **V 的邻域轴转置**。

原局部角度最大值：`A(p)=max_q where(V_transposed(q), acos(dot(U(Np),U(Nq))), 0)`，此处任何参与归约的 NaN 都使 A 为 NaN。

K2 改为输出 `E(p) = is_valid_angle_window(p) AND [A(p)>τn]`。只要选中的 dot 是 NaN 或超出 `[-1,1]`，E 为 false。后续 `nanmax(A邻域)>τn` 与邻域 E 的 OR 相同（默认正角阈值）；全 NaN 窗口的结果也为 false。

**没有直接使用四舍五入的 cos(τn)**。初始化时在 NPU 上用现有 `torch.acos` 比较，检查 cos 附近 257 个 FP32 值的单调转换，选择仍满足严格角度比较的最大 dot 值 d*，并用 `[-1,1]` 上 65,537 个值交叉检查。kernel 使用 `dot<=d*`。校准失败会抛错，不能静默继续近似；新阈值/新运行环境必须重新校准。本轮验收默认 5°。

校准和 JIT 必须在图捕获前完成，`NpuGraphRunner` 的 eager 预热负责此步骤。d* 作为 FP32 设备 tensor 传入，避免生成器输出浮点字面量时截断精度。校准只发生在首次调用，重放时不执行 acos 或 CPU 检查。

K3 保留：

- `Ed=(nanmax(where(M,D,-∞)) + nanmax(where(M,-D,-∞)))/D > τd`。
- `F=M AND NOT(dilate(E) AND Ed)`。
- `P′=P*F, C′=C*F, D′=D*F, Z′=Cz*F`。

不对深度取绝对值，不 clamp 分母，不用 where 替代最终乘法，保留 `NaN*0`/`Inf*0` 语义。max 的 NaN 忽略与边界语义由独立异常样例验证。

## Tiling 与同步

每个 AIV 每次负责一整行，20 个 block 使用 40 个 AIV，按行循环；列计算宽度补齐到 64。K2 将三行法线与 V 放入 UB，使用 Gather 形成左右邻居，九个邻域的归约在 UB 完成。K3 同样缓存三行 D/M/E，一次完成邻域判断与输出。

FP32 输出行补齐到 8 个元素（32 B），bool 行补齐到 32 B，避免不同核写相邻行尾部时覆盖同一个写入块。FakeTensor 明确复现这些 stride。跨 K1/K2/K3 的依赖依靠同一 NPU stream 的 kernel 顺序，不尝试在单个 kernel 内做跨核全局同步。

## 复验

在远端 `/root/workspace/MEOW-het`：

```bash
ASCEND_RT_VISIBLE_DEVICES=2 bash scripts/run_npu_tilelang.sh scripts/npu_validate_tilelang_edges.py --trace
ASCEND_RT_VISIBLE_DEVICES=2 bash scripts/run_npu_tilelang.sh scripts/npu_profile_tilelang_edges.py
ASCEND_RT_VISIBLE_DEVICES=2 bash scripts/run_npu_tilelang.sh scripts/npu_validate_graph_postprocess.py --fused-edges --out work/npu-edges/postprocess --trace
ASCEND_RT_VISIBLE_DEVICES=2 bash scripts/run_npu_tilelang.sh scripts/npu_validate_graph_model.py --stage full --runner --fused-edges --compare-unfused --tracks blk_mixed blk_erp blk_pinhole blk_fisheye --out work/npu-edges/full --trace
```

精度参考是既有 NPU Tensor 路径，不表示解决了与原 NumPy 后处理之间的历史 mask 差异。端到端每类采用 benchmark 第一个四视图 tuple；不是全量数据集精度验收。性能测量排除首次校准、JIT 和图捕获时间，device kernel 时长与 Python wall time 分开记录。

## 实测结果（910B2 / CANN 9.1.0）

六组模型 shape（B=1/4，H=252/392/518，W=518）、四类真实缓存，以及退化/阈值/非有限输入和非默认 stream 检查通过。四类完整推理每类三轮换输入重放，全字段与未融合 NPU 路径逐元素一致，未放宽阈值。

纯 device kernel 时间（每形状 profiler 两次执行均值，不含 Python/JIT）：

| B×H×W | K2 (μs) | K3 (μs) |
|---|---:|---:|
| 1×252×518 | 63.01 | 41.33 |
| 1×392×518 | 87.07 | 56.98 |
| 1×518×518 | 112.33 | 72.44 |
| 4×252×518 | 215.50 | 138.25 |
| 4×392×518 | 328.85 | 210.87 |
| 4×518×518 | 426.42 | 272.48 |

完整后处理，同进程同卡 ACL Graph A/B，10 次同步计时均值：

| 输入 | 未融合 (ms) | 融合 K2/K3 (ms) | 耗时下降 |
|---|---:|---:|---:|
| blk_mixed | 17.184 | 5.323 | 69.0% |
| blk_erp | 13.466 | 5.005 | 62.8% |
| blk_pinhole | 17.059 | 5.242 | 69.3% |
| blk_fisheye | 20.346 | 5.500 | 73.0% |

完整推理，同进程同卡 ACL Graph A/B，未融合 10 次、融合 5 次同步计时均值：

| 输入 | 未融合 (ms) | 融合 K2/K3 (ms) | 耗时下降 |
|---|---:|---:|---:|
| blk_mixed | 149.918 | 143.261 | 4.4% |
| blk_erp | 98.100 | 92.562 | 5.6% |
| blk_pinhole | 144.770 | 138.221 | 4.5% |
| blk_fisheye | 186.599 | 177.317 | 5.0% |

单 kernel profiling 使用物理卡 5，独立后处理 A/B 使用卡 3，整链路 A/B 使用卡 4；不跨卡做 A/B。设备上还有其他卡的既有任务，以上为本轮采样，非长期吞吐或并发性能承诺。

四类整链路 trace 每两次推理均有 2 次 ACL replay、2 次 MODEL_EXECUTE，K1/K2/K3 各 2 次；没有 `_local_scalar_dense` 或 `linalg_solve` 事件。K2/K3 已进入同一个主模型整图，不是在每次 replay 后额外 Python 调用。

复验入口：`scripts/run_npu_tilelang_edges_validation.sh`。本轮已逐项执行其中的单算子、后处理与整图验证；聚合脚本仅做 shell 语法检查，未重复整套测试。结果 JSON 与 trace 位于远端 `work/npu-edges/`；JSON 已同步到本仓库同路径。

收益主要来自减少 K2/K3 的小 kernel 与中间数据读写。完整后处理仍包含 RGB、内参和位姿等 Tensor 运算，主模型也仍占主要时延；不能把后处理约 3 倍的提速解释成整模型 3 倍提速。


## 2026-11-11 profiling 同步与流水线分析

本轮分析既有 2026-11-10 profiler 数据，没有重新运行性能测试。远端两个目录已完整同步到本地同路径：`work/npu-k1/tilelang/vector/`、`work/npu-edges/`。共 2,070 个文件、112,900,614 字节（约 107.7 MiB），包含导出的 CSV/trace JSON 和远端目录中保留的 CANN 原始采集文件。全部文件 SHA-256 与远端清单一致。

本地汇总：`work/npu-tilelang-profiling/pipelines.csv`、`pipelines.json`；同步校验：`remote_manifest.json`、`sync_verification.json`。复现分析：`python3 scripts/summarize_tilelang_pipelines.py`。

下面取同一份 mixed 整图 trace，B=4、392×518，每个 kernel 两次执行均值；不混用不同采集的单算子与整图时间：

| 算子 | kernel (μs) | Vector 活跃比例 | MTE2 活跃比例 | MTE3 活跃比例 | Scalar 活跃比例 |
|---|---:|---:|---:|---:|---:|
| meow_k1_tilelang_b20_kernel | 231.31 | 72.4% | 10.8% | 4.0% | 8.8% |
| meow_k2_tilelang_kernel | 335.59 | 80.1% | 8.1% | 2.3% | 7.7% |
| meow_k3_tilelang_kernel | 214.05 | 65.5% | 18.6% | 16.3% | 5.5% |

解释：这些是 CSV 的 AIV pipeline 活跃比例，不是 kernel wall time 的互斥组成。各流水线可能重叠，不能相加到 100%，不能用 MTE2+MTE3 直接估算 double buffer 可省时间；AIV 计数器统计也不直接揭示关键路径上的等待。Cube/AIC 指标不适合判断这些 Vector kernel。

- K1：Vector 活跃明显高于 MTE；当前数据支持先不做 double buffer，但不能证明收益必为零。独立 B=4 三种尺寸中 Vector 约 71%–73%，MTE2 约 11%，MTE3 约 4%。
- K2：Vector 更占主导。独立 B=4 三种尺寸中 Vector 约 81%–82%，MTE2 约 7%–8%，MTE3 约 2%；更应关注 Gather、向量指令与依赖链，而非优先复制整套 UB。
- K3：MTE2/MTE3 的活跃比例较高，与需要读取并写回多个几何字段的代码结构一致。若优化单 kernel，优先试 K3 的输入/输出 ping-pong；实际收益仍需生成代码中的同步依赖分析和同形状 A/B 测量。

mixed 整图内三个 TileLang kernel 累计约 0.781 ms，相对本轮此前测得约 143.3 ms 整链路占比约 0.55%。因此，以端到端收益排序时，DPT 的布局转换等热点仍比这三个 kernel 的 double buffer 更优先。该比较是量级判断，不是可直接兑现的优化收益预测。

## 当前源码整理与回归

2026-11-11：K1/K2/K3 统一到 `operators/tilelang_kernels.py`，旧模块保留兼容导入。9 个函数 AST 与迁移前一致；23 项边界测试、六组模型 shape、四类真实缓存通过；四类后处理 ACLGraph 各三次变更输入重放通过。结果位于本地 `work/npu-edges/consolidated-results.json` 和 `work/npu-edges/consolidated-graph/results.json`，不随 Git 分发。本次重构未重新执行完整模型性能测试。

2026-11-12：移除被当前实现替代的阶段性设计文档，统一入口见 [当前设计](ascend_910b_design.md)。既有测量表继续标识其原始采样口径，不当作本次文档整理的新测量。
