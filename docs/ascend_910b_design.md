# Ascend 910B 当前实现与设计

更新：2026-11-12。本文对应当前源码，取代早期融合候选、K1 伪代码和 Tensor 迁移阶段方案。原始项目的安装、训练、评测等文档继续保留。

## 当前入口与范围

主模型已增加 torch-npu 设备支持；固定桶部署使用 `NpuGraphRunner(model, processed_views, fused_edges=True)`，经 `torch.compile(..., backend='npugraph_ex', fullgraph=True, dynamic=False)` 捕获/重放 ACLGraph。此路径不要求 GE converter。默认 `model.infer` 没有自动切换到此部署入口。

图内路径：主模型 → RGB 反归一化、内参恢复、位姿及几何后处理。几何后处理将同尺寸视图合批，然后执行 K1→K2→K3，最后拆分输出。K1/K2/K3 为三个设备 kernel，不是单个完全融合 kernel；它们与主模型共同进入图重放。

图外：图像读取/resize、元数据分桶、导出及外部 ICP/KNN。未验收动态 shape、可选置信度/多视图一致性分支、几何先验输入和训练。严格一致性的参考是现有 NPU Tensor 路径，不表示解决了原 NumPy 路径阈值附近的 mask 差异。

## 源码索引

| 文件 | 用途 |
|---|---|
| `operators/tilelang_kernels.py` | 唯一 TileLang kernel 实现文件；K1/K2/K3、公式注释、校准与输入检查 |
| `operators/k1_tilelang/kernel.py`、`operators/geometry_tilelang/kernels.py` | 保留原 import 路径的兼容入口 |
| `operators/*tilelang/torch_op.py` | custom op 与精确 padded-stride FakeTensor 注册 |
| `mapanything/utils/geometry_tensor.py` | Tensor 数学参考及几何后处理 |
| `mapanything/utils/npu_graph.py` | 固定桶模型、后处理与图 runner |
| `mapanything/utils/npu_graph_ops.py` | 保留 eager 插值语义的不透明算子 |
| `mapanything/utils/device.py` | NPU 设备、同步、autocast 和缓存等基础支持 |
| `operators/k1/`、`mapanything/utils/k1_npu.py` | 已实现的 TQue/手动同步 Ascend C K1 对比基线 |
| `scripts/npu_validate_*.py` | 单算子、特殊值、后处理、主模型及图重放验证 |
| `scripts/npu_profile_*.py`、`scripts/summarize_*.py` | 采集及分析性能数据 |

原生 K1 的 TQue 路线使用队列管理流水依赖；manual 路线使用显式事件与单缓冲。它们不是 TileLang kernel 的外部调用目标，保留是为了可复现的性能比较。旧的不可编译同步伪代码已从发布文件中移除。

## K1：点图到法线

输入 `P: float32[B,H,W,3]`、`M: bool[B,H,W]`，输出 `N: float32[B,H,W,3]`、`V: bool[B,H,W]`。

定义 `U(x)=x/(sqrt((x0*x0+x1*x1)+x2*x2)+1e-12)`。四邻点 q 按上、左、下、右排列，下标模 4：

```text
d_i = P(q_i) - P(p)
t_i = M(p) & M(q_i) & M(q_(i+1))
s_i = t_i * U(cross(d_i, d_(i+1)))
S = ((s_0 + s_1) + s_2) + s_3
V = OR_i(t_i)
N = where(V, U(S), 0)
```

越界点坐标和 mask 补 0。保留先分别归一化再相加的顺序、NaN×0 和最终 where 的语义。禁用 FMA 收缩。

## K2：法线边缘候选

输入 N/V 及默认 5° 阈值，输出尚未膨胀的 `E: bool[B,H,W]`。邻域采用 replicate 边界，法线再次用 U 归一化。

```text
d_delta(p) = dot(U(N(p)), U(N(p+delta)))
a_delta(p) = where(V(p+transpose(delta)), acos(d_delta(p)), 0)
A(p) = max_delta(a_delta(p))
E(p) = A(p) > radians(degrees)
```

保留参考实现的 mask 邻域轴转置。被选中的 dot 为 NaN 或超出 [-1,1] 时，角度 max 为 NaN，E 为 false；不 clamp dot。

实际 kernel 不生成 acos 中间图：首次预热校准 NPU acos 的 FP32 接受边界 cutoff，使用 `dot<=cutoff` 并保留上述异常处理。校准检查阈值附近 257 个浮点值和 65,537 个全域采样点；这是工程验证，不是所有 FP32 值的穷举证明。校准失败抛错，阈值/运行环境改变需要重新验证。

## K3：深度边缘、膨胀和掩码应用

输入 E、径向深度 D、基础 mask M、世界点 P、相机点 C。对 replicate 的 3×3 邻域 Q，忽略无效 mask 和 NaN 深度：

```text
H = max(D(q)), q in Q and M(q) and not isnan(D(q))
L = max(-D(q)), same valid set
# 空集合 max 使用 -inf
Ed = ((H + L) / D(p)) > depth_tol   # 默认 depth_tol=0.03
En = OR_{q in Q}(E(q))
F = M & ~(En & Ed)
PO = P * F[..., None]
CO = C * F[..., None]
DO = D * F
ZO = CO[..., 2]
```

不对分母取 abs、clamp 或加 epsilon；保留 NaN/Inf 与乘法语义。输出顺序为 F/PO/CO/DO/ZO。

## Tiling、存储与同步

- 同一 NPU 上 FP32 几何张量、bool 掩码，inference-only。B/H>0，1≤W≤576。
- 默认 20 blocks，当前后端映射使用 40 AIV；每个 AIV 一次处理一行，循环覆盖 B×H。
- 计算宽度补齐到 64；三行邻域放入 UB，左右邻居通过字节偏移 gather 获取。没有显式双缓冲。
- FP32 输出行补齐到 8 个元素，bool 输出行补齐到 32 字节，以避免跨核部分块写入冲突。
- 点图法线输出 stride 为 `align_up(3W,8)`，标量输出为 `align_up(W,8)`，mask 为 `align_up(W,32)`；逻辑 shape 不包含 padding。
- P/C/D/M 要求连续；K2/K3 接受上一个 kernel 产生的行 padding。不要擅自把输出当成 contiguous。
- 核内开启自动同步与内存规划；核间依靠同一 NPU stream 的执行顺序。图捕获前完成 JIT 和阈值校准。
- `floor(j/3)` 的浮点索引实现包含限定 W 范围内的字面量舍入修正；扩宽必须重新分析，不能只解除输入检查。

## 模型与图适配

固定 panorama 索引与配置分支避免图内数据相关 Python 判断；常量尽量在设备构造。位置编码 bicubic 和 DPT/CroCo bilinear 使用 opaque custom op 保留原 eager 插值调用，避免当前 AOT 分解的数值偏差。

内参恢复保留采样/正规方程，以 NPU Tensor 2×2 部分主元 LU 替换 CPU fallback 的 solve。退化系统可产生 NaN/Inf，不保留 CPU 异常机制；图外需要按应用要求处理。

部署、复验与环境准备见 [ACLGraph 说明](ascend_910b_aclgraph.md)。K1 单算子对比见 [K1 实测](ascend_910b_k1_tilelang.md)，K2/K3 和整模型性能见 [融合结果](ascend_910b_tilelang_edges.md)。

## 当前验证状态

2026-11-11 合并到统一源码后：函数 AST 对比未改变计算逻辑；23 项边界用例、B=1/4 × H=252/392/518 × W=518 六组 shape、四类真实缓存均通过。四类后处理 ACLGraph 各三次变更输入重放与参考严格一致。

此前已完成四类完整模型固定桶的三次变更输入重放，全部输出与未融合 NPU 路径逐元素一致。每类取 benchmark 第一个四视图 tuple，不代表全量数据集或所有输入的证明。零容差比较及 equal_nan 不等于检查 NaN payload 的 bitwise 比较。

原始权重、数据、编译环境和 profiler trace 保留在本地/测试环境的 `checkpoints/`、`work/`，不纳入 Git。仓库提供源码、依赖清单、复验脚本与汇总结果；在新机器上必须先准备运行依赖及数据。

## 文档导航

- 本文：当前完整设计与接口契约。
- [ACLGraph 部署](ascend_910b_aclgraph.md)：环境准备、调用和验收范围。
- [K1 性能对比](ascend_910b_k1_tilelang.md)：当前 K1 设计细节及历史同进程 A/B。
- [K2/K3 与整图性能](ascend_910b_tilelang_edges.md)：融合设计、测量与流水分析。
