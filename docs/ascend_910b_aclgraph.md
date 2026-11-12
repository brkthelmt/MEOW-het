# Ascend 910B 固定桶 ACLGraph 部署

更新：2026-11-12。当前设计见 [实现与公式](ascend_910b_design.md)，最新融合性能见 [K2/K3 与整图结果](ascend_910b_tilelang_edges.md)。

## 运行环境与准备

已验收环境：ARM64 Ascend 910B2、CANN 9.1.0、Python 3.10、torch 2.10.0、torch-npu 2.10.0.post4。`requirements-npu.txt` 为依赖入口，`requirements-npu.lock.txt` 为测试环境快照。不要在该环境直接用 CUDA 版 requirements 覆盖 torch-npu 依赖。

TileLang-Ascend 固定源码 commit 为 `4805777d691b0c586e647fc0f975fa4cc306c00e`，包含其 submodule；构建说明见 [K1 环境记录](ascend_910b_k1_tilelang.md)。现有脚本期望：

- `.venv/bin/python`：安装模型及 torch-npu 依赖的解释器；`run_npu.sh` 可由 `MEOW_PYTHON` 指定。
- `work/npu-k1/tilelang/source`：上述 TileLang-Ascend 源码及构建产物。
- `work/npu-k1/toolchain/bin/aarch64-conda-linux-gnu-gcc/g++`：已测试 GCC13 工具链。
- CANN 环境脚本默认 `/usr/local/Ascend/ascend-toolkit/set_env.sh`，可用 `MEOW_CANN_ENV` 指定。
- 官方权重 `checkpoints/MEOW-sB25.pth`；benchmark 位于 `work/laser-benchmark/meow_laser_benchmark/v1`。
- 后处理独立验证需要 `work/npu-postprocess/raw/*_0.pt`，由已有采集流程生成。

`build_npu_tilelang.sh` 编译已准备的源码，不自动下载 SDK、工具链或权重。原生对比基线由 `build_npu_k1.sh` 编译；其 CANN/toolchain 路径同样基于验收环境。新机器应先核对这些路径。工作目录、权重、数据和构建产物不随 Git 发布。

## 调用

```python
from mapanything.utils.npu_graph import NpuGraphRunner

# model 已加载权重；processed_views 已预处理并位于 NPU。
# 使用专属模型实例，元数据固定；明确启用 TileLang K2/K3。
runner = NpuGraphRunner(model, processed_views, fused_edges=True)
outputs = runner(*(v['img'] for v in processed_views))
```

构造阶段完成 eager/JIT/阈值校准预热，再执行 `torch.compile(..., backend='npugraph_ex', fullgraph=True, dynamic=False)`。此入口使用 ACLGraph，不需要 GE converter。可选 pattern fusion/inplace/cat 改写按源码中的 `GRAPH_OPTIONS` 关闭，以保持已验收语义。

`fused_edges=False` 为默认构造参数，保留 K1 TileLang + Tensor 边缘路径作为比较基线。部署上面这份最新融合实现时必须显式传入 `True`。默认 `model.infer` 不会自动切换为 graph runner。

同一桶固定视图数、分辨率、aspect ratio、全景路由及其他模板元数据；不同组合应创建独立 runner。输出可能引用图内复用缓冲区，跨下一次 replay 保留时应自行 clone。

## 接入与限制

- `NpuCoreBucket`：主模型，使用静态全景路由。
- `NpuPostprocessBucket`：RGB、内参、位姿及几何输出；默认合批视图，`fused_edges=True` 使用 K1/K2/K3。
- `NpuInferenceBucket`：主模型和后处理串联。
- `NpuGraphRunner`：准备模型及图、验证输入约束并重放。

图像读取/resize、分桶、导出、ICP/KNN 不在图内。RGB-only 固定桶不接受尚未验收的几何先验输入；可选 confidence、多视图一致性及动态 shape 不在当前验收范围。训练不在此实现目标内。

图中内参用 2×2 部分主元 LU；相对原求解四类真实输入的历史最大绝对误差为 7.63e-5。退化系统可能输出 Inf/NaN，不保持 CPU 异常行为。ERP 针孔内参拟合也不能当作真实全景标定。

插值 custom op 保留 NPU eager bicubic/bilinear 的参数和实现，避免当前 AOT 分解引起的精度偏差。精度参考是新 NPU eager/Tensor；相对原 NumPy mask 的阈值附近差异仍未解决。

## 验证命令

从仓库根目录运行，选择空闲设备：

```bash
ASCEND_RT_VISIBLE_DEVICES=2 bash scripts/run_npu_tilelang.sh scripts/npu_validate_tilelang_edges.py
ASCEND_RT_VISIBLE_DEVICES=2 bash scripts/run_npu_tilelang.sh scripts/npu_validate_graph_postprocess.py --fused-edges --out work/npu-edges/postprocess
ASCEND_RT_VISIBLE_DEVICES=2 bash scripts/run_npu_tilelang.sh scripts/npu_validate_graph_model.py --stage full --runner --fused-edges --compare-unfused --tracks blk_mixed blk_erp blk_pinhole blk_fisheye --out work/npu-edges/full
```

可选 `--trace` 独立采集 profiler；一键组织上述融合验证使用 `scripts/run_npu_tilelang_edges_validation.sh`。`scripts/run_npu_graph_validation.sh` 保留未融合路径的验证；主模型数值定位见 `npu_validate_graph_primitives.py` 和 `npu_diagnose_graph_model.py`。

## 已验收结果与性能解释

四类固定桶完整模型各三次变更输入重放，与未融合 NPU 路径全部输出逐元素一致，包括 mask；每类仅取 benchmark 第一个四视图 tuple。2026-11-11 统一 kernel 文件后，单算子及四类后处理图重放再次通过，未重新执行完整模型测试。

历史完整推理同进程同卡 ACLGraph A/B：mixed 149.918→143.261 ms，ERP 98.100→92.562 ms，pinhole 144.770→138.221 ms，fisheye 186.599→177.317 ms。该收益来自 K2/K3 融合，与单纯 Eager→ACLGraph 的提交收益不同；完整口径见 [测量记录](ascend_910b_tilelang_edges.md)。

trace 验证 K1/K2/K3 与主模型共同进入图重放；没有 `_local_scalar_dense`/`linalg_solve` 事件不等于完全零拷贝。模型仍占主要时延，应通过设备 profiling 和端到端测量决定优化优先级。
