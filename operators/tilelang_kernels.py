"""MEOW 的全部 TileLang Ascend 后处理 kernel：K1 -> K2 -> K3。

逻辑布局：点/法线 [B,H,W,3]，标量/掩码 [B,H,W]；FP32 运算，bool 掩码。
物理布局：每个 AIV 处理一行，输出行按 32 字节对齐，避免核间部分块写冲突。
使用自动同步和 UB 内存规划；禁用 FMA 收缩以保持 Tensor 参考的运算顺序。
下述公式中的 U(x) = x / (sqrt((x0*x0 + x1*x1) + x2*x2) + 1e-12)。
旧模块仅作兼容导入；torch_op 模块保留 dispatcher/FakeTensor 注册。
"""
from functools import lru_cache
import math
import numpy as np
import torch
import tilelang
import tilelang.language as T

# K1：输入 P 点图、M 有效掩码；输出 N 法线、V 法线有效掩码。
# 令 q_i 依次为上、左、下、右邻点，d_i=P(q_i)-P(p)，下标模 4。
# t_i=M(p) & M(q_i) & M(q_(i+1))；越界邻点的坐标和掩码均补 0。
# s_i=t_i * U(d_i × d_(i+1))；S=((s_0+s_1)+s_2)+s_3。
# V=OR_i(t_i)，N=where(V, U(S), 0)。乘法保留 NaN*0 的 IEEE 语义。
@tilelang.jit(out_idx=[2, 3], target="ascendc", pass_configs={
    "tl.ascend_auto_sync": True,
    "tl.ascend_auto_sync_vs": True,
    "tl.ascend_memory_planning": True,
}, compile_flags=["-ffp-contract=off"])
def _build(batch, height, width, blocks=40):
    rows = batch * height
    wp = ((width + 63) // 64) * 64
    npad = ((width * 3 + 7) // 8) * 8
    vpad = ((width + 31) // 32) * 32

    @T.prim_func
    def main(P: T.Tensor((rows, width * 3), "float32"),
             M: T.Tensor((rows, width), "uint8"),
             N: T.Tensor((rows, npad), "float32"),
             V: T.Tensor((rows, vpad), "uint8")):
        with T.Kernel(blocks, is_npu=True) as (cid, vid):
            inp = T.alloc_ub((3, wp * 3), "float32")
            mb = T.alloc_ub((3, wp), "uint8")
            mh = T.alloc_ub((3, wp), "float16")
            mf = T.alloc_ub((3, wp), "float32")
            pos = T.alloc_ub((5, 3, wp), "float32")
            valid = T.alloc_ub((5, wp), "float32")
            ix = T.alloc_ub((3, wp), "uint32")
            im = T.alloc_ub((3, wp), "uint32")
            io = T.alloc_ub((3 * wp,), "uint32")
            base = T.alloc_ub((wp,), "int32")
            ixi = T.alloc_ub((3, wp), "int32")
            imi = T.alloc_ub((3, wp), "int32")
            ioi = T.alloc_ub((3 * wp,), "int32")
            iq = T.alloc_ub((3 * wp,), "int32")
            ir = T.alloc_ub((3 * wp,), "int32")
            cross = T.alloc_ub((3, wp), "float32")
            accum = T.alloc_ub((3, wp), "float32")
            tmp = T.alloc_ub((3, wp), "float32")
            length = T.alloc_ub((wp,), "float32")
            tri = T.alloc_ub((wp,), "float32")
            union = T.alloc_ub((wp,), "float32")
            out = T.alloc_ub((3 * wp,), "float32")
            vb = T.alloc_ub((wp,), "uint8")
            pred = T.alloc_ub((wp,), "uint8")
            with T.Scope("V"):
                # Vector IDs are independent AIVs. Use both halves of each pair.
                # Generate Gather byte offsets with vector arithmetic, once per core.
                T.tile.createvecindex(base, 0)
                T.tile.mul(imi[0, :], base, 4)
                T.tile.add(imi[1, :], base, -1)
                T.tile.max(imi[1, :], imi[1, :], 0)
                T.tile.mul(imi[1, :], imi[1, :], 4)
                T.tile.add(imi[2, :], base, 1)
                T.tile.min(imi[2, :], imi[2, :], wp - 1)
                T.tile.mul(imi[2, :], imi[2, :], 4)
                T.tile.mul(ixi, imi, 3)
                T.reinterpretcast(ix, ixi, "uint32_t")
                T.reinterpretcast(im, imi, "uint32_t")
                T.tile.createvecindex(ioi, 0)
                T.tile.cast(out, ioi, "CAST_NONE", 3 * wp)
                # C++ literal printing rounds 1/3 downward. Bias only this integer
                # index calculation; for W<=576, 1e-3 is below the 1/3 gap
                # and above the worst reciprocal rounding error.
                T.tile.mul(out, out, 1.0 / 3.0)
                T.tile.add(out, out, 1e-3)
                T.tile.cast(iq, out, "CAST_FLOOR", 3 * wp)
                T.tile.mul(ir, iq, 3)
                T.tile.sub(ir, ioi, ir)
                T.tile.mul(ir, ir, wp)
                T.tile.add(ioi, ir, iq)
                T.tile.mul(ioi, ioi, 4)
                T.reinterpretcast(io, ioi, "uint32_t")
                for it in T.serial(T.ceildiv(rows, blocks * 2)):
                    row = it * blocks * 2 + cid * 2 + vid
                    if row < rows:
                        T.tile.fill(inp, 0.0)
                        T.tile.fill(mh, 0.0)
                        T.copy(mh, mb)
                        T.copy(P[row, 0:width * 3], inp[0, 0:width * 3])
                        T.copy(M[row, 0:width], mb[0, 0:width])
                        if row % height > 0:
                            T.copy(P[row - 1, 0:width * 3], inp[1, 0:width * 3])
                            T.copy(M[row - 1, 0:width], mb[1, 0:width])
                        if row % height < height - 1:
                            T.copy(P[row + 1, 0:width * 3], inp[2, 0:width * 3])
                            T.copy(M[row + 1, 0:width], mb[2, 0:width])
                        T.tile.cast(mh, mb, "CAST_NONE", 3 * wp)
                        T.copy(mh, mf)
                        # center, up, left, down, right; offsets are in bytes.
                        for c in T.unroll(3):
                            T.tile.gather(pos[0, c, :], inp[0, :], ix[0, :], c * 4)
                            T.tile.gather(pos[1, c, :], inp[1, :], ix[0, :], c * 4)
                            T.tile.gather(pos[2, c, :], inp[0, :], ix[1, :], c * 4)
                            T.tile.gather(pos[3, c, :], inp[2, :], ix[0, :], c * 4)
                            T.tile.gather(pos[4, c, :], inp[0, :], ix[2, :], c * 4)
                            pos[2, c, 0] = 0.0
                            pos[4, c, width - 1] = 0.0
                        T.copy(mf[0, :], valid[0, :])
                        T.copy(mf[1, :], valid[1, :])
                        T.tile.gather(valid[2, :], mf[0, :], im[1, :], 0)
                        T.copy(mf[2, :], valid[3, :])
                        T.tile.gather(valid[4, :], mf[0, :], im[2, :], 0)
                        valid[2, 0] = 0.0
                        valid[4, width - 1] = 0.0
                        # 邻点坐标转差向量：d_i = P(q_i) - P(p)。
                        for d in T.unroll(1, 5):
                            for c in T.unroll(3):
                                T.tile.sub(pos[d, c, :], pos[d, c, :], pos[0, c, :])
                        T.tile.fill(union, 0.0)
                        # 叉积第 c 分量：a_(c+1)*b_(c+2) - a_(c+2)*b_(c+1)。
                        # 每个三角形先归一化，再乘 t_i，最后按固定顺序累加。
                        for d in T.unroll(4):
                            for c in T.unroll(3):
                                T.tile.mul(tmp[0, :], pos[d + 1, (c + 1) % 3, :], pos[(d + 1) % 4 + 1, (c + 2) % 3, :])
                                T.tile.mul(tmp[1, :], pos[d + 1, (c + 2) % 3, :], pos[(d + 1) % 4 + 1, (c + 1) % 3, :])
                                T.tile.sub(cross[c, :], tmp[0, :], tmp[1, :])
                            for c in T.unroll(3):
                                T.tile.mul(tmp[c, :], cross[c, :], cross[c, :])
                            T.tile.add(length, tmp[0, :], tmp[1, :])
                            T.tile.add(length, length, tmp[2, :])
                            T.tile.sqrt(length, length)
                            T.tile.add(length, length, 1e-12)
                            T.tile.mul(tri, valid[0, :], valid[d + 1, :])
                            T.tile.mul(tri, tri, valid[(d + 1) % 4 + 1, :])
                            T.tile.max(union, union, tri)
                            for c in T.unroll(3):
                                T.tile.div(cross[c, :], cross[c, :], length)
                                T.tile.mul(cross[c, :], cross[c, :], tri)
                                if d == 0:
                                    T.copy(cross[c, :], accum[c, :])
                                else:
                                    T.tile.add(accum[c, :], accum[c, :], cross[c, :])
                        for c in T.unroll(3):
                            T.tile.mul(tmp[c, :], accum[c, :], accum[c, :])
                        T.tile.add(length, tmp[0, :], tmp[1, :])
                        T.tile.add(length, length, tmp[2, :])
                        T.tile.sqrt(length, length)
                        T.tile.add(length, length, 1e-12)
                        T.tile.compare(pred, union, 0.0, "GT")
                        for c in T.unroll(3):
                            T.tile.div(accum[c, :], accum[c, :], length)
                            T.tile.select(accum[c, :], pred, accum[c, :], 0.0, "VSEL_TENSOR_SCALAR_MODE")
                        T.copy(union, mh[0, :])
                        T.copy(mh[0, :], vb)
                        T.tile.gather(out, accum, io, 0)
                        T.copy(out[0:width * 3], N[row, 0:width * 3])
                        T.copy(vb[0:width], V[row, 0:width])
    return main.with_attr("global_symbol", f"meow_k1_tilelang_b{blocks}")


@lru_cache(maxsize=64)
def get_kernel(batch, height, width, blocks=20):
    return _build(batch, height, width, blocks)


def pointmap_to_normals(points, mask, *, blocks=20):
    if points.device.type != "npu" or mask.device != points.device:
        raise ValueError("points and mask must be on the same NPU")
    if points.dtype != torch.float32 or mask.dtype != torch.bool:
        raise TypeError("Expected FP32 points and bool mask")
    if points.ndim != 4 or points.shape[-1] != 3 or points.shape[:-1] != mask.shape:
        raise ValueError("Expected [B,H,W,3] and [B,H,W]")
    if not points.is_contiguous() or not mask.is_contiguous():
        raise ValueError("Contiguous inputs required")
    if torch.is_grad_enabled() and points.requires_grad:
        raise RuntimeError("K1 is inference-only")
    b, h, w, _ = points.shape
    if b < 1 or h < 1 or w < 1 or w > 576 or not 1 <= blocks <= 40:
        raise ValueError("Expected B,H>0, 1<=W<=576, 1<=blocks<=40")
    with torch.npu.device(points.device):
        fn = get_kernel(b, h, w, blocks)
        n, v = fn(points.view(b * h, w * 3), mask.view(torch.uint8).view(b * h, w))
    return n[:, :w * 3].view(b, h, w, 3), v[:, :w].view(torch.bool).view(b, h, w)


PASS = {'tl.ascend_auto_sync': True, 'tl.ascend_auto_sync_vs': True,
        'tl.ascend_memory_planning': True}


@lru_cache(maxsize=32)
def threshold_tensor(device_index, degrees):
    """Calibrate outside capture; preserve the installed NPU acos comparison.

    Only the FP32 rounding interval near cos(tol) can affect classification.
    Check a 257-ULP neighborhood for a single monotone transition and retain
    the largest accepted dot product, rather than rounding cos(tol) blindly.
    """
    if not 0 < degrees < 180:
        raise ValueError('Expected 0 < normal threshold < 180 degrees')
    # 理想实数关系 acos(d)>theta 等价于 d<cos(theta)。实际 FP32 NPU
    # 使用 cutoff=max{d: acos_NPU(d)>theta}，运行时判断 d<=cutoff。
    # 预热时校准并验证单调边界，不能简单舍入 cos(theta) 后代替。
    center = np.float32(math.cos(math.radians(degrees)))
    lo, hi = center, center
    left, right = [], []
    for _ in range(128):
        lo = np.nextafter(lo, np.float32(-np.inf))
        hi = np.nextafter(hi, np.float32(np.inf))
        left.append(lo);right.append(hi)
    values = np.asarray(left[::-1] + [center] + right, dtype=np.float32)
    device = torch.device('npu', device_index)
    dots = torch.from_numpy(values).to(device)
    accepted = (torch.acos(dots) > math.radians(degrees)).cpu().numpy()
    if not accepted[0] or accepted[-1] or np.any(np.diff(accepted.astype(np.int32)) > 0):
        raise RuntimeError('NPU acos threshold calibration failed; use Tensor reference')
    cutoff = values[np.flatnonzero(accepted)[-1]]
    probe = torch.linspace(-1., 1., 65537, device=device, dtype=torch.float32)
    if not torch.equal(torch.acos(probe) > math.radians(degrees), probe <= float(cutoff)):
        raise RuntimeError('NPU acos classification is inconsistent with a single cutoff')
    return torch.full((8,), float(cutoff), dtype=torch.float32, device=device)


# K2：输入 K1 的 N/V 和校准阈值 Cut；输出未膨胀的 bool 候选 E。
# 所有邻域使用 replicate 边界。令 delta=(dy,dx)∈{-1,0,1}²：
# d_delta(p)=U(N(p))·U(N(p+delta))。
# a_delta=where(V(p+(dx,dy)), acos(d_delta), 0)；注意参考实现的掩码轴互换。
# A(p)=max_delta(a_delta)，E(p)=[A(p)>theta]，theta=degrees*pi/180。
# max 传播 NaN：任一被选中的 dot 为 NaN 或 |dot|>1，E 均为 false。
# 其余情况下 E=OR_delta[d_delta<=Cut]；无效掩码的 dot 替换为 1。
# 无需输出 acos/角度图：后续 nanmax_3x3(A)>theta 等价于 OR_3x3(E)。
@tilelang.jit(out_idx=[3], target='ascendc', pass_configs=PASS,
              compile_flags=['-ffp-contract=off'])
def build_k2(batch, height, width, ns, vs, blocks=20):
    rows = batch * height
    wp = T.ceildiv(width, 64) * 64
    ep = T.ceildiv(width, 32) * 32

    @T.prim_func
    def main(N: T.Tensor((rows, ns), 'float32'),
             V: T.Tensor((rows, vs), 'uint8'),
             Cut: T.Tensor((8,), 'float32'),
             E: T.Tensor((rows, ep), 'uint8')):
        with T.Kernel(blocks, is_npu=True) as (cid, vid):
            raw = T.alloc_ub((3, 3 * wp), 'float32')
            n = T.alloc_ub((3, 3, wp), 'float32')
            vb = T.alloc_ub((3, wp), 'uint8')
            vh = T.alloc_ub((3, wp), 'float16')
            vf = T.alloc_ub((3, wp), 'float32')
            base = T.alloc_ub((wp,), 'int32')
            ii = T.alloc_ub((3, wp), 'int32')
            ni = T.alloc_ub((3, wp), 'int32')
            ix = T.alloc_ub((3, wp), 'uint32')
            nx = T.alloc_ub((3, wp), 'uint32')
            near = T.alloc_ub((3, wp), 'float32')
            tmp = T.alloc_ub((3, wp), 'float32')
            dot = T.alloc_ub((wp,), 'float32')
            mask = T.alloc_ub((wp,), 'float32')
            bad = T.alloc_ub((wp,), 'float32')
            edge = T.alloc_ub((wp,), 'float32')
            one = T.alloc_ub((wp,), 'float32')
            pred = T.alloc_ub((wp,), 'uint8')
            cutoff = T.alloc_ub((8,), 'float32')
            cutvec = T.alloc_ub((wp,), 'float32')
            with T.Scope('V'):
                T.copy(Cut, cutoff)
                T.tile.fill(cutvec, cutoff[0])
                T.tile.createvecindex(base, 0)
                for s in T.unroll(3):
                    T.tile.add(ii[s, :], base, s - 1)
                    T.tile.max(ii[s, :], ii[s, :], 0)
                    T.tile.min(ii[s, :], ii[s, :], width - 1)
                    T.tile.mul(ii[s, :], ii[s, :], 4)
                T.tile.mul(ni, ii, 3)
                T.reinterpretcast(ix, ii, 'uint32_t')
                T.reinterpretcast(nx, ni, 'uint32_t')
                T.tile.fill(one, 1.0)
                for it in T.serial(T.ceildiv(rows, blocks * 2)):
                    row = it * blocks * 2 + cid * 2 + vid
                    if row < rows:
                        for dy in T.unroll(3):
                            src = row // height * height + T.min(T.max(row % height + dy - 1, 0), height - 1)
                            T.tile.fill(raw[dy, :], 0.0)
                            T.tile.fill(vh[dy, :], 0.0)
                            T.copy(vh[dy, :], vb[dy, :])
                            T.copy(N[src, 0:width * 3], raw[dy, 0:width * 3])
                            T.copy(V[src, 0:width], vb[dy, 0:width])
                            for c in T.unroll(3):
                                T.tile.gather(n[dy, c, :], raw[dy, :], nx[1, :], c * 4)
                                T.tile.mul(tmp[c, :], n[dy, c, :], n[dy, c, :])
                            T.tile.add(dot, tmp[0, :], tmp[1, :])
                            T.tile.add(dot, dot, tmp[2, :])
                            T.tile.sqrt(dot, dot)
                            T.tile.add(dot, dot, 1e-12)
                            for c in T.unroll(3):
                                T.tile.div(n[dy, c, :], n[dy, c, :], dot)
                        T.tile.cast(vh, vb, 'CAST_NONE', 3 * wp)
                        T.copy(vh, vf)
                        T.tile.fill(bad, 0.0)
                        T.tile.fill(edge, 0.0)
                        for dy in T.unroll(3):
                            for dx in T.unroll(3):
                                for c in T.unroll(3):
                                    T.tile.gather(near[c, :], n[dy, c, :], ix[dx, :], 0)
                                    T.tile.mul(tmp[c, :], n[1, c, :], near[c, :])
                                T.tile.add(dot, tmp[0, :], tmp[1, :])
                                T.tile.add(dot, dot, tmp[2, :])
                                # Preserve reference mask's transposed window axes.
                                T.tile.gather(mask, vf[dx, :], ix[dy, :], 0)
                                T.tile.compare(pred, mask, 0.0, 'GT')
                                T.tile.select(dot, pred, dot, 1.0, 'VSEL_TENSOR_SCALAR_MODE')
                                # 不 clamp dot：超出 [-1,1] 或 NaN 对应 acos 的 NaN。
                                T.tile.compare(pred, dot, dot, 'NE')
                                T.tile.select(mask, pred, one, 0.0, 'VSEL_TENSOR_SCALAR_MODE')
                                T.tile.max(bad, bad, mask)
                                T.tile.abs(tmp[0, :], dot)
                                T.tile.compare(pred, tmp[0, :], 1.0, 'GT')
                                T.tile.select(mask, pred, one, 0.0, 'VSEL_TENSOR_SCALAR_MODE')
                                T.tile.max(bad, bad, mask)
                                T.tile.compare(pred, dot, cutvec, 'LE')
                                T.tile.select(mask, pred, one, 0.0, 'VSEL_TENSOR_SCALAR_MODE')
                                T.tile.max(edge, edge, mask)
                        T.tile.compare(pred, bad, 0.0, 'EQ')
                        T.tile.select(edge, pred, edge, 0.0, 'VSEL_TENSOR_SCALAR_MODE')
                        T.copy(edge, vh[0, :])
                        T.copy(vh[0, :], vb[0, :])
                        T.copy(vb[0, 0:width], E[row, 0:width])
    return main.with_attr('global_symbol', 'meow_k2_tilelang')


# K3：输入 E 候选、D 射线深度、M 掩码、P 世界点、C 相机点。
# 输出 F 最终掩码，PO/CO/DO/ZO 分别为掩码后的世界点/相机点/深度/相机 Z。
# 在 replicate 的 3x3 邻域 Q(p) 上：
# H=max_{q∈Q,M(q),非NaN} D(q)，L=max_{q∈Q,M(q),非NaN}(-D(q))。
# 空集合的 max 取 -inf；H+L 在有效有限深度上等于 max(D)-min(D)。
# depth_edge=[(H+L)/D(p)>depth_tol]，分母是原始中心深度，不做 abs/clamp。
# normal_edge=OR_{q∈Q(p)} E(q)，F=M & ~(normal_edge & depth_edge)。
# PO=P*F，CO=C*F，DO=D*F，ZO=CO[...,2]；XYZ 共用 F。
# 使用乘法而非 where，保留 Inf*0/NaN*0；NaN 深度比值的比较为 false。
@tilelang.jit(out_idx=[5, 6, 7, 8, 9], target='ascendc', pass_configs=PASS,
              compile_flags=['-ffp-contract=off'])
def build_k3(batch, height, width, es, depth_tol=0.03, blocks=20):
    rows = batch * height
    wp = T.ceildiv(width, 64) * 64
    npad = T.ceildiv(width * 3, 8) * 8
    spad = T.ceildiv(width, 8) * 8
    mpad = T.ceildiv(width, 32) * 32

    @T.prim_func
    def main(E: T.Tensor((rows, es), 'uint8'),
             D: T.Tensor((rows, width), 'float32'),
             M: T.Tensor((rows, width), 'uint8'),
             P: T.Tensor((rows, width * 3), 'float32'),
             C: T.Tensor((rows, width * 3), 'float32'),
             F: T.Tensor((rows, mpad), 'uint8'),
             PO: T.Tensor((rows, npad), 'float32'),
             CO: T.Tensor((rows, npad), 'float32'),
             DO: T.Tensor((rows, spad), 'float32'),
             ZO: T.Tensor((rows, spad), 'float32')):
        with T.Kernel(blocks, is_npu=True) as (cid, vid):
            d = T.alloc_ub((3, wp), 'float32')
            center = T.alloc_ub((wp,), 'float32')
            eb = T.alloc_ub((3, wp), 'uint8')
            mb = T.alloc_ub((3, wp), 'uint8')
            eh = T.alloc_ub((3, wp), 'float16')
            mh = T.alloc_ub((3, wp), 'float16')
            ef = T.alloc_ub((3, wp), 'float32')
            mf = T.alloc_ub((3, wp), 'float32')
            dn = T.alloc_ub((3, wp), 'float32')
            base = T.alloc_ub((wp,), 'int32')
            ii = T.alloc_ub((3, wp), 'int32')
            ix = T.alloc_ub((3, wp), 'uint32')
            iz = T.alloc_ub((wp,), 'uint32')
            izi = T.alloc_ub((wp,), 'int32')
            repeat_i = T.alloc_ub((3 * wp,), 'int32')
            repeat_f = T.alloc_ub((3 * wp,), 'float32')
            repeat_x = T.alloc_ub((3 * wp,), 'uint32')
            tmp = T.alloc_ub((wp,), 'float32')
            high = T.alloc_ub((wp,), 'float32')
            low = T.alloc_ub((wp,), 'float32')
            edges = T.alloc_ub((wp,), 'float32')
            one = T.alloc_ub((wp,), 'float32')
            final = T.alloc_ub((wp,), 'float32')
            pred = T.alloc_ub((wp,), 'uint8')
            geom = T.alloc_ub((3 * wp,), 'float32')
            f3 = T.alloc_ub((3 * wp,), 'float32')
            with T.Scope('V'):
                T.tile.createvecindex(base, 0)
                for s in T.unroll(3):
                    T.tile.add(ii[s, :], base, s - 1)
                    T.tile.max(ii[s, :], ii[s, :], 0)
                    T.tile.min(ii[s, :], ii[s, :], width - 1)
                    T.tile.mul(ii[s, :], ii[s, :], 4)
                T.reinterpretcast(ix, ii, 'uint32_t')
                T.tile.mul(izi, ii[1, :], 3)
                T.reinterpretcast(iz, izi, 'uint32_t')
                # XYZ 广播：f3[j]=F[floor(j/3)]。1e-3 仅修正索引计算中
                # C++ 1/3 字面量舍入误差，在 W<=576 时不改变 floor 的目标值。
                T.tile.createvecindex(repeat_i, 0)
                T.tile.cast(repeat_f, repeat_i, 'CAST_NONE', 3 * wp)
                T.tile.mul(repeat_f, repeat_f, 1.0 / 3.0)
                T.tile.add(repeat_f, repeat_f, 1e-3)
                T.tile.cast(repeat_i, repeat_f, 'CAST_FLOOR', 3 * wp)
                T.tile.mul(repeat_i, repeat_i, 4)
                T.reinterpretcast(repeat_x, repeat_i, 'uint32_t')
                T.tile.fill(one, 1.0)
                for it in T.serial(T.ceildiv(rows, blocks * 2)):
                    row = it * blocks * 2 + cid * 2 + vid
                    if row < rows:
                        for dy in T.unroll(3):
                            src = row // height * height + T.min(T.max(row % height + dy - 1, 0), height - 1)
                            T.tile.fill(d[dy, :], 0.0)
                            T.tile.fill(eh[dy, :], 0.0)
                            T.copy(eh[dy, :], eb[dy, :])
                            T.copy(eh[dy, :], mb[dy, :])
                            T.copy(E[src, 0:width], eb[dy, 0:width])
                            T.copy(M[src, 0:width], mb[dy, 0:width])
                            T.copy(D[src, 0:width], d[dy, 0:width])
                        T.copy(d[1, :], center)
                        T.tile.cast(eh, eb, 'CAST_NONE', 3 * wp)
                        T.tile.cast(mh, mb, 'CAST_NONE', 3 * wp)
                        T.copy(eh, ef)
                        T.copy(mh, mf)
                        for dy in T.unroll(3):
                            T.tile.mul(dn[dy, :], d[dy, :], -1.0)
                            T.tile.compare(pred, mf[dy, :], 0.0, 'GT')
                            T.tile.select(d[dy, :], pred, d[dy, :], -T.infinity('float32'), 'VSEL_TENSOR_SCALAR_MODE')
                            T.tile.select(dn[dy, :], pred, dn[dy, :], -T.infinity('float32'), 'VSEL_TENSOR_SCALAR_MODE')
                            T.tile.compare(pred, d[dy, :], d[dy, :], 'EQ')
                            T.tile.select(d[dy, :], pred, d[dy, :], -T.infinity('float32'), 'VSEL_TENSOR_SCALAR_MODE')
                            T.tile.select(dn[dy, :], pred, dn[dy, :], -T.infinity('float32'), 'VSEL_TENSOR_SCALAR_MODE')
                        T.tile.fill(high, -T.infinity('float32'))
                        T.tile.fill(low, -T.infinity('float32'))
                        T.tile.fill(edges, 0.0)
                        for dy in T.unroll(3):
                            for dx in T.unroll(3):
                                T.tile.gather(tmp, d[dy, :], ix[dx, :], 0)
                                T.tile.max(high, high, tmp)
                                T.tile.gather(tmp, dn[dy, :], ix[dx, :], 0)
                                T.tile.max(low, low, tmp)
                                T.tile.gather(tmp, ef[dy, :], ix[dx, :], 0)
                                T.tile.max(edges, edges, tmp)
                        # 深度相对极差 r=(max(D)+max(-D))/D_center。
                        T.tile.add(tmp, high, low)
                        T.tile.div(tmp, tmp, center)
                        T.tile.compare(pred, tmp, depth_tol, 'GT')
                        T.tile.select(tmp, pred, one, 0.0, 'VSEL_TENSOR_SCALAR_MODE')
                        # bool 用 0/1 表示：F=M*(1-depth_edge*normal_edge)。
                        T.tile.mul(tmp, tmp, edges)
                        T.tile.sub(final, one, tmp)
                        T.tile.mul(final, final, mf[1, :])
                        T.copy(final, mh[0, :])
                        T.copy(mh[0, :], mb[0, :])
                        T.copy(mb[0, 0:width], F[row, 0:width])
                        T.tile.gather(f3, final, repeat_x, 0)
                        T.copy(P[row, 0:width * 3], geom[0:width * 3])
                        T.tile.mul(geom, geom, f3)
                        T.copy(geom[0:width * 3], PO[row, 0:width * 3])
                        T.copy(C[row, 0:width * 3], geom[0:width * 3])
                        T.tile.mul(geom, geom, f3)
                        T.copy(geom[0:width * 3], CO[row, 0:width * 3])
                        T.tile.gather(tmp, geom, iz, 8)
                        T.copy(tmp[0:width], ZO[row, 0:width])
                        T.tile.mul(tmp, center, final)
                        T.copy(tmp[0:width], DO[row, 0:width])
    return main.with_attr('global_symbol', 'meow_k3_tilelang')


def _shape(tensor):
    if tensor.device.type != 'npu' or tensor.dtype != torch.float32:
        raise ValueError('Expected FP32 NPU tensor')
    b, h, w, c = tensor.shape
    if c != 3 or min(b, h, w) < 1 or w > 576:
        raise ValueError('Expected [B,H,W,3], B,H>0, 1<=W<=576')
    if tensor.stride(-1) != 1 or (w > 1 and tensor.stride(-2) != 3) or (b > 1 and h > 1 and tensor.stride(0) != h * tensor.stride(1)):
        raise ValueError('Expected contiguous pixels, optionally padded rows')
    return b, h, w


def normal_candidates(normals, valid, degrees=5.0):
    if torch.is_grad_enabled() and normals.requires_grad:
        raise RuntimeError('K2 is inference-only')
    b, h, w = _shape(normals)
    if valid.shape != (b,h,w) or valid.dtype != torch.bool or valid.device != normals.device:
        raise ValueError('Expected matching bool normal validity')
    if valid.stride(-1) != 1 or (b > 1 and h > 1 and valid.stride(0) != h * valid.stride(1)):
        raise ValueError('Expected contiguous validity pixels, optionally padded rows')
    ns = normals.stride(1) if h > 1 else normals.stride(0) if b > 1 else 3*w
    vs = valid.stride(1) if h > 1 else valid.stride(0) if b > 1 else w
    with torch.npu.device(normals.device):
        cut = threshold_tensor(normals.device.index, float(degrees))
        e = build_k2(b,h,w,ns,vs)(
            torch.as_strided(normals,(b*h,ns),(ns,1)),
            torch.as_strided(valid.view(torch.uint8),(b*h,vs),(vs,1)),cut)
    return e.view(torch.bool).as_strided((b,h,w),(h*e.stride(0),e.stride(0),1))


def mask_and_apply(candidates, depth, mask, points, camera_points, depth_tol=0.03):
    b,h,w = _shape(points)
    if torch.is_grad_enabled() and any(x.requires_grad for x in (points,camera_points,depth)):
        raise RuntimeError('K3 is inference-only')
    if not math.isfinite(depth_tol):
        raise ValueError('Expected a finite depth threshold')
    if not points.is_contiguous() or not camera_points.is_contiguous() or camera_points.shape != points.shape:
        raise ValueError('Expected contiguous matching point maps')
    for x,dtype in ((depth,torch.float32),(mask,torch.bool),(candidates,torch.bool)):
        if x.shape != (b,h,w) or x.dtype != dtype or x.device != points.device:
            raise ValueError('Expected matching depth/mask/edge BHW NPU tensors')
    if not depth.is_contiguous() or not mask.is_contiguous():
        raise ValueError('Expected contiguous depth and mask')
    if camera_points.dtype != torch.float32 or camera_points.device != points.device:
        raise ValueError('Expected matching FP32 camera points')
    if candidates.stride(-1) != 1 or (b > 1 and h > 1 and candidates.stride(0) != h*candidates.stride(1)):
        raise ValueError('Expected contiguous edge pixels, optionally padded rows')
    es = candidates.stride(1) if h > 1 else candidates.stride(0) if b > 1 else w
    with torch.npu.device(points.device):
        f,p,c,d,z = build_k3(b,h,w,es,float(depth_tol))(
            torch.as_strided(candidates.view(torch.uint8),(b*h,es),(es,1)),
            depth.view(b*h,w),mask.view(torch.uint8).view(b*h,w),
            points.view(b*h,3*w),camera_points.view(b*h,3*w))
    return (f.view(torch.bool).as_strided((b,h,w),(h*f.stride(0),f.stride(0),1)),
            p.as_strided((b,h,w,3),(h*p.stride(0),p.stride(0),3,1)),
            c.as_strided((b,h,w,3),(h*c.stride(0),c.stride(0),3,1)),
            d.as_strided((b,h,w),(h*d.stride(0),d.stride(0),1)),
            z.as_strided((b,h,w),(h*z.stride(0),z.stride(0),1)))
