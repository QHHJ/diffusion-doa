#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_dataset_flom_matrix.py
生成用于“扩散+GNN”训练的 FLOM 矩阵数据集（矩阵版：7x7 Re/Im 两通道）

样本对：
  cond   = 归一化后的观测 FLOM 矩阵 C_y^(p)  -> [2, M, M]
  target = 归一化后的干净 FLOM 矩阵 C_x^(p)  -> [2, M, M]

归一化使用观测侧的尺度 c（trace 或 diag 平均），以避免信息泄漏。

噪声：
  物理层加在快拍矩阵 Y 上：W_α ~ SαS(alpha, β=0)
  SNR 采用 p-阶矩定义：SNR_p = 10*log10( E|signal|^p / E|noise|^p )
  通过缩放噪声的幅度使该比值满足给定 dB

依赖：仅 numpy（不依赖 scipy）
"""
import os
import argparse
import json
import numpy as np

# -----------------------------
# 工具：α稳定分布（CMS，β=0）
# -----------------------------
def salpha_stable_real(alpha: float, size, rng: np.random.Generator):
    """对称 α-stable（SαS），scale=1, beta=0, delta=0 的实数采样（CMS方法）"""
    U = rng.uniform(-np.pi/2, np.pi/2, size)
    if abs(alpha - 1.0) < 1e-12:
        # alpha=1, beta=0 → Cauchy：tan(U)
        X = np.tan(U)
    else:
        W = rng.exponential(1.0, size)
        # Chambers-Mallows-Stuck
        num = np.sin(alpha * U)
        den = (np.cos(U)) ** (1.0 / alpha)
        frac = num / den
        expo = (np.cos(U - alpha * U) / W) ** ((1.0 - alpha) / alpha)
        X = frac * expo
    return X

def salpha_stable_complex(alpha: float, size, rng: np.random.Generator):
    """复数 SαS：实部与虚部独立同分布（近似“圆对称”）"""
    x = salpha_stable_real(alpha, size, rng)
    y = salpha_stable_real(alpha, size, rng)
    return x + 1j * y

# -----------------------------
# 阵列/信号生成
# -----------------------------
def steering_vec_ula(M: int, d_over_lambda: float, theta_deg: float):
    """ULA 导向向量 a(θ) ，索引从 0 到 M-1"""
    m = np.arange(M)
    theta = np.deg2rad(theta_deg)
    phase = 2.0 * np.pi * d_over_lambda * np.sin(theta) * m
    return np.exp(1j * phase)  # shape [M]

def simulate_clean_snapshots(M, T, thetas_deg, d_over_lambda, rng):
    """生成干净快拍 Y_clean = A S ，S ~ CN(0,1)"""
    K = len(thetas_deg)
    A = np.stack([steering_vec_ula(M, d_over_lambda, th) for th in thetas_deg], axis=1)  # [M,K]
    # 复高斯 CN(0,1)
    S = (rng.standard_normal((K, T)) + 1j * rng.standard_normal((K, T))) / np.sqrt(2.0)
    Y = A @ S  # [M,T]
    return Y, A, S

# -----------------------------
# FLOM 矩阵（对称加权版本，稳定、Hermitian）
# R^(p) = E[ D(x) x x^H D(x) ],  D(x)=diag(|x|^{p/2 - 1})
# -----------------------------
def flom_matrix(Y: np.ndarray, p: float, eps: float = 1e-12):
    """
    Y: [M,T] 复矩阵
    p: 0 < p < 2（实际需 p < alpha）
    返回：C^(p) ∈ C^{M×M}
    """
    M, T = Y.shape
    absY = np.abs(Y) + eps
    w = absY ** (p / 2.0 - 1.0)        # [M,T]
    Yw = Y * w                         # [M,T]
    C = (Yw @ Yw.conj().T) / float(T)  # [M,M]
    return C

# -----------------------------
# 归一化 & 投影
# -----------------------------
def normalize_matrix(C: np.ndarray, method: str = "trace", eps: float = 1e-12):
    """
    返回 C_norm, scale_c
    method: 'trace' or 'diag'
    """
    if method == "trace":
        c = float(np.real(np.trace(C)))
    elif method == "diag":
        c = float(np.real(np.diag(C)).mean())
    else:
        raise ValueError("norm method must be 'trace' or 'diag'")
    c = max(c, eps)
    return C / c, c

def hermitian_symmetrize(C: np.ndarray):
    return 0.5 * (C + C.conj().T)

def toeplitz_project(C: np.ndarray):
    """沿副对角线平均（可选的“扶正”步骤）"""
    M = C.shape[0]
    out = np.zeros_like(C)
    for k in range(-(M-1), M):
        diag = np.diag(C, k)
        mean_val = diag.mean()
        out += np.diag([mean_val]* (M - abs(k)), k)
    # 保障 Hermitian（数值对称）
    return hermitian_symmetrize(out)

def psd_project(C: np.ndarray, eps: float = 0.0):
    """将负特征值截断至 >= eps"""
    vals, vecs = np.linalg.eigh(hermitian_symmetrize(C))
    vals = np.maximum(vals, eps)
    return (vecs * vals) @ vecs.conj().T

# -----------------------------
# p-阶矩 SNR 标定（基于样本）
# -----------------------------
def scale_noise_to_psnr(Y_clean: np.ndarray, W: np.ndarray, snr_db: float, p_for_snr: float, eps: float = 1e-12):
    """
    将噪声 W 线性缩放，使 p-阶矩意义下的 SNR 达到 snr_db
    SNR_p = 10 log10( E|signal|^p / E|noise|^p )
    """
    sig_p = np.mean(np.abs(Y_clean) ** p_for_snr)
    noi_p0 = np.mean(np.abs(W) ** p_for_snr) + eps
    ratio = 10.0 ** (snr_db / 10.0)
    s = (sig_p / (ratio * noi_p0)) ** (1.0 / p_for_snr)
    return s * W

# -----------------------------
# 主流程：生成一个样本
# -----------------------------
def make_one_sample(args, rng: np.random.Generator):
    # 1) 抽 K 与 DOA
    if args.K_fixed > 0:
        K = args.K_fixed
    else:
        K = rng.integers(args.K_min, args.K_max + 1)
    thetas = rng.uniform(args.doa_min, args.doa_max, size=K).tolist()

    # 2) 干净快拍
    Y_clean, A, S = simulate_clean_snapshots(args.M, args.T, thetas, args.d_over_lambda, rng)

    # 3) α稳定噪声 + p-SNR 标定
    alpha = rng.uniform(args.alpha_min, args.alpha_max)
    W = salpha_stable_complex(alpha, size=Y_clean.size, rng=rng).reshape(Y_clean.shape)
    # p 用于 FLOM；p_snr 用于 SNR 标定（默认等于 p）
    p_use = args.p if not args.p_rand else rng.uniform(args.p_min, min(args.p_max, alpha - 1e-3))
    p_snr = args.p_snr if args.p_snr > 0 else p_use
    snr_db = rng.uniform(args.snr_min, args.snr_max) if args.snr_grid is None else float(rng.choice(args.snr_grid))
    W = scale_noise_to_psnr(Y_clean, W, snr_db, p_for_snr=p_snr)
    Y_obs = Y_clean + W

    # 4) FLOM 矩阵（干净/观测），可选 Toeplitz 投影
    Cx = flom_matrix(Y_clean, p=p_use, eps=1e-12)
    Cy = flom_matrix(Y_obs,   p=p_use, eps=1e-12)

    if args.toeplitz_project:
        Cx = toeplitz_project(Cx)
        Cy = toeplitz_project(Cy)

    if args.hermitian_fix:
        Cx = hermitian_symmetrize(Cx)
        Cy = hermitian_symmetrize(Cy)

    # 5) 归一化（观测侧尺度 c）
    Cy_norm, c_scale = normalize_matrix(Cy, method=args.norm, eps=1e-12)
    Cx_norm = Cx / c_scale

    # 可选：写入 PSD 投影（通常训练数据不做；推断/采样时再投影）
    if args.psd_project:
        Cx_norm = psd_project(Cx_norm)
        Cy_norm = psd_project(Cy_norm)

    # 6) 打包（Re/Im 两通道）
    def to_reim(C):
        return np.stack([C.real, C.imag], axis=0).astype(np.float32)  # [2,M,M]

    sample = dict(
        cond=to_reim(Cy_norm),
        target=to_reim(Cx_norm),
        meta=dict(
            M=int(args.M), T=int(args.T), K=int(K), thetas_deg=thetas,
            alpha=float(alpha), p=float(p_use), p_snr=float(p_snr),
            snr_db=float(snr_db), norm=args.norm, scale_c=float(c_scale)
        )
    )
    return sample

# -----------------------------
# 入口
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Generate FLOM-matrix dataset for diffusion+GNN")
    # 基本尺寸
    ap.add_argument("--M", type=int, default=7, help="阵元数")
    ap.add_argument("--T", type=int, default=1024, help="快拍数")
    ap.add_argument("--d-over-lambda", type=float, default=0.5, dest="d_over_lambda",
                    help="阵元间距/波长")
    # 源与方位
    ap.add_argument("--K-min", type=int, default=2)
    ap.add_argument("--K-max", type=int, default=2)
    ap.add_argument("--K-fixed", type=int, default=0, help=">0 则固定为此 K")
    ap.add_argument("--doa-min", type=float, default=-60.0)
    ap.add_argument("--doa-max", type=float, default=60.0)
    # α 稳定噪声与 p
    ap.add_argument("--alpha-min", type=float, default=1.2)
    ap.add_argument("--alpha-max", type=float, default=1.8)
    ap.add_argument("--p", type=float, default=1.0, help="FLOM 次数 p；若 --p-rand 则忽略")
    ap.add_argument("--p-rand", action="store_true", help="每样本在 [p_min,p_max]∩(0,alpha) 随机 p")
    ap.add_argument("--p-min", type=float, default=0.8)
    ap.add_argument("--p-max", type=float, default=1.2)
    # SNR（p-阶矩定义）
    ap.add_argument("--snr-min", type=float, default=-10.0)
    ap.add_argument("--snr-max", type=float, default=0.0)
    ap.add_argument("--snr-grid", type=str, default=None,
                    help="离散 SNR 列表，如 \"-10,-5,0,5,10\"；设置后覆盖 min/max")
    ap.add_argument("--p-snr", type=float, default=0.0,
                    help="用于 SNR 标定的 p 值；<=0 时与 FLOM 的 p 相同")
    # 归一化与投影
    ap.add_argument("--norm", type=str, default="trace", choices=["trace", "diag"],
                    help="归一化尺度 c 的计算方式（观测侧）")
    ap.add_argument("--hermitian-fix", action="store_true", default=True,
                    help="保存前做 Hermitian 对称化")
    ap.add_argument("--toeplitz-project", action="store_true", default=False,
                    help="保存前做 Toeplitz 投影（沿副对角平均）")
    ap.add_argument("--psd-project", action="store_true", default=False,
                    help="保存前做 PSD 投影（一般训练集不做）")
    # 数据量与输出
    ap.add_argument("--num-samples", type=int, default=2000)
    ap.add_argument("--chunk-size", type=int, default=10000, help="每个文件包含的样本数")
    ap.add_argument("--out-dir", type=str, default="dataset_flom_mat_snr_-10_to_0_eval",
                    help="输出目录，将写入 .npz 分片与 meta.json")
    ap.add_argument("--seed", type=int, default=2025)
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # 解析 SNR 网格
    if args.snr_grid is not None:
        grid = [float(x) for x in args.snr_grid.split(",") if x.strip() != ""]
        args.snr_grid = np.array(grid, dtype=np.float32)

    # 写 meta
    meta = {
        "M": args.M, "T": args.T, "d_over_lambda": args.d_over_lambda,
        "K_min": args.K_min, "K_max": args.K_max, "K_fixed": args.K_fixed,
        "doa_min": args.doa_min, "doa_max": args.doa_max,
        "alpha_min": args.alpha_min, "alpha_max": args.alpha_max,
        "p_fixed": args.p, "p_rand": args.p_rand, "p_min": args.p_min, "p_max": args.p_max,
        "snr_min": args.snr_min, "snr_max": args.snr_max, "snr_grid": None if args.snr_grid is None else args.snr_grid.tolist(),
        "p_snr": args.p_snr, "norm": args.norm,
        "hermitian_fix": args.hermitian_fix, "toeplitz_project": args.toeplitz_project, "psd_project": args.psd_project,
        "num_samples": args.num_samples, "chunk_size": args.chunk_size, "seed": args.seed
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    N = args.num_samples
    B = args.chunk_size
    n_files = (N + B - 1) // B

    idx = 0
    for fi in range(n_files):
        n_this = min(B, N - idx)
        conds = np.zeros((n_this, 2, args.M, args.M), dtype=np.float32)
        targets = np.zeros_like(conds)
        metas = []

        for k in range(n_this):
            sample = make_one_sample(args, rng)
            conds[k] = sample["cond"]
            targets[k] = sample["target"]
            metas.append(sample["meta"])
            idx += 1

        out_path = os.path.join(args.out_dir, f"flom_mat_{fi:03d}.npz")
        np.savez_compressed(out_path, cond=conds, target=targets, meta=json.dumps(metas, ensure_ascii=False))
        print(f"[{fi+1}/{n_files}] saved {out_path} with {n_this} samples")

    print("Done.")

if __name__ == "__main__":
    main()
