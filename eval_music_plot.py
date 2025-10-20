# guided_diffusion/eval_music_3way.py
# 评估扩散+UNet 去噪对 MUSIC 的增益：分别画 Noisy / Denoised / Clean 三个子图的伪谱
# 并统计 SR@0.5°、RMSE(°) 等指标。

import os, glob, json, argparse
import numpy as np
import torch as th
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast

# 解决 OpenMP 重复库问题（Windows 常见）
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 直接导入同目录下模块（与你的训练脚本一致）
from models.unet import UNetModel
from models.gaussian_diffusion import get_named_beta_schedule


# ---------- 数据集：返回 cond/target + meta ----------
class FlomMatNPZWithMeta(Dataset):
    def __init__(self, data_dir, split_ratio=(0.90, 0.10), train=False, use_all=False):
        """
        Args:
            data_dir: 数据目录
            split_ratio: (训练比例, 验证比例)，仅在 use_all=False 时生效
            train: True=使用训练集, False=使用验证集，仅在 use_all=False 时生效
            use_all: 如果为True，使用全部数据（评估独立测试集时使用）
        """
        files = sorted(glob.glob(os.path.join(data_dir, "flom_mat_*.npz")))
        assert files, f"No npz found in {data_dir}"
        self.items = []  # [(file, idx, meta_dict)]
        for f in files:
            with np.load(f, allow_pickle=True) as npz:
                N = npz["cond"].shape[0]
                meta_raw = npz["meta"]
                # 兼容：字符串JSON / 0-d array 内嵌JSON / object 数组
                if isinstance(meta_raw, np.ndarray):
                    if meta_raw.ndim == 0:
                        metas = json.loads(meta_raw.item())
                    elif meta_raw.dtype == object:
                        metas = list(meta_raw)
                    else:
                        metas = json.loads(str(meta_raw))
                else:
                    metas = json.loads(meta_raw)
                assert len(metas) == N, f"meta length mismatch in {f}"
                for i in range(N):
                    self.items.append((f, i, metas[i]))
        
        # 如果 use_all=True，使用全部数据；否则按比例划分
        if not use_all:
            n_all = len(self.items)
            n_train = int(n_all * split_ratio[0])
            self.items = self.items[:n_train] if train else self.items[n_train:]
        
        self.mask = th.zeros(1, 8, 8, dtype=th.float32); self.mask[:, :7, :7] = 1.0

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        f, i, meta = self.items[idx]
        with np.load(f) as npz:
            cond = npz["cond"][i]    # [2,7,7]
            tgt  = npz["target"][i]  # [2,7,7]
        c8 = np.zeros((2,8,8), np.float32); c8[:, :7, :7] = cond
        t8 = np.zeros((2,8,8), np.float32); t8[:, :7, :7] = tgt
        return th.from_numpy(c8), th.from_numpy(t8), self.mask.clone(), meta


# ---------- 结构投影：Toeplitz → Hermitian → PSD ----------
def project_struct_numpy(C):
    # C: complex (7,7)
    C = 0.5 * (C + C.conj().T)
    M = C.shape[0]
    Tproj = np.zeros_like(C, dtype=np.complex64)
    for k in range(-(M-1), M):
        d = np.diag(C, k); m = d.mean()
        Tproj += np.diag(np.full(M-abs(k), m, dtype=np.complex64), k)
    C = 0.5 * (Tproj + Tproj.conj().T)
    w, V = np.linalg.eigh(C); w = np.maximum(w, 0.0)
    return (V * w) @ V.conj().T


# ---------- 简化版 ULA-MUSIC ----------
def steering_vec_ula(M, d_over_lambda, theta_deg):
    m = np.arange(M)
    th_rad = np.deg2rad(theta_deg)
    phase = 2*np.pi*d_over_lambda*np.sin(th_rad)*m
    return np.exp(1j*phase)[:, None]  # [M,1]

def music_spectrum(R, K, d_over_lambda, theta_grid_deg):
    R = 0.5*(R + R.conj().T)
    w, V = np.linalg.eigh(R)
    En = V[:, :R.shape[0]-K]  # 噪声子空间
    pseu = []
    for th_deg in theta_grid_deg:
        a = steering_vec_ula(R.shape[0], d_over_lambda, th_deg)
        denom = np.linalg.norm(En.conj().T @ a)**2
        pseu.append(1.0 / (denom + 1e-12))
    return np.asarray(pseu)

def run_music(R, K, d_over_lambda, theta_grid_deg):
    pseu = music_spectrum(R, K, d_over_lambda, theta_grid_deg)
    idx = pseu.argsort()[-K:][::-1]
    return theta_grid_deg[idx], pseu

def match_metrics(pred_deg, gt_deg, thr=0.5):
    pred = list(pred_deg); gt = list(gt_deg)
    used = [False]*len(pred); hits=0; sq=[]
    for g in gt:
        j_best, d_best = -1, 1e9
        for j,p in enumerate(pred):
            if used[j]: continue
            d = abs(p-g)
            if d < d_best: d_best, j_best = d, j
        if j_best>=0: used[j_best]=True
        if d_best<=thr: hits += 1
        sq.append(d_best**2)
    sr = hits / max(1,len(gt))
    rmse = (np.mean(sq)**0.5) if sq else np.nan
    return sr, rmse


# ---------- 条件 UNet 包装（x_t 与 cond 拼通道） ----------
class CondUNet(th.nn.Module):
    def __init__(self, unet): super().__init__(); self.unet = unet
    def forward(self, x_t, t, cond): return self.unet(th.cat([x_t, cond], dim=1), t)


@th.no_grad()
def sample_denoised(model, cond, steps, sqrt_ab, sqrt_1m_ab, do_project_each_step=True):
    """DDPM-like 反推+每步结构投影，返回 (B,7,7) 复矩阵"""
    device = cond.device
    B = cond.size(0)
    img = th.randn_like(cond)
    for i in reversed(range(steps)):
        t = th.full((B,), i, device=device, dtype=th.long)
        ab  = sqrt_ab.index_select(0, t).view(B,1,1,1)
        abm = sqrt_1m_ab.index_select(0, t).view(B,1,1,1)
        eps_hat = model(img, t, cond)
        x0_hat = (img - abm * eps_hat) / (ab + 1e-8)
        if do_project_each_step:
            for b in range(B):
                C = x0_hat[b,0,:7,:7].cpu().numpy() + 1j*x0_hat[b,1,:7,:7].cpu().numpy()
                C = project_struct_numpy(C)
                # 修复：明确指定 device 和 dtype
                x0_hat[b,0,:7,:7] = th.from_numpy(C.real).to(device=img.device, dtype=img.dtype)
                x0_hat[b,1,:7,:7] = th.from_numpy(C.imag).to(device=img.device, dtype=img.dtype)
        if i>0:
            t1 = th.full((B,), i-1, device=device, dtype=th.long)
            ab_prev = sqrt_ab.index_select(0, t1).view(B,1,1,1)
            img = ab_prev * x0_hat + th.sqrt(th.clamp(1 - (ab_prev**2), min=0)) * th.randn_like(img)
        else:
            img = x0_hat
    re = img[:,0,:7,:7].cpu().numpy(); im = img[:,1,:7,:7].cpu().numpy()
    return re + 1j*im


# ---------- 三子图绘制：Noisy / Denoised / Clean ----------
def plot_three_spectra(theta_grid, pseu_noisy, pseu_deno, pseu_clean,
                       gt_deg, title, out_path,
                       normalize="per", yscale="log"):
    """
    normalize: 'per' 每条曲线各自除以最大值；'global' 用三条共同最大值；'none' 不归一化
    yscale: 'log' 或 'linear'
    """
    import matplotlib.pyplot as plt
    curves = [("Noisy", pseu_noisy), ("Denoised", pseu_deno), ("Clean", pseu_clean)]
    if normalize != "none":
        if normalize == "global":
            gmax = max([c[1].max() for c in curves]) + 1e-12
        for i, (name, y) in enumerate(curves):
            m = (y.max() + 1e-12) if normalize == "per" else gmax
            curves[i] = (name, y / m)

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    for ax, (name, y) in zip(axes, curves):
        if yscale == "log":
            ax.semilogy(theta_grid, y)
        else:
            ax.plot(theta_grid, y)
        for g in gt_deg:
            ax.axvline(g, linestyle="--", linewidth=1.0, color="C7")
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.25)
    axes[0].set_title(title)
    axes[-1].set_xlabel("θ (deg)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser("3-way MUSIC eval (three subplots): noisy vs denoised vs clean")
    parser.add_argument("--ckpt", type=str, default="ckpt_flo_unet/10111424/best_model_loss=0.000541.pt", help="训练好的权重路径")
    parser.add_argument("--data-dir", type=str, default="dataset_flom_mat_snr_-10_to_0_eval", help="npz 数据目录")
    parser.add_argument("--steps", type=int, default=30, help="采样反推步数（建议30-100，不要超过100）")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--project-inputs", action="store_true",
                        help="在跑 MUSIC 前对 noisy/clean 也做一次 Toeplitz→Hermitian→PSD 投影（建议开）")
    parser.add_argument("--eval-num", type=int, default=200, help="统计指标用的样本数上限")
    parser.add_argument("--use-all-data", action="store_true", default=True,
                        help="使用测试集的全部数据（默认开启）")
    # —— 模型架构参数（需与训练时保持一致）——
    parser.add_argument("--image-size", type=int, default=8, help="输入图像尺寸")
    parser.add_argument("--model-ch", type=int, default=256, help="UNet 基础通道数（应与训练时一致）")
    parser.add_argument("--num-res-blocks", type=int, default=2, help="残差块数量")
    parser.add_argument("--attn-res", type=str, default="8,4,2", help="注意力分辨率级别，逗号分隔")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout 概率")
    # —— 绘图选项 ——
    parser.add_argument("--plot-num", type=int, default=100, help="保存谱图的样本数量")
    parser.add_argument("--plot-out", type=str, default="music_plots", help="谱图输出目录")
    parser.add_argument("--show", action="store_true", help="是否交互显示（服务器上可不加）")
    parser.add_argument("--yscale", type=str, default="log", choices=["linear", "log"], help="谱图 y 轴刻度")
    parser.add_argument("--norm", type=str, default="per", choices=["per", "global", "none"], help="谱图归一化方式")
    args = parser.parse_args()

    # 尝试从模型检查点同目录下的 args.json 读取训练配置
    ckpt_dir = os.path.dirname(args.ckpt)
    args_json_path = os.path.join(ckpt_dir, "args.json")
    train_steps = 400  # 默认扩散步数
    
    if os.path.exists(args_json_path):
        print(f"[Config] 从 {args_json_path} 加载训练配置")
        with open(args_json_path, "r", encoding="utf-8") as f:
            train_args = json.load(f)
        
        # 用训练时的模型架构参数覆盖（确保评估时与训练一致）
        args.image_size = train_args.get("image_size", args.image_size)
        args.model_ch = train_args.get("model_ch", args.model_ch)
        args.num_res_blocks = train_args.get("num_res_blocks", args.num_res_blocks)
        args.attn_res = train_args.get("attn_res", args.attn_res)
        args.dropout = train_args.get("dropout", args.dropout)
        train_steps = train_args.get("steps", 400)  # 读取训练时的扩散步数
        
        print(f"[Config] 模型配置: model_ch={args.model_ch}, attn_res={args.attn_res}, "
              f"num_res_blocks={args.num_res_blocks}, dropout={args.dropout}, train_steps={train_steps}")
    else:
        print(f"[Config] 未找到 {args_json_path}，使用默认或命令行指定的模型配置")

    # 读数据 meta（全局参数）
    with open(os.path.join(args.data_dir, "meta.json"), "r", encoding="utf-8") as f:
        meta_all = json.load(f)
    M = int(meta_all["M"]); d_over_lambda = float(meta_all["d_over_lambda"])
    theta_grid = np.linspace(meta_all["doa_min"], meta_all["doa_max"], 721)  # 0.1° 分辨

    # DataLoader（自定义 collate 以保留 meta 列表）
    def collate_fn(batch):
        conds, tgts, masks, metas = zip(*batch)
        return th.stack(conds, 0), th.stack(tgts, 0), th.stack(masks, 0), list(metas)

    val_set = FlomMatNPZWithMeta(args.data_dir, train=False, use_all=args.use_all_data)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True, collate_fn=collate_fn)
    print(f"[Data] 加载了 {len(val_set)} 个测试样本 (use_all={args.use_all_data})")

    # 噪声日程（使用从训练配置读取的步数）
    betas = get_named_beta_schedule("cosine", train_steps)
    alphas = 1. - betas
    alpha_bar = np.cumprod(alphas, axis=0).astype(np.float32)
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    sqrt_ab = th.from_numpy(np.sqrt(alpha_bar)).to(device)
    sqrt_1m_ab = th.from_numpy(np.sqrt(1. - alpha_bar)).to(device)

    # 模型（结构需与训练一致）
    # 解析注意力分辨率参数
    attention_ds = []
    for r in args.attn_res.split(","):
        r = r.strip()
        if not r: continue
        attention_ds.append(args.image_size // int(r))
    
    unet = UNetModel(
        image_size=args.image_size, 
        in_channels=4, 
        model_channels=args.model_ch, 
        out_channels=2,
        num_res_blocks=args.num_res_blocks, 
        attention_resolutions=tuple(attention_ds), 
        dropout=args.dropout,
        channel_mult=(1,2,4) if args.image_size >= 16 else (1,2),
        use_checkpoint=False, 
        use_fp16=args.fp16, 
        use_scale_shift_norm=True
    ).to(device)
    model = CondUNet(unet).to(device)

    # 加载权重（若 ckpt 包含 EMA，优先用 EMA）
    ckpt = th.load(args.ckpt, map_location=device)
    if "ema" in ckpt and isinstance(ckpt["ema"], (list, tuple)):
        for p, s in zip(model.parameters(), ckpt["ema"]):
            p.data.copy_(s.to(device))
        print("Loaded EMA weights.")
    else:
        model.load_state_dict(ckpt["model"])
        print("Loaded model weights.")
    
    # 设置为评估模式（关键！）
    model.eval()
    print("Model set to evaluation mode.")

    # —— 准备绘图输出 —— 
    os.makedirs(args.plot_out, exist_ok=True)
    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: F401  # 仅触发后端

    # 评测统计器
    sr_noisy = []; rmse_noisy = []
    sr_deno  = []; rmse_deno  = []
    sr_clean = []; rmse_clean = []
    n_done = 0
    n_plotted = 0

    # 确定设备类型用于 autocast
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'

    for cond8, tgt8, mask, meta in val_loader:
        B = cond8.size(0)
        cond8 = cond8.to(device)
        with autocast(device_type, enabled=args.fp16):
            C_deno = sample_denoised(model, cond8, steps=args.steps,
                                     sqrt_ab=sqrt_ab, sqrt_1m_ab=sqrt_1m_ab)

        cond_np = cond8.cpu().numpy()
        tgt_np  = tgt8.cpu().numpy()
        for b in range(B):
            # 三路矩阵（复数 7x7）
            C_noisy = cond_np[b,0,:7,:7] + 1j*cond_np[b,1,:7,:7]
            C_clean = tgt_np [b,0,:7,:7] + 1j*tgt_np [b,1,:7,:7]
            C_d     = C_deno[b]

            if args.project_inputs:
                C_noisy = project_struct_numpy(C_noisy)
                C_clean = project_struct_numpy(C_clean)
                C_d     = project_struct_numpy(C_d)  # 采样时已步步投影，这里再保险一次

            sm = meta[b]; K = int(sm["K"])
            gt = [float(x) for x in sm["thetas_deg"]]
            alpha = float(sm.get("alpha", np.nan))
            snr_db = float(sm.get("snr_db", np.nan))

            # 伪谱与峰
            pred_noisy, pseu_noisy = run_music(C_noisy, K, d_over_lambda, theta_grid)
            pred_deno , pseu_deno  = run_music(C_d,     K, d_over_lambda, theta_grid)
            pred_clean, pseu_clean = run_music(C_clean, K, d_over_lambda, theta_grid)

            # 指标
            sr, rm = match_metrics(pred_noisy, gt, thr=0.5); sr_noisy.append(sr); rmse_noisy.append(rm)
            sr, rm = match_metrics(pred_deno,  gt, thr=0.5); sr_deno .append(sr); rmse_deno .append(rm)
            sr, rm = match_metrics(pred_clean, gt, thr=0.5); sr_clean.append(sr); rmse_clean.append(rm)
            n_done += 1

            # —— 三子图绘制 —— 
            if n_plotted < args.plot_num:
                out_path = os.path.join(args.plot_out, f"spectrum_{n_plotted:04d}.png")
                title = f"K={K} | α={alpha:.2f}, SNR={snr_db:.1f} dB"
                plot_three_spectra(theta_grid, pseu_noisy, pseu_deno, pseu_clean,
                                   gt, title, out_path, normalize=args.norm, yscale=args.yscale)
                print(f"[plot] saved {out_path}")
                n_plotted += 1

            if n_done >= args.eval_num: break
        if n_done >= args.eval_num: break

    def _mean(x):
        x = np.array(x, dtype=np.float64)
        return float(np.nanmean(x)) if x.size else float('nan')

    print("--------------------------------------------------")
    print(f"Evaluated {n_done} samples with {args.steps} sampling steps")
    print("SR@0.5°  noisy:   {:.3f} | denoised: {:.3f} | clean:   {:.3f}".format(_mean(sr_noisy), _mean(sr_deno), _mean(sr_clean)))
    print("RMSE(°)  noisy:   {:.3f} | denoised: {:.3f} | clean:   {:.3f}".format(_mean(rmse_noisy), _mean(rmse_deno), _mean(rmse_clean)))
    print(f"Saved spectra to: {args.plot_out}")
    print("--------------------------------------------------")


if __name__ == "__main__":
    main()
