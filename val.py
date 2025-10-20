# ---- 验证脚本：评估训练好的扩散模型在FLOM矩阵重建和DOA估计上的性能 ----

import os, json, argparse
import numpy as np
import torch as th
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 解决OpenMP重复库问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 导入训练脚本中的模块
from unet import UNetModel
from gaussian_diffusion import get_named_beta_schedule
from train import FlomMatNPZ, CondUNet, project_struct

@th.no_grad()
def eval_val_loss(model, val_loader, sqrt_ab, sqrt_1m_ab, steps, device, fp16=False):
    model.eval()
    tot, denom = 0.0, 0.0
    for cond, x0, mask in val_loader:
        B = x0.size(0)
        cond = cond.to(device); x0 = x0.to(device); mask = mask.to(device)
        t   = th.randint(0, steps, (B,), device=device)
        eps = th.randn_like(x0)
        ab  = sqrt_ab.index_select(0, t).view(B,1,1,1)
        abm = sqrt_1m_ab.index_select(0, t).view(B,1,1,1)
        x_t = ab * x0 + abm * eps
        with th.cuda.amp.autocast(enabled=fp16):
            eps_hat = model(x_t, t, cond)
            mse = ((eps_hat - eps)**2) * mask
        tot += mse.sum().item()
        denom += (mask.sum().item() * x0.size(1))
    return tot / max(denom, 1.0)

def steering_vec_ula_np(M, d_over_lambda, theta_deg):
    m = np.arange(M)
    th_rad = np.deg2rad(theta_deg)
    phase = 2*np.pi*d_over_lambda*np.sin(th_rad)*m
    return np.exp(1j*phase)

def run_music_doas(R, K, d_over_lambda, theta_grid_deg):
    """简化 ULA-MUSIC：返回 K 个峰位置（度）"""
    R = 0.5*(R + R.conj().T)
    w, V = np.linalg.eigh(R)
    En = V[:, :R.shape[0]-K]  # 小特征值对应的噪声子空间
    pseu = []
    for th_deg in theta_grid_deg:
        a = steering_vec_ula_np(R.shape[0], d_over_lambda, th_deg)[:,None]
        denom = np.linalg.norm(En.conj().T @ a)**2
        pseu.append(1.0 / (denom + 1e-12))
    pseu = np.array(pseu)
    # 取 K 个峰
    idx = pseu.argsort()[-K:][::-1]
    return theta_grid_deg[idx], pseu

@th.no_grad()
def sample_conditional(model, cond, steps, sqrt_ab, sqrt_1m_ab, do_project=True):
    """DDPM-like 反推（等距 t），每步可做 Toeplitz→Hermitian→PSD 投影，返回左上 7x7 复矩阵"""
    device = cond.device
    B = cond.size(0)
    img = th.randn_like(cond)  # 形状与 x0 一样：[B,2,8,8]
    for i in reversed(range(steps)):
        t = th.full((B,), i, device=device, dtype=th.long)
        ab  = sqrt_ab.index_select(0, t).view(B,1,1,1)
        abm = sqrt_1m_ab.index_select(0, t).view(B,1,1,1)
        eps_hat = model(img, t, cond)
        x0_hat = (img - abm * eps_hat) / (ab + 1e-8)
        if do_project:
            x0_hat = project_struct(x0_hat)
        if i > 0:
            t1 = th.full((B,), i-1, device=device, dtype=th.long)
            ab_prev = sqrt_ab.index_select(0, t1).view(B,1,1,1)
            img = ab_prev * x0_hat + th.sqrt(th.clamp(1 - (ab_prev**2), min=0)) * th.randn_like(img)
        else:
            img = x0_hat
    # 取左上 7x7，转复矩阵
    C_re = img[:,0,:7,:7].cpu().numpy()
    C_im = img[:,1,:7,:7].cpu().numpy()
    return C_re + 1j*C_im

def match_and_metrics(pred_deg, gt_deg, thr=0.5):
    """贪心匹配，算 SR@thr 和 RMSE（度）"""
    pred = list(pred_deg); gt = list(gt_deg)
    used = [False]*len(pred)
    hits, sqerr = 0, []
    for g in gt:
        # 找最近的未用 pred
        best_j, best_d = -1, 1e9
        for j,p in enumerate(pred):
            if used[j]: continue
            d = abs(p-g)
            if d < best_d:
                best_d, best_j = d, j
        if best_j >= 0:
            used[best_j] = True
            if best_d <= thr:
                hits += 1
            sqerr.append(best_d**2)
    sr = hits / max(1, len(gt))
    rmse = (np.mean(sqerr)**0.5) if sqerr else np.nan
    return sr, rmse

def evaluate_checkpoint(ckpt_path, data_dir, steps=30, fp16=False, batch_size=64, workers=4, 
                       model_ch=64, num_res_blocks=2, eval_samples=1024):
    # 读数据 meta（拿 M, d/λ）
    with open(os.path.join(data_dir, "meta.json"), "r", encoding="utf-8") as f:
        meta_all = json.load(f)
    M = int(meta_all["M"]); d_over_lambda = float(meta_all["d_over_lambda"])

    # 数据
    val_set = FlomMatNPZ(data_dir, train=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    # 噪声日程
    betas = get_named_beta_schedule("cosine", 400)
    alphas = 1. - betas
    alpha_bar = np.cumprod(alphas, axis=0).astype(np.float32)
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    sqrt_ab = th.from_numpy(np.sqrt(alpha_bar)).to(device)
    sqrt_1m_ab = th.from_numpy(np.sqrt(1. - alpha_bar)).to(device)

    # 模型（结构需与训练一致：in=4, out=2, image=8…）
    attention_ds = (1,2)  # 对应 8×8/4×4
    unet = UNetModel(image_size=8, in_channels=4, model_channels=model_ch, out_channels=2,
                     num_res_blocks=num_res_blocks, attention_resolutions=attention_ds,
                     dropout=0.0, channel_mult=(1,2), use_checkpoint=False,
                     use_fp16=fp16, use_scale_shift_norm=True)
    model = CondUNet(unet).to(device)

    # 加载权重（支持含 EMA 的 ckpt）
    ckpt = th.load(ckpt_path, map_location=device)
    if "ema" in ckpt:  # 用 EMA 权重更稳
        for p, s in zip(model.parameters(), ckpt["ema"]):
            p.data.copy_(s.to(device))
    else:
        model.load_state_dict(ckpt["model"])

    # 1) val ε-MSE
    # 这里重建一个 val_loader（含 cond/x0/mask），与上 val_set 相同
    vset = FlomMatNPZ(data_dir, train=False)
    vloader = DataLoader(vset, batch_size=batch_size*2, shuffle=False, num_workers=workers)
    val_mse = eval_val_loss(model, vloader, sqrt_ab, sqrt_1m_ab, steps=400, device=device, fp16=fp16)
    print(f"val ε-MSE: {val_mse:.6f}")

    # 2) 采样 + DOA 指标
    theta_grid = np.linspace(meta_all["doa_min"], meta_all["doa_max"], 721)  # 0.1° 分辨
    SR_list, RMSE_list = [], []
    count = 0
    for cond, x0, mask in val_loader:
        cond = cond.to(device)
        C_hat_b = sample_conditional(model, cond, steps=steps, sqrt_ab=sqrt_ab, sqrt_1m_ab=sqrt_1m_ab, do_project=True)
        B = C_hat_b.shape[0]
        # 读取样本 meta（在 npz["meta"] 里是 JSON 串，需解析；这里只示意从文件旁的 meta.json 读全局 K，不严谨）
        # 更严谨的做法：你在数据生成时把每条样本的 K 与 thetas 存在 npz["meta"]，这里逐条解析。
        for b in range(B):
            C_hat = C_hat_b[b]
            # 从对应分片的 meta 解析 K、真 DOA（如果你保存在 npz["meta"] 里）
            # 这里用 x0 的“真”矩阵跑 EVD 拿 rank 粗估 K 也可以：K = (# 大于中位数的特征值)
            # 为简洁：假设 K=2，你也可以从 meta 里读
            # K = ...
            # gt = ...
            # 这里示例当 K=1/2 时如何跑：
            # ----
            # 用噪声子空间 MUSIC
            K_guess = 2
            pred, pseu = run_music_doas(C_hat, K_guess, d_over_lambda, theta_grid)
            # 如果你能读到 gt：
            # sr, rmse = match_and_metrics(pred, gt, thr=0.5)
            # SR_list.append(sr); RMSE_list.append(rmse)
        count += B
        if count >= eval_samples: break  # 先评估一小部分看看趋势
    # print("SR@0.5° =", np.mean(SR_list), " RMSE =", np.nanmean(RMSE_list))
    
    print(f"Evaluated {count} samples with {steps} sampling steps")
    return val_mse

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained diffusion model on FLOM matrices")
    
    # 必需参数
    parser.add_argument("--ckpt", type=str, default="ckpt_flo_unet/10011447/best_model_loss=0.000821.pt", help="Path to checkpoint file")
    parser.add_argument("--data-dir", type=str, default="dataset_flom_mat_1", help="Path to dataset directory")
    
    # 模型参数（需与训练时一致）
    parser.add_argument("--model-ch", type=int, default=64, help="Model channels")
    parser.add_argument("--num-res-blocks", type=int, default=2, help="Number of residual blocks")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision")
    
    # 采样参数
    parser.add_argument("--sampling-steps", type=int, default=30, help="Number of sampling steps (faster than training steps)")
    parser.add_argument("--diffusion-steps", type=int, default=400, help="Number of diffusion steps (should match training)")
    
    # 评估参数
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("--workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--eval-samples", type=int, default=1024, help="Number of samples to evaluate")
    parser.add_argument("--doa-threshold", type=float, default=0.5, help="DOA matching threshold in degrees")
    
    # 其他选项
    parser.add_argument("--seed", type=int, default=2025, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Print detailed results")
    
    args = parser.parse_args()
    
    # 设置随机种子
    th.manual_seed(args.seed)
    th.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    
    print(f"Evaluating checkpoint: {args.ckpt}")
    print(f"Data directory: {args.data_dir}")
    print(f"Sampling steps: {args.sampling_steps}")
    print(f"Batch size: {args.batch_size}")
    print(f"Eval samples: {args.eval_samples}")
    print("-" * 50)
    
    # 运行评估
    val_mse = evaluate_checkpoint(
        ckpt_path=args.ckpt,
        data_dir=args.data_dir,
        steps=args.sampling_steps,
        fp16=args.fp16,
        batch_size=args.batch_size,
        workers=args.workers,
        model_ch=args.model_ch,
        num_res_blocks=args.num_res_blocks,
        eval_samples=args.eval_samples
    )
    
    print("-" * 50)
    print(f"Final validation MSE: {val_mse:.6f}")
    print("Evaluation completed!")

if __name__ == "__main__":
    main()
