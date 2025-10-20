# guided_diffusion/train_flo_unet.py
# 训练 UNet 在 FLOM 矩阵（7x7，Re/Im 两通道，条件=观测矩阵）上的扩散去噪器
# 改动要点：
# 1) 运行开始时在 --out 目录下创建以“MMDDHHMM”命名的子文件夹（例：10010027）
# 2) 每个 epoch 结束计算一次验证 ε-MSE，遇到更优就只在该子文件夹里保存最优模型
#    文件名：best_model_loss={val_mse:.6f}_ep{epoch:03d}.pt
# 3) 同步保存 args.json 与 train_log.txt，便于复现实验

import os, glob, json, math, argparse, time
from datetime import datetime
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# 解决 OpenMP 重复库问题（Windows 常见）
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 直接导入同目录下的模块
from models.unet import UNetModel
from models.gaussian_diffusion import get_named_beta_schedule

# ---------------------------
# 数据集：读取 npz 分片，取 cond/target，pad 到 8x8，并给出有效 mask
# ---------------------------
class FlomMatNPZ(Dataset):
    def __init__(self, data_dir, split_ratio=(0.9, 0.1), train=True):
        files = sorted(glob.glob(os.path.join(data_dir, "flom_mat_*.npz")))
        assert files, f"No npz found in {data_dir}"
        self.files = files
        # 构建索引表（每个分片有 N 条）
        self.index = []
        for f in files:
            with np.load(f) as npz:
                n = npz["cond"].shape[0]
            self.index += [(f, i) for i in range(n)]
        n_all = len(self.index)
        n_train = int(n_all * split_ratio[0])
        if train:
            self.index = self.index[:n_train]
        else:
            self.index = self.index[n_train:]
        # 7x7 -> 8x8 的 mask（只在有效区域计损失）
        self.mask = th.zeros(1, 8, 8, dtype=th.float32)
        self.mask[:, :7, :7] = 1.0

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        f, i = self.index[idx]
        with np.load(f) as npz:
            cond = npz["cond"][i]    # [2,7,7], float32
            target = npz["target"][i]# [2,7,7], float32
        # pad 到 8x8
        cond8 = np.zeros((2, 8, 8), dtype=np.float32)
        tgt8  = np.zeros((2, 8, 8), dtype=np.float32)
        cond8[:, :7, :7] = cond
        tgt8[:,  :7, :7] = target
        return th.from_numpy(cond8), th.from_numpy(tgt8), self.mask.clone()

# ---------------------------
# 工具：EMA、结构投影（采样用）
# ---------------------------
class EMA:
    def __init__(self, model, decay=0.9995):  # 提高到0.9995，对8x8小模型更平滑
        self.decay = decay
        self.shadow = [p.clone().detach() for p in model.parameters()]
        self.backup = []  # 用于临时存储训练参数
        for p in self.shadow:
            p.requires_grad_(False)

    @th.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow, model.parameters()):
            s.mul_(self.decay).add_(p, alpha=1.0 - self.decay)

    @th.no_grad()
    def copy_to(self, parameters):
        """将EMA权重复制到模型参数"""
        for s, p in zip(self.shadow, parameters):
            p.data.copy_(s.data)
    
    @th.no_grad()
    def store(self, parameters):
        """存储当前训练参数（切换到EMA前）"""
        self.backup = [p.data.clone() for p in parameters]
    
    @th.no_grad()
    def restore(self, parameters):
        """恢复训练参数（从EMA切换回来）"""
        for p, b in zip(parameters, self.backup):
            p.data.copy_(b)
        self.backup = []

@th.no_grad()
def project_struct(x_complex_2chw):
    """
    x_complex_2chw: [B,2,8,8] -> 在每个样本上：
      1) 取 7x7 有效区作为复矩阵 C
      2) Hermitian 对称: (C+C^H)/2
      3) Toeplitz 投影: 每条副对角线平均
      4) PSD 投影: 负特征值截断到 0
      5) 写回到 8x8 的左上角区域，其他保持原样
    """
    B = x_complex_2chw.size(0)
    out = x_complex_2chw.clone()
    for b in range(B):
        C = out[b, 0, :7, :7].cpu().numpy() + 1j * out[b, 1, :7, :7].cpu().numpy()
        # Hermitian
        C = 0.5 * (C + C.conj().T)
        # Toeplitz（沿副对角线平均）
        M = C.shape[0]
        Tproj = np.zeros_like(C, dtype=np.complex64)
        for k in range(-(M - 1), M):
            d = np.diag(C, k)
            m = d.mean()
            Tproj += np.diag(np.full(M - abs(k), m, dtype=np.complex64), k)
        C = 0.5 * (Tproj + Tproj.conj().T)
        # PSD
        w, V = np.linalg.eigh(C)
        w = np.maximum(w, 0.0)
        C = (V * w) @ V.conj().T
        out[b, 0, :7, :7] = th.from_numpy(C.real).to(out.device, out.dtype)
        out[b, 1, :7, :7] = th.from_numpy(C.imag).to(out.device, out.dtype)
    return out

# ---------------------------
# 条件 UNet（把 cond 拼到输入通道）
# ---------------------------
class CondUNet(nn.Module):
    def __init__(self, unet: UNetModel):
        super().__init__()
        self.unet = unet
    def forward(self, x_t, timesteps, cond):
        # x_t: [B,2,8,8], cond: [B,2,8,8]  -> 拼通道
        x_in = th.cat([x_t, cond], dim=1)
        return self.unet(x_in, timesteps)

# ---------------------------
# 验证：计算验证集 ε-MSE（与训练一致的指标）
# ---------------------------
@th.no_grad()
def eval_val_metrics(model, loader, sqrt_ab, sqrt_1m_ab, steps, device, fp16=False):
    model.eval()
    total, denom = 0.0, 0.0
    for cond, x0, mask in loader:
        B = x0.size(0)
        cond = cond.to(device); x0 = x0.to(device); mask = mask.to(device)
        t = th.randint(0, steps, (B,), device=device, dtype=th.long)
        eps = th.randn_like(x0)
        ab  = sqrt_ab.index_select(0, t).view(B,1,1,1)
        abm = sqrt_1m_ab.index_select(0, t).view(B,1,1,1)
        x_t = ab * x0 + abm * eps
        with th.amp.autocast('cuda', enabled=fp16):
            eps_hat = model(x_t, t, cond)
            mse = ((eps_hat - eps) ** 2) * mask
        total += mse.sum().item()
        denom += (mask.sum().item() * x0.size(1))
    avg_loss = total / max(denom, 1.0)
    rmse = math.sqrt(avg_loss)
    return avg_loss, rmse

# ---------------------------
# 运行目录工具：创建 “MMDDHHMM” 子目录；保存 args.json 与日志
# ---------------------------
def make_run_dir(out_root: str) -> str:
    run_id = datetime.now().strftime("%m%d%H%M")  # 例如 10010027
    run_dir = os.path.join(out_root, run_id)
    suffix = 1
    while os.path.exists(run_dir):
        run_dir = os.path.join(out_root, f"{run_id}_{suffix}")
        suffix += 1
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def save_args_json(args, run_dir):
    with open(os.path.join(run_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

# ---------------------------
# 主训练逻辑
# ---------------------------
def main():
    ap = argparse.ArgumentParser("Train UNet on FLOM matrices (7x7 -> pad 8x8) with timed run folder & best checkpoint")
    
    # ========== 数据相关参数 ==========
    ap.add_argument("--data-dir", type=str, default="dataset_flom_mat_snr_-10_to_0_train", 
                    help="npz 数据目录路径，包含 flom_mat_*.npz 文件")
    ap.add_argument("--workers", type=int, default=20, 
                    help="数据加载器的工作进程数，建议设置为 CPU 核心数")
    
    # ========== 模型架构参数 ==========
    ap.add_argument("--image-size", type=int, default=8, 
                    help="输入图像尺寸，FLOM 矩阵从 7x7 填充到 8x8")
    ap.add_argument("--model-ch", type=int, default=384, 
                    help="UNet 基础通道数，影响模型容量和参数量")
    ap.add_argument("--num-res-blocks", type=int, default=2, 
                    help="每个分辨率级别的残差块数量")
    ap.add_argument("--dropout", type=float, default=0.0, 
                    help="Dropout 概率，用于防止过拟合")
    ap.add_argument("--attn-res", type=str, default="8,4,2", 
                    help="注意力机制的分辨率级别，在 8×8/4×4 处加注意力")
    
    # ========== 扩散过程参数 ==========
    ap.add_argument("--steps", type=int, default=400, 
                    help="扩散步数，影响生成质量和计算复杂度")
    ap.add_argument("--noise-schedule", type=str, default="cosine", 
                    choices=["cosine", "linear"],
                    help="噪声调度策略：cosine（余弦）或 linear（线性）")
    
    # ========== 训练超参数 ==========
    ap.add_argument("--batch-size", type=int, default=512, 
                    help="训练批次大小，影响内存使用和训练稳定性")
    ap.add_argument("--epochs", type=int, default=150, 
                    help="训练轮数")
    ap.add_argument("--lr", type=float, default=8e-5, 
                    help="学习率，控制参数更新步长")
    ap.add_argument("--wd", type=float, default=1e-4, 
                    help="权重衰减系数，L2 正则化强度")
    ap.add_argument("--fp16", action="store_true", 
                    help="启用半精度训练，节省显存并加速训练")
    ap.add_argument("--ema", type=float, default=0.9995, 
                    help="指数移动平均衰减率，用于模型权重的平滑")
    ap.add_argument("--log-every", type=int, default=150, 
                    help="每隔多少步记录一次训练日志")
    
    # ========== 输出和随机种子 ==========
    ap.add_argument("--out", type=str, default="ckpt_flo_unet", 
                    help="输出根目录（其下创建 MMDDHHMM 格式的时间戳子目录）")
    ap.add_argument("--seed", type=int, default=2025, 
                    help="随机种子，确保实验可复现")
    args = ap.parse_args()

    th.manual_seed(args.seed)
    th.cuda.manual_seed_all(args.seed)
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    # === 运行目录 ===
    os.makedirs(args.out, exist_ok=True)
    run_dir = make_run_dir(args.out)  # e.g., ckpt_flo_unet/10010027
    print(f"[Run] 输出目录：{run_dir}")
    save_args_json(args, run_dir)
    log_path = os.path.join(run_dir, "train_log.txt")

    # 数据
    train_set = FlomMatNPZ(args.data_dir, train=True)
    val_set   = FlomMatNPZ(args.data_dir, train=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True, drop_last=False)

    # 扩散日程
    betas = get_named_beta_schedule(args.noise_schedule, args.steps)
    alphas = 1.0 - betas
    alpha_bar = np.cumprod(alphas, axis=0).astype(np.float32)
    sqrt_ab = th.from_numpy(np.sqrt(alpha_bar)).to(device)                # [T]
    sqrt_1m_ab = th.from_numpy(np.sqrt(1.0 - alpha_bar)).to(device)       # [T]

    # 模型（in= x_t(2) + cond(2) → 4通道；out= ε(2)）
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
        channel_mult=(1,2,4) if args.image_size >= 16 else (1,2),  # 8x8 用 (1,2) 足够
        use_checkpoint=False,
        use_fp16=args.fp16,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_new_attention_order=False,
    ).to(device)
    model = CondUNet(unet).to(device)

    opt = th.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scaler = th.amp.GradScaler('cuda', enabled=args.fp16)
    ema = EMA(model, decay=args.ema)
    
    # 学习率调度器：验证loss不降则降低lr
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(
        opt, mode='min',           # 监控指标越小越好
        factor=0.5,                # lr降低为原来的0.5倍
        patience=8,                # 8个epoch不改善才降lr
        min_lr=1e-6,              # lr最小值
        threshold=1e-5,           # 改善的最小阈值
        cooldown=2                # 降lr后等2个epoch再监控
    )

    # 记录最佳（按验证 ε-MSE 判定）
    best_val = float('inf')
    best_path = None

    # 训练循环
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        
        # 用于计算epoch平均训练损失
        train_loss_sum = 0.0
        train_batches = 0
        
        for cond, x0, mask in train_loader:
            B = x0.size(0)
            cond = cond.to(device)          # [B,2,8,8]
            x0   = x0.to(device)            # [B,2,8,8]
            mask = mask.to(device)          # [B,1,8,8]

            # 采样 t、噪声 ε，并合成 x_t
            t = th.randint(0, args.steps, (B,), device=device, dtype=th.long)  # [B]
            eps = th.randn_like(x0)
            ab  = sqrt_ab.index_select(0, t).view(B,1,1,1)          # [B,1,1,1]
            abm = sqrt_1m_ab.index_select(0, t).view(B,1,1,1)
            x_t = ab * x0 + abm * eps

            # 前向 & 损失（只在 7x7 有效区计算）
            opt.zero_grad(set_to_none=True)
            with th.amp.autocast('cuda', enabled=args.fp16):
                eps_hat = model(x_t, t, cond)                       # [B,2,8,8]
                mse = ((eps_hat - eps) ** 2) * mask                 # [B,2,8,8]
                loss = mse.sum() / (mask.sum() * x0.size(1) + 1e-8) # 平均到像素

            scaler.scale(loss).backward()
            th.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            ema.update(model)
            global_step += 1
            
            # 累积训练损失
            train_loss_sum += loss.item()
            train_batches += 1

            if global_step % args.log_every == 0:
                msg = f"[ep {epoch:03d}] step {global_step:06d} | train loss {loss.item():.6f}"
                print(msg)
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(msg + "\n")

        # === 每个 epoch 结束：计算训练和验证指标 ===
        train_avg_loss = train_loss_sum / max(train_batches, 1)
        train_rmse = math.sqrt(train_avg_loss)
        
        # 使用 EMA 权重进行验证（更稳定、更好的性能）
        ema.store(model.parameters())
        ema.copy_to(model.parameters())
        val_loss, val_rmse = eval_val_metrics(model, val_loader, sqrt_ab, sqrt_1m_ab, args.steps, device, fp16=args.fp16)
        ema.restore(model.parameters())
        
        # 更新学习率调度器（基于验证loss）
        scheduler.step(val_loss)
        
        # 检查是否是最佳模型
        is_best = val_loss < best_val
        if is_best:
            # 删除之前的最佳模型文件
            if best_path and os.path.exists(best_path):
                os.remove(best_path)
                
            best_val = val_loss
            # 以 "best_model_loss=xx.pt" 命名（不包含epoch信息，保持简洁）
            best_filename = f"best_model_loss={best_val:.6f}.pt"
            best_path = os.path.join(run_dir, best_filename)
            
            # 保存 EMA 权重作为主模型（推理时性能更好）
            ema.store(model.parameters())
            ema.copy_to(model.parameters())
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),  # 保存的是EMA权重
                "ema": [p.cpu() for p in ema.shadow],  # 同时保存shadow backup
                "args": vars(args),
                "train_loss": train_avg_loss,
                "train_rmse": train_rmse,
                "val_loss": val_loss,
                "val_rmse": val_rmse,
            }
            th.save(ckpt, best_path)
            ema.restore(model.parameters())
            
            # 输出最佳模型保存信息
            best_msg = f"最佳模型已保存于 Epoch {epoch}，测试损失: {val_loss:.6f}, 测试RMSE: {val_rmse:.6f}"
            print(best_msg)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(best_msg + "\n")
        
        # 输出epoch总结（包含当前学习率）
        current_lr = opt.param_groups[0]['lr']
        epoch_msg = f"Epoch {epoch}, LR:{current_lr:.6f} 训练损失:{train_avg_loss:.4f} 训练RMSE:{train_rmse:.4f} 测试损失: {val_loss:.4f} 测试RMSE: {val_rmse:.4f}"
        print(epoch_msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(epoch_msg + "\n")

        # （可选）小批量采样快速自检 —— 如不需要可以注释掉
        # 这里不另存 epoch 检查点，避免目录里产生多余文件
        model.eval()
        with th.no_grad():
            for cond, x0, mask in val_loader:
                cond = cond.to(device); x0 = x0.to(device); mask = mask.to(device)
                B = cond.size(0)
                img = th.randn_like(x0)
                steps = min(30, args.steps)  # 轻量采样看趋势
                ema.copy_to(model.parameters())  # 用 EMA 权重更稳
                for i in reversed(range(steps)):
                    t = th.full((B,), i, device=device, dtype=th.long)
                    ab  = sqrt_ab.index_select(0, t).view(B,1,1,1)
                    abm = sqrt_1m_ab.index_select(0, t).view(B,1,1,1)
                    eps_hat = model(img, t, cond)
                    x0_hat = (img - abm * eps_hat) / (ab + 1e-8)
                    x0_hat = project_struct(x0_hat)  # 结构投影（只投影 7x7）
                    if i > 0:
                        t_1 = th.full((B,), i-1, device=device, dtype=th.long)
                        ab_prev = sqrt_ab.index_select(0, t_1).view(B,1,1,1)
                        img = ab_prev * x0_hat + th.sqrt(th.clamp(1 - (ab_prev**2), min=0)) * th.randn_like(img)
                    else:
                        img = x0_hat
                break  # 只做一小批

    print("Training done.")
    print(f"[Run] 本次运行目录：{run_dir}")
    if best_path:
        print(f"[Run] 最优模型：{best_path}")

if __name__ == "__main__":
    main()
