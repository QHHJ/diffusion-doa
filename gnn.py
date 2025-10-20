# models/graphcnn_denoiser.py
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn.pytorch as dglnn

# --- 时间步嵌入（sinusoidal + MLP），供扩散用 ---
class TimestepEmbedding(nn.Module):
    def __init__(self, dim=128, mlp_dim=128):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.SiLU(),
            nn.Linear(mlp_dim, mlp_dim),
        )
    def forward(self, t):  # t: [B] long/int
        t = t.float()
        half = self.dim // 2
        freqs = torch.exp(torch.arange(half, device=t.device) *
                          (-torch.log(torch.tensor(10000.0))/half))
        ang = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)
        if self.dim % 2 == 1: emb = F.pad(emb, (0,1))
        return self.proj(emb)  # [B, mlp_dim]

def init_weights(m):
    if isinstance(m, (nn.Linear,)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class GraphCNN_Denoiser(nn.Module):
    """
    节点级 ε-pred 去噪器（用于扩散）：
    - 输入：节点特征 x_t（Re/Im，2维）与条件 cond（r_y 的 Re/Im，2维）
    - 还接收时间步 t（扩散需要）
    - 输出：节点级 ε̂（Re/Im，2维），形状与 x_t 对齐
    """
    def __init__(
        self,
        num_layers: int = 4,
        in_dim: int = 2,       # x_t: Re/Im
        cond_dim: int = 2,     # r_y: Re/Im
        hidden_dim: int = 64,
        t_embed_dim: int = 128,
        gnn_dropout: float = 0.5,
    ):
        super().__init__()
        self.gnn_dropout = gnn_dropout
        self.t_embed = TimestepEmbedding(dim=t_embed_dim, mlp_dim=hidden_dim)
        # 输入投影：拼接 x_t 与 cond
        self.input_projection = nn.Linear(in_dim + cond_dim, hidden_dim)

        # 图卷积堆叠
        self.convs = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                dglnn.GraphConv(hidden_dim, hidden_dim, allow_zero_in_degree=True)
            )
            self.layer_norms.append(nn.LayerNorm(hidden_dim))

        # 节点级输出头：预测 ε 的 Re/Im
        self.node_head = nn.Linear(hidden_dim, in_dim)

        self.apply(init_weights)

    def forward(self, g: dgl.DGLGraph, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor):
        """
        g: batched DGLGraph
        x_t:   (sum_N, 2)  节点特征（带噪首列 r_t 的 Re/Im）
        cond:  (sum_N, 2)  条件（观测首列 r_y 的 Re/Im 或 FLOM 版本）
        t:     (B,)        时间步（按图一个 t）
        """
        # 将每个图的 t 嵌入广播到对应节点
        sizes = g.batch_num_nodes().tolist()            # [N1, N2, ..., NB]
        t_emb = self.t_embed(t.to(x_t.device))          # [B, hidden_dim]
        t_per_node = torch.cat([t_emb[i].expand(n, -1)  # (Ni, H)
                                for i, n in enumerate(sizes)], dim=0)  # (sum_N, H)

        # 输入拼接与投影
        h = torch.cat([x_t, cond], dim=-1)              # (sum_N, in_dim+cond_dim)
        h = self.input_projection(h) + t_per_node       # 加 t 信息
        h = F.gelu(h)

        # 图卷积堆叠 + 残差
        for i, (conv, ln) in enumerate(zip(self.convs, self.layer_norms)):
            h_in = h
            h = conv(g, h)
            h = ln(h)
            h = F.gelu(h)
            h = F.dropout(h, p=self.gnn_dropout, training=self.training)
            if i != 0:
                h = h + h_in

        # 节点级 ε̂
        eps_hat = self.node_head(h)                     # (sum_N, 2)
        return eps_hat
