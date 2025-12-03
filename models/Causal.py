import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np
from mamba_ssm import Mamba
from models.Predict import MGCN_block

class CausalIntervention(nn.Module):
    def __init__(self, num_nodes, state_size, d_model, device):
        super().__init__()
        self.device = device
        self.S = nn.Parameter(torch.randn(num_nodes, state_size)*0.1)
        self.confounder_net = nn.Sequential(
            nn.Linear(d_model, 2*state_size),
            nn.GELU(),
            nn.Linear(2*state_size, state_size)
        )
        self.env_net = nn.Sequential(
            nn.Linear(state_size, 2*state_size),
            nn.ReLU(),
            nn.Linear(2*state_size, state_size),
            nn.Dropout(0.1)
        )
        self.gate_net = nn.Sequential(
            nn.Linear(2*state_size, state_size),
            nn.Sigmoid()
        )
        
        self.proj = nn.Linear(state_size, d_model)
         
    def forward(self, x, adj):
        B, N, T, D = x.shape
        
        confounder = self.confounder_net(x.mean(dim=2))

        S_env = self.env_net(self.S)  # [N, S]
        S_env = torch.einsum('nm,ms->ns', torch.tensor(adj.astype(np.float32)).to(self.device), S_env) 
        S_env = S_env.unsqueeze(0).expand(B, -1, -1) 

        S_combined = torch.cat([S_env, confounder], dim=-1)
        gate = self.gate_net(S_combined)
        S_adjusted = S_env * gate + confounder * (1 - gate)

        adjustment = self.proj(S_adjusted)
        adjustment = adjustment.unsqueeze(2).expand(-1, -1, T, -1)
        
        x_hat = x + x * adjustment
        
        return x_hat


class SpatioTemporalEncoder(nn.Module):
    def __init__(self, hidden_dim, num_nodes, d_model, device):
        super().__init__()
        self.mgcn = MGCN_block(device, in_channels=1, K=2, nb_chev_filter=64, nb_time_filter=64, time_strides=1, len_input=12)
        
        self.spatial_convs = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim) for _ in range(3)
        ])
        self.temporal_blocks = nn.ModuleList([
            Mamba(d_model=64)
            for _ in range(3)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(3)
        ])
        self.linear_1 = nn.Linear(hidden_dim, d_model)
        self.linear_2 = nn.Linear(d_model, hidden_dim)
        
    def forward(self, x, adj):
        B, N, T, D = x.shape  # D=hidden_dim=64
        
        # 首先将特征维度从hidden_dim投影到d_model=1
        x = self.linear_1(x)  # [B, N, T, 1]
        x = x.permute(0, 1, 3, 2)  # [B, N, 1, T]
        
        # 创建边索引
        adj_tensor = torch.from_numpy(adj).float().to(x.device)
        edge_index = adj_tensor.nonzero().t().contiguous()
        
        # 确保边索引在有效范围内
        if edge_index.max().item() >= N:
            mask = (edge_index[0] < N) & (edge_index[1] < N)
            edge_index = edge_index[:, mask]
        
        # 通过MGCN块
        x = self.mgcn(x, edge_index)  # 输出: [B, N, 1, T]
        
        # 调整维度
        x = x.permute(0, 3, 1, 2)  # [B, T, N, 1]
        
        # 将特征维度从1投影回hidden_dim
        x = self.linear_2(x)  # [B, T, N, hidden_dim]
        
        return x


class CausalModule(nn.Module):
    def __init__(self, hidden_dim, num_nodes, his_len, pred_len, d_model, state_size, device):
        super().__init__()
        self.his_len = his_len
        self.pred_len = pred_len
        self.num_nodes = num_nodes
        self.device = device
        self.linear = nn.Linear(hidden_dim, d_model)
        self.encoder = SpatioTemporalEncoder(hidden_dim, num_nodes, d_model, device)
        self.causal_layer = CausalIntervention(num_nodes, state_size, hidden_dim, device)
        self.decoder = nn.Linear(hidden_dim, d_model)
        
    def forward(self, fused_feat, t_feat, matrix):
        # t_feat 形状: [B, N, T, D] 其中 D=hidden_dim=64
        B, N, T, D = t_feat.shape
        
        # 处理时间特征
        encoded = self.encoder(t_feat, matrix)  # 输出: [B, T, N, hidden_dim]
        
        # 直接通过decoder
        corrected = self.decoder(encoded)  # [B, T, N, d_model] 其中 d_model=1
        
        # 去掉最后一个维度，然后转置维度以匹配期望的形状 [B, N, T]
        corrected = corrected.squeeze(-1).permute(0, 2, 1)  # [B, N, T]
        
        # 提取不同长度的预测
        pred_3 = corrected[..., :3]  # [B, N, 3]
        pred_6 = corrected[..., :6]  # [B, N, 6]
        pred_12 = corrected  # [B, N, T]

        # 处理融合特征
        m_encoded = self.encoder(fused_feat, matrix)  # [B, T, N, hidden_dim]
        
        # 调整维度以匹配CausalIntervention的期望输入 [B, N, T, hidden_dim]
        m_encoded_adj = m_encoded.permute(0, 2, 1, 3)  # [B, N, T, hidden_dim]
        m_corrected = self.causal_layer(m_encoded_adj, matrix)  # [B, N, T, hidden_dim]
        m_corrected = m_corrected.permute(0, 1, 2, 3)  # 保持不变 [B, N, T, hidden_dim]
        m_corrected = self.decoder(m_corrected)  # [B, N, T, d_model]
        m_corrected = m_corrected.squeeze(-1)  # [B, N, T]

        m_pred_3 = m_corrected[..., :3]  # [B, N, 3]
        m_pred_6 = m_corrected[..., :6]  # [B, N, 6]
        m_pred_12 = m_corrected  # [B, N, T]
        
        return pred_3, pred_6, pred_12, m_pred_3, m_pred_6, m_pred_12