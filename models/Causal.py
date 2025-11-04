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
        self.mgcn=MGCN_block(device, in_channels=1, K=2, nb_chev_filter=64, nb_time_filter=64, time_strides=1 ,len_input=12)

        self.spatial_convs = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim) for _ in range(3)
        ])
        self.temporal_blocks = nn.ModuleList([
            Mamba(d_model = 64)
            for _ in range(3)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(3)
        ])
        self.linear_1=nn.Linear(hidden_dim, d_model)
        self.linear_2=nn.Linear(d_model, hidden_dim)
        
    def forward(self, x, adj):
        B, N, T, D = x.shape
        x=self.linear_1(x).permute(0,1,3,2)

        adj_tensor = torch.from_numpy(adj).float().to(x.device)
        edge_index = adj_tensor.nonzero().t().contiguous()
        x=self.mgcn(x, edge_index)
        output_final=x.permute(0,2,1)
        
        # for spatial_conv, temporal_block, norm in zip(self.spatial_convs, 
        #                                              self.temporal_blocks, 
        #                                              self.norms):
        #     residual = x
    
        #     x_spatial = x.permute(0, 2, 1, 3).reshape(B*T, N, D)
        #     x_spatial = spatial_conv(x_spatial, edge_index) 
        #     x_spatial = x_spatial.view(B, T, N, D).permute(0, 2, 1, 3) 

        #     x_temporal = temporal_block(x.reshape(B*N, T, D)).reshape(B, N, T, D)
          
        #     x = x_spatial + x_temporal
        #     x = norm(x + residual)
            
        return x

class CausalModule(nn.Module):
    def __init__(self, hidden_dim, num_nodes, his_len, pred_len, d_model, state_size, device):
        super().__init__()
        self.his_len = his_len
        self.pred_len = pred_len
        self.num_nodes = num_nodes
        self.device = device
        self.linear=nn.Linear(hidden_dim, d_model)        
        self.encoder = SpatioTemporalEncoder(hidden_dim, num_nodes, d_model, device)

        self.causal_layer = CausalIntervention(num_nodes, state_size, hidden_dim, device)

        self.decoder = nn.Linear(hidden_dim, d_model)
        
        
    def forward(self, fused_feat, t_feat, matrix):
        encoded = self.encoder(t_feat, matrix)
        corrected=self.decoder(encoded)
        pred_3=corrected.squeeze(-1)[..., :3]
        pred_6=corrected.squeeze(-1)[..., :6]
        pred_12=corrected.squeeze(-1)


        m_encoded = self.encoder(fused_feat, matrix) 
        m_corrected = self.causal_layer(m_encoded, matrix) 
        m_corrected=self.decoder(m_corrected)

        m_pred_3 = m_corrected.squeeze(-1)[..., :3]
        m_pred_6 = m_corrected.squeeze(-1)[..., :6]
        m_pred_12 = m_corrected.squeeze(-1)
        
        return pred_3, pred_6, pred_12, m_pred_3, m_pred_6, m_pred_12 