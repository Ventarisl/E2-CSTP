import torch
import torch.nn as nn
import torch.nn.functional as F
class CrossModalAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.num_heads = 4
        self.scale = (dim // self.num_heads) ** 0.5
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, query, key, value):
        Q = self.query(query).view(query.shape[0], query.shape[1], query.shape[2], self.num_heads, -1).transpose(2, 3)  
        K = self.key(key).view(key.shape[0], key.shape[1], key.shape[2], self.num_heads, -1).transpose(2, 3)  
        V = self.value(value).view(value.shape[0], value.shape[1], value.shape[2], self.num_heads, -1).transpose(2, 3)

        attn_weights = self.softmax(torch.matmul(Q, K.transpose(-2, -1)) / self.scale)  
        attn_output = torch.matmul(attn_weights, V)  
        attn_output = attn_output.transpose(2, 3).contiguous().view(query.shape[0], query.shape[1], query.shape[2], -1)  

        return attn_output

class FeatureFusionModule(nn.Module):
    def __init__(self, t_dim, text_dim, img_dim, hidden_dim):
        super().__init__()
        self.align_t = nn.Linear(t_dim, hidden_dim)
        self.align_text = nn.Linear(text_dim, hidden_dim)
        self.align_img = nn.Linear(img_dim, hidden_dim)

        self.cross_t2text = CrossModalAttention(hidden_dim)
        self.cross_t2img = CrossModalAttention(hidden_dim)
        
        self.fusion_gate = nn.Sequential(
            nn.Linear(3*hidden_dim, 3),
            nn.Softmax(dim=-1)
        )
        self.gate = nn.Sequential(
            nn.Linear(3*hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )


    def forward(self, t_feat, text_feat, img_feat):
        aligned_t = self.align_t(t_feat)

        aligned_text = self.align_text(text_feat).mean(dim=2).unsqueeze(2).expand_as(aligned_t) 

        aligned_img = self.align_img(img_feat).unsqueeze(2).expand_as(aligned_t) 

        attn_t2text = self.cross_t2text(aligned_t, aligned_text, aligned_text)
        attn_t2img = self.cross_t2img(aligned_t, aligned_img, aligned_img)

        combined = torch.cat([aligned_t, attn_t2text, attn_t2img], dim=-1) 
    
        gates = self.fusion_gate(combined)   
        gates = gates.unsqueeze(-1) 

        fused_features = torch.stack([aligned_t, attn_t2text, attn_t2img], dim=3)
        fused = (fused_features * gates).sum(dim=3)

        return aligned_t, fused
