import torch
import torch.nn as nn
import shap
import numpy as np
from mamba_ssm import Mamba


class DeepSHAPModule(nn.Module):
    def __init__(self, nodes, his_len, d_model, state_size, device):
        super().__init__()
        self.nodes=nodes
        self.model=Mamba(seq_len = his_len, d_model = d_model, state_size = state_size, device = device)
    def forward(self,X):
        X=X.mean(dim=0, keepdim=True)
        background_data = X.squeeze(0)
        X_data = X.squeeze(0)

        explainer = shap.DeepExplainer(self.model, background_data)
        shap_values = explainer.shap_values(X_data)
        shap_values_array = np.stack(shap_values, axis=0)
        shap_values_array = np.transpose(shap_values_array, (1, 0, 2, 3)).squeeze(-1).reshape(X_data.shape[0], -1)
        
        if np.all(shap_values_array == 0):
            correlation_matrix = np.zeros((self.nodes, self.nodes))
        else:
            correlation_matrix = np.corrcoef(shap_values_array)
            correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)

        return correlation_matrix

