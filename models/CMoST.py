import torch.nn as nn
from models.FeatureExtraction import FeatureExtractionModule
from models.FeatureFusion import FeatureFusionModule
from models.Causal import CausalModule
from models.DeepSHAPModule import DeepSHAPModule
import numpy as np

class CMoST(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.extraction = FeatureExtractionModule()
        self.fusion = FeatureFusionModule(args.t_dim, args.text_dim, args.img_dim, args.hidden_dim)
        self.deepshap=DeepSHAPModule(args.num_nodes,args.his_len, args.d_model, args.state_size, args.device)
        self.causal = CausalModule(args.hidden_dim, args.num_nodes, args.his_len, args.pred_len, args.d_model, args.state_size, args.device)
        
    def forward(self, I, T, X, e, l): 
        precomputed_text_feat = self.extraction.encode_text(T[0])

        precomputed_img_feat = self.extraction.encode_img(I[0])
        t_feat, text_feat, img_feat = self.extraction(X, precomputed_text_feat, precomputed_img_feat)

        if e % 5 == 0:
            matrix=self.deepshap(t_feat)
            np.save('cache/deepshap/matrix.npy',matrix)
            matrix=l*np.load('cache/deepshap/prior_matrix/matrix.npy')+(1-l)*np.load('cache/deepshap/matrix.npy')
        else: 
            matrix=l*np.load('cache/deepshap/prior_matrix/matrix.npy')+(1-l)*np.load('cache/deepshap/matrix.npy')
        
        align_t, fused_feat = self.fusion(t_feat, text_feat, img_feat)
        pred = self.causal(fused_feat, align_t, matrix)
        return pred