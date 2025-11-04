import torch
import torch.nn as nn
from transformers import BertModel


class FeatureExtractionModule(nn.Module):
    def __init__(self):
        super().__init__()        
        self.text_encoder = BertModel.from_pretrained('models/bert-base-uncased')

        self.img_encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.AdaptiveAvgPool2d((1,1))
        )
    def encode_text(self, T_input):
        with torch.no_grad(): 
            text_feat = self.text_encoder(T_input).last_hidden_state
        return text_feat

    def encode_img(self, I_input):
        img_feat = self.img_encoder(I_input.unsqueeze(1))
        img_feat = img_feat.squeeze(-1).squeeze(-1)
        return img_feat 
    
    def forward(self, X, T, I):
        temporal_feat=X.unsqueeze(-1)
        text_feat = T.unsqueeze(0).expand(X.shape[0], -1, -1, -1)
        img_feat = I.unsqueeze(0).expand(X.shape[0], -1, -1)

        return temporal_feat, text_feat, img_feat