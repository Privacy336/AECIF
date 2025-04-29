import math
import torch
import torch.nn as nn
from torch.nn import functional as F



class Grid_feature_fuse_extractor(nn.Module):
    def __init__(self, normalize_input=True, config=None):
        super(Grid_feature_fuse_extractor, self).__init__()
        self.dim = 512
        self.normalize_input = normalize_input
        self.encoder_change = nn.Linear(1024, self.dim)
        self.layer_norm = nn.LayerNorm(self.dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, origial_enc_output_1):
        batch_size = origial_enc_output_1.size(0)
        size = int(math.sqrt(origial_enc_output_1.size(1)))
        channel_dimension = origial_enc_output_1.size(2)
        enc_output_1 = origial_enc_output_1.reshape(batch_size, size, size, channel_dimension) #[10,49,1024]-----》[10, 7, 7, 1024]
        tensor_list_1 = [enc_output_1[:, i, :, :] for i in range(size)]
        enc_output_1 = torch.cat(tensor_list_1, dim=1)
        enc_output_1 = enc_output_1.unsqueeze(-1).permute(0, 2, 1, 3)
        if self.normalize_input:
            enc_output_1 = F.normalize(enc_output_1, p=2, dim=1)
        enc_output_1 = self.encoder_change(enc_output_1.transpose(1,3)).transpose(1,3)
        grad_enc_outpu_1 = enc_output_1.squeeze(3).permute(0, 2, 1)
        spatial_attention_1 = grad_enc_outpu_1[:, 0:size, :].unsqueeze(1)
        for j in range(1, size):
            spatial_attention_1 = torch.cat((spatial_attention_1, grad_enc_outpu_1[:, 7*j : 7*(j+1), :].unsqueeze(1)), dim=1)
        spatial_attention_orgial_1 = spatial_attention_1.permute(0, 3, 1, 2)  #10,512,7,7
        grid_feature_1 = spatial_attention_orgial_1.reshape(batch_size, size*size, self.dim) #[5, 49, 512]

        return grid_feature_1
