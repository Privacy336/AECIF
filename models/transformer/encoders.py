from torch.nn import functional as F
from models.transformer.utils import PositionWiseFeedForward
from models.transformer.attention import MultiHeadAttention
import torch
import torch.nn as nn

from models.containers import Module, ModuleList



class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
                 attention_module=None, attention_module_kwargs=None,config=None):
        super(EncoderLayer, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                        attention_module=attention_module,
                                        attention_module_kwargs=attention_module_kwargs)

        self.dropout = nn.Dropout(dropout)
        self.lnorm = nn.LayerNorm(d_model)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):

        att, _ = self.mhatt(queries, keys, values, attention_mask, attention_weights)
        att = self.lnorm(queries + self.dropout(att))
        ff = self.pwff(att)
        return ff


class MultiLevelEncoder(nn.Module):
    def __init__(self, N, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 identity_map_reordering=False, attention_module=None, attention_module_kwargs=None):
        super(MultiLevelEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                  identity_map_reordering=identity_map_reordering,
                                                  attention_module=attention_module,
                                                  attention_module_kwargs=attention_module_kwargs)
                                     for _ in range(N)])
        self.padding_idx = padding_idx

    def forward(self, input, attention_weights=None):
        # input (b_s, seq_len, d_in)
        attention_mask = (torch.sum(input, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)

        out = input
        for l in self.layers:
            out = l(out, out, out, attention_mask, attention_weights)
            # outs.append(out.unsqueeze(1))

        # outs = torch.cat(outs, 1)
        return out, attention_mask


class TransformerEncoder(MultiLevelEncoder):
    def __init__(self, N, padding_idx, d_in=2048, **kwargs):
        super(TransformerEncoder, self).__init__(N, padding_idx, **kwargs)
        self.fc = nn.Linear(d_in, self.d_model)
        self.dropout = nn.Dropout(p=self.dropout)
        self.layer_norm = nn.LayerNorm(self.d_model)

    def forward(self, input, attention_weights=None):
        mask = (torch.sum(input, dim=-1) == 0).unsqueeze(-1)
        out = F.relu(self.fc(input))
        out = self.dropout(out)
        out = self.layer_norm(out)
        out = out.masked_fill(mask, 0)
        return super(TransformerEncoder, self).forward(out, attention_weights=attention_weights)

class m2_grid_transformer(nn.Module):
    def __init__(self, N, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 identity_map_reordering=False, attention_module=None, attention_module_kwargs=None, config=None):
        super(m2_grid_transformer, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.padding_idx = padding_idx
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout, identity_map_reordering=identity_map_reordering,attention_module=attention_module,
                                                  attention_module_kwargs=attention_module_kwargs, config=config) for _ in range(N)])

    def forward(self, grid_input_1, grid_input_2, attention_weights=None, config=None):

        grid_f = grid_input_1

        grid_f_attention_mask = (torch.sum(grid_f, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)

        # m2模式
        outs = []
        for l in self.layers:
            out = l(grid_f, grid_f, grid_f, grid_f_attention_mask, attention_weights)
            outs.append(out.unsqueeze(1))
            grid_f = out
        outs = torch.cat(outs, 1)
        return outs, grid_f_attention_mask


class singal_grid_transformer(nn.Module):
    def __init__(self, N, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 identity_map_reordering=False, attention_module=None, attention_module_kwargs=None, config=None):
        super(singal_grid_transformer, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.padding_idx = padding_idx
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout, identity_map_reordering=identity_map_reordering,attention_module=attention_module,
                                                  attention_module_kwargs=attention_module_kwargs, config=config) for _ in range(N)])

    def forward(self, grid_input_1, attention_weights=None, config=None):

        grid_f = grid_input_1
        grid_f_attention_mask = (torch.sum(grid_input_1, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)
        for l in self.layers:
            out = l(grid_f, grid_f, grid_f, grid_f_attention_mask, attention_weights)
            grid_f = out
        return out, grid_f_attention_mask



class DecoderLayer(Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, self_att_module=None,
                 enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
        super(DecoderLayer, self).__init__()
        self.enc_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=False,
                                          attention_module=enc_att_module,
                                          attention_module_kwargs=enc_att_module_kwargs)

        self.dropout2 = nn.Dropout(dropout)
        self.lnorm2 = nn.LayerNorm(d_model)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

    def forward(self, input, enc_output, mask_enc_att):
        enc_att ,_ = self.enc_att(input, enc_output, enc_output, mask_enc_att)
        enc_att = self.lnorm2(input + self.dropout2(enc_att))
        ff = self.pwff(enc_att)
        return ff











