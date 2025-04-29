import torch
from torch import nn
from torch.nn import functional as F
from models.transformer.attention import MultiHeadAttention
from models.transformer.utils import sinusoid_encoding_table, PositionWiseFeedForward
from models.containers import Module, ModuleList
import numpy as np



################# standard-decoder
class DecoderLayer(Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, self_att_module=None,
                 enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
        super(DecoderLayer, self).__init__()
        self.self_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=True,
                                           attention_module=self_att_module,
                                           attention_module_kwargs=self_att_module_kwargs)
        self.enc_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=False,
                                          attention_module=enc_att_module,
                                          attention_module_kwargs=enc_att_module_kwargs)

        self.dropout1 = nn.Dropout(dropout)
        self.lnorm1 = nn.LayerNorm(d_model)

        self.dropout2 = nn.Dropout(dropout)
        self.lnorm2 = nn.LayerNorm(d_model)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

    def forward(self, input, enc_output, mask_pad, mask_self_att, mask_enc_att):
        # MHA+AddNorm
        self_att, _ = self.self_att(input, input, input, mask_self_att)
        self_att = self.lnorm1(input + self.dropout1(self_att))
        self_att = self_att * mask_pad
        # MHA+AddNorm
        enc_att, att = self.enc_att(self_att, enc_output, enc_output, mask_enc_att)
        enc_att = self.lnorm2(self_att + self.dropout2(enc_att))
        enc_att = enc_att * mask_pad
        # FFN+AddNorm
        ff = self.pwff(enc_att)
        ff = ff * mask_pad
        return ff, att


class TransformerDecoderLayer(Module):
    def __init__(self, vocab_size, max_len, N_dec, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 self_att_module=None, enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
        super(TransformerDecoderLayer, self).__init__()
        self.d_model = d_model
        self.word_emb = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len + 1, d_model, 0), freeze=True)
        self.layers = ModuleList(
            [DecoderLayer(d_model, d_k, d_v, h, d_ff, dropout, self_att_module=self_att_module, enc_att_module=enc_att_module, self_att_module_kwargs=self_att_module_kwargs, enc_att_module_kwargs=enc_att_module_kwargs) for _ in range(N_dec)])
        self.fc = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len
        self.padding_idx = padding_idx
        self.N = N_dec

        self.register_state('running_mask_self_attention', torch.zeros((1, 1, 0)).byte())
        self.register_state('running_seq', torch.zeros((1,)).long())

    def forward(self, input, encoder_output, mask_encoder, device):
        # input (b_s, seq_len)
        input = input.to(device)   # 李环宇加
        b_s, seq_len = input.shape[:2]
        mask_queries = (input != self.padding_idx).unsqueeze(-1).float()  # (b_s, seq_len, 1)
        mask_self_attention = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8, device=input.device),
                                         diagonal=1)
        mask_self_attention = mask_self_attention.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        mask_self_attention = mask_self_attention + (input == self.padding_idx).unsqueeze(1).unsqueeze(1).byte()
        mask_self_attention = mask_self_attention.gt(0)  # (b_s, 1, seq_len, seq_len)
        if self._is_stateful:
            self.running_mask_self_attention = torch.cat([self.running_mask_self_attention.type_as(mask_self_attention), mask_self_attention], -1)
            mask_self_attention = self.running_mask_self_attention

        seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(input.device)  # (b_s, seq_len)
        seq = seq.masked_fill(mask_queries.squeeze(-1) == 0, 0)
        if self._is_stateful:
            self.running_seq.add_(1)
            seq = self.running_seq

        out = self.word_emb(input) + self.pos_emb(seq)

        for i, l in enumerate(self.layers):
            out, att = l(out, encoder_output, mask_queries, mask_self_attention, mask_encoder)

        out = self.fc(out)
        return F.log_softmax(out, dim=-1) #zjk


class TransformerDecoderLayer_att_map(Module):
    def __init__(self, vocab_size, max_len, N_dec, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 self_att_module=None, enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
        super(TransformerDecoderLayer_att_map, self).__init__()
        self.d_model = d_model
        self.word_emb = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len + 1, d_model, 0), freeze=True)
        self.layers = ModuleList(
                          [DecoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                          self_att_module=self_att_module,
                          enc_att_module=enc_att_module,
                          self_att_module_kwargs=self_att_module_kwargs,
                          enc_att_module_kwargs=enc_att_module_kwargs) for _ in range(N_dec)])



        self.fc = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len
        self.padding_idx = padding_idx
        self.N = N_dec

        self.register_state('running_mask_self_attention', torch.zeros((1, 1, 0)).byte())
        self.register_state('running_seq', torch.zeros((1,)).long())

    def forward(self, input, encoder_output, mask_encoder, device):
        # input (b_s, seq_len)
        input = input.to(device)   # 李环宇加
        b_s, seq_len = input.shape[:2]
        mask_queries = (input != self.padding_idx).unsqueeze(-1).float()  # (b_s, seq_len, 1)
        mask_self_attention = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8, device=input.device),
                                         diagonal=1)
        mask_self_attention = mask_self_attention.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        mask_self_attention = mask_self_attention + (input == self.padding_idx).unsqueeze(1).unsqueeze(1).byte()
        mask_self_attention = mask_self_attention.gt(0)  # (b_s, 1, seq_len, seq_len)
        if self._is_stateful:
            self.running_mask_self_attention = torch.cat([self.running_mask_self_attention.type_as(mask_self_attention), mask_self_attention], -1)
            mask_self_attention = self.running_mask_self_attention

        seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(input.device)  # (b_s, seq_len)
        seq = seq.masked_fill(mask_queries.squeeze(-1) == 0, 0)
        if self._is_stateful:
            self.running_seq.add_(1)
            seq = self.running_seq

        out = self.word_emb(input) + self.pos_emb(seq)

        attention_weights = []  #创建一个注意力权重矩阵

        for i, l in enumerate(self.layers):
            out, att = l(out, encoder_output, mask_queries, mask_self_attention, mask_encoder)
            if i == self.N - 1:  # 如果是最后一层
                attention_weights = att
                print(att.shape)
                #attention_weights.append(att)  # 保存注意力权重

        out = self.fc(out)

        return F.log_softmax(out, dim=-1), attention_weights  # 返回注意力权重

class MeshedDecoderLayer(Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, n_dec=3,self_att_module=None,
                 enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
        super(MeshedDecoderLayer, self).__init__()
        self.self_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=True,
                                           attention_module=self_att_module,
                                           attention_module_kwargs=self_att_module_kwargs)
        self.enc_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=False,
                                          attention_module=enc_att_module,
                                          attention_module_kwargs=enc_att_module_kwargs)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.n_dec = n_dec


        if  self.n_dec == 1:
            self.fc_alpha1 = nn.Linear(d_model + d_model, d_model)
            self.init_weights()

        if self.n_dec == 3:
            self.fc_alpha1 = nn.Linear(d_model + d_model, d_model)
            self.fc_alpha2 = nn.Linear(d_model + d_model, d_model)
            self.fc_alpha3 = nn.Linear(d_model + d_model, d_model)
            self.init_weights()

        if self.n_dec == 5:
            self.fc_alpha1 = nn.Linear(d_model + d_model, d_model)
            self.fc_alpha2 = nn.Linear(d_model + d_model, d_model)
            self.fc_alpha3 = nn.Linear(d_model + d_model, d_model)
            self.fc_alpha4 = nn.Linear(d_model + d_model, d_model)
            self.fc_alpha5 = nn.Linear(d_model + d_model, d_model)
            self.init_weights()

        if self.n_dec == 7:
            self.fc_alpha1 = nn.Linear(d_model + d_model, d_model)
            self.fc_alpha2 = nn.Linear(d_model + d_model, d_model)
            self.fc_alpha3 = nn.Linear(d_model + d_model, d_model)
            self.fc_alpha4 = nn.Linear(d_model + d_model, d_model)
            self.fc_alpha5 = nn.Linear(d_model + d_model, d_model)
            self.fc_alpha6 = nn.Linear(d_model + d_model, d_model)
            self.fc_alpha7 = nn.Linear(d_model + d_model, d_model)
            self.init_weights()

    def init_weights(self):
        if self.n_dec == 1:
            nn.init.xavier_uniform_(self.fc_alpha1.weight)
            nn.init.constant_(self.fc_alpha1.bias, 0)
        if self.n_dec == 3:
            nn.init.xavier_uniform_(self.fc_alpha1.weight)
            nn.init.xavier_uniform_(self.fc_alpha2.weight)
            nn.init.xavier_uniform_(self.fc_alpha3.weight)
            nn.init.constant_(self.fc_alpha1.bias, 0)
            nn.init.constant_(self.fc_alpha2.bias, 0)
            nn.init.constant_(self.fc_alpha3.bias, 0)
        if self.n_dec == 5:
            nn.init.xavier_uniform_(self.fc_alpha1.weight)
            nn.init.xavier_uniform_(self.fc_alpha2.weight)
            nn.init.xavier_uniform_(self.fc_alpha3.weight)
            nn.init.xavier_uniform_(self.fc_alpha4.weight)
            nn.init.xavier_uniform_(self.fc_alpha5.weight)
            nn.init.constant_(self.fc_alpha1.bias, 0)
            nn.init.constant_(self.fc_alpha2.bias, 0)
            nn.init.constant_(self.fc_alpha3.bias, 0)
            nn.init.constant_(self.fc_alpha4.bias, 0)
            nn.init.constant_(self.fc_alpha5.bias, 0)
        if self.n_dec == 7:
            nn.init.xavier_uniform_(self.fc_alpha1.weight)
            nn.init.xavier_uniform_(self.fc_alpha2.weight)
            nn.init.xavier_uniform_(self.fc_alpha3.weight)
            nn.init.xavier_uniform_(self.fc_alpha4.weight)
            nn.init.xavier_uniform_(self.fc_alpha5.weight)
            nn.init.xavier_uniform_(self.fc_alpha6.weight)
            nn.init.xavier_uniform_(self.fc_alpha7.weight)
            nn.init.constant_(self.fc_alpha1.bias, 0)
            nn.init.constant_(self.fc_alpha2.bias, 0)
            nn.init.constant_(self.fc_alpha3.bias, 0)
            nn.init.constant_(self.fc_alpha4.bias, 0)
            nn.init.constant_(self.fc_alpha5.bias, 0)
            nn.init.constant_(self.fc_alpha6.bias, 0)
            nn.init.constant_(self.fc_alpha7.bias, 0)

    def forward(self, input, enc_output, mask_pad, mask_self_att, mask_enc_att):
        self_att,_ = self.self_att(input, input, input, mask_self_att)
        self_att = self_att * mask_pad

        if self.n_dec == 1:
            enc_att1,_ = self.enc_att(self_att, enc_output[:, 0], enc_output[:, 0], mask_enc_att)
            enc_att1 = enc_att1 * mask_pad

            alpha1 = torch.sigmoid(self.fc_alpha1(torch.cat([self_att, enc_att1], -1)))
            enc_att = (enc_att1 * alpha1) / np.sqrt(1)
            enc_att = enc_att * mask_pad

        if self.n_dec == 3:
            enc_att1,_ = self.enc_att(self_att, enc_output[:, 0], enc_output[:, 0], mask_enc_att)
            enc_att1 = enc_att1 * mask_pad
            enc_att2,_ = self.enc_att(self_att, enc_output[:, 1], enc_output[:, 1], mask_enc_att)
            enc_att2 = enc_att2 * mask_pad
            enc_att3,_ = self.enc_att(self_att, enc_output[:, 2], enc_output[:, 2], mask_enc_att)
            enc_att3 = enc_att3 * mask_pad

            alpha1 = torch.sigmoid(self.fc_alpha1(torch.cat([self_att, enc_att1], -1)))
            alpha2 = torch.sigmoid(self.fc_alpha2(torch.cat([self_att, enc_att2], -1)))
            alpha3 = torch.sigmoid(self.fc_alpha3(torch.cat([self_att, enc_att3], -1)))

            enc_att = (enc_att1 * alpha1 + enc_att2 * alpha2 + enc_att3 * alpha3) / np.sqrt(3)
            enc_att = enc_att * mask_pad

        if self.n_dec == 5:
            enc_att1,_ = self.enc_att(self_att, enc_output[:, 0], enc_output[:, 0], mask_enc_att)
            enc_att1 = enc_att1 * mask_pad
            enc_att2,_ = self.enc_att(self_att, enc_output[:, 1], enc_output[:, 1], mask_enc_att)
            enc_att2 = enc_att2 * mask_pad
            enc_att3,_ = self.enc_att(self_att, enc_output[:, 2], enc_output[:, 2], mask_enc_att)
            enc_att3 = enc_att3 * mask_pad
            enc_att4,_ = self.enc_att(self_att, enc_output[:, 3], enc_output[:, 3], mask_enc_att)
            enc_att4 = enc_att4 * mask_pad
            enc_att5,_ = self.enc_att(self_att, enc_output[:, 4], enc_output[:, 4], mask_enc_att)
            enc_att5 = enc_att5 * mask_pad

            alpha1 = torch.sigmoid(self.fc_alpha1(torch.cat([self_att, enc_att1], -1)))
            alpha2 = torch.sigmoid(self.fc_alpha2(torch.cat([self_att, enc_att2], -1)))
            alpha3 = torch.sigmoid(self.fc_alpha3(torch.cat([self_att, enc_att3], -1)))
            alpha4 = torch.sigmoid(self.fc_alpha4(torch.cat([self_att, enc_att4], -1)))
            alpha5 = torch.sigmoid(self.fc_alpha5(torch.cat([self_att, enc_att5], -1)))

            enc_att = (enc_att1 * alpha1 + enc_att2 * alpha2 + enc_att3 * alpha3+ enc_att4 * alpha4+ enc_att5 * alpha5) / np.sqrt(5)
            enc_att = enc_att * mask_pad

        if self.n_dec == 7:
            enc_att1,_ = self.enc_att(self_att, enc_output[:, 0], enc_output[:, 0], mask_enc_att)
            enc_att1 = enc_att1 * mask_pad
            enc_att2,_ = self.enc_att(self_att, enc_output[:, 1], enc_output[:, 1], mask_enc_att)
            enc_att2 = enc_att2 * mask_pad
            enc_att3,_ = self.enc_att(self_att, enc_output[:, 2], enc_output[:, 2], mask_enc_att)
            enc_att3 = enc_att3 * mask_pad
            enc_att4,_ = self.enc_att(self_att, enc_output[:, 3], enc_output[:, 3], mask_enc_att)
            enc_att4 = enc_att4 * mask_pad
            enc_att5,_ = self.enc_att(self_att, enc_output[:, 4], enc_output[:, 4], mask_enc_att)
            enc_att5 = enc_att5 * mask_pad
            enc_att6,_ = self.enc_att(self_att, enc_output[:, 5], enc_output[:, 5], mask_enc_att)
            enc_att6 = enc_att6 * mask_pad
            enc_att7,_ = self.enc_att(self_att, enc_output[:, 6], enc_output[:, 6], mask_enc_att)
            enc_att7 = enc_att7 * mask_pad


            alpha1 = torch.sigmoid(self.fc_alpha1(torch.cat([self_att, enc_att1], -1)))
            alpha2 = torch.sigmoid(self.fc_alpha2(torch.cat([self_att, enc_att2], -1)))
            alpha3 = torch.sigmoid(self.fc_alpha3(torch.cat([self_att, enc_att3], -1)))
            alpha4 = torch.sigmoid(self.fc_alpha4(torch.cat([self_att, enc_att4], -1)))
            alpha5 = torch.sigmoid(self.fc_alpha5(torch.cat([self_att, enc_att5], -1)))
            alpha6 = torch.sigmoid(self.fc_alpha6(torch.cat([self_att, enc_att6], -1)))
            alpha7 = torch.sigmoid(self.fc_alpha7(torch.cat([self_att, enc_att7], -1)))

            enc_att = (  enc_att1 * alpha1
                       + enc_att2 * alpha2
                       + enc_att3 * alpha3
                       + enc_att4 * alpha4
                       + enc_att5 * alpha5
                       + enc_att6 * alpha6
                       + enc_att7 * alpha7) / np.sqrt(7)
            enc_att = enc_att * mask_pad

        ff = self.pwff(enc_att)
        ff = ff * mask_pad
        return ff



class MeshedDecoder(Module):
    def __init__(self, vocab_size, max_len, N_dec, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 self_att_module=None, enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
        super(MeshedDecoder, self).__init__()
        self.d_model = d_model
        self.N = N_dec
        self.word_emb = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len + 1, d_model, 0), freeze=True)
        self.layers = ModuleList(
            [MeshedDecoderLayer(d_model, d_k, d_v, h, d_ff, dropout, n_dec=self.N, self_att_module=self_att_module,
                                enc_att_module=enc_att_module, self_att_module_kwargs=self_att_module_kwargs,
                                enc_att_module_kwargs=enc_att_module_kwargs) for _ in range(self.N)])
        self.fc = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len
        self.padding_idx = padding_idx

        #self.register_state('running_mask_self_attention', torch.zeros((1, 1, 0)).byte())
        self.register_state('running_mask_self_attention', torch.zeros((1, 1, 0)).bool())
        self.register_state('running_seq', torch.zeros((1,)).long())

    def forward(self, input, encoder_output, mask_encoder, device):
        # input (b_s, seq_len)
        input = input.to(device)   # 李环宇加
        b_s, seq_len = input.shape[:2]
        mask_queries = (input != self.padding_idx).unsqueeze(-1).float()  # (b_s, seq_len, 1)
        mask_self_attention = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8, device=input.device), diagonal=1)
        mask_self_attention = mask_self_attention.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

        #mask_self_attention = mask_self_attention + (input == self.padding_idx).unsqueeze(1).unsqueeze(1).byte()
        mask_self_attention = mask_self_attention + (input == self.padding_idx).unsqueeze(1).unsqueeze(1).bool()

        mask_self_attention = mask_self_attention.gt(0)  # (b_s, 1, seq_len, seq_len)

        if self._is_stateful:
            self.running_mask_self_attention = torch.cat([self.running_mask_self_attention, mask_self_attention], -1)
            mask_self_attention = self.running_mask_self_attention

        seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(input.device)  # (b_s, seq_len)

        seq = seq.masked_fill(mask_queries.squeeze(-1) == 0, 0)

        if self._is_stateful:
            self.running_seq.add_(1)
            seq = self.running_seq

        out = self.word_emb(input) + self.pos_emb(seq)

        for i, l in enumerate(self.layers):
            out = l(out, encoder_output, mask_queries, mask_self_attention, mask_encoder)

        out = self.fc(out)

        return F.log_softmax(out, dim=-1)


# -------------------------------
# 基于PTSM的文本融合策略
# -------------------------------
class PTSMDecoderLayer(Module):
    def __init__(self, vocab_size, max_len, N_dec, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 self_att_module=None, enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
        super(TransformerDecoderLayer, self).__init__()
        self.d_model = d_model
        # -------------------------------
        # 加载clip_embeds的嵌入字典
        # -------------------------------
        self.word_emb = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        weight = torch.load('./word_embeds.pth')['clip_embeds'].float()
        self.word_emb = nn.Embedding.from_pretrained(weight, freeze=False)
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len + 1, d_model, 0), freeze=False)


        self.layers = ModuleList(
            [DecoderLayer(d_model, d_k, d_v, h, d_ff, dropout, self_att_module=self_att_module, enc_att_module=enc_att_module, self_att_module_kwargs=self_att_module_kwargs, enc_att_module_kwargs=enc_att_module_kwargs) for _ in range(N_dec)])
        self.fc = nn.Linear(d_model, vocab_size, bias=False)
        self.fc.weight.data = weight
        self.max_len = max_len
        self.padding_idx = padding_idx
        self.N = N_dec

        self.register_state('running_mask_self_attention', torch.zeros((1, 1, 0)).byte())
        self.register_state('running_seq', torch.zeros((1,)).long())

    def forward(self, input, encoder_output, mask_encoder, device):
        # input (b_s, seq_len)
        input = input.to(device)   # 李环宇加
        b_s, seq_len = input.shape[:2]
        mask_queries = (input != self.padding_idx).unsqueeze(-1).float()  # (b_s, seq_len, 1)
        mask_self_attention = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8, device=input.device),
                                         diagonal=1)
        mask_self_attention = mask_self_attention.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        mask_self_attention = mask_self_attention + (input == self.padding_idx).unsqueeze(1).unsqueeze(1).byte()
        mask_self_attention = mask_self_attention.gt(0)  # (b_s, 1, seq_len, seq_len)
        if self._is_stateful:
            self.running_mask_self_attention = torch.cat([self.running_mask_self_attention.type_as(mask_self_attention), mask_self_attention], -1)
            mask_self_attention = self.running_mask_self_attention

        seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(input.device)  # (b_s, seq_len)
        seq = seq.masked_fill(mask_queries.squeeze(-1) == 0, 0)
        if self._is_stateful:
            self.running_seq.add_(1)
            seq = self.running_seq

        out = self.word_emb(input) + self.pos_emb(seq)
        for i, l in enumerate(self.layers):
            out, att = l(out, encoder_output, mask_queries, mask_self_attention, mask_encoder)

        out = self.fc(out)
        return F.log_softmax(out, dim=-1)
