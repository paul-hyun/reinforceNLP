import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


# 참고: https://github.com/jadore801120/attention-is-all-you-need-pytorch
#      https://github.com/JayParks/transformer
#      https://github.com/modudeepnlp/code_implementation/blob/master/codes/transformer/Transformer-Torch.py


def get_sinusoid_encoding_table(n_seq, d_embed):
    def cal_angle(position, i_embed):
        return position / np.power(10000, 2 * (i_embed // 2) / d_embed)
    def get_posi_angle_vec(position):
        return [cal_angle(position, i_embed) for i_embed in range(d_embed)]

    sinusoid_table = np.array([get_posi_angle_vec(i_seq) for i_seq in range(n_seq)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return sinusoid_table


def get_attn_pad_mask(seq_q, seq_k, pad):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(pad).unsqueeze(1).expand(batch_size, len_q, len_k)  # <pad>
    return pad_attn_mask.byte()


def get_attn_subsequent_mask(seq):
    subsequent_mask = torch.ones_like(seq).unsqueeze(-1).expand(seq.size(0), seq.size(1), seq.size(1))
    subsequent_mask = subsequent_mask.triu(diagonal=1) # upper triangular part of a matrix(2-D)
    return subsequent_mask.byte()


"""
Configuration
"""
class Config(dict): 
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


class ScaledDotProductAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.config.d_k)
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        attn = self.dropout(attn)
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.W_Q = nn.Linear(self.config.d_embed, self.config.d_k * self.config.n_heads)
        self.W_K = nn.Linear(self.config.d_embed, self.config.d_k * self.config.n_heads)
        self.W_V = nn.Linear(self.config.d_embed, self.config.d_k * self.config.n_heads)
        self.scaled_dot_attn = ScaledDotProductAttention(self.config)
        self.linear = nn.Linear(self.config.n_heads * self.config.d_v, self.config.d_embed)
        self.layer_norm = nn.LayerNorm(self.config.d_embed)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, Q, K, V, attn_mask):
        residual, batch_size = Q, Q.size(0)

        q_s = self.W_Q(Q).view(batch_size, -1, self.config.n_heads, self.config.d_k).transpose(1,2)
        k_s = self.W_K(K).view(batch_size, -1, self.config.n_heads, self.config.d_k).transpose(1,2)
        v_s = self.W_V(V).view(batch_size, -1, self.config.n_heads, self.config.d_v).transpose(1,2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.config.n_heads, 1, 1)

        context, attn = self.scaled_dot_attn(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.config.n_heads * self.config.d_v)
        output = self.linear(context)
        output = self.dropout(output)
        return self.layer_norm(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.conv1 = nn.Conv1d(in_channels=self.config.d_embed, out_channels=self.config.d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=self.config.d_ff, out_channels=self.config.d_embed, kernel_size=1)
        self.layer_norm = nn.LayerNorm(self.config.d_embed)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, inputs):
        residual = inputs

        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        output = self.dropout(output)
        return self.layer_norm(output + residual)


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.enc_self_attn = MultiHeadAttention(self.config)
        self.pos_ffn = PoswiseFeedForwardNet(self.config)
    
    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, enc_self_attn_mask


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.enc_emb = nn.Embedding(self.config.n_enc_vocab, self.config.d_embed)
        sinusoid_table = torch.tensor(get_sinusoid_encoding_table(self.config.n_enc_seq + 1, self.config.d_embed), dtype=torch.float).to(config.device)
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_table, freeze=True)

        self.layers = nn.ModuleList([EncoderLayer(self.config) for _ in range(self.config.n_layer)])
    
    def forward(self, enc_inputs):
        possitions = torch.cumsum(torch.ones(enc_inputs.size(1), dtype=torch.long), dim=0).to(self.config.device) * (1 - enc_inputs.eq(self.config.i_pad)).to(torch.long)
        enc_outputs = self.enc_emb(enc_inputs) + self.pos_emb(possitions)

        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs, self.config.i_pad)

        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.dec_self_attn = MultiHeadAttention(self.config)
        self.dec_enc_attn = MultiHeadAttention(self.config)
        self.pos_ffn = PoswiseFeedForwardNet(self.config)
    
    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.dec_emb = nn.Embedding(self.config.n_dec_vocab, self.config.d_embed)
        sinusoid_table = torch.tensor(get_sinusoid_encoding_table(self.config.n_dec_seq + 1, self.config.d_embed), dtype=torch.float).to(config.device)
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_table, freeze=True)

        self.layers = nn.ModuleList([DecoderLayer(self.config) for _ in range(self.config.n_layer)])
    
    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        possitions = torch.cumsum(torch.ones(dec_inputs.size(1), dtype=torch.long), dim=0).to(self.config.device) * (1 - dec_inputs.eq(self.config.i_pad)).to(torch.long)
        dec_outputs = self.dec_emb(dec_inputs) + self.pos_emb(possitions)

        dec_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs, self.config.i_pad)
        dec_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)
        dec_self_attn_mask = torch.gt((dec_attn_pad_mask + dec_attn_subsequent_mask), 0)
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs, self.config.i_pad)

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = Encoder(self.config)
        self.decoder = Decoder(self.config)
        self.projection = nn.Linear(self.config.d_embed, self.config.n_dec_vocab, bias=False)
    
    def forward(self, enc_inputs, dec_inputs):
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        dec_logits = self.projection(dec_outputs)
        return dec_logits, enc_self_attns, dec_self_attns, dec_enc_attns


class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

