from math import sqrt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.masking import TriangularCausalMask

'''

attention_block = Attention_Block(d_model=34, n_heads=8, dropout=0.1, activation="relu")
x = torch.randn(64, 96, 34)
output = attention_block(x)
print(output.shape)

'''


class Attention_Block(nn.Module):
    def __init__(self, d_model, d_ff=None, n_heads=4, dropout=0.05, activation="relu"):
        super(Attention_Block, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = self_attention(FullAttention, d_model, n_heads=n_heads)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        # x = torch.squeeze(x).transpose(1, 2)
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y)

class self_attention(nn.Module):
    def __init__(self, attention, d_model ,n_heads):
        super(self_attention, self).__init__()
        d_keys =  d_model // n_heads
        d_values = d_model // n_heads

        self.inner_attention = attention( attention_dropout = 0.1)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads


    def forward(self, queries ,keys ,values, attn_mask= None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
                    queries,
                    keys,
                    values,
                    attn_mask
                )
        out = out.view(B, L, -1)
        out = self.out_projection(out)
        return out , attn


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        # return V.contiguous()
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

