import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import argparse
from layers.Embed import DataEmbedding
from layers.Spatial_Block import GraphBlock
from layers.Temporal_Block import Attention_Block



def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]

class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.k = configs.top_k
        
        self.conv = nn.Sequential(
            GraphBlock(configs.num_nodes),
            nn.GELU(),
            Attention_Block(configs.num_nodes)
            # temporal blocks and spatial blocks
            # get [batch, stations, nums, periods]
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)
        # print(period_list)
        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len) % period != 0:
                length = (((self.seq_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len
                out = x
            # reshape into # [Batch, num_periods, period, num_nodes]
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            # out = out.permute(0, 2, 1)
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            
            # reshape back
            out = out.reshape(B, -1, N)
            res.append(out[:, :self.seq_len, :])
     
        res = torch.stack(res, dim=-1)
        
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res

