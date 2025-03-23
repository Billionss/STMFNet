import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.PeriodTS import TimesBlock
from layers.BiFPN import BiStreamFPN, MultiScaleFusion, Conv
from layers.CrossModalFusion import MultiModalAttentionGraph, ModalFusion
from layers.CrossModalFusion import ModalityFusion_1

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        # print(x.permute(0, 2, 1))
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        fusion_dim = configs.num_channels + configs.num_nodes

        # TimesBlock with seasonal and trend decomposition
        self.visual_graph = configs.visual_graph
        kernel_size = 25
        self.seq_len = configs.seq_len
        self.pred_len = configs.label_len
        self.channels = configs.num_nodes
        self.individual = configs.individual

        self.decomposition = series_decomp(kernel_size)
        self.seasonal_block = nn.ModuleList([TimesBlock(configs) for _ in range(configs.ts_layers)])

        if self.individual:
            self.trend_block = nn.ModuleList()
            for i in range(self.channels):
                self.trend_block.append(nn.Linear(self.seq_len,self.seq_len))
        else:
            self.trend_block = nn.Linear(self.seq_len,self.seq_len)

        self.layers= configs.ts_layers

        self.rs_block = Conv(configs.seq_len // 24, configs.num_channels, configs.embed_dim)
        # self.rs_block = BiStreamFPN(configs.seq_len // 24 , configs.num_channels)
        # self.scale_fusion = MultiScaleFusion()

        # Fusion Block
        self.fusion = ModalityFusion_1(configs.num_nodes, configs.num_channels, configs.embed_dim)

        # self.multi_modal_graph = MultiModalAttentionGraph(configs.num_channels, configs.num_nodes, configs.embed_dim)
        # self.modal_fusion = ModalFusion(dim=fusion_dim)

        # linear
        self.ts_linear = nn.Linear(configs.seq_len, configs.embed_dim)
        self.rs_linear = nn.Linear(196, configs.embed_dim)

        # output projection
        self.proj1 = nn.Linear(configs.embed_dim, configs.seq_len)
        self.proj2 = nn.Conv1d(fusion_dim, configs.num_nodes, kernel_size=1, stride=1, padding=0)

        self.out_linear1 = nn.Conv1d(configs.seq_len, configs.seq_len, kernel_size=1, stride=1, padding=0)
        self.out_linear2 = nn.Conv1d(configs.seq_len, configs.label_len, kernel_size=1, stride=1, padding=0)

    def forward(self, data_ts, data_rs):


        seasonal, trend = self.decomposition(data_ts) # [batch, seq_len, num_nodes]
        for i in range(self.layers):
            seasonal = self.seasonal_block[i](seasonal)
            # trend = self.trend_block[i](trend)
        trend = trend.permute(0, 2, 1)
        if self.individual:
            trend_output = torch.zeros([trend.size(0),trend.size(1),self.seq_len],dtype=trend.dtype).to(trend.device)
            for i in range(self.channels):
                trend_output[:,i,:] = self.trend_block[i](trend[:,i,:])
        else:
            trend_output = self.trend_block(trend)

        ts_out = seasonal + trend_output.permute(0, 2, 1)  # [batch, seq_len, num_nodes]

        # for fusion block
        ts_fusion = self.ts_linear(ts_out.transpose(1, 2))

        rs_out = self.rs_block(data_rs)

        fusion_out = self.fusion(ts_fusion, rs_out)

        res_out = self.proj1(fusion_out)
        res_out = F.relu(res_out)
        res_out = self.proj2(res_out).transpose(1, 2)

        # add the residual link    waiting for test
        out = ts_out + res_out
        out = self.out_linear1(out)
        out = F.relu(out)
        out = self.out_linear2(out)

        # if self.visual_graph and not self.training:
        #     return out, modal_graph
        # else:
        return out


if __name__ == "__main__":
    class Configs:
        def __init__(self):
            self.seq_len = 96
            self.top_k = 5
            self.num_nodes = 34
            self.num_channels = 64
            self.embed_dim = 256
            self.pred_len = 48

    configs = Configs()
    model = model(configs)

    ts = torch.rand([32, 96, 34])
    rs = torch.rand([32, 3, 1, 132, 209])

    out = model(ts, rs)

    print(out.shape)