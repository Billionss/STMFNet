import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.PeriodTS import TimesBlock


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
        # TimesBlock with seasonal and trend decomposition
        kernel_size = 13
        self.decomposition = series_decomp(kernel_size)
        self.sea_block = nn.ModuleList([TimesBlock(configs) for _ in range(configs.ts_layers)])
        # self.sea_block = TimesBlock(configs)
        self.trend_block = nn.ModuleList([TimesBlock(configs) for _ in range(configs.ts_layers)])
        self.layers= configs.ts_layers

        self.out_linear1 = nn.Conv1d(configs.seq_len, configs.seq_len, kernel_size=1, stride=1, padding=0)
        self.out_linear2 = nn.Conv1d(configs.seq_len, configs.label_len, kernel_size=1, stride=1, padding=0)

    def forward(self, data_ts, data_rs):

        seasonal, trend = self.decomposition(data_ts)
        for i in range(self.layers):
            seasonal = self.sea_block[i](seasonal)
            trend = self.trend_block[i](trend)


        ts_out = seasonal + trend

        out = self.out_linear1(ts_out)
        out = F.relu(out)
        out = self.out_linear2(out)
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