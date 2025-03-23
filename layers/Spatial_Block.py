import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch





class GraphBlock(nn.Module):
    def __init__(self, c_out, gcn_depth=1, dropout=0.05, propalpha=0.5, node_dim=10):
        super(GraphBlock, self).__init__()

        self.nodevec1 = nn.Parameter(torch.randn(c_out, node_dim), requires_grad=True)
        self.nodevec2 = nn.Parameter(torch.randn(node_dim, c_out), requires_grad=True)
        # self.start_conv = nn.Conv2d(1 , conv_channel, kernel_size=(1, 1))
        self.gconv1 = mixprop(c_out, c_out, gcn_depth, dropout, propalpha)
        self.gelu = nn.GELU()
        # self.end_conv = nn.Conv2d(skip_channel, 1 , (1,1) )
        self.norm = nn.LayerNorm(c_out)

    # x in (B, T, d_model)
    # Here we use a mlp to fit a complex mapping f (x)

    def forward(self, x):   # [Batch, num_nodes, num_periods, period]
        B , N= x.size(0), x.size(1)
        adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        out = x
        # out = self.start_conv(out)
        out = self.gelu(self.gconv1(out , adp))
        # out = self.end_conv(out).squeeze().transpose(1, 2)

        return self.norm((x + out).reshape(B, N, -1).transpose(1, 2))


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('nwcl,vw->nvcl',(x,A))
        # x = torch.einsum('nwc,vw->nvc',(x,A))
        # x = torch.einsum('ncwl,wv->nclv',(x,A)
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out,bias=True):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=bias)
        # self.mlp = torch.nn.Conv1d(c_in, c_out, kernel_size=1, padding=0, stride=1, bias=bias)

    def forward(self,x):
        return self.mlp(x)

class mixprop(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha):
        super(mixprop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear((gdep+1)*c_in,c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x, adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        out = [h]
        a = adj / d.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,a)
            out.append(h)
        ho = torch.cat(out,dim=1)
        ho = self.mlp(ho)
        return ho




