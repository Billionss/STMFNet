import torch 
import torch.nn as nn
import torch.nn.functional as F


class IndependentLinear(nn.Module):
    def __init__(self, num_features, input_dim, output_dim):
        super(IndependentLinear, self).__init__()
        self.num_features = num_features
        self.linears = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(num_features)])

    def forward(self, x):
        outputs = []
        for i in range(self.num_features):
            outputs.append(self.linears[i](x[:, i, :]))
        return torch.stack(outputs, dim=1)

class DecouplingModule(nn.Module):
    def __init__(self, nodes, channels, dims):
        super(DecouplingModule, self).__init__()
        self.nodes = nodes
        self.channels = channels
        self.dims = dims

        # Independent linear layers for modality-specific gating
        self.gateA = nn.Linear(dims, dims)
        self.gateB = nn.Linear(dims, dims)
        self.gateC = nn.Linear(dims, dims)
        # self.gateA = IndependentLinear(nodes, dims, dims)
        # self.gateB = IndependentLinear(channels, dims, dims)
        # self.gateC = IndependentLinear(nodes+channels, dims, dims)

    def forward(self, modalA, modalB):
        # Ensure modalA shape is [batch, nodes, dims] and modalB shape is [batch, channels, dims]
        assert modalA.shape[1:] == (self.nodes, self.dims)
        assert modalB.shape[1:] == (self.channels, self.dims)

        # Calculate the specific features for modalA
        gateA = torch.sigmoid(self.gateA(modalA))
        specificA = modalA * gateA

        # Calculate the specific features for modalB
        gateB = torch.sigmoid(self.gateB(modalB))
        specificB = modalB * gateB

        # Calculate the common features
        commonA = modalA - specificA
        commonB = modalB - specificB

        # Concatenate the common features along the nodes and channels dimension
        common = torch.cat((commonA, commonB), dim=1)
        gateC = torch.sigmoid(self.gateC(common))
        common = common * gateC

        return specificA, specificB, common

class SelfHierarchicalAttention(nn.Module):
    def __init__(self, dim, num_heads=2):
        super(SelfHierarchicalAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)

    def forward(self, x):
        N = x.shape[0]
        qlen, klen = x.shape[1], x.shape[1]

        query = self.query(x)
        key = self.key(x)

        query = query.view(N, qlen, self.num_heads, self.head_dim)
        key = key.view(N, klen, self.num_heads, self.head_dim)

        attn = torch.mean( torch.einsum('nqhd,nkhd->nhqk', query, key), dim=1)

        weighted = torch.mean( F.softmax(attn / (self.dim ** (1/2)  ), dim=-1), dim=0)

        return weighted

class CrossHierarchicalAttention(nn.Module):
    def __init__(self, dim_x, dim_y, embed_dim, num_heads=2):
        super(CrossHierarchicalAttention, self).__init__()
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query_x = nn.Linear(embed_dim, embed_dim)
        self.key_x = nn.Linear(embed_dim, embed_dim)

        self.query_y = nn.Linear(embed_dim, embed_dim)
        self.key_y = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, y):

        # [N, len_x, embed_dim]
        # [N, len_y, embed_dim]
        N = x.shape[0]

        qlen_x , klen_x = x.shape[1], x.shape[1]
        qlen_y, klen_y = y.shape[1], y.shape[1]

        query_x = self.query_x(x)
        key_y = self.key_y(y)

        # [N, len_x, num_heads, head_dim]
        query_x = query_x.view(N, qlen_x, self.num_heads, self.head_dim) #
        key_y = key_y.view(N, klen_y, self.num_heads, self.head_dim)

        attn = torch.mean( torch.einsum('nqhd,nkhd->nhqk', query_x, key_y), dim=1)

        weighted = torch.mean(F.softmax(attn / (self.embed_dim ** (1/2)  ), dim=-1), dim=0)

        return weighted


class MultiModalAttentionGraph(nn.Module):
    def __init__(self, dim_x, dim_y, embed_dim, num_heads=4):
        super(MultiModalAttentionGraph, self).__init__()
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.query_modalA = nn.Linear(embed_dim, embed_dim)
        self.key_modalA = nn.Linear(embed_dim, embed_dim)
        
        self.query_modalB = nn.Linear(embed_dim, embed_dim)
        self.key_modalB = nn.Linear(embed_dim, embed_dim)

        
    def forward(self, modalA, modalB):
        
        # [N, len_a, embed_dim]
        # [N, len_b, embed_dim]
        N = modalA.shape[0]

        qlen_A , klen_A= modalA.shape[1], modalA.shape[1]
        qlen_B, klen_B = modalB.shape[1], modalB.shape[1]
        
        query_A = self.query_modalA(modalA)
        key_A = self.key_modalA(modalA)
        
        query_B = self.query_modalB(modalB)
        key_B = self.key_modalB(modalB)
        
        # [N, len_a, num_heads, head_dim]
        query_A = query_A.view(N, qlen_A, self.num_heads, self.head_dim)
        key_A = key_A.view(N, klen_A, self.num_heads, self.head_dim)
        
        query_B = query_B.view(N, qlen_B, self.num_heads, self.head_dim)
        key_B = key_B.view(N, klen_B, self.num_heads, self.head_dim)
        
        # Self Modal Attention 
        attn_A = torch.mean( torch.einsum('nqhd,nkhd->nhqk', query_A, key_A), dim=1) 
        attn_B = torch.mean( torch.einsum('nqhd,nkhd->nhqk', query_B, key_B), dim=1) 
        
        weighted_A = F.softmax(attn_A / (self.embed_dim ** (1/2)  ), dim=-1)
        weighted_B = F.softmax(attn_B / (self.embed_dim ** (1/2)  ), dim=-1)
        
        # Cross Modal Attention
        attn_AB = torch.mean( torch.einsum('nqhd,nkhd->nhqk', query_A, key_B), dim=1) 
        attn_BA = torch.mean( torch.einsum('nqhd,nkhd->nhqk', query_B, key_A), dim=1) 
        
        weighted_AB = F.softmax(attn_AB / (self.embed_dim ** (1/2)  ), dim=-1)
        weighted_BA = F.softmax(attn_BA / (self.embed_dim ** (1/2)  ), dim=-1)
        
        cat_upper = torch.cat([weighted_B, weighted_BA], dim=-1)  # [34, 34] [34, 64] --> [34, 98]
        cat_lower = torch.cat([weighted_A, weighted_AB], dim=-1)  # [64, 64] [64, 34] --> [64, 98]
        
        cat = torch.mean(  torch.cat([cat_lower, cat_upper], dim=1),  dim=0 )
        return cat
    


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('nwd,wv->nvd',(x,A))
        # x = torch.einsum('ncwl,wv->nclv',(x,A)
        return x.contiguous()

class gcn(nn.Module):
    def __init__(self,c_in,c_out,gdep=2,dropout=0.05):
        super(gcn, self).__init__()
        self.nconv = nconv()

        self.mlp = nn.Linear( gdep * c_in, c_out)
        self.gdep = gdep
        self.dropout = dropout

    def forward(self, x, adj):
        h = x
        out = []
        for i in range(self.gdep):
            h = self.nconv(x, adj)
            out.append(h)
        ho = torch.cat(out,dim=1)
        ho = self.mlp(ho.transpose(1, 2)).transpose(1,2)
        return ho


class ModalFusion(nn.Module):
    def __init__(self, dim ):
        super(ModalFusion, self).__init__()
        self.gcn = gcn(dim, dim)
    
    def forward(self, modalA, modalB, graph):
        cat = torch.cat([modalA, modalB], dim=1)
        out = self.gcn(cat, graph)
        return out

class HierarchicalConv(nn.Module):
    def __init__(self, dim ):
        super(HierarchicalConv, self).__init__()
        self.gcn = gcn(dim, dim)

    def forward(self, modal, graph):
        out = self.gcn(modal, graph)
        return out

class ModalityFusion(nn.Module):
    def __init__(self, dima, dimb, embed_dim):
        super(ModalityFusion, self).__init__()

        fusion_dim = dima + dimb

        self.decouple = DecouplingModule(nodes=dima, channels=dimb, dims=embed_dim)

        self.sha_modalS = SelfHierarchicalAttention(dim=embed_dim)
        self.sha_modalR = SelfHierarchicalAttention(dim=embed_dim)
        self.sha_modalI = SelfHierarchicalAttention(dim=embed_dim)

        self.cha_modalSI = CrossHierarchicalAttention(dim_x=dima, dim_y=fusion_dim, embed_dim=embed_dim)
        self.cha_modalRI = CrossHierarchicalAttention(dim_x=dimb, dim_y=fusion_dim, embed_dim=embed_dim)

        self.shgcn_ms = HierarchicalConv(dim=dima)

        self.shgcn_mr = HierarchicalConv(dim=dimb)
        self.shgcn_mi = HierarchicalConv(dim=fusion_dim)

        self.chgcn_si = HierarchicalConv(dim=fusion_dim)
        self.chgcn_ri = HierarchicalConv(dim=fusion_dim)

        self.mlp = nn.Linear(fusion_dim*3, fusion_dim)

    def forward(self, modalA, modalB):

        out = []

        spec_ms, spec_mr, common = self.decouple(modalA, modalB)

        graph_ms = self.sha_modalS(spec_ms)
        graph_mr = self.sha_modalR(spec_mr)
        graph_mi = self.sha_modalI(common)

        graph_si = self.cha_modalSI(spec_ms, common)
        graph_ri = self.cha_modalRI(spec_mr, common)

        spec_ms = self.shgcn_ms(spec_ms, graph_ms)
        spec_mr = self.shgcn_mr(spec_mr, graph_mr)
        common = self.shgcn_mi(common, graph_mi)
        out.append(common)

        si_out = self.chgcn_si(spec_ms, graph_si)
        out.append(si_out)

        ri_out = self.chgcn_ri(spec_mr, graph_ri)
        out.append(ri_out)

        out = torch.cat(out, dim=1)
        out = self.mlp(out.transpose(1, 2)).transpose(1, 2)

        return out


class ModalityFusion_1(nn.Module):
    def __init__(self, dima, dimb, embed_dim):
        super(ModalityFusion_1, self).__init__()

        fusion_dim = dima + dimb

        self.decouple = DecouplingModule(nodes=dima, channels=dimb, dims=embed_dim)

        self.sha_modal = SelfHierarchicalAttention(dim=embed_dim)


        self.cha_modalSI = CrossHierarchicalAttention(dim_x=dima, dim_y=fusion_dim, embed_dim=embed_dim)
        self.cha_modalRI = CrossHierarchicalAttention(dim_x=dimb, dim_y=fusion_dim, embed_dim=embed_dim)

        self.shgcn_ms = HierarchicalConv(dim=dima)
        self.shgcn_mr = HierarchicalConv(dim=dimb)
        # self.shgcn_mi = HierarchicalConv(dim=fusion_dim)

        self.chgcn = HierarchicalConv(dim=fusion_dim)

        self.mlp = nn.Linear(fusion_dim*3, fusion_dim)

        self.mlp_v = nn.Linear(fusion_dim*2, fusion_dim)



    def forward(self, modalA, modalB):

        out = []

        spec_ms, spec_mr, common = self.decouple(modalA, modalB)

        graph_ms = self.sha_modal(spec_ms)
        graph_mr = self.sha_modal(spec_mr)
        graph_mi = self.sha_modal(common)

        graph_si = self.cha_modalSI(spec_ms, common)
        graph_ri = self.cha_modalRI(spec_mr, common)

        spec_ms = self.shgcn_ms(spec_ms, graph_ms)
        spec_mr = self.shgcn_mr(spec_mr, graph_mr)
        common = self.chgcn(common, graph_mi)
        out.append(common)

        si_out = self.chgcn(spec_ms, graph_si)
        out.append(si_out)

        ri_out = self.chgcn(spec_mr, graph_ri)
        out.append(ri_out)

        out = torch.cat(out, dim=1)
        out = self.mlp(out.transpose(1, 2)).transpose(1, 2)

        # out = torch.cat([common, torch.cat([spec_ms, spec_mr], dim=1)], dim=1)
        # out = self.mlp_v(out.transpose(1, 2)).transpose(1, 2)

        return out










        return


if __name__ == "__main__":
    a = torch.rand([64, 34, 96])
    b = torch.rand([64, 2, 96])

    decouple = DecouplingModule(nodes=34, channels=2, dims=96)

    specific_ms, specific_mr, common = decouple(a, b)

    print(specific_ms.shape)
    print(specific_mr.shape)
    print(common.shape)

    sha_modalS = SelfHierarchicalAttention(dim=96)
    graph_ms = sha_modalS(specific_ms)

    sha_modalR = SelfHierarchicalAttention(dim=96)
    graph_mr = sha_modalR(specific_mr)

    sha_modalI = SelfHierarchicalAttention(dim=96)
    graph_mi = sha_modalI(common)

    cha_modalSI = CrossHierarchicalAttention(dim_x=34, dim_y=36, embed_dim=96)
    graph_si = cha_modalSI(specific_ms, common)

    cha_modalRI = CrossHierarchicalAttention(dim_x=2, dim_y=36, embed_dim=96)
    graph_ri = cha_modalRI(specific_mr, common)

    print(graph_ms.shape)
    print(graph_mr.shape)
    print(graph_mi.shape)

    print(graph_si.shape)
    print(graph_ri.shape)

    shgcn_ms = HierarchicalConv(dim=34)
    ms_out = shgcn_ms(specific_ms, graph_ms)

    shgcn_mr = HierarchicalConv(dim=2)
    mr_out = shgcn_mr(specific_mr, graph_mr)

    shgcn_mi = HierarchicalConv(dim=36)
    mi_out = shgcn_mi(common, graph_mi)

    chgcn_si = HierarchicalConv(dim=36)
    si_out = chgcn_si(specific_ms, graph_si)

    chgcn_ri = HierarchicalConv(dim=36)
    ri_out = chgcn_ri(specific_mr, graph_ri)

    print(ms_out.shape)
    print(mr_out.shape)
    print(mi_out.shape)
    print(si_out.shape)
    print(ri_out.shape)




    


