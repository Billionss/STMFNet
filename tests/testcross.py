import torch 
import torch.nn as nn
import torch.nn.functional as F


class MultiModalAttentionFusion(nn.Module):
    def __init__(self, dim_x, dim_y, embed_dim, num_heads=4):
        super(MultiModalAttentionFusion, self).__init__()
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
        
        cat = torch.cat([cat_lower, cat_upper], dim=1)
        return cat
    
if __name__ == "__main__":
    a = torch.rand([64, 64, 96])
    b = torch.rand([64, 34, 96])

    fusion = MultiModalAttentionFusion(dim_x=64, dim_y=34, embed_dim=96)

    cat  = fusion(a, b)
    print(cat.shape)

