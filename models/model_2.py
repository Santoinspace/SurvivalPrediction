import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import einsum

class FeedForward(nn.Module):
    def __init__(self, dim, mult=1, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(self.norm(x))

class MMAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        residual = True,
        residual_conv_kernel = 33,
        eps = 1e-8,
        dropout = 0.,
        num_pathways = 281,
    ):
        super().__init__()
        self.num_pathways = num_pathways
        self.eps = eps
        inner_dim = heads * dim_head

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.residual = residual
        if residual:
            kernel_size = residual_conv_kernel
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(heads, heads, (kernel_size, 1), padding = (padding, 0), groups = heads, bias = False)

    def forward(self, x, mask=None, return_attn=False):
        b, n, _, h, m, eps = *x.shape, self.heads, self.num_pathways, self.eps

        # derive query, keys, values
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # set masked positions to 0 in queries, keys, values
        if mask != None:
            mask = rearrange(mask, 'b n -> b () n')
            q, k, v = map(lambda t: t * mask[..., None], (q, k, v))

        # regular transformer scaling
        q = q * self.scale

        # extract the pathway/histology queries and keys
        q_pathways = q[:, :, :self.num_pathways, :]  # bs x head x num_pathways x dim
        k_pathways = k[:, :, :self.num_pathways, :]

        q_histology = q[:, :, self.num_pathways:, :]  # bs x head x num_patches x dim
        k_histology = k[:, :, self.num_pathways:, :]
        
        # similarities
        einops_eq = '... i d, ... j d -> ... i j'
        cross_attn_histology = einsum(einops_eq, q_histology, k_pathways)
        attn_pathways = einsum(einops_eq, q_pathways, k_pathways)
        cross_attn_pathways = einsum(einops_eq, q_pathways, k_histology)
        
        # softmax
        pre_softmax_cross_attn_histology = cross_attn_histology
        cross_attn_histology = cross_attn_histology.softmax(dim=-1)
        attn_pathways_histology = torch.cat((attn_pathways, cross_attn_pathways), dim=-1).softmax(dim=-1)

        # compute output 
        out_pathways =  attn_pathways_histology @ v
        out_histology = cross_attn_histology @ v[:, :, :self.num_pathways]

        out = torch.cat((out_pathways, out_histology), dim=2)
        
        # add depth-wise conv residual of values
        if self.residual:
            out += self.res_conv(v)

        # merge and combine heads
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)

        if return_attn:  
            # return three matrices
            return out, attn_pathways.squeeze().detach().cpu(), cross_attn_pathways.squeeze().detach().cpu(), pre_softmax_cross_attn_histology.squeeze().detach().cpu()

        return out

class MMAttentionLayer(nn.Module):
    """
    Applies layer norm --> attention
    """

    def __init__(
        self,
        norm_layer=nn.LayerNorm,
        dim=512,
        dim_head=64,
        heads=6,
        residual=True,
        dropout=0.,
        num_pathways = 281,
    ):

        super().__init__()
        self.norm = norm_layer(dim)
        self.num_pathways = num_pathways
        self.attn = MMAttention(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            residual=residual,
            dropout=dropout,
            num_pathways=num_pathways
        )

    def forward(self, x=None, mask=None, return_attention=False):

        if return_attention:
            x, attn_pathways, cross_attn_pathways, cross_attn_histology = self.attn(x=self.norm(x), mask=mask, return_attn=True)
            return x, attn_pathways, cross_attn_pathways, cross_attn_histology
        else:
            x = self.attn(x=self.norm(x), mask=mask)

        return x
    
"""SurvPath model"""
def SNN_Block(dim1, dim2, dropout=0.25):
    r"""
    Multilayer Reception Block w/ Self-Normalization (Linear + ELU + Alpha Dropout)

    args:
        dim1 (int): Dimension of input features
        dim2 (int): Dimension of output features
        dropout (float): Dropout rate
    """
    import torch.nn as nn

    return nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.ELU(),
            nn.AlphaDropout(p=dropout, inplace=False))

class SurvPath(nn.Module):
    def __init__(
        self, 
        pet_embedding_dim=512,  # 新增PET特征维度
        ct_embedding_dim=512,   # 新增CT特征维度
        tabular_dim=21,         # 新增表格数据维度
        dropout=0.1,
        num_classes=4,
        projection_dim=256,     # 统一投影维度
    ):
        super(SurvPath, self).__init__()
        
        #--- 三模态特征投影 ---
        self.pet_projection = nn.Linear(pet_embedding_dim, projection_dim)
        self.ct_projection = nn.Linear(ct_embedding_dim, projection_dim)
        self.tabular_projection = nn.Sequential(
            nn.Linear(tabular_dim, 256),
            nn.ReLU(),
            nn.Linear(256, projection_dim)
        )

        #--- 三模态交叉注意力 ---
        self.cross_attender = MMAttentionLayer(
            dim=projection_dim,
            dim_head=projection_dim // 2,
            heads=1,
            num_pathways=3  # PET/CT/Tabular三个模态
        )

        #--- 融合后处理 ---
        self.to_logits = nn.Sequential(
            nn.Linear(projection_dim*3, int(projection_dim/2)),  # 拼接三模态特征
            nn.ReLU(),
            nn.Linear(int(projection_dim/2), num_classes)
        )

    def forward(self, **kwargs):
        # 获取三模态输入
        pet = kwargs['pet']
        ct = kwargs['ct']
        tabular = kwargs['tabular']
        
        # 特征投影
        pet_proj = self.pet_projection(pet).unsqueeze(1)  # [B,1,D]
        ct_proj = self.ct_projection(ct).unsqueeze(1)     # [B,1,D]
        tab_proj = self.tabular_projection(tabular).unsqueeze(1) # [B,1,D]
        
        # 拼接模态特征
        tokens = torch.cat([pet_proj, ct_proj, tab_proj], dim=1) # [B,3,D]
        
        # 交叉注意力
        mm_embed = self.cross_attender(tokens)  # [B,3,D]
        
        # 特征聚合（沿序列维度取平均）
        aggregated = torch.mean(mm_embed, dim=1)  # [B,D]
        
        # 输出预测
        logits = self.to_logits(aggregated)
        return logits

if __name__ == '__main__':
    print('SurvPath model test')
    model = SurvPath(
        pet_embedding_dim=512,
        ct_embedding_dim=512,
        tabular_dim=21,
        num_classes=4
    )

    # 假设输入特征维度
    pet_feat = torch.randn(32, 512)  # PET影像特征
    ct_feat = torch.randn(32, 512)   # CT影像特征
    tabular = torch.randn(32, 21)    # 表格数据

    output = model(pet=pet_feat, ct=ct_feat, tabular=tabular)
    print(output.shape)  # torch.Size([32, 4])
    print(output)