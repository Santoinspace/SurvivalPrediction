import math
from typing import Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch import einsum
import numpy as np
from scipy.interpolate import interp1d

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.blocks.transformerblock import TransformerBlock
from monai.networks.layers import Conv
from monai.utils import ensure_tuple_rep, optional_import
from monai.utils.module import look_up_option
from typing import Sequence, Union
from einops import rearrange

Rearrange, _ = optional_import("einops.layers.torch", name="Rearrange")
SUPPORTED_EMBEDDING_TYPES = {"conv", "perceptron"}

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

def flatten_layers(arr):
    return [i for sub in arr for i in sub]

class TMSS(nn.Module):
    def __init__(self, t_dim, interval_num):
        super().__init__()
        self.model = UNETR(in_channels=2,
                            out_channels=1,
                            img_size=(112,112,112),
                            t_dim=t_dim,
                            interval_num=interval_num,
                            feature_size = 16,
                            hidden_size = 768,
                            mlp_dim = 3072,
                            num_heads = 12,
                            pos_embed = "conv",
                            norm_name = "instance",
                            conv_block = True,
                            res_block = True,
                            dropout_rate = 0.0,
                            spatial_dims = 3,)

    def forward(self, pt, ct, ta):
        return self.model(pt, ct, ta)

class UNETR(nn.Module):
    """
    UNETR based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(self,
                in_channels: int,
                out_channels: int,
                img_size: Union[Sequence[int], int],
                t_dim: int,
                interval_num=int,
                feature_size: int = 16,
                hidden_size: int = 768,
                mlp_dim: int = 3072,
                num_heads: int = 12,
                pos_embed: str = "conv",
                norm_name: Union[Tuple, str] = "instance",
                conv_block: bool = True,
                res_block: bool = True,
                dropout_rate: float = 0.0,
                spatial_dims: int = 3,
    ) -> None:

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.num_layers = 12
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.patch_size = ensure_tuple_rep(16, spatial_dims)
        self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, self.patch_size))
        self.hidden_size = hidden_size

        self.num_pathways = 1
        self.num_classes = interval_num
                  
        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            t_dim=t_dim,
            hidden_size=hidden_size,
            num_heads=num_heads,
            pos_embed=pos_embed,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )

        self.cross_attender = MMAttentionLayer(
            dim=self.hidden_size,
            dim_head=self.hidden_size // 2,
            heads=2,
            residual=False,
            dropout=0.1,
            num_pathways = self.num_pathways
        )

        self.sigmoid = nn.Sigmoid()

        # feedforward network
        self.feed_forward = FeedForward(self.hidden_size, dropout=dropout_rate)
        self.layer_norm = nn.LayerNorm(self.hidden_size)

        # when both top and bottom blocks 
        self.to_logits = nn.Sequential(
                nn.Linear(self.hidden_size * 2, int(self.hidden_size/4)),
                nn.ReLU(),
                nn.Linear(int(self.hidden_size/4), self.num_classes)
            )

    def proj_feat(self, x, hidden_size, feat_size):
        new_view = (x.size(0), *feat_size, hidden_size)
        x = x.view(new_view)
        new_axes = (0, len(x.shape) - 1) + tuple(d + 1 for d in range(len(feat_size)))
        x = x.permute(new_axes).contiguous()
        return x

    def forward(self, pt, ct, tab):
        img = torch.cat([pt, ct], dim=1)
        x_in = (img, tab)

        x = self.patch_embedding(x_in)

        feats = torch.mean(x, dim=1)
        x = self.cross_attender(x)

        mm_embed = self.feed_forward(x)
        mm_embed = self.layer_norm(x)

        #---> aggregate 
        # modality specific mean 
        paths_postSA_embed = mm_embed[:, :self.num_pathways, :]
        paths_postSA_embed = torch.mean(paths_postSA_embed, dim=1)

        wsi_postSA_embed = mm_embed[:, self.num_pathways:, :]
        wsi_postSA_embed = torch.mean(wsi_postSA_embed, dim=1)

        # when both top and bottom block
        embedding = torch.cat([paths_postSA_embed, wsi_postSA_embed], dim=1)

        x = self.to_logits(embedding)
        pred = self.sigmoid(x)
        
        return pred

class PatchEmbeddingBlock(nn.Module):
    """
    A patch embedding block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"

    Example::

        >>> from monai.networks.blocks import PatchEmbeddingBlock
        >>> PatchEmbeddingBlock(in_channels=4, img_size=32, patch_size=8, hidden_size=32, num_heads=4, pos_embed="conv")

    """

    def __init__(
        self,
        in_channels: int,
        img_size: Union[Sequence[int], int],
        patch_size: Union[Sequence[int], int],
        t_dim: int,
        hidden_size: int,
        num_heads: int,
        pos_embed: str,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            img_size: dimension of input image.
            patch_size: dimension of patch size.
            hidden_size: dimension of hidden layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dimensions.
        """
        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden size should be divisible by num_heads.")

        self.pos_embed = look_up_option(pos_embed, SUPPORTED_EMBEDDING_TYPES)

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        for m, p in zip(img_size, patch_size):
            if m < p:
                raise ValueError("patch_size should be smaller than img_size.")
            if self.pos_embed == "perceptron" and m % p != 0:
                raise ValueError("patch_size should be divisible by img_size for perceptron.")
        self.n_patches = np.prod([im_d // p_d for im_d, p_d in zip(img_size, patch_size)]) + 1  # +1 for EHR
        self.patch_dim = in_channels * np.prod(patch_size)

        self.patch_embeddings: nn.Module
        if self.pos_embed == "conv":
            self.patch_embeddings = Conv[Conv.CONV, spatial_dims](
                in_channels=in_channels, out_channels=hidden_size, kernel_size=patch_size, stride=patch_size
            )
        elif self.pos_embed == "perceptron":
            # for 3d: "b c (h p1) (w p2) (d p3)-> b (h w d) (p1 p2 p3 c)"
            chars = (("h", "p1"), ("w", "p2"), ("d", "p3"))[:spatial_dims]
            from_chars = "b c " + " ".join(f"({k} {v})" for k, v in chars)
            to_chars = f"b ({' '.join([c[0] for c in chars])}) ({' '.join([c[1] for c in chars])} c)"
            axes_len = {f"p{i+1}": p for i, p in enumerate(patch_size)}
            self.patch_embeddings = nn.Sequential(
                Rearrange(f"{from_chars} -> {to_chars}", **axes_len), nn.Linear(self.patch_dim, hidden_size)
            )
            
        self.EHR_proj = nn.Sequential(nn.Linear(t_dim, hidden_size))

        self.position_embeddings = nn.Parameter(torch.zeros(1, self.n_patches, hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.dropout = nn.Dropout(dropout_rate)
        self.trunc_normal_(self.position_embeddings, mean=0.0, std=0.02, a=-2.0, b=2.0)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            self.trunc_normal_(m.weight, mean=0.0, std=0.02, a=-2.0, b=2.0)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def trunc_normal_(self, tensor, mean, std, a, b):
        # From PyTorch official master until it's in a few official releases - RW
        # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
        def norm_cdf(x):
            return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

        with torch.no_grad():
            l = norm_cdf((a - mean) / std)
            u = norm_cdf((b - mean) / std)
            tensor.uniform_(2 * l - 1, 2 * u - 1)
            tensor.erfinv_()
            tensor.mul_(std * math.sqrt(2.0))
            tensor.add_(mean)
            tensor.clamp_(min=a, max=b)
            return tensor

    def forward(self, x):
        img, clin_var = x
        x = self.patch_embeddings(img)
        
        clin_var = self.EHR_proj(clin_var)
        clin_var = clin_var.unsqueeze(dim=1)
      
        if self.pos_embed == "conv":
            x = x.flatten(2).transpose(-1, -2)
            
        x = torch.cat([clin_var, x], dim=1)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

    def reset_parameters(self):
        """Resets the model parameters."""
        nn.init.xavier_normal_(self.mtlr_weight)
        nn.init.constant_(self.mtlr_bias, 0.)

    def __repr__(self):
        return (f"{self.__class__.__name__}(in_features={self.in_features},"
                f" num_time_bins={self.num_time_bins})")

if __name__ == '__main__':
    net = TMSS(t_dim=21)
     # 生成模拟数据
    batch_size = 2
    # PET图像 [B, 1, 112, 112, 112]
    pt = torch.randn(batch_size, 1, 112, 112, 112)
    # CT图像 [B, 1, 112, 112, 112] 
    ct = torch.randn(batch_size, 1, 112, 112, 112)
    # 临床数据 [B, 15]
    ta = torch.randn(batch_size, 21)
    
    # 前向传播测试
    pred, feats = net(pt, ct, ta)
    
    # 输出形状验证
    print(f"Prediction shape: {pred.shape}")  # 应输出 torch.Size([2, 10])
    print(f"Features shape: {feats.shape}")   # 应输出 torch.Size([2, 768])
    
    total = sum([param.nelement() for param in net.parameters() if param.requires_grad])
    # 精确地计算：1MB=1024KB=1048576字节
    print('Number of parameter: % .4fM' % (total / 1024 / 1024))
