import math
from typing import Sequence, Tuple, Union

import torch
import torch.nn as nn
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


def flatten_layers(arr):
    return [i for sub in arr for i in sub]


class TMSS(nn.Module):
    def __init__(self, t_dim):
        super().__init__()
        self.model = UNETR(in_channels=2,
                            out_channels=1,
                            img_size=(112,112,112),
                            t_dim=t_dim,
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
        self.classification = False
        self.vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            t_dim=t_dim,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            classification=self.classification,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )
        
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)
        
        fc_layers = [[nn.Linear(768, 256 * 2), 
                        nn.BatchNorm1d(256 * 2),
                        nn.ReLU(inplace=True), 
                        nn.Dropout(0.2)],
                     [nn.Linear(256 * 2, 64 * 2),
                        nn.BatchNorm1d(64 * 2),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.2)]]
        fc_layers = flatten_layers(fc_layers)
        self.mtlr = nn.Sequential(*fc_layers, MTLR(64 * 2, 10),)
        self.sigmoid = nn.Sigmoid()

    def proj_feat(self, x, hidden_size, feat_size):
        new_view = (x.size(0), *feat_size, hidden_size)
        x = x.view(new_view)
        new_axes = (0, len(x.shape) - 1) + tuple(d + 1 for d in range(len(feat_size)))
        x = x.permute(new_axes).contiguous()
        return x

    def forward(self, pt, ct, tab):
        img = torch.cat([pt, ct], dim=1)
        x_in = (img, tab)
        
        x, hidden_states_out = self.vit(x_in)
        
        enc1 = self.encoder1(img)
        x2 = hidden_states_out[3][:,1:,:]
        enc2 = self.encoder2(self.proj_feat(x2, self.hidden_size, self.feat_size))
        x3 = hidden_states_out[6][:,1:,:]
        enc3 = self.encoder3(self.proj_feat(x3, self.hidden_size, self.feat_size))
        x4 = hidden_states_out[9][:,1:,:]
        enc4 = self.encoder4(self.proj_feat(x4, self.hidden_size, self.feat_size))
        dec4 = self.proj_feat(x[:,1:,:], self.hidden_size, self.feat_size)
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        out = self.decoder2(dec1, enc1)
        seg_pred = self.out(out)
        seg_pred = self.sigmoid(seg_pred)
        
        feats = torch.mean(x, dim=1)
        x = self.mtlr(feats)
        pred = self.sigmoid(x)
        
        return seg_pred, pred, feats 


class ViT(nn.Module):
    """
    Vision Transformer (ViT), based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """

    def __init__(
        self,
        in_channels: int,
        img_size: Union[Sequence[int], int],
        patch_size: Union[Sequence[int], int],
        t_dim: int,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        pos_embed: str = "conv",
        classification: bool = False,
        num_classes: int = 2,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            img_size: dimension of input image.
            patch_size: dimension of patch size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_layers: number of transformer blocks.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            classification: bool argument to determine if classification is used.
            num_classes: number of classes if classification is used.
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dimensions.

        Examples::

            # for single channel input with image size of (96,96,96), conv position embedding and segmentation backbone
            >>> net = ViT(in_channels=1, img_size=(96,96,96), pos_embed='conv')

            # for 3-channel with image size of (128,128,128), 24 layers and classification backbone
            >>> net = ViT(in_channels=3, img_size=(128,128,128), pos_embed='conv', classification=True)

            # for 3-channel with image size of (224,224), 12 layers and classification backbone
            >>> net = ViT(in_channels=3, img_size=(224,224), pos_embed='conv', classification=True, spatial_dims=2)

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.classification = classification
        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            t_dim=t_dim,
            hidden_size=hidden_size,
            num_heads=num_heads,
            pos_embed=pos_embed,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )
        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate) for i in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_size)
        if self.classification:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
            self.classification_head = nn.Sequential(nn.Linear(hidden_size, num_classes), nn.Tanh())
            
        ## Projection of EHR
        self.EHR_proj = nn.Linear(hidden_size, hidden_size)


    def forward(self, x):
        
        x = self.patch_embedding(x) #img, clin_var = x
        
        if self.classification:
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states_out.append(x)
        x = self.norm(x)
        if self.classification:
            x = self.classification_head(x[:, 0])
            
        return x, hidden_states_out
    
    

    # Copyright 2020 - 2021 MONAI Consortium

Rearrange, _ = optional_import("einops.layers.torch", name="Rearrange")
SUPPORTED_EMBEDDING_TYPES = {"conv", "perceptron"}


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


class MTLR(nn.Module):
    """Multi-task logistic regression for individualised
    survival prediction.

    The MTLR time-logits are computed as:
    `z = sum_k x^T w_k + b_k`,
    where `w_k` and `b_k` are learnable weights and biases for each time
    interval.

    Note that a slightly more efficient reformulation is used here, first
    proposed in [2]_.

    References
    ----------
    ..[1] C.-N. Yu et al., ‘Learning patient-specific cancer survival
    distributions as a sequence of dependent regressors’, in Advances in neural
    information processing systems 24, 2011, pp. 1845–1853.
    ..[2] P. Jin, ‘Using Survival Prediction Techniques to Learn
    Consumer-Specific Reservation Price Distributions’, Master's thesis,
    University of Alberta, Edmonton, AB, 2015.
    """

    def __init__(self, in_features: int, num_time_bins: int):
        """Initialises the module.

        Parameters
        ----------
        in_features
            Number of input features.
        num_time_bins
            The number of bins to divide the time axis into.
        """
        super().__init__()
        if num_time_bins < 1:
            raise ValueError("The number of time bins must be at least 1")
        if in_features < 1:
            raise ValueError("The number of input features must be at least 1")
        self.in_features = in_features
        self.num_time_bins = num_time_bins

        self.mtlr_weight = nn.Parameter(torch.Tensor(self.in_features,
                                                     self.num_time_bins))
        self.mtlr_bias = nn.Parameter(torch.Tensor(self.num_time_bins))

        # `G` is the coding matrix from [2]_ used for fast summation.
        # When registered as buffer, it will be automatically
        # moved to the correct device and stored in saved
        # model state.
        self.register_buffer(
            "G",
            torch.tril(
                torch.ones(self.num_time_bins,
                           self.num_time_bins,
                           requires_grad=True)))
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass on a batch of examples.

        Parameters
        ----------
        x : torch.Tensor, shape (num_samples, num_features)
            The input data.

        Returns
        -------
        torch.Tensor, shape (num_samples, num_time_bins)
            The predicted time logits.
        """
        out = torch.matmul(x, self.mtlr_weight) + self.mtlr_bias
        return torch.matmul(out, self.G)

    def reset_parameters(self):
        """Resets the model parameters."""
        nn.init.xavier_normal_(self.mtlr_weight)
        nn.init.constant_(self.mtlr_bias, 0.)

    def __repr__(self):
        return (f"{self.__class__.__name__}(in_features={self.in_features},"
                f" num_time_bins={self.num_time_bins})")


def masked_logsumexp(x: torch.Tensor,
                     mask: torch.Tensor,
                     dim: int = -1) -> torch.Tensor:
    """Computes logsumexp over elements of a tensor specified by a mask
    in a numerically stable way.

    Parameters
    ----------
    x
        The input tensor.
    mask
        A tensor with the same shape as `x` with 1s in positions that should
        be used for logsumexp computation and 0s everywhere else.
    dim
        The dimension of `x` over which logsumexp is computed. Default -1 uses
        the last dimension.

    Returns
    -------
    torch.Tensor
        Tensor containing the logsumexp of each row of `x` over `dim`.
    """
    max_val, _ = (x * mask).max(dim=dim)
    max_val = torch.clamp_min(max_val, 0)
    return torch.log(
        torch.sum(torch.exp(x - max_val.unsqueeze(dim)) * mask,
                  dim=dim)) + max_val


def mtlr_neg_log_likelihood(logits: torch.Tensor,
                            target: torch.Tensor,
                            model: torch.nn.Module,
                            C1: float,
                            average: bool = False) -> torch.Tensor:
    """Computes the negative log-likelihood of a batch of model predictions.

    Parameters
    ----------
    logits : torch.Tensor, shape (num_samples, num_time_bins)
        Tensor with the time-logits (as returned by the MTLR module) for one
        instance in each row.
    target : torch.Tensor, shape (num_samples, num_time_bins)
        Tensor with the encoded ground truth survival.
    model
        PyTorch Module with at least `MTLR` layer.
    C1
        The L2 regularization strength.
    average
        Whether to compute the average log likelihood instead of sum
        (useful for minibatch training).

    Returns
    -------
    torch.Tensor
        The negative log likelihood.
    """
    censored = target.sum(dim=1) > 1
    nll_censored = masked_logsumexp(logits[censored], target[censored]).sum() if censored.any() else 0
    nll_uncensored = (logits[~censored] * target[~censored]).sum() if (~censored).any() else 0

    # the normalising constant
    norm = torch.logsumexp(logits, dim=1).sum()

    nll_total = -(nll_censored + nll_uncensored - norm)
    if average:
        nll_total = nll_total / target.size(0)

    # L2 regularization
    for k, v in model.named_parameters():
        if "mtlr_weight" in k:
            nll_total += C1/2 * torch.sum(v**2)

    return nll_total


def mtlr_survival(logits: torch.Tensor) -> torch.Tensor:
    """Generates predicted survival curves from predicted logits.

    Parameters
    ----------
    logits
        Tensor with the time-logits (as returned by the MTLR module) for one
        instance in each row.

    Returns
    -------
    torch.Tensor
        The predicted survival curves for each row in `pred` at timepoints used
        during training.
    """
    # TODO: do not reallocate G in every call
    G = torch.tril(torch.ones(logits.size(1),
                              logits.size(1))).to(logits.device)
    density = torch.softmax(logits, dim=1)
    return torch.matmul(density, G)


def mtlr_survival_at_times(logits: torch.Tensor,
                           train_times: Union[torch.Tensor, np.ndarray],
                           pred_times: np.ndarray) -> np.ndarray:
    """Generates predicted survival curves at arbitrary timepoints using linear
    interpolation.

    Notes
    -----
    This function uses scipy.interpolate internally and returns a Numpy array,
    in contrast with `mtlr_survival`.

    Parameters
    ----------
    logits
        Tensor with the time-logits (as returned by the MTLR module) for one
        instance in each row.
    train_times
        Time bins used for model training. Must have the same length as the
        first dimension of `pred`.
    pred_times
        Array of times used to compute the survival curve.

    Returns
    -------
    np.ndarray
        The survival curve for each row in `pred` at `pred_times`. The values
        are linearly interpolated at timepoints not used for training.
    """
    train_times = np.pad(train_times, (1, 0))
    surv = mtlr_survival(logits).detach().cpu().numpy()
    interpolator = interp1d(train_times, surv)
    return interpolator(np.clip(pred_times, 0, train_times.max()))


def mtlr_hazard(logits: torch.Tensor) -> torch.Tensor:
    """Computes the hazard function from MTLR predictions.

    The hazard function is the instantenous rate of failure, i.e. roughly
    the risk of event at each time interval. It's computed using
    `h(t) = f(t) / S(t)`,
    where `f(t)` and `S(t)` are the density and survival functions at t,
    respectively.

    Parameters
    ----------
    logits
        The predicted logits as returned by the `MTLR` module.

    Returns
    -------
    torch.Tensor
        The hazard function at each time interval in `y_pred`.
    """
    return torch.softmax(
        logits, dim=1)[:, :-1] / (mtlr_survival(logits) + 1e-15)[:, 1:]


def mtlr_risk(logits: torch.Tensor) -> torch.Tensor:
    """Computes the overall risk of event from MTLR predictions.

    The risk is computed as the time integral of the cumulative hazard,
    as defined in [1]_.

    Parameters
    ----------
    logits
        The predicted logits as returned by the `MTLR` module.

    Returns
    -------
    torch.Tensor
        The predicted overall risk.
    """
    hazard = mtlr_hazard(logits)
    return torch.sum(hazard.cumsum(1), dim=1)


net = TMSS(t_dim=15)
total = sum([param.nelement() for param in net.parameters() if param.requires_grad])
# 打印每一层的参数量
for name, param in net.named_parameters():
    if param.requires_grad:
        print(name, param.numel())
# 精确地计算：1MB=1024KB=1048576字节
print('Number of parameter: % .4fM' % (total / 1024 / 1024))
