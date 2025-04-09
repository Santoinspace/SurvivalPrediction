from typing import Union

import torch
import torch.nn as nn
import numpy as np
from scipy.interpolate import interp1d


def conv_3d_block (in_c, out_c, act='relu', norm='bn', num_groups=8, *args, **kwargs):
    activations = nn.ModuleDict ([
        ['relu', nn.ReLU(inplace=True)],
        ['lrelu', nn.LeakyReLU(0.1, inplace=True)]
    ])
    
    normalizations = nn.ModuleDict ([
        ['bn', nn.BatchNorm3d(out_c)],
        ['gn', nn.GroupNorm(int(out_c/num_groups), out_c)]
    ])
    
    return nn.Sequential(
        nn.Conv3d(in_c, out_c, *args, **kwargs),
        normalizations[norm],
        activations[act],
    )


def flatten_layers(arr):
    return [i for sub in arr for i in sub]


class DeepMTLR(nn.Module):
    def __init__(self, t_dim, interval_num=10):
        super().__init__()

        self.cnn1 = nn.Sequential(conv_3d_block(1, 32, kernel_size=3),
                                 conv_3d_block(32, 64, kernel_size=5),
                                 nn.MaxPool3d(kernel_size=2, stride=2),
                                 
                                 conv_3d_block(64, 128, kernel_size=3),
                                 conv_3d_block(128, 256, kernel_size=5),
                                 nn.MaxPool3d(kernel_size=2, stride=2),
                                 
                                 nn.AdaptiveAvgPool3d(1),
                                 nn.Flatten())
        
        self.cnn2 = nn.Sequential(conv_3d_block(1, 32, kernel_size=3),
                                 conv_3d_block(32, 64, kernel_size=5),
                                 nn.MaxPool3d(kernel_size=2, stride=2),
                                 
                                 conv_3d_block(64, 128, kernel_size=3),
                                 conv_3d_block(128, 256, kernel_size=5),
                                 nn.MaxPool3d(kernel_size=2, stride=2),
                                 
                                 nn.AdaptiveAvgPool3d(1),
                                 nn.Flatten())

        fc_layers = [[nn.Linear(512 + t_dim, 1024), 
                        nn.BatchNorm1d(1024),
                        nn.ReLU(inplace=True), 
                        nn.Dropout(0.2)],
                     [nn.Linear(1024, 1024),
                        nn.BatchNorm1d(1024),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.2)]]
        fc_layers = flatten_layers(fc_layers)
        
        self.mtlr = nn.Sequential(*fc_layers, MTLR(1024, interval_num))
        self.sigmoid = nn.Sigmoid()

    def forward(self, pt, ct, ta):
        pt = self.cnn1(pt)
        ct = self.cnn2(ct)
        feats = torch.cat((pt, ct, ta), dim=1)
        x = self.mtlr(feats)
        pred = self.sigmoid(x)
        return pred


"""
Inspired from the work of
Credits:
@article{
  kim_deep-cr_2020,
	title = {Deep-{CR} {MTLR}: a {Multi}-{Modal} {Approach} for {Cancer} {Survival} {Prediction} with {Competing} {Risks}},
	shorttitle = {Deep-{CR} {MTLR}},
	url = {https://arxiv.org/abs/2012.05765v1},
	language = {en},
	urldate = {2021-03-16},
	author = {Kim, Sejin and Kazmierski, Michal and Haibe-Kains, Benjamin},
	month = dec,
	year = {2020}
}
"""


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

if __name__ == '__main__':
    net = DeepMTLR(t_dim=21,interval_num=4)
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

