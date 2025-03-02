import torch

def loss_nll(y_true, y_pred, n_intervals=4):
    """
    Args:
        y_true(Tensor): First half: 1 if individual survived that interval, 0 if not.
                        Second half: 1 for time interval before which failure has occured, 0 for other intervals.
        y_pred(Tensor): Predicted survival probability (1-hazard probability) for each time interval.
    """
    cens_uncens = torch.clamp(1.0 + y_true[:, 0 : n_intervals] * (y_pred - 1.0), min=1e-5)
    uncens = torch.clamp(1.0 - y_true[:, n_intervals : 2 * n_intervals] * y_pred, min=1e-5)
    loss = -torch.mean(torch.log(cens_uncens) + torch.log(uncens))
    return loss