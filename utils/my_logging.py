import os
from tensorboardX import SummaryWriter

class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.max = float('-inf')
        self.min = float('inf')
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        try:
            self.max = max(val, self.max)
            self.min = min(val, self.min)
        except Exception:
            pass
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Logger:
    def __init__(self, log_dir, use_tensorboard=True):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.use_tensorboard = use_tensorboard
        if use_tensorboard:
            self.writer = SummaryWriter(log_dir=log_dir)
    
    def log(self, scalar_name, value, global_step):
        """Log a scalar value to both console and TensorBoard."""
        print(f"{scalar_name}: {value} at step {global_step}")
        if self.use_tensorboard:
            self.writer.add_scalar(scalar_name, value, global_step)
    
    def close(self):
        """Close the TensorBoard writer."""
        if self.use_tensorboard:
            self.writer.close()