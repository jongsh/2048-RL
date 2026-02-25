import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineLR(_LRScheduler):
    """Learning rate scheduler with linear warmup and cosine annealing."""

    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps, total_steps, eta_min=0, last_epoch=-1):
        super(WarmupCosineLR, self).__init__(optimizer, last_epoch)
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_min = eta_min
        if last_epoch != -1:
            for group in optimizer.param_groups:
                initial_lr = group["lr"]
                group.setdefault("initial_lr", initial_lr)

    def get_lr(self):
        step = self.last_epoch + 1
        if step <= self.warmup_steps:
            # linear warmup
            return [base_lr * step / self.warmup_steps for base_lr in self.base_lrs]
        else:
            # cosine annealing
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return [
                self.eta_min + (base_lr - self.eta_min) * 0.5 * (1 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs
            ]


if __name__ == "__main__":
    model = nn.Linear(10, 1)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler2 = WarmupCosineLR(optimizer, warmup_steps=5, total_steps=50, eta_min=1e-5, last_epoch=0)

    for epoch in range(1, 50):
        optimizer.step()
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Step {epoch+1}, lr = {current_lr:.6f}")
        scheduler2.step()
