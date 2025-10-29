import numpy as np


class RunningNormalizer:
    """A running normalizer to normalize inputs online"""

    def __init__(self, momentum=0.99, eps=1e-8):
        self.mean = 0.0
        self.var = 1.0
        self.momentum = momentum
        self.eps = eps
        self.initialized = False

    def update(self, x):
        x = np.array(x, dtype=np.float32)
        batch_mean = np.mean(x)
        batch_var = np.var(x)

        if not self.initialized:
            self.mean = batch_mean
            self.var = batch_var
            self.initialized = True
        else:
            self.mean = self.momentum * self.mean + (1 - self.momentum) * batch_mean
            self.var = self.momentum * self.var + (1 - self.momentum) * batch_var

    def normalize(self, x):
        if not self.initialized:
            return x
        return (x - self.mean) / (np.sqrt(self.var) + self.eps)
