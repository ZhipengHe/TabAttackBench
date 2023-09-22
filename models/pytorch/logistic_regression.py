from typing import Optional, List

# import utils

import torch.nn as nn
import torch.nn.functional as F


from tqdm import tqdm
import wandb
wandb.login()


class LogisticRegression(nn.Module):
    def __init__(self,
        input_dim: int,
        output_dim: int,
    ) -> None:
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out