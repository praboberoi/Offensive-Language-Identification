import torch
import torch.nn as nn

class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 3)

    def forward(self, x):
        out = self.linear(x)
        return nn.functional.softmax(out, dim=1)
























