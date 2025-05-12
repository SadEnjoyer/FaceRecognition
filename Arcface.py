import torch
import torch.nn as nn
import torch.nn.functional as F

class ArcFace(nn.Module):
    def __init__(self, cin, cout, s=8.0, m=0.5):
        super().__init__()
        self.s = s
        self.m = m
        self.cout = cout
        self.weight = nn.Parameter(torch.randn(cout, cin))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, label=None):
        cos = F.linear(F.normalize(x, dim=1), F.normalize(self.weight, dim=1))

        if label is not None:
            theta = torch.acos(cos)
            target_logit = torch.cos(theta + self.m)
            one_hot = F.one_hot(label, num_classes=self.cout).type_as(cos)
            cos = cos * (1 - one_hot) + target_logit * one_hot

        return cos * self.s
