import os
import torch
from torch import nn


class LanguageModel(nn.Module):
    def __init__(self, vocab_len):
        super().__init__()

        self.hidden1 = nn.Linear(75, 2000)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(2000,2000)
        self.act2 = nn.ReLU()
        self.output = nn.Linear(2000, vocab_len)
        self.act_out = nn.Sigmoid()

    def forward(self, x):

        x = self.act1(self.hidden1(torch.as_tensor(x)))
        x = self.act2(self.hidden2(x))
        x = self.act_out(self.output(x))
        return x
