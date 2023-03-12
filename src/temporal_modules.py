import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class Conv2dRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size=3, act_fn=nn.ReLU()):
        super(Conv2dRNNCell, self).__init__()

        padding = kernel_size // 2

        self.conv_xh = nn.Conv2d(in_channels=input_size, out_channels=hidden_size, kernel_size=kernel_size, padding=padding)
        self.conv_hh = nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=kernel_size, padding=padding)

        self.act_fn = act_fn

        # Initialize the hidden state parameter
        self.h_init = False
        self.h = None

    def forward(self, x):
        # x: input sequence, shape (batch_size, input_size, sequence_length)
        # h: hidden state, shape (batch_size, hidden_size, sequence_length)

        if self.h_init == False:
            self.h = torch.randn(x.shape[0], self.conv_hh.out_channels, x.shape[2], x.shape[3])
            self.h = self.h.to(x.device)

        # Compute the convolutional output
        c = self.conv_xh(x) + self.conv_hh(self.h)
        
        # Apply the non-linear activation function
        c = self.act_fn(c)
        
        # Update the hidden state
        h_new = c
        
        return h_new

