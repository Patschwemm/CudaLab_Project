import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class Conv2dRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size=3, act_fn=nn.ReLU()):
        super(Conv2dRNNCell, self).__init__()

        padding = kernel_size // 2

        W_xh = nn.Conv2d(in_channels=input_size, out_channels=hidden_size, kernel_size=kernel_size, padding=padding)
        W_hh = nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=kernel_size, padding=padding)

        self.act_fn = act_fn

    def forward(self, x, h):
        # x: input sequence, shape (batch_size, input_size, sequence_length)
        # h: hidden state, shape (batch_size, hidden_size, sequence_length)

        # Compute the convolutional output
        c = self.conv_xh(x) + self.conv_hh(h)
        
        # Apply the non-linear activation function
        c = self.activation(c)
        
        # Update the hidden state
        h_new = c
        
        return h_new

