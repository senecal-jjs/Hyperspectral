""" A sequence to sequence recurrent model to predict life remaining based on a sequence of past spectral 
    reflectance curves"""

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
   
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super (EncoderRNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, bidirectional=False)
        
    def forward(self, input, hidden=None):
        output, hidden = self.gru(input, hidden)
        return output, hidden


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        
    def forward(self, input, hidden=None):
        output = F.relu(input)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden
