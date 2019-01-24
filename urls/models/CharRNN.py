import torch
import torch.nn as nn


class CharRNN(nn.Module):

    def __init__(self, _input_size, _hidden_size, _output_size):
        super(CharRNN, self).__init__()
        self.hidden_size = _hidden_size
        self.input_to_hidden = nn.Linear(_input_size + _hidden_size, _hidden_size)
        self.input_to_output = nn.Linear(_input_size + _hidden_size, _output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, _input, _hidden):
        concat = torch.cat((_input, _hidden), 1)
        hidden = self.input_to_hidden(concat)
        out_first = self.input_to_output(concat)
        out_soft = self.softmax(out_first)
        return out_soft, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)
