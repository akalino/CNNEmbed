import torch
import torch.nn as nn


class DropCharRNN(nn.Module):

    def __init__(self, _input_size, _hidden_size, _output_size, _batch_size):
        super(DropCharRNN, self).__init__()
        self.hidden_size = _hidden_size
        self.batch_size = _batch_size
        self.input_to_hidden = nn.Linear(_input_size + _hidden_size, _hidden_size)
        self.input_to_output = nn.Linear(_input_size + _hidden_size, _output_size)
        self.drop_layer = nn.Dropout(0.00)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, _input, _hidden):
        concat = torch.cat((_input, _hidden), 1)
        hidden = self.input_to_hidden(self.drop_layer(concat))
        out_first = self.input_to_output(self.drop_layer(concat))
        pre_out = self.drop_layer(out_first)
        second_linear = nn.Linear(pre_out.shape[0], pre_out.shape[1])
        second_out = second_linear(pre_out)
        out_soft = self.softmax(second_out)
        return out_soft, hidden

    def init_hidden(self):
        return torch.zeros(self.batch_size, self.hidden_size)
