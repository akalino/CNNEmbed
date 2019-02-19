import torch

from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import functional as F


class CharLSTM(torch.nn.Module):

    def __init__(self, _input_size, _output_size,
                 _embedding_size, _n_layers,
                 _hidden_size, _batch_size,
                 _max_sequence, _linear_size):
        super(CharLSTM, self).__init__()
        self.alphabet = _input_size
        self.hidden_size = _hidden_size
        self.embedding_size = _embedding_size
        self.n_layers = _n_layers
        self.batch_size = _batch_size
        self.max_sequence = _max_sequence
        self.linear_size = _linear_size
        self.embedding = torch.nn.Embedding(_input_size, self.embedding_size)
        self.recurring = torch.nn.LSTM(self.embedding_size, self.hidden_size,
                                       dropout=0.5, num_layers=self.n_layers, bidirectional=True)
        self.hidden_out = torch.nn.Linear(2*self.hidden_size, _output_size)
        self.drop_layer = torch.nn.Dropout(0.5)
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def init_hidden(self):
        h0 = Variable(torch.zeros(self.n_layers * 2, self.batch_size, self.hidden_size))
        c0 = Variable(torch.zeros(self.n_layers * 2, self.batch_size, self.hidden_size))
        return h0.cuda(), c0.cuda()

    def forward(self, _input):
        self.hidden = self.init_hidden()
        embedded = self.embedding(_input)
        print(embedded.shape)
        tpe = embedded.transpose(1, 2)
        print(tpe.shape)
        packed = pack_padded_sequence(tpe, [self.max_sequence]*self.batch_size, batch_first=True)
        outputs, self.hidden = self.recurring(packed, self.hidden)
        output, output_lengths = pad_packed_sequence(outputs, batch_first=False)
        output = torch.transpose(output, 0, 1)
        output = torch.transpose(output, 1, 2)
        output = torch.tanh(output)
        output, indices = F.max_pool1d(output, output.size(2), return_indices=True)
        output = torch.tanh(output)
        output = output.squeeze(2)
        output = self.drop_layer(output)
        output = self.hidden_out(output)
        output = self.softmax(output)
        return output, self.hidden
