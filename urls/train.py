import os
import torch
import torch.nn as nn

from models.CharRNN import CharRNN
from utilities import *


HIDDEN_SIZE = 256
LEARNING_RATE = 1e-2
N_ITERS = 500000
PATH = 'data/*.txt'
USE_CUDA = torch.cuda.is_available()


def train(_model, _x_tensor, _y_tensor, _target):
    hidden = _model.init_hidden()
    _model.zero_grad()

    for i in range(_x_tensor.size()[0]):
        out, hidden = _model(_x_tensor[i], hidden)

    _loss = _target(out, _y_tensor)
    _loss.backward()

    for p in _model.parameters():
        p.data.add_(-LEARNING_RATE, p.grad.data)

    return out, _loss.item()


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda:0" if USE_CUDA else "cpu")
    alphabet = generate_web_alphabet()
    category_lines, all_categories = load_data(PATH)
    model = CharRNN(len(alphabet), HIDDEN_SIZE, len(all_categories))
    criterion = nn.NLLLoss()
    total_loss = 0
    loss_history = []
    tracking = N_ITERS / 10000
    for j in range(1, N_ITERS + 1):
        x_tensor, y_tensor, actual = random_example(category_lines, all_categories)
        output, loss = train(model, x_tensor, y_tensor, criterion)
        total_loss += loss
        if j % tracking == 0:
            print('On iteration: {i}'.format(i=j))
            print('Average Loss is {s}'.format(s=float(total_loss)/j))
            pred_class = map_output_categories(output, all_categories)
            print('Last prediction was {p} while actual class was {a}'.format(p=pred_class,
                                                                              a=actual))
