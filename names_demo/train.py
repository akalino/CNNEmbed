import torch.nn as nn

from model import CharRNN
from utilities import *


HIDDEN_SIZE = 128
LEARNING_RATE = 5e-3
N_ITERS = 50000
PATH = 'data/names/*.txt'


def train(_model, _x_tensor, _y_tensor, _target):
    hidden = _model.init_hidden()
    _model.zero_grad()

    for i in range(_x_tensor.size()[0]):
        out, hidden = _model(_x_tensor[i], hidden)

    _loss = _target(output, _y_tensor)
    _loss.backward()

    for p in _model.parameters():
        p.data.add_(-LEARNING_RATE, p.grad.data)

    return out, _loss.item()


if __name__ == "__main__":
    alphabet = generate_alphabet()
    category_lines, all_categories = load_data(PATH)
    model = CharRNN(len(alphabet), HIDDEN_SIZE, len(all_categories))
    criterion = nn.NLLLoss()
    total_loss = 0
    loss_history = []
    tracking = N_ITERS / 1000
    for j in range(1, N_ITERS + 1):
        x_tensor, y_tensor, actual = random_example(category_lines, all_categories)
        output, loss = train(model, x_tensor, y_tensor, criterion)
        total_loss += loss
        if j % tracking == 0:
            print('Average Loss is {s}'.format(s=float(total_loss)/j))
            pred_class = map_output_categories(output, all_categories)
            print('Last prediction was {p} while actual class was {a}'.format(p=pred_class,
                                                                              a=actual))
