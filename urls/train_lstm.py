import os
import torch
import torch.nn as nn

from models.CharLSTM import CharLSTM
from models.CLR import CyclicLR
from numpy.random import seed
from sklearn.metrics import accuracy_score
from torch.autograd import Variable
from utilities import *


HIDDEN_SIZE = 128
BATCH_SIZE = 128
MAX_SEQUENCE = 128
LINEAR_SIZE = 1
LEARNING_RATE = 1e-3
MOMENTUM = 0.8
EPOCHS = 5001
RANDOM_SEED = seed(17)
PATH = 'data/source/*.txt'
USE_CUDA = torch.cuda.is_available()


def train_batch(_model, _x_tensor, _y_tensor, _target):
    _model.zero_grad()
    # input should be batch, alphabet, hidden_size
    out = _model(_x_tensor.type(torch.long))
    _loss = _target(out.cuda(), _y_tensor.long())
    _loss.backward()

    for p in _model.parameters():
        if p.grad is not None:
            p.data.add_(-LEARNING_RATE, p.grad.data)

    return out, _loss.item()


def evaluate(_model, _x_val, _y_val):
    out = _model(_x_val)
    out_proba, out_class = torch.max(out, 1)
    acc = accuracy_score(_y_val.type(torch.int32).numpy(),
                         out_class.cpu().type(torch.int32).numpy())
    return acc


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda:0" if USE_CUDA else "cpu")
    torch.cuda.empty_cache()
    df = CharSeqData(PATH, MAX_SEQUENCE)
    train, valid = train_valid_split(df, split_fold=BATCH_SIZE, random_seed=RANDOM_SEED)
    train_batched = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    valid_set = DataLoader(valid, batch_size=len(valid), shuffle=False)
    model = CharLSTM(len(df.alphabet), df.count_classes(), 256, 8, HIDDEN_SIZE,
                     BATCH_SIZE, MAX_SEQUENCE, LINEAR_SIZE).cuda()
    criterion = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    scheduler = CyclicLR(optimizer)
    for epoch in range(EPOCHS):
        el = 0
        for batch in train_batched:
            scheduler.batch_step()
            x = Variable(batch[0].cuda(),  requires_grad=True)
            y = Variable(batch[1].cuda())
            preds, loss = train_batch(model, x, y, criterion)
            el += loss
        if epoch % 1000 == 0:
            for val_batch in valid_set:
                x_valid = Variable(val_batch[0].cuda())
                y_valid = Variable(val_batch[1])
                acc = evaluate(model, x_valid, y_valid)
                print(acc)
            print('Epoch {e} had loss {l}'.format(e=epoch, l=el))
        lr_list = scheduler.get_lr()
    print('Final learning rate was {x}'.format(x=lr_list[-1]))
