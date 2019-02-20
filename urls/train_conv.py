import os
import torch
import torch.nn as nn
torch.manual_seed(17)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from models.CharConv import CharConv
from models.CLR import CyclicLR
from models.Stopper import EarlyStopping
from sklearn.metrics import accuracy_score, f1_score
from torch.autograd import Variable
from utilities import *


HIDDEN_SIZE = 512
BATCH_SIZE = 32
MAX_SEQUENCE = 256
LINEAR_SIZE = 2048
LEARNING_RATE = 0.005
REG_PARAM = 1e-2
MOMENTUM = 0.8
EPOCHS = 5001
RANDOM_SEED = 17
PATH = 'data/source/*.txt'
USE_CUDA = torch.cuda.is_available()


def train_batch(_model, _x_tensor, _y_tensor, _target):
    _model.zero_grad()
    # input should be batch, alphabet, hidden_size
    out = _model(_x_tensor)
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
    f1 = f1_score(_y_val.type(torch.int32).numpy(),
                  out_class.cpu().type(torch.int32).numpy(), average='weighted')
    return acc, f1


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda:0" if USE_CUDA else "cpu")
    df = PaddedCharSeqData(PATH, MAX_SEQUENCE)
    train, valid, test = train_valid_test_split(df, split_fold=BATCH_SIZE*4)
    print('Training set has {m} entries'.format(m=len(train)))
    print('Validation set has {n} entries'.format(n=len(valid)))
    print('Test set has {t} entries'.format(t=len(test)))
    train_batched = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    valid_set = DataLoader(valid, batch_size=len(valid), shuffle=False)
    test_set = DataLoader(test, batch_size=len(test), shuffle=False)
    model = CharConv(len(df.alphabet), df.count_classes(), HIDDEN_SIZE, BATCH_SIZE, MAX_SEQUENCE, LINEAR_SIZE).cuda()
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=REG_PARAM)  # , momentum=MOMENTUM)
    # scheduler = CyclicLR(optimizer)
    stopper = EarlyStopping(patience=10, verbose=False, saver=False)
    for epoch in range(EPOCHS):
        el = 0
        model.train()
        for batch in train_batched:
            # scheduler.batch_step()
            x = Variable(batch[0].cuda(),  requires_grad=True)
            y = Variable(batch[1].cuda())
            preds, loss = train_batch(model, x, y, criterion)
            el += loss

        model.eval()
        for val_batch in valid_set:
            x_val = Variable(val_batch[0].cuda())
            y_val = Variable(val_batch[1].cuda())
            val_out = model(x_val)
            val_loss = criterion(val_out.cuda(), y_val.long())
        stopper(val_loss, model)

        if stopper.early_stop:
            print("Stopping training at epoch {cur}".format(cur=epoch))
            for test_batch in test_set:
                x_test = Variable(test_batch[0].cuda())
                y_test = Variable(test_batch[1])
                acc, f1 = evaluate(model, x_test, y_test)
                print('Test set accuracy of {a} and F1 of {f}'.format(a=acc, f=f1))
            break

        if epoch % 100 == 0:
            for test_batch in test_set:
                x_test = Variable(test_batch[0].cuda())
                y_test = Variable(test_batch[1])
                acc, f1 = evaluate(model, x_test, y_test)
                print('Test set accuracy of {a} and F1 of {f}'.format(a=acc, f=f1))
        print('Epoch {e} had loss {ls}'.format(e=epoch, ls=el))
        # lr_list = scheduler.get_lr()
    # print('Final learning rate was {x}'.format(x=lr_list[-1]))
