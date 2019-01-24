import glob
import os
import pandas as pd
import random
import string
import torch
import unicodedata

from torch.utils.data import Dataset, DataLoader


class CharTensorData(Dataset):
    def __init__(self, _path):
        self.df = load_data_pandas(_path)
        self.alphabet = generate_web_alphabet()
        self.df['char_tensor'] = self.df.url.apply(lambda x: self.tensor_rep(x))
        self.df.labels = pd.Categorical(self.df.labels)
        self.df['class_label'] = self.df.labels.cat.codes

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, _idx):
        x = self.df.char_tensor[_idx]
        y = self.df.class_label[_idx]
        return x, y

    def tensor_rep(self, _s):
        t = torch.zeros(1, len(self.alphabet))
        for _char in _s:
            t[char_to_idx(_char, self.alphabet)] = 1  # _s.count(_char)
        return t


def find_files(_path):
    return glob.glob(_path)


def generate_alphabet():
    alpha = string.ascii_letters + " .,;'"
    return alpha


def generate_web_alphabet():
    reg_alpha = string.ascii_letters
    extras = ' ./:!@#$%^&*()-_=+[]{};"<>,.|~1234567890'
    web_alpha = reg_alpha + extras
    return web_alpha


def uni_to_ascii(_s, _alphabet):
    _o = ''.join(c for c in unicodedata.normalize('NFD', _s)
                 if unicodedata.category(c) != 'Mn'
                 and c in _alphabet)
    return _o


def read_and_convert(_fn):
    lines = open(_fn, encoding='utf-8').read().strip().split('\n')
    alphabet = generate_web_alphabet()
    return [uni_to_ascii(line, alphabet) for line in lines]


def load_data(_path):
    cat_lines = {}
    all_categories = []
    for filename in find_files(_path):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = read_and_convert(filename)
        cat_lines[category] = lines
    return cat_lines, all_categories


def load_data_pandas(_path):
    df_list = []
    for filename in find_files(_path):
        category = os.path.splitext(os.path.basename(filename))[0]
        lines = read_and_convert(filename)
        cur_df = pd.DataFrame({'url': lines})
        cur_df['label'] = category
        df_list.append(cur_df)
    out_df = pd.concat(df_list).reset_index(drop=True)
    out_df['id'] = out_df.index
    return out_df


def char_to_idx(_char, _alphabet):
    return _alphabet.find(_char)


def char_to_tensor(_char, _alphabet):
    _alpha_size = len(_alphabet)
    t = torch.zeros(1, _alpha_size)
    t[0][char_to_idx(_char, _alphabet)] = 1
    return t


def string_to_tensor(_lines, _alphabet):
    # TODO: replace with URL to tensor
    t = torch.zeros(len(_lines), 1, len(_alphabet))
    for line,  char in enumerate(_lines):
        t[line][0][char_to_idx(char, _alphabet)] = 1
    return t


def map_output_categories(_output, _categories):
    top_n, top_i = _output.topk(1)
    category_i = top_i[0].item()
    return _categories[category_i], category_i


def random_choice(_list):
    return _list[random.randint(0, len(_list) - 1)]


def random_example(_lines, _categories):
    # TODO: this is going to go one at a time, better to mini-batch
    alphabet = generate_web_alphabet()
    rand_cat = random_choice(_categories)
    line = random_choice(_lines[rand_cat])
    cat_tensor = torch.tensor([_categories.index(rand_cat)], dtype=torch.long)
    line_tensor = string_to_tensor(line, alphabet)
    return line_tensor, cat_tensor, rand_cat


def precompute_batch():
    # TODO:
    pass


def split_train_val_test():
    # TODO:
    pass


def cheat_batching(_lines, _categories, _batch_size):
    xes = []
    yes = []
    for i in range(_batch_size):
        lt, ct, rc = random_example(_lines, _categories)
        xes.append(lt)
        yes.append(ct)
    x_batch = torch.stack(xes)
    y_batch = torch.stack(yes)
    return None
