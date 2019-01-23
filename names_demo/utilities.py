import glob
import os
import random
import string
import torch
import unicodedata


def find_files(_path):
    return glob.glob(_path)


def generate_alphabet():
    alpha = string.ascii_letters + " .,;'"
    return alpha


def uni_to_ascii(_s, _alphabet):
    _o = ''.join(c for c in unicodedata.normalize('NDF', _s)
                 if unicodedata.category(c) != 'Mn'
                 and c in _alphabet)
    return _o


def read_and_convert(_fn):
    lines = open(_fn, encoding='utf-8').read().strip().split('\n')
    return [uni_to_ascii(line) for line in lines]


def load_data(_path):
    cat_lines = {}
    all_categories = []
    for filename in find_files(_path):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = read_and_convert(filename)
        cat_lines[category] = lines
    return cat_lines, all_categories


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
        t[line][0][char_to_idx(char)] = 1
    return t


def map_output_categories(_output, _categories):
    top_n, top_i = _output.topk(1)
    category_i = top_i[0].item()
    return _categories[category_i], category_i


def random_choice(_list):
    return _list[random.randint(0, len(_list) - 1)]


def random_example(_lines, _categories):
    # TODO: this is going to go one at a time, better to mini-batch
    rand_cat = random_choice(_categories)
    line = random_choice(_lines[rand_cat])
    cat_tensor = torch.tensor([_categories.index(rand_cat)], dtype=torch.long)
    line_tensor = string_to_tensor(line)
    return line_tensor, cat_tensor, rand_cat
