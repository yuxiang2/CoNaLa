import json
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from preprocessing.tokenizer import tokenize_intent, tokenize_code, canonicalize_code
import torch.nn as nn
import torch.nn.functional as F
import os

# should return a list of words
def process_intent(intent):
    intent_tokens, slot_map = tokenize_intent(intent)
    return intent_tokens, slot_map


# should return a code tokens
def process_code(code, slot_map):
    code_tokens = tokenize_code(code, slot_map)
    return code_tokens


def process_data(data, mine=False):
    intents = []
    codes = []
    slot_maps = []
    if mine == False:
        for e in data:
            intent_tokens, slot_map = process_intent(e['intent'])
            intents.append(intent_tokens)
            codes.append(process_code(e['code'], slot_map))
            slot_maps.append(slot_map)

    return intents, codes, slot_maps


# return English vocabularies occur more than cut_freq times
def vocab_list(sentences, sos_eos=True, cut_freq=2):
    vocab = Counter()
    for sentence in sentences:
        for word in sentence:
            vocab[word] += 1

    if cut_freq > 0:
        vocab = [k for k in vocab if vocab[k] >= cut_freq]
    vocab.append('<UNK>')
    if sos_eos:
        vocab.append('<sos>')
        vocab.append('<eos>')

    return vocab


def code_list(codes, sos_eos=True, cut_freq=2):
    vocab = Counter()
    for code in codes:
        for word in code:
            vocab[word] += 1

    if cut_freq > 0:
        vocab = [k for k in vocab if vocab[k] >= cut_freq]
    vocab.append('<UNK>')
    vocab.append('str_0')
    vocab.append('<pad>')

    if sos_eos:
        vocab.append('<sos>')
        vocab.append('<eos>')

    return vocab


# ------------------------------------------------------------------------------
# the following create a pytorch data loader
class code_intent_pair(Dataset):
    def __init__(self, intents, code_lst, word2num, code2num):

        # convert each intent into numbers
        self.intents = []
        for intent in intents:
            num_intent = []
            for word in intent:
                if word in word2num:
                    num_intent.append(word2num[word])
                else:
                    num_intent.append(word2num['<UNK>'])
            num_intent.append(word2num['<eos>'])
            self.intents.append(num_intent)

        self.codes = []
        for code in code_lst:
            num_intent = []
            num_intent.append(code2num['<sos>'])
            for word in code:
                if word in code2num:
                    num_intent.append(code2num[word])
                else:
                    num_intent.append(code2num['<UNK>'])
            num_intent.append(code2num['<eos>'])
            self.codes.append(num_intent)

    def __len__(self):
        return len(self.intents)

    def __getitem__(self, idx):
        return (self.intents[idx], self.codes[idx])


class intent_set(Dataset):
    def __init__(self, intents, word2num):
        self.intents = intents
        self.num_intents = []
        for intent in intents:
            num_intent = []
            for word in intent:
                if word in word2num:
                    num_intent.append(word2num[word])
                else:
                    num_intent.append(word2num['<UNK>'])
                num_intent.append(word2num['<eos>'])
            self.num_intents.append(np.array(num_intent))

    def __len__(self):
        return len(self.intents)

    def __getitem__(self, idx):
        return self.num_intents[idx], self.intents[idx]


# sort the data by length, so we can do packed sequence learning
def collate_lines(seq_list):
    inputs, targets = zip(*seq_list)
    lens = [len(seq) for seq in inputs]
    seq_order = sorted(range(len(lens)), key=lens.__getitem__, reverse=True)
    inputs = [torch.LongTensor(inputs[i]) for i in seq_order]
    targets = [targets[i] for i in seq_order]

    max_len_code = 0
    for i in range(len(targets)):
        if len(targets[i]) > max_len_code:
            max_len_code = len(targets[i])

    for i in range(len(targets)):
        targets[i] = targets[i] + [754] * (max_len_code - len(targets[i]))
        # print((len(targets[i])))

    return inputs, torch.LongTensor(targets)


def get_train_loader(intents, labels, word2num, code2num, batch_size=16):
    dataset = code_intent_pair(intents, labels, word2num, code2num)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_lines)


def get_test_loader(intents, word2num, batch_size=16, shuffle=False):
    dataset = intent_set(intents, word2num)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_lines)


def load_dataset(batch_size):
    directory = os.path.dirname(os.path.abspath(__file__))+'/../corpus/'
    train_file = directory + 'train.json'
    test_file = directory + 'test.json'

    with open(train_file) as f:
        train_data = json.load(f)

    with open(test_file) as f:
        test_data = json.load(f)

    train_intent, train_codes, train_slot_maps = process_data(train_data)
    test_intent, test_codes, test_slot_maps = process_data(test_data)

    word_lst = vocab_list(train_intent)
    print(word_lst)
    code_lst = code_list(train_codes)
    print(code_lst)

    word2num = dict(zip(word_lst, range(0, len(word_lst))))
    code2num = dict(zip(code_lst, range(0, len(code_lst))))
    print(word2num)
    print(code2num)

    train_loader = get_train_loader(train_intent, train_codes, word2num, code2num, batch_size=batch_size)
    val_loader = get_train_loader(test_intent, test_codes, word2num, code2num, batch_size=batch_size)
    test_loader = get_test_loader(test_intent, word2num, batch_size=1)

    return train_loader, val_loader, test_loader, word2num, code2num


if __name__ == "__main__":
    try:
        load_dataset(batch_size=36)
    except KeyboardInterrupt as e:
        print("[STOP]", e)
