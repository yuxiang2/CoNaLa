import torch
from torch.utils.data import Dataset, DataLoader
import json
import time
import os


class TrainSet(Dataset):
    def __init__(self, code_intent_pair):
        self.code_intent_pair = code_intent_pair

    def __len__(self):
        return len(self.code_intent_pair)

    def __getitem__(self, idx):
        intent_idx = self.code_intent_pair[idx]['intent_indx']
        code_idx = self.code_intent_pair[idx]['code_indx_nocopy']
        return (intent_idx, code_idx)


def collate_lines(seq_list, special_symbols):
    inputs, targets = zip(*seq_list)
    lens = [len(seq) for seq in inputs]
    seq_order = sorted(range(len(lens)), key=lens.__getitem__, reverse=True)
    inputs = [torch.LongTensor(inputs[i]) for i in seq_order]
    targets = [targets[i] for i in seq_order]

    # get valid target lengths
    valid_target_lengths = [len(target) - 1 for target in targets]
    # -1 because padding

    # pad target to the same length
    max_len_target = max(len(target) for target in targets)
    code_pad = special_symbols['code_eos']
    padded_targets = []
    for i in range(len(targets)):
        padded_target = targets[i] + [code_pad] * (max_len_target - len(targets[i]))
        padded_targets.append(padded_target)
    padded_targets = torch.LongTensor(padded_targets)

    original_targets = []
    for target in targets:
        original_targets += target[1:]
    original_targets = torch.LongTensor(original_targets)

    return inputs, original_targets, padded_targets, valid_target_lengths


def get_train_loader(train_entries, special_symbols, hyperP):
    trainset = TrainSet(train_entries)
    batch_size = hyperP['batch_size']
    return DataLoader(trainset, batch_size=hyperP['batch_size'],
                      collate_fn=lambda b: collate_lines(b, special_symbols))


class get_test_loader(Dataset):
    def __init__(self, code_intent_pair):
        self.code_intent_pair = code_intent_pair

    def __len__(self):
        return len(self.code_intent_pair)

    def __getitem__(self, idx):
        intent_idx = self.code_intent_pair[idx]['intent_indx']
        code = self.code_intent_pair[idx]['code']
        slot_map = self.code_intent_pair[idx]['slot_map']
        intent = self.code_intent_pair[idx]['intent']
        return (torch.LongTensor([intent_idx]), slot_map, code, intent)

# def get_test_loader(test_entries):
# testset = TestSet(test_entries)
# return DataLoader(testset, batch_size = 1, shuffle=False)


def write_answer_json(code_list):
    directory = os.path.dirname(os.path.abspath(__file__)) + '/../answer.txt'
    with open(directory, 'w') as outfile:
        json.dump(code_list, outfile, indent=1)
        outfile.flush()
