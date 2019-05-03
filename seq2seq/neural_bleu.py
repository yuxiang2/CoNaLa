import torch
import torch.nn as nn
import torch.nn.utils as U
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import softmax
import random
import json
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR


def to_tensor(array):
    return torch.from_numpy(np.array(array))


class ScoreDataset(Dataset):
    def __init__(self, intent_lists, code_lists, slot_nums, scores):
        self.intent_lists = intent_lists
        self.code_lists = code_lists
        self.slot_nums = slot_nums
        self.scores = scores

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        intent_list = to_tensor(self.intent_lists[idx])
        code_list = to_tensor(self.code_lists[idx])
        slot_num = self.slot_nums[idx]
        score = self.scores[idx]
        print(intent_list)
        return (intent_list, code_list, slot_num, score)


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
    trainset = ScoreDataset(intent_lists, code_lists, slot_nums, scores)
    # batch_size = hyperP['batch_size']
    return DataLoader(trainset, batch_size=hyperP['batch_size'],
                      collate_fn=lambda b: collate_lines(b, special_symbols))


class Encoder(nn.Module):
    def __init__(self, word_size, hyperP):
        super(Encoder, self).__init__()

        embed_size = hyperP['encoder_embed_size']
        hidden_size = hyperP['encoder_hidden_size']
        n_layers = hyperP['encoder_layers']
        dropout = hyperP['encoder_dropout_rate']

        self.hidden_size = hidden_size
        self.embed = nn.Sequential(
            nn.Embedding(word_size, embed_size),
            nn.Dropout(dropout, inplace=True)
        )
        self.gru = nn.GRU(embed_size, hidden_size, n_layers,
                          dropout=dropout, bidirectional=True)

    def forward(self, src, hidden=None):
        embeddings = [self.embed(datapoint) for datapoint in src]
        packed = U.rnn.pack_sequence(embeddings)
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = U.rnn.pad_packed_sequence(outputs)

        # sum bidirectional outputs
        outputs = (outputs[:, :, :self.hidden_size] +
                   outputs[:, :, self.hidden_size:])
        return outputs, output_lengths, hidden


class ScoreNet(nn.Module):
    def __init__(self, word_size, code_size, hyperP):
        super(ScoreNet, self).__init__()
        self.intent_encoder = Encoder(word_size, hyperP)
        self.code_encoder = Encoder(code_size, hyperP)

        # self.fc = nn.Linear(, 1)

        self.encoder_hidden_size = hyperP['encoder_hidden_size']
        self.code_size = code_size

    def forward(self, intent_src_seq, code_src_seq, slot_nums_seq, trgt_seq):
        batch_size = len(intent_src_seq)
        intent_encoder_outputs, intent_encoder_valid_lengths, _ = self.intent_encoder(intent_src_seq)
        encoder_outputs_reshaped = intent_encoder_outputs.permute(1, 2, 0).contiguous()

        code_encoder_outputs, code_encoder_valid_lengths, _ = self.code_encoder(intent_src_seq)
        encoder_outputs_reshaped = code_encoder_outputs.permute(1, 2, 0).contiguous()

        # return score


hyperP = {
    ## training parameters
    'batch_size': 32,
    'lr': 1e-3,
    'teacher_force_rate': 0.90,
    'max_epochs': 50,
    'lr_keep_rate': 0.95,  # set to 1.0 to not decrease lr overtime
    'load_pretrain_code_embed': False,
    'freeze_embed': False,

    ## encoder architecture
    'encoder_layers': 2,
    'encoder_embed_size': 128,
    'encoder_hidden_size': 384,
    'encoder_dropout_rate': 0.3,

    ## visualization
    'print_every': 10
}


def train(model, trainloader, optimizer, loss_f, hyperP):
    model.train()
    total_loss = 0
    loss_sum = 0
    total_correct = 0
    size = 0
    print_every = hyperP['print_every']

    for i, (inp_seq, original_out_seq, padded_out_seq, out_lens) in enumerate(trainloader):
        logits = model(inp_seq, padded_out_seq, out_lens)
        loss = loss_f(logits, original_out_seq)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # show stats
        loss_sum += loss.item()
        total_loss += loss.item()
        _, predictions = torch.max(logits, dim=1)
        total_correct += (predictions == original_out_seq).sum()
        size += len(original_out_seq)

        if (i + 1) % print_every == 0:
            print('Train: loss:{}\tacc:{}'.format(loss_sum / print_every, float(total_correct) / size), end='\r')
            loss_sum = 0
            total_correct = 0
            size = 0


if __name__ == '__main__':
    # Training settings
    with open('../corpus/rerank_data.json', 'r') as f:
        array = json.load(f)
    intent_lists = [x[0] for x in array]
    code_lists = [x[1] for x in array]
    slot_nums = [x[2] for x in array]
    scores = [x[3] for x in array]
    word_size = max(max(intent_lists))
    code_size = max(max(code_lists))

    # trainset = ScoreDataset(intent_lists, code_lists, slot_nums, scores)
    trainloader = get_train_loader()

    model = ScoreNet(word_size, code_size, hyperP)
    optimizer = optim.Adam(model.parameters(), lr=hyperP['lr'], weight_decay=1e-4)
    lr_keep_rate = hyperP['lr_keep_rate']
    if lr_keep_rate != 1.0:
        lr_reduce_f = lambda epoch: lr_keep_rate ** epoch
        scheduler = LambdaLR(optimizer, lr_lambda=lr_reduce_f)

    loss_f = torch.nn.CrossEntropyLoss()

    for idx, (intent_lists, code_lists, slot_nums, scores) in enumerate(trainloader):
        print(intent_lists.size(), scores.size())

    # losses = []
    # for e in range(hyperP['max_epochs']):
    #     loss = train(model, trainloader, optimizer, loss_f, hyperP)
    #     losses.append(loss)
    #     model.save()
    #     print('model saved')
    #     if lr_keep_rate != 1.0:
    #         scheduler.step()
