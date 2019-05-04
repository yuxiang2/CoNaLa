import torch
import torch.nn as nn
import torch.nn.utils as U
from torch.utils.data import Dataset, DataLoader
import json
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

def collate_lines(seq_list):
    intents, codes, slot_nums, scores = zip(*seq_list)
    lens = [len(seq) for seq in intents]
    intents_seq_order = sorted(range(len(lens)), key=lens.__getitem__, reverse=True)
    intents = [torch.LongTensor(intents[i]) for i in intents_seq_order]
    intents_reverse_order = [0] * len(intents_seq_order)
    for i,j in enumerate(intents_seq_order):
        intents_reverse_order[j] = i

    lens = [len(seq) for seq in codes]
    codes_seq_order = sorted(range(len(lens)), key=lens.__getitem__, reverse=True)
    codes = [torch.LongTensor(codes[i]) for i in codes_seq_order]
    codes_reverse_order = [0] * len(codes_seq_order)
    for i,j in enumerate(codes_seq_order):
        codes_reverse_order[j] = i

    slot_nums = 0.1 * torch.Tensor(slot_nums).unsqueeze(1)
    scores = torch.Tensor(scores).unsqueeze(1)

    return intents, codes, slot_nums, scores, intents_reverse_order, codes_reverse_order


def get_train_loader(trainset, hyperP):
    return DataLoader(trainset, batch_size=hyperP['batch_size'], shuffle=True, collate_fn=collate_lines)


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

    def forward(self, src, order, hidden=None):
        embeddings = [self.embed(datapoint.cuda()) for datapoint in src]
        packed = U.rnn.pack_sequence(embeddings).cuda()
        outputs, hidden = self.gru(packed, hidden)

        hidden = hidden.permute(1, 0, 2).contiguous().view(hidden.size(1), -1)
        return hidden[order]


class ScoreNet(nn.Module):
    def __init__(self, word_size, code_size, hyperP):
        super(ScoreNet, self).__init__()
        self.intent_encoder = Encoder(word_size, hyperP)
        self.code_encoder = Encoder(code_size, hyperP)

        self.fc1 = nn.Sequential(
            nn.Linear(4 * 2 * hyperP['encoder_hidden_size'], 200),
            nn.ReLU(),
            nn.BatchNorm1d(200),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(200+1, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
            nn.Sigmoid()
        )

        self.encoder_hidden_size = hyperP['encoder_hidden_size']
        self.code_size = code_size

    def forward(self, intents, codes, slot_nums, intent_order, code_order):
        batch_size = len(intents)
        intent_hidden = self.intent_encoder(intents, intent_order)
        code_hidden = self.code_encoder(codes, code_order)

        inp = torch.cat((intent_hidden, code_hidden), dim=1)
        x =  self.fc1(inp)
        x = torch.cat((x, slot_nums.cuda()),dim = 1)
        return self.fc2(x)

    def save(self):
        torch.save(self.state_dict(), 'nerual_bleu.t7')

    def load(self):
        self.load_state_dict(torch.load('nerual_bleu.t7'))
        
    def get_idx(self, inputs):
        intents, codes, slot_nums = zip(*inputs)
        
        lens = [len(seq) for seq in intents]
        intents_seq_order = sorted(range(len(lens)), key=lens.__getitem__, reverse=True)
        intents = [torch.LongTensor(intents[i]) for i in intents_seq_order]
        intents_reverse_order = [0] * len(intents_seq_order)
        for i,j in enumerate(intents_seq_order):
            intents_reverse_order[j] = i

        lens = [len(seq) for seq in codes]
        codes_seq_order = sorted(range(len(lens)), key=lens.__getitem__, reverse=True)
        codes = [torch.LongTensor(codes[i]) for i in codes_seq_order]
        codes_reverse_order = [0] * len(codes_seq_order)
        for i,j in enumerate(codes_seq_order):
            codes_reverse_order[j] = i

        slot_nums = 0.1 * torch.Tensor(slot_nums).unsqueeze(1)
        
        with torch.no_grad():
            scores = self.forward(intents, codes, slot_nums, intents_reverse_order, codes_reverse_order)
            scores = scores.view(-1)
            _, idx = torch.max(scores, dim=0)
        return idx.item()