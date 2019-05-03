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
#         print(intent_list)
        return (intent_list, code_list, slot_num, score)


def collate_lines(seq_list):
    intents, codes, slot_nums, scores = zip(*seq_list)
    lens = [len(seq) for seq in intents]
    intents_seq_order = sorted(range(len(lens)), key=lens.__getitem__, reverse=True)
    intents = [torch.LongTensor(intents[i]) for i in intents_seq_order]
    
    lens = [len(seq) for seq in codes]
    codes_seq_order = sorted(range(len(lens)), key=lens.__getitem__, reverse=True)
    codes = [torch.LongTensor(codes[i]) for i in codes_seq_order]
    
    slot_nums = 0.02 * torch.Tensor(slot_nums).unsqueeze(1)
    scores = torch.Tensor(scores).unsqueeze(1)

    return intents, codes, slot_nums, scores, intents_seq_order, codes_seq_order


def get_train_loader(intent_lists, code_lists, slot_nums, scores, hyperP):
    trainset = ScoreDataset(intent_lists, code_lists, slot_nums, scores)
    # batch_size = hyperP['batch_size']
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
        try:
            embeddings = [self.embed(datapoint) for datapoint in src]
        except:
            print(src)
        packed = U.rnn.pack_sequence(embeddings)
        outputs, hidden = self.gru(packed, hidden) 
        new_order = [i for i,j in enumerate(order)]
        
        hidden = hidden.permute(1,0,2).contiguous().view(hidden.size(1), -1) 
        return hidden[new_order]


class ScoreNet(nn.Module):
    def __init__(self, word_size, code_size, hyperP):
        super(ScoreNet, self).__init__()
        self.intent_encoder = Encoder(word_size, hyperP)
        self.code_encoder = Encoder(code_size, hyperP)

        self.fc = nn.Sequential(
            nn.Linear(4 * 2 * hyperP['encoder_hidden_size'] + 1, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            nn.Linear(50,1),
            nn.Sigmoid(),
        )

        self.encoder_hidden_size = hyperP['encoder_hidden_size']
        self.code_size = code_size

    def forward(self, intent_src_seq, code_src_seq, slot_nums_seq, intent_order, code_order):
        batch_size = len(intent_src_seq)
        intent_hidden = self.intent_encoder(intent_src_seq, intent_order)
        code_hidden = self.code_encoder(code_src_seq, code_order)
        inp = torch.cat((intent_hidden, code_hidden, slot_nums_seq), dim=1)
        return self.fc(inp)

hyperP = {
    ## training parameters
    'batch_size': 8,
    'lr': 1e-3,
    'teacher_force_rate': 0.90,
    'max_epochs': 50,
    'lr_keep_rate': 0.95,  # set to 1.0 to not decrease lr overtime
    'load_pretrain_code_embed': False,
    'freeze_embed': False,

    ## encoder architecture
    'encoder_layers': 2,
    'encoder_embed_size': 128,
    'encoder_hidden_size': 256,
    'encoder_dropout_rate': 0.3,

    ## visualization
    'print_every': 10
}


def train(model, trainloader, optimizer, loss_f, hyperP):
    model.train()
    total_loss = 0
    loss_sum = 0
    print_every = hyperP['print_every']

    for i, (intents, codes, slot_nums, scores, intents_seq_order, codes_seq_order) in enumerate(trainloader):
        predict_scores = model(intents, codes, slot_nums, intents_seq_order, codes_seq_order)
        loss = loss_f(predict_scores, scores)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # show stats
        loss_sum += loss.item()
        total_loss += loss.item()

        if (i + 1) % print_every == 0:
            print('Train loss:{}\t'.format(loss_sum / print_every))
            loss_sum = 0
            
    return total_loss / len(trainloader)


if __name__ == '__main__':
    # Training settings
    with open('../corpus/rerank_data.json', 'r') as f:
        array = json.load(f)
    intent_lists = [x[0] for x in array]
    code_lists = [x[1] for x in array]
    slot_nums = [x[2] for x in array]
    scores = [x[3] for x in array]
    
    intent_flat_list = []
    for intent_list in intent_lists:
        intent_flat_list.extend(intent_list)

    code_flat_list = []
    for code_list in code_lists:
        code_flat_list.extend(code_list)
    
    word_size = max(intent_flat_list) + 1
    code_size = max(code_flat_list) + 1

    # trainset = ScoreDataset(intent_lists, code_lists, slot_nums, scores)
    trainloader = get_train_loader(intent_lists, code_lists, slot_nums, scores, hyperP)

    model = ScoreNet(word_size, code_size, hyperP)
    
    optimizer = optim.Adam(model.parameters(), lr=hyperP['lr'], weight_decay=1e-4)
    lr_keep_rate = hyperP['lr_keep_rate']
    if lr_keep_rate != 1.0:
        lr_reduce_f = lambda epoch: lr_keep_rate ** epoch
        scheduler = LambdaLR(optimizer, lr_lambda=lr_reduce_f)

    loss_f = torch.nn.MSELoss()

    losses = []
    for e in range(hyperP['max_epochs']):
        loss = train(model, trainloader, optimizer, loss_f, hyperP)
        losses.append(loss)
        model.save()
        print('model saved')
        if lr_keep_rate != 1.0:
            scheduler.step()
