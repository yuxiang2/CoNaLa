# coding=utf-8
from collections import OrderedDict
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils as U
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from asdl.hypothesis import Hypothesis, GenTokenAction
from asdl.transition_system import ApplyRuleAction, ReduceAction, Action
from common.registerable import Registrable
from common.utils import update_args, init_arg_parser
from dataset.action_info import ActionInfo
from dataset.decode_hypothesis import DecodeHypothesis
from model import nn_utils
from model.attention_util import AttentionUtil
from model.nn_utils import LabelSmoothing
from model.pointer_net import PointerNet


class Encoder(nn.Module):
    def __init__(self, embed_size, hidden_size, word_size, lstm_layers=3):
        super(Encoder, self).__init__()
        self.emb = nn.Embedding(word_size, embed_size)
        self.rnn = nn.LSTM(embed_size, int(hidden_size / 2), num_layers=lstm_layers, bidirectional=True)
        nn.init.xavier_normal(self.emb.weight.data)
        
    def forward(self, x):
        embed = [self.emb(datapoint.long()) for datapoint in x]
        packed = U.rnn.pack_sequence(embed)
        outputs, hidden = self.rnn(packed, None)
        hidden_h, hidden_c = hidden 
        hidden_h = hidden_h[-1] + hidden_h[-2]
        hidden_c = hidden_c[-1] + hidden_c[-2]
        return outputs, (hidden_h, hidden_c)
        
class Decoder(nn.Module): 
    def __init__(self, action_embed_size, encoder_hidden_size, hidden_size, action_size, token_size):
        super(Decoder, self).__init__()
        
        self.action_embed_size = action_embed_size
        self.encoder_hidden_size = encoder_hidden_size
        self.hidden_size = hidden_size
        self.action_size = action_size + 1 # add one for padding
        self.token_size = token_size

        self.emb = nn.Embedding(self.action_size, self.action_embed_size)
        self.cell1 = nn.LSTMCell(self.encoder_hidden_size, self.hidden_size)
        self.cell2 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.cell3 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.pointer_net = PointerNet() # TODO: arguments
        self.linear = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.action_size)
        )
        self.linear_gen = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.token_size),
        )

        for layer in self.linear.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal(layer.weight)
        for layer in self.linear_gen.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal(layer.weight)
        nn.init.xavier_normal(self.emb.weight.data)
        
    ## pad code within a batch to the same length so 
    ## that we can do batch rnn
    def process_code(self, batched_actions_info):
        batch_idxs = []
        max_leng = 0
        end_symbol = self.action_size
        for actions_info in batched_actions_info:
            idxs = [action_info.idx for action_info in actions_info]
            idxs.append(end_symbol)
            max_leng = max(max_leng, len(idxs))
            batch_idxs.append(idxs)
            
        tensor_batch_idxs = torch.LongTensor(len(batched_actions_info), maxlen)
        for i in range(len(batch_idxs)):
            origin_leng = len(batch_idxs[i])
            tensor_batch_idxs[i,:origin_leng] = batch_idxs[i]
            tensor_batch_idxs[i,origin_leng:] = end_symbol
        return tensor_batch_idxs
            
    def decode_step(self, action_embed_tm1, hiddens, sentence_encoding):
        assert len(hiddens) == 3
        att_t = None  # TODO: calculate attention using sentence_encoding
        h_t0, cell_t0 = self.cell1(action_embed_tm1, hiddens[0])
        h_t1, cell_t1 = self.cell2(h_t0, hiddens[1])
        h_t2, cell_t2 = self.cell3(h_t1, hiddens[2])
        return [(h_t0, cell_t0), (h_t1, cell_t1), (h_t2, cell_t2)], att_t

    def decode(self, x, encoder_hidden, sentence_encoding, action_index_copy, action_index_gen):
        batch_size = len(x)
        padded_x = self.process_code(x)
        length = len(padded_x[0])
        embed = self.emb(padded_x)
        
        ## initialize hidden states
        hidden1, hidden2, hidden3 = encoder_hidden, encoder_hidden, encoder_hidden
        
        ## logits
        logits_action_type = torch.DoubleTensor(batch_size, length, self.action_size)
        logits_copy_list = []
        tgt_copy_list = []
        logits_gen_list = []
        tgt_gen_list = []
        
        ## for each time step
        att_vecs = []

        for t in range(length):
            # previous action embedding
            if t == 0:
                embed_tm1 = Variable(self.new_tensor(batch_size, self.decoder_lstm.input_size).zero_(), requires_grad=False)
            else:
                embed_tm1 = embed[:, t - 1, :]

            # decode one step
            hiddens, att_t = decode_step(embed_tm1, [hidden1, hidden2, hidden3], sentence_encoding)
            att_vecs.append(att_t)

            # update previous hidden state
            hidden1, hidden2, hidden3 = **hiddens

            ## do linear inside for loop is inefficient, but it allows teacher forcing
            logits_action_type[:, t, :] = self.linear(hiddens[2][0])


            for perform_copy_ind in [i for i, num in enumerate(padded_x[:, t].tolist()) if num == action_index_copy]:
                encoding_info = sentence_encoding[perform_copy_ind, :, :]
                hidden_state = hidden3[0][perform_copy_ind, :]
                copy_logits = self.pointer_net(encoding_info, hidden_state)
                src_token_ind = x[perform_copy_ind, t].src_token_position
                assert src_token_ind != -1
                logits_copy_list.append(copy_logits)
                tgt_copy_list.append(src_token_ind)
                
            for perform_gen_ind in [i for i, num in enumerate(padded_x[:, t].tolist()) if num == action_index_gen]:
                hidden_state = hidden3[0][perform_gen_ind, :]
                gen_logits = self.linear_gen(hidden_state)
                token_ind = x[perform_gen_ind, t].token
                assert token_ind is not None
                logits_gen_list.append(gen_logits)
                tgt_gen_list.append(token_ind)

        ## padded eos symbols are not removed, thus
        ## calculated accuracy can be too high
        return (logits_action_type.view(batch_size * length, -1), padded_x), \
               (torch.stack(logits_copy_list), torch.LongTensor(tgt_copy_list)), \
               (torch.stack(logits_gen_list), torch.LongTensor(tgt_gen_list))

        
class Model(nn.Module):
    def __init__(self, action_size, hyperParams, token_size, word_size, best_acc=0.0, encoder_lstm_layers=3):
        super(Model, self).__init__()
        self.hyperParams = hyperParams
        self.encoder = Encoder(hyperParams.embed_size, hyperParams.hidden_size, word_size, lstm_layers=encoder_lstm_layers)
        self.decoder = Decoder(hyperParams.action_embed_size, hyperParams.embed_size, 
                               hyperParams.hidden_size, 
                               action_size, token_size)
        self.loss = nn.CrossEntropyLoss()
        self.opt = torch.optim.Adam(self.parameters(), lr=hyperParams.lr)
        self.best_acc = best_acc
        
    def forward(self, intent, code, train=True):
        hidden = self.encoder(intent)
        scores, labels = self.decoder(code, hidden)
        
        # get statistics
        _, predicted = torch.max(scores, 1)
        num_correct = (predicted == labels).sum().item()
        acc = float(num_correct) / len(predicted)
        
        if train:
            # gradient descent
            loss = self.loss(scores, labels)
            self.opt.zero_grad()
            # uncomment to use gradient clipping
            # U.clip_grad_norm_(self.parameters(), 5.0)
            loss.backward()
            self.opt.step()
            
            return loss.item(), acc
        else:
            return acc
        
    def save(self, path):
        dir_name = os.path.dirname(path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        params = {
            'state_dict': self.state_dict()
        }
        torch.save(self.params, path)

    @classmethod
    def load(cls, action_size, hyperParams, token_size, word_size, encoder_lstm_layers=3):
        params = torch.load(hyperParams.load_model, map_location=lambda storage, loc: storage)
        saved_state = params['state_dict']

        parser = cls(action_size, hyperParams, token_size, word_size, encoder_lstm_layers=encoder_lstm_layers)
        parser.load_state_dict(saved_state)

        if hyperParams.cuda: 
            parser = parser.cuda()
        parser.eval()
        return parser
