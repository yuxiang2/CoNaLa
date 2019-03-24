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

# from asdl.hypothesis import Hypothesis, GenTokenAction
from common.registerable import Registrable
from dataset.action_info import ActionInfo
# from dataset.decode_hypothesis import DecodeHypothesis
# from model import nn_utils
from model.pointer_net import PointerNet


class Encoder(nn.Module):
    def __init__(self, embed_size, word_size, hidden_size, lstm_layers=3):
        super(Encoder, self).__init__()
        self.emb = nn.Embedding(word_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers=lstm_layers, bidirectional=True)
        nn.init.xavier_normal_(self.emb.weight.data)

    def forward(self, x):
        embed = [self.emb(datapoint.long()) for datapoint in x]
        packed = U.rnn.pack_sequence(embed)
        outputs_packed, hidden = self.rnn(packed, None)
        outputs, lens = U.rnn.pad_packed_sequence(outputs_packed)
        hidden_h, hidden_c = hidden
        hidden_h = hidden_h[-1] + hidden_h[-2]
        hidden_c = hidden_c[-1] + hidden_c[-2]
        outputs = outputs.permute(1, 0, 2)
        outputs = outputs.contiguous()
        return outputs, lens, (hidden_h, hidden_c)


class Decoder(nn.Module):
    def __init__(self, action_embed_size, attn_size, encoder_hidden_size, hidden_size, action_size, token_size):
        super(Decoder, self).__init__()

        self.action_embed_size = action_embed_size
        self.action_size = action_size
        self.attn_size = attn_size
        self.encoder_hidden_size = encoder_hidden_size
        self.hidden_size = hidden_size
        self.action_size = action_size + 1  # add one for padding
        self.token_size = token_size
        self.emb = nn.Embedding(self.action_size, self.action_embed_size)

        # these two are for attention
        self.attn = nn.Sequential(
            nn.Linear(self.hidden_size + self.encoder_hidden_size, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
            nn.Tanh()
        )
        self.attn_combine = nn.Linear(self.encoder_hidden_size + self.action_embed_size, self.action_embed_size)

        self.cell1 = nn.LSTMCell(self.action_embed_size, self.hidden_size)
        self.cell2 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.cell3 = nn.LSTMCell(self.hidden_size, self.hidden_size)

        self.pointer_net = PointerNet(self.hidden_size)

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
                nn.init.xavier_normal_(layer.weight)
        for layer in self.linear_gen.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
        nn.init.xavier_normal_(self.emb.weight.data)

    ## pad code within a batch to the same length so 
    ## that we can do batch rnn
    def process_code(self, batched_actions_info):
        batch_idxs = []
        max_leng = 0
        end_symbol = self.action_size - 1
        for actions_info in batched_actions_info:
            idxs = [action_info.idx for action_info in actions_info]
            idxs.append(end_symbol)
            max_leng = max(max_leng, len(idxs))
            batch_idxs.append(idxs)

        tensor_batch_idxs = torch.LongTensor(len(batched_actions_info), max_leng)
        for i in range(len(batch_idxs)):
            origin_leng = len(batch_idxs[i])
            tensor_batch_idxs[i, :origin_leng] = torch.LongTensor(batch_idxs[i])
            tensor_batch_idxs[i, origin_leng:] = end_symbol
        return tensor_batch_idxs

    def decode_step(self, action_embed_tm1, hiddens, sentence_encoding, batch_lens, att_context):
        """
        :param action_embed_tm1: prev action embedding as input （batch, hidden_size)
        :param hiddens: decoder hiddens (3, batch, hidden_size)
        :param sentence_encoding: encoder output (batch, seq_len, hidden_size)
        :return:
        """
        assert len(hiddens) == 3

        ##to_combine_tmp (batch, 2* hidden_size)
        to_combine_tmp = torch.cat((action_embed_tm1, att_context), 1)
        att_t = self.attn_combine(to_combine_tmp)

        h_t0, cell_t0 = self.cell1(att_t, hiddens[0])
        h_t1, cell_t1 = self.cell2(h_t0, hiddens[1])
        h_t2, cell_t2 = self.cell3(h_t1, hiddens[2])

        ################################################################################
        ## attentioned result for encoder output
        ## （batch, seq_len, hidden_size)
        max_length = sentence_encoding.size()[1]
        batch_size = h_t2.size()[0]

        hidden_for_att = h_t2.repeat(max_length, 1, 1).permute(1, 0, 2).contiguous()
        # intput = action_embed_tm1
        encoder_outputs = sentence_encoding

        att_features = torch.cat((encoder_outputs, hidden_for_att), 2).view(batch_size * max_length, -1)
        attn_input = self.attn(att_features)
        attn_input = attn_input.view(batch_size, max_length)
        for i, length in enumerate(batch_lens):
            attn_input[i, length:] = -float('inf')
        attn_weights = F.softmax(attn_input, dim=1)

        ## (batch, 1, maxlen) bmm (batch, seq_len，hidden_size)
        att_context = torch.bmm(attn_weights.unsqueeze(1),
                                encoder_outputs).squeeze()

        return [(h_t0, cell_t0), (h_t1, cell_t1), (h_t2, cell_t2)], att_context

    def decode(self, batch_act_infos, encoder_hidden, sentence_encoding, action_index_copy, action_index_gen,
               batch_lens):
        batch_size = len(batch_act_infos)
        act_lens = [len(act_infos) for act_infos in batch_act_infos]

        padded_x = self.process_code(batch_act_infos)
        length = len(padded_x[0])
        embed = self.emb(padded_x)

        ## initialize hidden states
        hiddens = [encoder_hidden, encoder_hidden, encoder_hidden]

        ## logits for classify action_types, tokens, copy_pos
        logits_action_type = torch.DoubleTensor(batch_size, length, self.action_size)
        logits_copy_list = []
        tgt_copy_list = []
        logits_gen_list = []
        tgt_gen_list = []
        att_context = torch.zeros(batch_size, self.encoder_hidden_size)
        ## for each time step
        for t in range(length):
            # previous action embedding
            if t == 0:
                ## if no previous action, initialize to zero vector
                embed_tm1 = torch.zeros(batch_size, self.action_embed_size)
            else:
                embed_tm1 = embed[:, t - 1, :]

            # decode one step
            # att_t (batch, 1, hidden_size)
            hiddens, att_context = self.decode_step(embed_tm1, hiddens, sentence_encoding, batch_lens, att_context)

            ## do linear inside for loop is inefficient, but it allows teacher forcing
            logits_action_type[:, t, :] = self.linear(hiddens[2][0])

            for perform_copy_ind in [i for i, num in enumerate(padded_x[:, t].tolist()) if num == action_index_copy]:
                encoding_info = sentence_encoding[:, perform_copy_ind, :]
                hidden_state = hidden3[0][perform_copy_ind, :]
                copy_logits = self.pointer_net(encoding_info, act_lens[perform_copy_ind], hidden_state)
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
    def __init__(self, hyperParams, action_size, token_size, word_size, action_index_copy, action_index_gen,
                 encoder_lstm_layers=3):
        super(Model, self).__init__()
        self.hyperParams = hyperParams
        self.encoder = Encoder(hyperParams.embed_size, word_size, hyperParams.hidden_size,
                               lstm_layers=encoder_lstm_layers)
        self.decoder = Decoder(hyperParams.action_embed_size,
                               hyperParams.att_vec_size,
                               hyperParams.hidden_size * 2,
                               hyperParams.hidden_size,
                               action_size,
                               token_size)
        self.action_index_copy = action_index_copy
        self.action_index_gen = action_index_gen

    def forward(self, x):
        intent, batch_act_infos = x
        sentence_encoding, batch_lens, hidden = self.encoder(intent)
        return self.decoder.decode(batch_act_infos, hidden, sentence_encoding, self.action_index_copy,
                                   self.action_index_gen, batch_lens)

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
