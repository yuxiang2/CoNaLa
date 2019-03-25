import math
import os

import torch
import torch.nn as nn
import torch.nn.utils as U
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import model.pointer_net as Pointer_Net
from asdl.hypothesis import Hypotheses

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
    def __init__(self, action_embed_size, encoder_hidden_size, decoder_hidden_size, action_size, token_size):
        super(Decoder, self).__init__()
        
        ## fields:
        # embedding dimension for action
        self.action_embed_size = action_embed_size  
        
        # encoder output dimensions, used for attention
        self.encoder_hidden_size = encoder_hidden_size 
        self.decoder_hidden_size = decoder_hidden_size
        self.action_size = action_size + 1          # add one for padding
        self.token_size = token_size
        
        ## network structures:
        self.emb = nn.Embedding(self.action_size, self.action_embed_size)

        # these two are for attention, attn is for calculating energy for attention
        self.attn = nn.Sequential(
            nn.Linear(self.decoder_hidden_size + self.encoder_hidden_size, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
            nn.Tanh()
        )
        # attn_combine is used to combine action embedding with context vector
        self.attn_combine = nn.Linear(self.encoder_hidden_size + self.action_embed_size, self.decoder_hidden_size)

        # main LSTM structure
        self.cell1 = nn.LSTMCell(self.decoder_hidden_size, self.decoder_hidden_size)
        self.cell2 = nn.LSTMCell(self.decoder_hidden_size, self.decoder_hidden_size)
        self.cell3 = nn.LSTMCell(self.decoder_hidden_size, self.decoder_hidden_size)

        # pointer network is to determine which src words to copy
        self.pointer_net = Pointer_Net.PointerNet(self.encoder_hidden_size, self.decoder_hidden_size)

        # linear is to classify action types
        self.linear = nn.Sequential(
            nn.Linear(self.decoder_hidden_size + self.encoder_hidden_size, self.decoder_hidden_size),
            nn.ReLU(),
            nn.Linear(self.decoder_hidden_size, self.action_size)
        )
        # linear_gen is to classify tokens
        self.linear_gen = nn.Sequential(
            nn.Linear(self.decoder_hidden_size + self.encoder_hidden_size, self.decoder_hidden_size),
            nn.ReLU(),
            nn.Linear(self.decoder_hidden_size, self.token_size),
        )

        for layer in self.linear.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
        for layer in self.linear_gen.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
        nn.init.xavier_normal_(self.emb.weight.data)

    ## extract action types and pad actions within a batch to the same length so 
    ## that we can do batch rnn
    def process_code(self, batched_actions_info):
        batch_idxs = []
        max_leng = 0
        end_symbol = self.action_size - 1
        
        # extract action types and find max length
        for actions_info in batched_actions_info:
            idxs = [action_info.idx for action_info in actions_info]
            max_leng = max(max_leng, len(idxs))
            batch_idxs.append(idxs)

        # pad to the same length and return tensor
        tensor_batch_idxs = torch.LongTensor(len(batched_actions_info), max_leng)
        for i in range(len(batch_idxs)):
            origin_leng = len(batch_idxs[i])
            tensor_batch_idxs[i, :origin_leng] = torch.LongTensor(batch_idxs[i])
            tensor_batch_idxs[i, origin_leng:] = end_symbol
        return tensor_batch_idxs

    def decode_step(self, action_embed_tm1, hiddens, encoder_outputs, batch_lens, att_context):
        assert len(hiddens) == 3
        batch_size, max_length, _ = encoder_outputs.size()

        # combine action embedding with context vector
        to_combine_tmp = torch.cat((action_embed_tm1, att_context), 1)
        att_t = self.attn_combine(to_combine_tmp)

        # main rnn cells
        h_t0, cell_t0 = self.cell1(att_t, hiddens[0])
        h_t1, cell_t1 = self.cell2(h_t0, hiddens[1])
        h_t2, cell_t2 = self.cell3(h_t1, hiddens[2])

        # concatenate decoder hiddens with encoder outputs, and then flatten it
        hidden_for_att = h_t2.repeat(max_length, 1, 1).permute(1, 0, 2).contiguous()
        att_features = torch.cat((encoder_outputs, hidden_for_att), 2).view(batch_size * max_length, -1)
        
        # goes to attention layer
        attn_input = self.attn(att_features)
        attn_input = attn_input.view(batch_size, max_length)
        attn_weights = F.softmax(attn_input, dim=1)

        # calculate context vector
        att_context = torch.bmm(attn_weights.unsqueeze(1),
                                encoder_outputs).squeeze()

        return ((h_t0, cell_t0), (h_t1, cell_t1), (h_t2, cell_t2)), att_context

    def decode(self, batch_act_infos, encoder_hidden, sentence_encoding, action_index_copy, action_index_gen,
               batch_lens):
        batch_size = len(batch_act_infos)
        # number of valid actions within a batch
        act_lens = [len(act_infos) for act_infos in batch_act_infos]

        padded_actions = self.process_code(batch_act_infos)
        length = len(padded_actions[0])
        embed = self.emb(padded_actions)

        ## initialize hidden states
        hiddens = [encoder_hidden, encoder_hidden, encoder_hidden]

        ## store logits and ground truth for classify action_types, tokens, copy_pos
        logits_action_type = torch.DoubleTensor(batch_size, length, self.action_size)
        logits_copy_list = []
        tgt_copy_list = []
        logits_gen_list = []
        tgt_gen_list = []
        
        # initialize context vector
        att_context = torch.zeros(batch_size, self.encoder_hidden_size)
        
        ## for each time step
        for t in range(length):
            # previous action embedding
            if t == 0:
                ## if no previous action, initialize to zero vector
                embed_tm1 = torch.zeros(batch_size, self.action_embed_size)
            else:
                ## TODO add teacher forcing here, also add token embedding
                embed_tm1 = embed[:, t - 1, :]

            # decode one step
            hiddens, att_context = self.decode_step(embed_tm1, hiddens, sentence_encoding, batch_lens, att_context)

            # classify action types
            hiddens_with_attention = torch.cat((hiddens[2][0], att_context), dim=1)
            logits_action_type[:, t, :] = self.linear(hiddens_with_attention)

            # get copy predictions and copy ground truth
            for perform_copy_ind in [i for i, num in enumerate(padded_actions[:, t].tolist()) if num == action_index_copy]:
                encoding_info = sentence_encoding[perform_copy_ind, :, :]
                hidden_state = hiddens[2][0][perform_copy_ind, :]
                copy_logits = self.pointer_net(encoding_info, act_lens[perform_copy_ind], hidden_state)
                src_token_ind = batch_act_infos[perform_copy_ind][t].src_token_position
                assert src_token_ind != -1
                logits_copy_list.append(copy_logits)
                tgt_copy_list.append(src_token_ind)

            # get token predictions and token ground truth
            for perform_gen_ind in [i for i, num in enumerate(padded_actions[:, t].tolist()) if num == action_index_gen]:
                hidden_state = hiddens[2][0][perform_gen_ind, :]
                att_context_gen = att_context[perform_gen_ind, :]
                gen_hidden_with_att = torch.cat((hidden_state, att_context_gen), dim=0)

                gen_logits = self.linear_gen(gen_hidden_with_att)
                token_ind = batch_act_infos[perform_gen_ind][t].token
                assert token_ind is not None
                logits_gen_list.append(gen_logits)
                tgt_gen_list.append(token_ind)

        ## padded eos actions are not removed, thus acc for actions maybe too high
        return (logits_action_type.view(batch_size * length, -1), padded_actions.view(-1)), \
               (torch.stack(logits_copy_list), torch.LongTensor(tgt_copy_list)), \
               (torch.stack(logits_gen_list), torch.LongTensor(tgt_gen_list))
    
    def decode_evaluate(self, intent, encoder_hidden, sentence_encoding, action_index_copy, action_index_gen, 
                        word_lst, act_lst, token_lst,
                        batch_lens, beam_size=1):
        """
        TODO: implement dis
        return: a list of hypotheses, ranked by decreasing score.
                (In the case of greedy search, returns a list with length 1)
        """
        assert beam_size == 1
        assert len(sentence_encoding) == 1

        ## initialize hidden states
        batch_size = 1
        hiddens = [encoder_hidden, encoder_hidden, encoder_hidden]
        action_tm1 = 
        action_embed_tm1 = torch.zeros(batch_size, self.action_embed_size)

        ## for each time step
        hyp = Hypotheses()
        while not hyp.completed:
            # decode one step
            # att_t (batch, 1, hidden_size)
            hiddens, att_context = self.decode_step(action_embed_tm1, hiddens, sentence_encoding, 
                                                    batch_lens, att_context)

            # classify action types
            hiddens_with_attention = torch.cat((hiddens[2][0], att_context), dim=1)
            logits_action_type = self.linear(hiddens_with_attention)
            _, inds = torch.sort(logits_action_type, descending=True)

            found_valid_next_action = False
            for ind in inds:
                if ind != self.action_index_copy and ind != self.action_index_gen:
                    try:
                        hyp.apply_action(act_lst[ind])
                        action_embed_tm1 = 
                        found_valid_next_action = True
                        break
                    except:
                        pass
                elif ind == self.action_index_copy:
                    try:
                        copy_action = act_lst[ind]
                        encoding_info = sentence_encoding[0, :, :]
                        hidden_state = hiddens[2][0][0, :]
                        copy_logits = self.pointer_net(encoding_info, batch_lens[0], hidden_state)
                        _, copy_ind = torch.max(copy_logits)
                        copy_action.token = word_lst[intent[copy_ind]]
                        hyp.apply_action(copy_action)
                        action_embed_tm1 = 
                        found_valid_next_action = True
                    except:
                        pass
                else: # genToken
                    try:
                        hidden_state = hiddens[2][0][0, :]
                        att_context_gen = att_context[0, :]
                        gen_hidden_with_att = torch.cat((hidden_state, att_context_gen), dim=0)
                        gen_logits = self.linear_gen(gen_hidden_with_att)
                        _, gen_ind = torch.max(gen_logits)
                        hyp.apply_action(act_lst[ind])
                        action_embed_tm1 = 
                        found_valid_next_action = True
                    except:
                        pass

            assert found_valid_next_action
        return [hyp]


class Model(nn.Module):
    def __init__(self, hyperParams, action_size, token_size, word_size, action_index_copy, action_index_gen,
                 encoder_lstm_layers=3):
        super(Model, self).__init__()
        self.hyperParams = hyperParams
        self.encoder = Encoder(hyperParams.embed_size, word_size, hyperParams.hidden_size,
                               lstm_layers=encoder_lstm_layers)
        self.decoder = Decoder(hyperParams.action_embed_size,
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
    
    def parse(self, src_sentence):
        """
        src_sentence: tensor of size (1, sentence_length). 1 is the batch size.
        return: a list of hypotheses, ranked by decreasing score.
                (In the case of greedy search, returns a list with length 1)
        """
        # can only handle batch size of 1
        assert len(src_sentence) == 1
        sentence_encoding, batch_lens, hidden = self.encoder(src_sentence)
        return self.decode_evaluate(intent=src_sentence,
                                    encoder_hidden=hidden, 
                                    sentence_encoding=sentence_encoding,
                                    word_lst=hyperParams.word_lst,
                                    action_lst=hyperParams.action_lst,
                                    token_lst=hyperParams.token_lst,
                                    action_index_copy=self.action_index_copy, 
                                    action_index_gen=self.action_index_gen, 
                                    batch_lens=batch_lens,
                                    beam_size=self.hyperParams.beam_size)

    def save(self, path):
        dir_name = os.path.dirname(path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        params = {
            'state_dict': self.state_dict()
        }
        torch.save(params, path)

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