import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils as U
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import model.pointer_net as Pointer_Net
from asdl.hypothesis import Hypothesis
from asdl.lang.py3.py3_transition_system import *

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
        self.action_size = action_size + 1  # add one for padding
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
                                encoder_outputs).squeeze(1)

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
                if np.random.uniform() < 0.90:   #0.95 is teacher forcing rate
                    embed_tm1 = embed[:, t - 1, :]
                else:
                    __, best_act = torch.max(action_logits, 1)
                    embed_tm1 = self.emb(best_act)

            # decode one step
            hiddens, att_context = self.decode_step(embed_tm1, hiddens, sentence_encoding, batch_lens, att_context)

            # classify action types
            hiddens_with_attention = torch.cat((hiddens[2][0], att_context), dim=1)
            action_logits = self.linear(hiddens_with_attention)
            logits_action_type[:, t, :] = action_logits

            # get copy predictions and copy ground truth
            for perform_copy_ind in [i for i, num in enumerate(padded_actions[:, t].tolist()) if
                                     num == action_index_copy]:
                encoding_info = sentence_encoding[perform_copy_ind, :, :]
                hidden_state = hiddens[2][0][perform_copy_ind, :]
                copy_logits = self.pointer_net(encoding_info, act_lens[perform_copy_ind], hidden_state)
                src_token_ind = batch_act_infos[perform_copy_ind][t].src_token_position
                assert src_token_ind != -1
                logits_copy_list.append(copy_logits)
                tgt_copy_list.append(src_token_ind)

            # get token predictions and token ground truth
            for perform_gen_ind in [i for i, num in enumerate(padded_actions[:, t].tolist()) if
                                    num == action_index_gen]:
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
    
    def decode_evaluate(self, intent, intent_text, encoder_hidden, sentence_encoding, 
                        action_index_copy, action_index_gen, act_lst, token_lst,
                        batch_lens, ast_action, unknown_token_index=0):
        """
        return: a list of hypotheses, ranked by decreasing score.
                (In the case of greedy search, returns a list with length 1)
        """
        assert len(sentence_encoding) == 1

        ## initialize hidden states
        batch_size = 1
        hiddens = [encoder_hidden, encoder_hidden, encoder_hidden]
        action_embed_tm1 = torch.zeros(batch_size, self.action_embed_size)

        ## initialize context vector
        att_context = torch.zeros(batch_size, self.encoder_hidden_size)

        ## for each time step
        my_early_stop_cnt = 1
        hyp = Hypothesis()
        while not hyp.completed and my_early_stop_cnt<200:
            my_early_stop_cnt = my_early_stop_cnt+1
            # decode one step
            hiddens, att_context = self.decode_step(action_embed_tm1, hiddens, sentence_encoding,
                                                    batch_lens, att_context)

            # classify action types
            hiddens_with_attention = torch.cat((hiddens[2][0], att_context), dim=1)
            logits_action_type = self.linear(hiddens_with_attention).view(-1).numpy()
            logits_action_type[-1] = - float('inf')
            inds = list(np.argsort(logits_action_type))
            inds.reverse()

            found_valid_next_action = False
            for ind in inds:
                # if apply rule
                if ind != action_index_copy and ind != action_index_gen:
                    act = act_lst[ind]
                    if not ast_action.is_valid_action(hyp, act):
                        continue
                    hyp.apply_action(act)
                    found_valid_next_action = True

                # if copy token from src
                elif ind == action_index_copy:
                    copy_action = GenTokenAction('')
                    if not ast_action.is_valid_action(hyp, copy_action):
                        continue
                    encoding_info = sentence_encoding[0, :, :]
                    hidden_state = hiddens[2][0][0, :]
                    copy_logits = self.pointer_net(encoding_info, batch_lens[0], hidden_state)
                    _, copy_ind = torch.max(copy_logits, 0)
                    copy_token = intent_text[0][copy_ind]
                    copy_action.token = copy_token
                    hyp.apply_action(copy_action)
                    found_valid_next_action = True

                # if use known tokens
                else:
                    gen_action = GenTokenAction('')
                    if not ast_action.is_valid_action(hyp, gen_action):
                        continue
                    hidden_state = hiddens[2][0][0, :]
                    att_context_gen = att_context[0, :]
                    gen_hidden_with_att = torch.cat((hidden_state, att_context_gen), dim=0)
                    gen_logits = self.linear_gen(gen_hidden_with_att)
                    _, gen_ind = torch.topk(gen_logits.view(-1), 2)
                    if gen_ind[0].item() == unknown_token_index:
                        gen_token = token_lst[gen_ind[1]]
                    else:
                        gen_token = token_lst[gen_ind[0]]
                    gen_action.token = gen_token
                    hyp.apply_action(gen_action)
                    found_valid_next_action = True

                if found_valid_next_action:
                    action_embed_tm1 = self.emb(torch.LongTensor([ind]))
                    break
            
            assert found_valid_next_action
        return hyp

    def __get_valid_continue_action_list(self, hyp, inds, act_lst, ast_action, action_index_copy, action_index_gen):
        valid_lst = []
        for ind in inds:
            if ind == action_index_copy or ind == action_index_gen:
                continue
            if ast_action.is_valid_action(hyp, act_lst[ind]):
                valid_lst.append(ind)

        if ast_action.is_valid_action(hyp, GenTokenAction('')):
            valid_lst += [action_index_copy, action_index_gen]
        return valid_lst

    def decode_evaluate_beam(self, intent, intent_text, 
                             encoder_hidden, sentence_encoding, 
                             action_index_copy, action_index_gen, act_lst, token_lst,
                             batch_lens, ast_action, beam_size=100, unknown_token_index=0,
                             max_time_step=40):
        """
        return: a list of hypotheses, ranked by decreasing score.
        """
        assert len(sentence_encoding) == 1

        ## initialize hidden states and context vector
        batch_size = 1
        hiddens = [encoder_hidden, encoder_hidden, encoder_hidden]
        action_embed_tm1 = torch.zeros(batch_size, self.action_embed_size)
        att_context = torch.zeros(batch_size, self.encoder_hidden_size)

        ## Generate beam_size of initial hypotheses
        valid_action_inds = self.__get_valid_continue_action_list(hyp, range(self.action_size - 1), 
                                                                  act_lst, ast_action, action_index_copy, 
                                                                  action_index_gen)
        hiddens, att_context = self.decode_step(action_embed_tm1, hiddens, sentence_encoding, 
                                                batch_lens, att_context)
        hiddens_with_attention = torch.cat((hiddens[2][0], att_context), dim=1)
        logits_action_type = self.linear(hiddens_with_attention).view(-1)
        valid_logits = logits_action_type[valid_action_inds]
        log_probs_action_type = F.log_softmax(valid_logits, dim=0).view(-1).numpy()
        sorted_ind = np.argsort(-log_probs_action_type, axis=None)  
        hyp_infos = []   # format: [(hyp, action_embed_tm1, hiddens, att_context)_1, (hyp, action_embed_tm1, hiddens, att_context)_2, ...]

        for i in range(min(beam_size, len(sorted_ind))):
            ind = sorted_ind[i]
            if ind == action_index_copy or ind == action_index_gen or ind == self.action_size - 1:
                continue

            # push new hyp to list
            assert ast_action.is_valid_action(hyp, act_lst[ind])
            hyp = Hypothesis()
            hyp.apply_action(act_lst[ind])
            hyp.score += log_probs_action_type[ind].item()
            action_embed_tm1 = self.emb(torch.LongTensor([ind]))
            hyp_infos.append((hyp, action_embed_tm1, hiddens, att_context))

        ## for each time step...
        t = 0
        print("beam searching with size {}".format(beam_size))
        # if t < max_time_step, continue training if any one of the hyp is incomplete;
        # otherwise, continue training until we have one complete hyp
        while (t < max_time_step and len(completed_hyps) < beam_size \
            and any((not hyp_info[0].completed) for hyp_info in hyp_infos)):

            tmp_hyp_infos = []
            if t % 20 == 0:
                print("beam search step {}".format(t))

            for hyp_info in hyp_infos:
                if hyp_info[0].completed:
                    # Save only the hypothesis for completed ones
                    completed_hyps.append(hyp_info[0])
                    continue

                # Unwrap information
                hyp, action_embed_tm1, hiddens, att_context = hyp_info
                valid_action_inds = self.__get_valid_continue_action_list(hyp, range(self.action_size - 1), 
                                                                          act_lst, ast_action, action_index_copy, 
                                                                          action_index_gen)

                # decode one step
                hiddens, att_context = self.decode_step(action_embed_tm1, hiddens, sentence_encoding, 
                                                        batch_lens, att_context)

                # classify action types
                hiddens_with_attention = torch.cat((hiddens[2][0], att_context), dim=1)
                logits_action_type = self.linear(hiddens_with_attention).view(-1)
                valid_logits = logits_action_type[valid_action_inds]
                log_probs_action_type = F.log_softmax(valid_logits, dim=0).view(-1).numpy()
                sorted_ind = np.argsort(-log_probs_action_type, axis=None)
                
                # Generate beam_size of new hypotheses for each old hypothesis
                for i in range(min(beam_size, len(sorted_ind))):
                    # Choose an action index randomly (first action can't be copy or gen)
                    valid_lst_ind = sorted_ind[i]
                    ind = valid_action_inds[valid_lst_ind]
                    new_action_score = log_probs_action_type[valid_lst_ind]
                    action_embed_tm1 = self.emb(torch.LongTensor([ind]))

                    # if apply rule
                    if ind != action_index_copy and ind != action_index_gen:
                        act = act_lst[ind]
                        assert ast_action.is_valid_action(hyp, act)
                    
                    # if copy token from src
                    elif ind == action_index_copy:
                        act = GenTokenAction('')
                        assert ast_action.is_valid_action(hyp, act)
                        encoding_info = sentence_encoding[0, :, :]
                        hidden_state = hiddens[2][0][0, :]
                        copy_logits = self.pointer_net(encoding_info, batch_lens[0], hidden_state).view(-1)
                        _, copy_ind = torch.max(copy_logits, 0)
                        act.token = intent_text[0][copy_ind]
                        
                    # if use known tokens
                    else:
                        act = GenTokenAction('')
                        assert ast_action.is_valid_action(hyp, act)
                        hidden_state = hiddens[2][0][0, :]
                        att_context_gen = att_context[0, :]
                        gen_hidden_with_att = torch.cat((hidden_state, att_context_gen), dim=0)
                        gen_logits = self.linear_gen(gen_hidden_with_att)
                        _, gen_ind = torch.topk(gen_logits.view(-1), 2)
                        if gen_ind[0].item() == unknown_token_index:
                            act.token = token_lst[gen_ind[1]]
                        else:
                            act.token = token_lst[gen_ind[0]]
                    
                    # push new hyp to list
                    new_hyp = hyp.clone_and_apply_action(act)
                    new_hyp.score += new_action_score
                    tmp_hyp_infos.append((new_hyp, action_embed_tm1, hiddens, att_context))

            # Truncate the list to top beam_size of hypotheses
            hyp_infos = sorted(tmp_hyp_infos, key=lambda x: x[0].score / float(len(x[0].actions)), reverse=True)[:beam_size]
            t += 1

        assert len(completed_hyps) > 0  
        print("beam search done")
        return sorted(completed_hyps, key=lambda x: x.score / float(len(x.actions)), reverse=True)[0]

    def decode_evaluate_random(self, intent, intent_text, 
                               encoder_hidden, sentence_encoding, 
                               action_index_copy, action_index_gen, act_lst, token_lst,
                               batch_lens, ast_action, random_size=200, unknown_token_index=0,
                               max_time_step=200):
        """
        return: a list of hypotheses, ranked by decreasing score.
        """
        assert len(sentence_encoding) == 1

        ## initialize hidden states and context vector
        batch_size = 1
        hiddens = [encoder_hidden, encoder_hidden, encoder_hidden]
        action_embed_tm1 = torch.zeros(batch_size, self.action_embed_size)
        att_context = torch.zeros(batch_size, self.encoder_hidden_size)

        # print("Random searching with size {}...".format(random_size))
        completed_hyps = []
        while len(completed_hyps) < random_size:
            # if len(completed_hyps) % 20:
                # print("random search {}".format(t))

            ## for each time step
            hyp = Hypothesis()
            t = 0
            while not hyp.completed and t < max_time_step:
                t += 1

                # decode one step
                hiddens, att_context = self.decode_step(action_embed_tm1, hiddens, sentence_encoding,
                                                        batch_lens, att_context)

                # get valid next action
                valid_action_inds = self.__get_valid_continue_action_list(hyp, range(self.action_size - 1), 
                                                                          act_lst, ast_action, action_index_copy, 
                                                                          action_index_gen)
                rand_ind = np.random.choice(valid_action_inds)
                action_embed_tm1 = self.emb(torch.LongTensor([rand_ind]))

                # get score
                hiddens_with_attention = torch.cat((hiddens[2][0], att_context), dim=1)
                logits_action_type = self.linear(hiddens_with_attention).view(-1)
                log_probs_action_type = F.log_softmax(logits_action_type, dim=0).numpy()
                hyp.score += log_probs_action_type[rand_ind].item()

                # if apply rule
                if rand_ind != action_index_copy and rand_ind != action_index_gen:
                    act = act_lst[rand_ind]
                    assert ast_action.is_valid_action(hyp, act)
                    hyp.apply_action(act)

                # if copy token from src
                elif rand_ind == action_index_copy:
                    copy_action = GenTokenAction('')
                    assert ast_action.is_valid_action(hyp, copy_action)
                    encoding_info = sentence_encoding[0, :, :]
                    hidden_state = hiddens[2][0][0, :]
                    copy_logits = self.pointer_net(encoding_info, batch_lens[0], hidden_state)
                    _, copy_ind = torch.max(copy_logits, 0)
                    copy_token = intent_text[0][copy_ind]
                    copy_action.token = copy_token
                    hyp.apply_action(copy_action)

                # if use known tokens
                else:
                    gen_action = GenTokenAction('')
                    assert ast_action.is_valid_action(hyp, gen_action)
                    hidden_state = hiddens[2][0][0, :]
                    att_context_gen = att_context[0, :]
                    gen_hidden_with_att = torch.cat((hidden_state, att_context_gen), dim=0)
                    gen_logits = self.linear_gen(gen_hidden_with_att)
                    _, gen_ind = torch.topk(gen_logits.view(-1), 2)
                    if gen_ind[0].item() == unknown_token_index:
                        gen_token = token_lst[gen_ind[1]]
                    else:
                        gen_token = token_lst[gen_ind[0]]
                    gen_action.token = gen_token
                    hyp.apply_action(gen_action)

            if hyp.completed:
                completed_hyps.append(hyp)
        return sorted(completed_hyps, key=lambda x: x.score / float(len(x.actions)), reverse=True)[0]


class Model(nn.Module):
    def __init__(self, hyperParams, action_size, token_size, word_size, 
                 action_index_copy, action_index_gen,
                 encoder_lstm_layers=3, unknown_token_index=0):
        super(Model, self).__init__()
        self.hyperParams = hyperParams
        self.encoder = Encoder(hyperParams.embed_size,
                               word_size,
                               hyperParams.hidden_size,
                               lstm_layers=encoder_lstm_layers)
        self.decoder = Decoder(hyperParams.action_embed_size,
                               hyperParams.hidden_size * 2,
                               hyperParams.hidden_size,
                               action_size,
                               token_size)
        self.action_index_copy = action_index_copy
        self.action_index_gen = action_index_gen
        self.unknown_token_index = unknown_token_index

    def forward(self, x):
        intent, batch_act_infos = x
        sentence_encoding, batch_lens, hidden = self.encoder(intent)
        return self.decoder.decode(batch_act_infos, hidden, sentence_encoding, self.action_index_copy,
                                   self.action_index_gen, batch_lens)
    
    def parse(self, intent, intent_texts, act_lst, token_lst, ast_action, decode_method='random', random_size=100):
        """
        src_sentence: tensor of size (1, sentence_length). 1 is the batch size.
        return: a list of hypotheses, ranked by decreasing score.
                (In the case of greedy search, returns a list with length 1)
        """
        # can only handle batch size of 1
        assert len(intent) == 1
        hyperParams = self.hyperParams
        sentence_encoding, batch_lens, hidden = self.encoder(intent)

        with torch.no_grad():
            if decode_method == 'random':
                return self.decoder.decode_evaluate_random(intent=intent,
                                        intent_text = intent_texts,
                                        encoder_hidden=hidden, 
                                        sentence_encoding=sentence_encoding,
                                        action_index_copy=self.action_index_copy, 
                                        action_index_gen=self.action_index_gen, 
                                        act_lst=act_lst,
                                        token_lst=token_lst,
                                        batch_lens=batch_lens,
                                        ast_action=ast_action,
                                        random_size=random_size, 
                                        unknown_token_index=self.unknown_token_index, 
                                        max_time_step=200)
            elif decode_method == "beam":
                return self.decoder.decode_evaluate_beam(intent=intent,
                                    intent_text = intent_texts,
                                    encoder_hidden=hidden, 
                                    sentence_encoding=sentence_encoding,
                                    action_index_copy=self.action_index_copy, 
                                    action_index_gen=self.action_index_gen, 
                                    act_lst=act_lst,
                                    token_lst=token_lst,
                                    batch_lens=batch_lens,
                                    ast_action=ast_action,
                                    beam_size=self.hyperParams.beam_size,
                                    unknown_token_index=self.unknown_token_index, 
                                    max_time_step=80)
            else:
                return self.decoder.decode_evaluate(intent=intent,
                                    intent_text = intent_texts,
                                    encoder_hidden=hidden, 
                                    sentence_encoding=sentence_encoding,
                                    action_index_copy=self.action_index_copy, 
                                    action_index_gen=self.action_index_gen, 
                                    act_lst=act_lst,
                                    token_lst=token_lst,
                                    batch_lens=batch_lens,
                                    ast_action=ast_action,
                                    unknown_token_index=self.unknown_token_index)
                

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
