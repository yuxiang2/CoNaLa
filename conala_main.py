#!/usr/bin/env python
# coding=utf-8

""" This script includes the high level training and evaluating routines. """

from __future__ import print_function
from collections import namedtuple
from itertools import chain
import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np
import os
import os.path
#import pickle
import sys
import time
#import token
#import tokenize
import torch
#from tqdm import tqdm
#import traceback



from common.registerable import Registrable
#from dataset import Dataset, Example
from dataset import *
#import dataset.bleu_score as bleu_score
#from dataset.evaluator import ConalaEvaluator
from dataset.util import tokenize_for_bleu_eval
#from model import nn_utils, utils
from model.parsers import Model
#from model.utils import GloveHelper, get_parser_class
import preprocess_temp as P



hyperParamMap = {
    #### General configuration ####
    'cuda': torch.cuda.is_available(),  # Use gpu
    'asdl_file': '',                    # Path to ASDL grammar specification
    'mode': 'train',                    # train or test

    #### Modularized configuration ####
    'parser': 'tranX',  # which parser model to use

    #### Model configuration ####
    'lstm': 'lstm',     # Type of LSTM used, currently only standard LSTM cell is supported

    #### Embedding sizes ####
    'embed_size': 128,         # Size of word embeddings
    'action_embed_size': 128,  # Size of ApplyRule/GenToken action embeddings
    'field_embed_size': 64,    # Embedding size of ASDL fields
    'type_embed_size': 64,     # Embeddings ASDL types

    #### Hidden sizes ####
    'hidden_size': 256,        # Size of LSTM hidden states
    'ptrnet_hidden_dim': 32,   # Hidden dimension used in pointer network
    'att_vec_size': 256,       # Size of attentional vector

    #### readout layer ####
    'no_query_vec_to_action_map': False,    # Do not use additional linear layer to transform the attentional vector for computing action probabilities
    'readout': 'identity',                  # Type of activation if using additional linear layer
    'query_vec_to_action_diff_map': False,  # Use different linear mapping 

    #### parent information switch for decoder LSTM ####
    'no_parent_production_embed': False,    # Do not use embedding of parent ASDL production to update decoder LSTM state
    'no_parent_field_embed': False,         # Do not use embedding of parent field to update decoder LSTM state
    'no_parent_field_type_embed': False,    # Do not use embedding of the ASDL type of parent field to update decoder LSTM state
    'no_parent_state': True,                # Do not use the parent hidden state to update decoder LSTM state
    'no_input_feed': False,                 # Do not use input feeding in decoder LSTM
    'no_copy': False,                       # Do not use copy mechanism

    #### Training ####
    'vocab': '',                            # Path of the serialized vocabulary
    'train_file': '',                       # path to the training target file
    'dev_file': '',                         # path to the dev source file
    'batch_size': 10,                       # Batch size
    'dropout': 0.,                          # dropout rate
    'word_dropout': 0.,                     # Word dropout rate
    'decoder_word_dropout', 0.,             # Word dropout rate on decoder
    'primitive_token_label_smoothing': 0.0, # Apply label smoothing when predicting primitive tokens
    'src_token_label_smoothing': 0.0,       # Apply label smoothing in reconstruction model when predicting source tokens
    'negative_sample_type': 'best',         # 
    'action_lst': None,
    'word_lst': None,
    'token_lst': None,

    #### training schedule details ####
    'valid_metric': 'acc',                # Metric used for validation
    'valid_every_epoch': 1,               # Perform validation every x epoch
    'log_every': 10,                      # Log training statistics every n iterations
    'save_to': 'model',                   # Save trained model to
    'save_all_models': False,             # Save all intermediate checkpoints
    'patience': 5,                        # Training patience
    'max_num_trial': 10,                  # Stop training after x number of trials
    'glorot_init': False,                 # Use glorot initialization
    'clip_grad': 5.,                      # Clip gradients
    'max_epoch': 10,                      # Maximum number of training epoches
    'optimizer': 'Adam',                  # optimizer
    'lr': 0.001,                          # Learning rate
    'lr_decay': 0.5,                      # decay learning rate if the validation performance drops
    'lr_decay_after_epoch': 0,            # Decay learning rate after x epoch
    'decay_lr_every_epoch': False,        # force to decay learning rate after each epoch
    'reset_optimizer': False,             # Whether to reset optimizer when loading the best checkpoint
    'verbose': True,                      # Verbose mode

    #### decoding/validation/testing ####
    'load_model': None,                   # Load a pre-trained model
    'beam_size': 5,                       # Beam size for beam search
    'random_size': 50,                    # Random size for random search
    'decode_max_time_step': 100,          # Maximum number of time steps used in decoding and sampling
    'sample_size': 5,                     # Sample size
    'test_file': '',                      # Path to the test file
    'save_decode_to': None,               # Save decoding results to file
}

HyperParams = namedtuple('HyperParams', list(hyperParamMap.keys()), verbose=False)
hyperParams = HyperParams(**hyperParamMap)



for e in range(20):
    epoch_begin = time.time()
    for batch_ind, x in enumerate(train_loader):
        optimizer.zero_grad()

        (action_logits, action_labels), (copy_logits, copy_labels), (token_logits, token_labels) = model(x)

        loss1 = lossFunc(action_logits, action_labels)
        loss2 = torch.DoubleTensor([0.0])
        if len(copy_logits) > 0:
            loss2 = lossFunc(copy_logits, copy_labels)
        loss3 = torch.DoubleTensor([0.0])
        if len(token_logits) > 0:
            loss3 = lossFunc(token_logits, token_labels)

        total_loss = loss1 + loss2.double() + loss3.double()
        total_loss.backward()

        # clip gradient
        if hyperParams.clip_grad > 0.:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hyperParams.clip_grad)

        optimizer.step()

        if batch_ind % hyperParams.log_every == hyperParams.log_every - 1:
            print("Action loss: {}".format(loss1.data))
            print("Copy loss: {}".format(loss2.data))
            print("Token loss: {}".format(loss3.data))
            print('-------------------------------------------------------')
            report_loss = report_examples = 0.

    print('epoch elapsed %ds' % (time.time() - epoch_begin))


def test(model, params, ast_action, test_loader, target_lst):
    model.eval()
    decode_results = []
    for example_ind, (src_sentence, _) in enumerate(test_loader):
        decoded_hyp = model.parse(src_sentence)
        code = ast_action.actions2code(decoded_hyp.actions)
        code_token_list = tokenize_for_bleu_eval(code)
        target_token_list = tokenize_for_bleu_eval(target_lst[example_ind])

        # First argument should be list of list of words from ground truth (in our case only one ground truth)
        # Second argument should be our prediction, and it's a list of word
        bleu = sentence_bleu([target_token_list],
                             code_token_list,
                             smoothing_function=SmoothingFunction().method3)

        print("Intent:       {}".format(src_sentence))
        print("Ground Truth: {}".format(target_lst[example_ind]))
        print("Predicted:    {}".format(code))
        print("BLEU score:   {}\n".format(bleu))
        decode_results.append((src_sentence, target_lst[example_ind], code, bleu))
        
    # if params.save_decode_to:
    #     pickle.dump(decode_results, open(params.save_decode_to, 'wb'))
    return decode_results
    




if __name__ == '__main__':
    # load train and test dataset
    directory = './conala-corpus/'
    train_file = directory + 'train.json'
    test_file = directory + 'test.json'
    with open(train_file) as f:
        train_data = json.load(f)
    with open(test_file) as f:
        test_data = json.load(f)

    # intent processing includes lowercase, remove punctuation'?'
    train_intent, train_codes = P.process_data(train_data)
    test_intent, test_codes = P.process_data(test_data)

    # this class is used for code2actions and actions2code
    ast_action = P.Ast_Action()
    train_actions = []
    for code in train_codes:
        train_actions.append(ast_action.code2actions(code))

    # word list, action list and token list.
    word_lst = P.vocab_list(train_intent, cut_freq=2)
    act_lst, token_lst = P.action_list(train_actions, cut_freq=5)
    word2num = dict(zip(word_lst, range(0,len(word_lst))))
    act2num = dict(zip(act_lst, range(0,len(act_lst))))
    token2num = dict(zip(token_lst, range(0,len(token_lst))))
    hyperParams.word_lst = word_lst
    hyperParams.action_lst = act_lst
    hyperParams.token_lst = token_lst

    # Get dataloaders
    train_loader = P.get_train_loader(train_intent, train_actions, word2num, act2num, token2num)
    test_loader = P.get_test_loader(test_intent, word2num)
    action_index_copy = act2num[P.GenTokenAction('copy')]
    action_index_gen = act2num[P.GenTokenAction('token')]

    # TODO: fix this so that we have more control over training, testing and where the model comes from (trained or loaded)
    if hyperParams.mode == 'train':
        model = Model(hyperParams=hyperParams, action_size=len(act_lst), token_size=len(token_lst), 
                      word_size=len(word_lst), action_index_copy=action_index_copy, action_index_copy=action_index_gen, 
                      encoder_lstm_layers=3, unknown_token_index=token2num['<UNK>'])
        train(model=model, train_loader=train_loader, dev_loader=None, params=hyperParams)
    elif hyperParams.mode == 'test':
        assert hyperParams.load_model
        print('load model from [%s]' % params.load_model, file=sys.stderr)
        model = Model.load(hyperParams=hyperParams, action_size=len(act_lst), token_size=len(token_lst), 
                           word_size=len(word_lst), encoder_lstm_layers=3):
        test(model=model, params=hyperParams, test_loader=test_loader)
    else:
        raise RuntimeError('unknown mode')
