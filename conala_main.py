#!/usr/bin/env python
# coding=utf-8

""" This script includes the high level training and evaluating routines. """


from __future__ import print_function
from itertools import chain

from collections import namedtuple
import json
import numpy as np
import os
import os.path
import sys
import time
import token
import tokenize
import torch
from tqdm import tqdm
import traceback


from asdl import *
from asdl.asdl import ASDLGrammar
from common.registerable import Registrable
from common.utils import update_args, init_arg_parser
#from dataset import Dataset, Example
from dataset import *
#import dataset.bleu_score as bleu_score
#from dataset.evaluator import ConalaEvaluator
from dataset.util import tokenize_for_bleu_eval
from model import nn_utils, utils
from model.parsers import Parser
from model.utils import GloveHelper, get_parser_class


hyperParamMap = {
    #### General configuration ####
    'cuda': True,      # Use gpu
    'asdl_file': '',   # Path to ASDL grammar specification
    'mode': 'train',   # train or test

    #### Modularized configuration ####
    'parser': 'default_parser',  # which parser model to use

    #### Model configuration ####
    'lstm': 'lstm',    # Type of LSTM used, currently only standard LSTM cell is supported

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
    'verbose': False,                     # Verbose mode

    #### decoding/validation/testing ####
    'load_model': None,                   # Load a pre-trained model
    'beam_size': 5,                       # Beam size for beam search
    'decode_max_time_step': 100,          # Maximum number of time steps used in decoding and sampling
    'sample_size': 5,                     # Sample size
    'test_file': '',                      # Path to the test file
    'save_decode_to': None,               # Save decoding results to file
}

HyperParams = namedtuple('HyperParams', list(hyperParamMap.keys()), verbose=True)
hyperParams = HyperParams(**hyperParamMap)


def train(model, train_dataloader, dev_dataloader, params):
    model.train()
    if params.cuda: 
        model.cuda()

    optimizer_cls = eval('torch.optim.%s' % params.optimizer)  # FIXME: this is evil!
    optimizer = optimizer_cls(model.parameters(), lr=params.lr)

    lossFunc = nn.CrossEntropyLoss()

    # print('use glorot initialization', file=sys.stderr)
    # nn_utils.glorot_init(model.parameters())

    epoch = 0
    report_loss = report_examples = 0.
    while True:
        epoch += 1
        epoch_begin = time.time()

        for batch_ind, x in enumerate(train_dataloader):
            optimizer.zero_grad()

            (action_logits, action_labels), (copy_logits, copy_labels), (token_logits, token_labels) = model(x)
            loss1 = lossFunc(action_logits, action_labels)
            loss2 = 0.0
            if len(copy_logits) > 0:
                loss2 = lossFunc(copy_logits, copy_labels)
            loss3 = 0.0
            if len(token_logits) > 0:
                loss3 = lossFunc(token_logits, token_labels)

            total_loss.backward()

            # clip gradient
            if params.clip_grad > 0.:
                grad_norm = torch.nn.utils.clip_grad_norm(model.parameters(), params.clip_grad)

            optimizer.step()

            if train_iter % params.log_every == 0:
                total_loss = loss1 + loss2 + loss3
                print("Action loss: {}".format(loss1.data))
                print("Copy loss: {}".format(loss2.data))
                print("Token loss: {}".format(loss3.data))
                print("GenToken total loss: {}".format(loss2.data + loss3.data))
                report_loss = report_examples = 0.

        print('[Epoch %d] epoch elapsed %ds' % (epoch, time.time() - epoch_begin), file=sys.stdout)

        # if params.save_all_models:
        #     model_file = params.save_to + '.iter%d.bin' % train_iter
        #     print('save model to [%s]' % model_file, file=sys.stderr)
        #     model.save(model_file)

        if params.decay_lr_every_epoch and epoch > params.lr_decay_after_epoch:
            lr = optimizer.param_groups[0]['lr'] * params.lr_decay
            print('decay learning rate to %f' % lr, file=sys.stderr)

            # set new lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        if epoch == params.max_epoch:
            print('reached max epoch, stop!', file=sys.stderr)
            exit(0)

if __name__ == '__main__':
    if hyperParams.mode == 'train':
        
        train(hyperParams)
    # elif hyperParams.mode == 'test':
    #     test(hyperParams)
    else:
        raise RuntimeError('unknown mode')



# def decode(examples, model, beam_size=1, verbose=False, **kwargs):
#     if verbose:
#         print('evaluating %d examples' % len(examples))
#     was_training = model.training
#     model.eval()

#     decode_results = []
#     count = 0
#     for example in tqdm(examples, desc='Decoding', file=sys.stdout, total=len(examples)):
#         hyps = model.parse(example.src_sent, context=None, beam_size=beam_size)
#         decoded_hyps = []
#         for hyp_id, hyp in enumerate(hyps):
#             got_code = False
#             try:
#                 hyp.code = model.transition_system.ast_to_surface_code(hyp.tree)
#                 got_code = True
#                 decoded_hyps.append(hyp)
#             except:
#                 if verbose:
#                     print("Exception in converting tree to code:", file=sys.stdout)
#                     print('-' * 60, file=sys.stdout)
#                     print('Example: %s\nIntent: %s\nTarget Code:\n%s\nHypothesis[%d]:\n%s' % (example.idx,
#                                                                                              ' '.join(example.src_sent),
#                                                                                              example.tgt_code,
#                                                                                              hyp_id,
#                                                                                              hyp.tree.to_string()), file=sys.stdout)
#                     if got_code:
#                         print()
#                         print(hyp.code)
#                     traceback.print_exc(file=sys.stdout)
#                     print('-' * 60, file=sys.stdout)

#         count += 1
#         decode_results.append(decoded_hyps)

#     if was_training: model.train()
#     return decode_results

# def evaluate(examples, parser, evaluator, beam_size=1, verbose=False, return_decode_result=False, eval_top_pred_only=False):
#     decode_results = decode(examples, parser, beam_size=beam_size, verbose=verbose)
#     eval_result = evaluator.evaluate_dataset(examples, decode_results, fast_mode=eval_top_pred_only)
#     if return_decode_result:
#         return eval_result, decode_results
#     else:
#         return eval_result



# def test(params):
#     test_set = Dataset.from_bin_file(params.test_file)
#     assert params.load_model

#     print('load model from [%s]' % params.load_model, file=sys.stderr)
#     loaded_params = torch.load(params.load_model, map_location=lambda storage, loc: storage)
#     transition_system = loaded_params['transition_system']
#     saved_args = loaded_params['args']
#     saved_args.cuda = params.cuda
#     # set the correct domain from saved arg

#     parser_cls = Registrable.by_name(params.parser)
#     parser = parser_cls.load(model_path=params.load_model, cuda=params.cuda)
#     parser.eval()
#     evaluator = ConalaEvaluator(transition_system, args=params)
#     eval_results, decode_results = evaluate(test_set.examples, parser, evaluator, params,
#                                             verbose=params.verbose, return_decode_result=True)
#     print(eval_results, file=sys.stderr)
#     if params.save_decode_to:
#         pickle.dump(decode_results, open(params.save_decode_to, 'wb'))













# # ============================= Start of Helper Functions ===============================

# """ Parses a file in the natural .jsonl format that the Conala corpus comes in.
#     @param f: .jsonl file containing snippets
#     @return: list of lists of tokens
# """
# def parse_file_json(f):
#     snippet_list = json.load(f)
#     result = []
#     for snippet in snippet_list:
#         toks = tokenize_for_bleu_eval(snippet['snippet'])
#         result.append(toks)
#     return result

# """ This runs the built-in Python tokenizer. Note that it only works on correctly parseable Python programs.
#     @param string: string containing a Python tokenizable code snippet
#     @return: list of code tokens
# """
# def tokenize_code(string, concat_symbol=None):
#     tokens = []
#     string = string.strip().decode('utf-8').encode('ascii', 'strict') #.decode('string_escape')
#     for toknum, tokval, _, _, _  in tokenize.generate_tokens(StringIO(string).readline):
#         # We ignore these tokens during evaluation.
#         if toknum not in [token.ENDMARKER, token.INDENT, token.DEDENT]:
#             tokens.append(tokval.lower())

#     return tokens

# """ This builds the reference list for BLEU scoring
#     @param reference_file_name: The reference file can be downloaded from https://conala-corpus.github.io/ as
#                                 conala_annotations.v1.0.zip/examples.annotated.test.json
#     @return: list of references ready for BLEU scoring
# """
# def get_reference_list(reference_file_name):
#     f_reference = open(reference_file_name)
#     a = parse_file_json(f_reference)
#     a = [[l] for l in a]
#     return a

# """ This scores hypotheses against references using BLEU.
#     @param reference_list: reference list returned by get_reference_list.
#     @param hypothesis_list: list of lists of tokens that a model generates.
#     @return: 3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
#              precisions and brevity penalty.
# """
# def evaluate_bleu(reference_list, hypothesis_list):
#     b = [tokenize_for_bleu_eval(s) for s in hypothesis_list]
#     return bleu_score.compute_bleu(reference_list, b, smooth=False)


# # ============================= End of Helper Functions ===============================



# def main():
#     p = argparse.ArgumentParser(description="Evaluator for CoNaLa",
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#     p.add_argument("--input_dir",
#                    help="input directory, containing 'res/answer.txt' and 'ref/truth.txt'",
#                    default=None)
#     p.add_argument("--input_ref",
#                    help="input reference file",
#                    default=None)
#     p.add_argument("--input_hyp",
#                    help="input hypothesis file",
#                    default=None)
#     p.add_argument("--output_file",
#                    help="output score file",
#                    default=None)
#     p.add_argument("--output_dir",
#                    help="output score directory which will contain output_dir/scores.txt",
#                    default=None)
#     p.add_argument("--no_exact_match",
#                    help="only output bleu scores and not exact_match score",
#                    action="store_true")
#     p.add_argument("--strip_ref_metadata",
#                    help="strip metadata from the reference and get only the code",
#                    action="store_true")

#     args = p.parse_args()

#     if not (args.input_dir or (args.input_ref and args.input_hyp)):
#         raise ValueError("Must specify input_dir or input_ref+input_hyp")

#     input_hyp = args.input_hyp if args.input_hyp else os.path.join(args.input_dir, 'res', 'answer.txt')
#     input_ref = args.input_ref if args.input_ref else os.path.join(args.input_dir, 'ref', 'truth.txt')

#     with open(input_hyp, 'r') as f_hyp:
#         c_hyp = json.load(f_hyp)
#         c_hyp = [tokenize_for_bleu_eval(s) for s in c_hyp]
#     with open(input_ref, 'r') as f_ref:
#         c_ref = json.load(f_ref)
#         if args.strip_ref_metadata:
#           c_ref = [x['snippet'] for x in c_ref]
#         c_ref = [tokenize_for_bleu_eval(s) for s in c_ref]

#     if len(c_hyp) != len(c_ref):
#         raise ValueError('Length of hypothesis and reference don\'t match: {} != {}'.format(len(c_hyp), len(c_ref)))

#     if args.output_file:
#         f_out = open(args.output_file, 'w')
#     elif args.output_dir:
#         f_out = open(os.path.join(args.output_dir, 'scores.txt'), 'w')
#     else:
#         f_out = sys.stdout

#     bleu_tup = bleu_score.compute_bleu([[x] for x in c_ref], c_hyp, smooth=False)
#     bleu = bleu_tup[0]
#     exact = sum([1 if h == r else 0 for h, r in zip(c_hyp, c_ref)])/len(c_hyp)

#     f_out.write('bleu:{0:.2f}\n'.format(bleu * 100))
#     if not args.no_exact_match:
#         f_out.write('exact:{0:.2f}\n'.format(exact * 100))

#     f_out.close()

# if __name__ == '__main__':
#     main()