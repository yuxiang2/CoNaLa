# coding=utf-8

# Common
from collections import OrderedDict
import torch
import numpy as np
try:
    import cPickle as pickle
except:
    import pickle
from torch.autograd import Variable
from asdl.transition_system import ApplyRuleAction, ReduceAction
from common.utils import cached_property
from model import nn_utils

# CoNaLa
import json
import sys

from asdl.lang.py3.py3_transition_system import python_ast_to_asdl_ast, asdl_ast_to_python_ast, Python3TransitionSystem
from asdl.hypothesis import *
from asdl.transition_system import *
from .action_info import ActionInfo
from .action_info import get_action_infos
from .util import *
from .vocab import Vocab, VocabEntry


# Common
class Dataset(object):
    def __init__(self, examples):
        self.examples = examples

    @property
    def all_source(self):
        return [e.src_sent for e in self.examples]

    @property
    def all_targets(self):
        return [e.tgt_code for e in self.examples]

    @staticmethod
    def from_bin_file(file_path):
        examples = pickle.load(open(file_path, 'rb'))
        return Dataset(examples)

    def batch_iter(self, batch_size, shuffle=False):
        index_arr = np.arange(len(self.examples))
        if shuffle:
            np.random.shuffle(index_arr)

        batch_num = int(np.ceil(len(self.examples) / float(batch_size)))
        for batch_id in range(batch_num):
            batch_ids = index_arr[batch_size * batch_id: batch_size * (batch_id + 1)]
            batch_examples = [self.examples[i] for i in batch_ids]
            batch_examples.sort(key=lambda e: -len(e.src_sent))

            yield batch_examples

    def __len__(self):
        return len(self.examples)

    def __iter__(self):
        return iter(self.examples)


class Example(object):
    def __init__(self, src_sent, tgt_actions, tgt_code, tgt_ast, idx=0, meta=None):
        self.src_sent = src_sent
        self.tgt_code = tgt_code
        self.tgt_ast = tgt_ast
        self.tgt_actions = tgt_actions

        self.idx = idx
        self.meta = meta


class Batch(object):
    def __init__(self, examples, grammar, vocab, copy=True, cuda=False):
        self.examples = examples
        self.max_action_num = max(len(e.tgt_actions) for e in self.examples)

        self.src_sents = [e.src_sent for e in self.examples]
        self.src_sents_len = [len(e.src_sent) for e in self.examples]

        self.grammar = grammar
        self.vocab = vocab
        self.copy = copy
        self.cuda = cuda

        self.init_index_tensors()

    def __len__(self):
        return len(self.examples)

    def get_frontier_field_idx(self, t):
        ids = []
        for e in self.examples:
            if t < len(e.tgt_actions):
                ids.append(self.grammar.field2id[e.tgt_actions[t].frontier_field])
                # assert self.grammar.id2field[ids[-1]] == e.tgt_actions[t].frontier_field
            else:
                ids.append(0)

        return Variable(torch.cuda.LongTensor(ids)) if self.cuda else Variable(torch.LongTensor(ids))

    def get_frontier_prod_idx(self, t):
        ids = []
        for e in self.examples:
            if t < len(e.tgt_actions):
                ids.append(self.grammar.prod2id[e.tgt_actions[t].frontier_prod])
                # assert self.grammar.id2prod[ids[-1]] == e.tgt_actions[t].frontier_prod
            else:
                ids.append(0)

        return Variable(torch.cuda.LongTensor(ids)) if self.cuda else Variable(torch.LongTensor(ids))

    def get_frontier_field_type_idx(self, t):
        ids = []
        for e in self.examples:
            if t < len(e.tgt_actions):
                ids.append(self.grammar.type2id[e.tgt_actions[t].frontier_field.type])
                # assert self.grammar.id2type[ids[-1]] == e.tgt_actions[t].frontier_field.type
            else:
                ids.append(0)

        return Variable(torch.cuda.LongTensor(ids)) if self.cuda else Variable(torch.LongTensor(ids))

    def init_index_tensors(self):
        self.apply_rule_idx_matrix = []
        self.apply_rule_mask = []
        self.primitive_idx_matrix = []
        self.gen_token_mask = []
        self.primitive_copy_mask = []
        self.primitive_copy_token_idx_mask = np.zeros((self.max_action_num, len(self), max(self.src_sents_len)), dtype='float32')

        for t in range(self.max_action_num):
            app_rule_idx_row = []
            app_rule_mask_row = []
            token_row = []
            gen_token_mask_row = []
            copy_mask_row = []

            for e_id, e in enumerate(self.examples):
                app_rule_idx = app_rule_mask = token_idx = gen_token_mask = copy_mask = 0
                if t < len(e.tgt_actions):
                    action = e.tgt_actions[t].action
                    action_info = e.tgt_actions[t]

                    if isinstance(action, ApplyRuleAction):
                        app_rule_idx = self.grammar.prod2id[action.production]
                        # assert self.grammar.id2prod[app_rule_idx] == action.production
                        app_rule_mask = 1
                    elif isinstance(action, ReduceAction):
                        app_rule_idx = len(self.grammar)
                        app_rule_mask = 1
                    else:
                        src_sent = self.src_sents[e_id]
                        token = str(action.token)
                        token_idx = self.vocab.primitive[action.token]

                        token_can_copy = False

                        if self.copy and token in src_sent:
                            token_pos_list = [idx for idx, _token in enumerate(src_sent) if _token == token]
                            self.primitive_copy_token_idx_mask[t, e_id, token_pos_list] = 1.
                            copy_mask = 1
                            token_can_copy = True

                        if token_can_copy is False or token_idx != self.vocab.primitive.unk_id:
                            # if the token is not copied, we can only generate this token from the vocabulary,
                            # even if it is a <unk>.
                            # otherwise, we can still generate it from the vocabulary
                            gen_token_mask = 1

                        if token_can_copy:
                            assert action_info.copy_from_src
                            assert action_info.src_token_position in token_pos_list

                        # # cannot copy, only generation
                        # # could be unk!
                        # if not action_info.copy_from_src:
                        #     gen_token_mask = 1
                        # else:  # copy
                        #     copy_mask = 1
                        #     copy_pos = action_info.src_token_position
                        #     if token_idx != self.vocab.primitive.unk_id:
                        #         # both copy and generate from vocabulary
                        #         gen_token_mask = 1

                app_rule_idx_row.append(app_rule_idx)
                app_rule_mask_row.append(app_rule_mask)

                token_row.append(token_idx)
                gen_token_mask_row.append(gen_token_mask)
                copy_mask_row.append(copy_mask)

            self.apply_rule_idx_matrix.append(app_rule_idx_row)
            self.apply_rule_mask.append(app_rule_mask_row)

            self.primitive_idx_matrix.append(token_row)
            self.gen_token_mask.append(gen_token_mask_row)

            self.primitive_copy_mask.append(copy_mask_row)

        T = torch.cuda if self.cuda else torch
        self.apply_rule_idx_matrix = Variable(T.LongTensor(self.apply_rule_idx_matrix))
        self.apply_rule_mask = Variable(T.FloatTensor(self.apply_rule_mask))
        self.primitive_idx_matrix = Variable(T.LongTensor(self.primitive_idx_matrix))
        self.gen_token_mask = Variable(T.FloatTensor(self.gen_token_mask))
        self.primitive_copy_mask = Variable(T.FloatTensor(self.primitive_copy_mask))
        self.primitive_copy_token_idx_mask = Variable(torch.from_numpy(self.primitive_copy_token_idx_mask))
        if self.cuda: self.primitive_copy_token_idx_mask = self.primitive_copy_token_idx_mask.cuda()

    @property
    def primitive_mask(self):
        return 1. - torch.eq(self.gen_token_mask + self.primitive_copy_mask, 0).float()

    @cached_property
    def src_sents_var(self):
        return nn_utils.to_input_variable(self.src_sents, self.vocab.source,
                                          cuda=self.cuda)

    @cached_property
    def src_token_mask(self):
        return nn_utils.length_array_to_mask_tensor(self.src_sents_len,
                                                    cuda=self.cuda)

    @cached_property
    def token_pos_list(self):
        # (batch_size, src_token_pos, unique_src_token_num)

        batch_src_token_to_pos_map = []
        for e_id, e in enumerate(self.examples):
            aggregated_primitive_tokens = OrderedDict()
            for token_pos, token in enumerate(e.src_sent):
                aggregated_primitive_tokens.setdefault(token, []).append(token_pos)



# CoNaLa
def preprocess_conala_dataset(train_file, test_file, grammar_file, src_freq=3, code_freq=3):
    np.random.seed(1234)

    asdl_text = open(grammar_file).read()
    grammar = ASDLGrammar.from_text(asdl_text)
    transition_system = Python3TransitionSystem(grammar)

    print('process training data...')
    train_examples = preprocess_dataset(train_file, name='train', transition_system=transition_system)

    # held out 200 examples for development
    full_train_examples = train_examples[:]
    np.random.shuffle(train_examples)
    dev_examples = train_examples[:200]
    train_examples = train_examples[200:]

    # full_train_examples = train_examples[:]
    # np.random.shuffle(train_examples)
    # dev_examples = []
    # dev_questions = set()
    # dev_examples_id = []
    # for i, example in enumerate(full_train_examples):
    #     qid = example.meta['example_dict']['question_id']
    #     if qid not in dev_questions and len(dev_examples) < 200:
    #         dev_questions.add(qid)
    #         dev_examples.append(example)
    #         dev_examples_id.append(i)

    # train_examples = [e for i, e in enumerate(full_train_examples) if i not in dev_examples_id]
    print(f'{len(train_examples)} training instances', file=sys.stderr)
    print(f'{len(dev_examples)} dev instances', file=sys.stderr)

    print('process testing data...')
    test_examples = preprocess_dataset(test_file, name='test', transition_system=transition_system)
    print(f'{len(test_examples)} testing instances', file=sys.stderr)

    src_vocab = VocabEntry.from_corpus([e.src_sent for e in train_examples], size=5000,
                                       freq_cutoff=src_freq)
    primitive_tokens = [map(lambda a: a.action.token,
                            filter(lambda a: isinstance(a.action, GenTokenAction), e.tgt_actions))
                        for e in train_examples]
    primitive_vocab = VocabEntry.from_corpus(primitive_tokens, size=5000, freq_cutoff=code_freq)

    # generate vocabulary for the code tokens!
    code_tokens = [transition_system.tokenize_code(e.tgt_code, mode='decoder') for e in train_examples]
    code_vocab = VocabEntry.from_corpus(code_tokens, size=5000, freq_cutoff=code_freq)

    vocab = Vocab(source=src_vocab, primitive=primitive_vocab, code=code_vocab)
    print('generated vocabulary %s' % repr(vocab), file=sys.stderr)

    action_lens = [len(e.tgt_actions) for e in train_examples]
    print('Max action len: %d' % max(action_lens), file=sys.stderr)
    print('Avg action len: %d' % np.average(action_lens), file=sys.stderr)
    print('Actions larger than 100: %d' % len(list(filter(lambda x: x > 100, action_lens))), file=sys.stderr)

    pickle.dump(train_examples, open('data/conala/train.var_str_sep.bin', 'wb'))
    pickle.dump(full_train_examples, open('data/conala/train.var_str_sep.full.bin', 'wb'))
    pickle.dump(dev_examples, open('data/conala/dev.var_str_sep.bin', 'wb'))
    pickle.dump(test_examples, open('data/conala/test.var_str_sep.bin', 'wb'))
    pickle.dump(vocab, open('data/conala/vocab.var_str_sep.new_dev.src_freq%d.code_freq%d.bin' % (src_freq, code_freq), 'wb'))


def preprocess_dataset(file_path, transition_system, name='train'):
    dataset = json.load(open(file_path))
    examples = []
    evaluator = ConalaEvaluator(transition_system)

    f = open(file_path + '.debug', 'w')

    for i, example_json in enumerate(dataset):
        example_dict = preprocess_example(example_json)
        if example_json['question_id'] in (18351951, 9497290, 19641579, 32283692):
            print(example_json['question_id'])
            continue

        python_ast = ast.parse(example_dict['canonical_snippet'])
        canonical_code = astor.to_source(python_ast).strip()
        tgt_ast = python_ast_to_asdl_ast(python_ast, transition_system.grammar)
        tgt_actions = transition_system.get_actions(tgt_ast)

        # sanity check
        hyp = Hypothesis()
        for t, action in enumerate(tgt_actions):
            assert action.__class__ in transition_system.get_valid_continuation_types(hyp)
            if isinstance(action, ApplyRuleAction):
                assert action.production in transition_system.get_valid_continuating_productions(hyp)

            p_t = -1
            f_t = None
            if hyp.frontier_node:
                p_t = hyp.frontier_node.created_time
                f_t = hyp.frontier_field.field.__repr__(plain=True)

            # print('\t[%d] %s, frontier field: %s, parent: %d' % (t, action, f_t, p_t))
            hyp = hyp.clone_and_apply_action(action)

        assert hyp.frontier_node is None and hyp.frontier_field is None
        hyp.code = code_from_hyp = astor.to_source(asdl_ast_to_python_ast(hyp.tree, transition_system.grammar)).strip()
        assert code_from_hyp == canonical_code

        decanonicalized_code_from_hyp = decanonicalize_code(code_from_hyp, example_dict['slot_map'])
        assert compare_ast(ast.parse(example_json['snippet']), ast.parse(decanonicalized_code_from_hyp))
        assert transition_system.compare_ast(transition_system.surface_code_to_ast(decanonicalized_code_from_hyp),
                                             transition_system.surface_code_to_ast(example_json['snippet']))

        tgt_action_infos = get_action_infos(example_dict['intent_tokens'], tgt_actions)

        example = Example(idx=f'{i}-{example_json["question_id"]}',
                          src_sent=example_dict['intent_tokens'],
                          tgt_actions=tgt_action_infos,
                          tgt_code=canonical_code,
                          tgt_ast=tgt_ast,
                          meta=dict(example_dict=example_json,
                                    slot_map=example_dict['slot_map']))
        assert evaluator.is_hyp_correct(example, hyp)

        examples.append(example)

        # log!
        f.write(f'Example: {example.idx}\n')
        f.write(f"Original Utterance: {example.meta['example_dict']['rewritten_intent']}\n")
        f.write(f"Original Snippet: {example.meta['example_dict']['snippet']}\n")
        f.write(f"\n")
        f.write(f"Utterance: {' '.join(example.src_sent)}\n")
        f.write(f"Snippet: {example.tgt_code}\n")
        f.write(f"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")

    f.close()

    return examples


def preprocess_example(example_json):
    intent = example_json['intent']
    rewritten_intent = example_json['rewritten_intent']
    snippet = example_json['snippet']
    question_id = example_json['question_id']

    if rewritten_intent is None:
        rewritten_intent = intent

    canonical_intent, slot_map = canonicalize_intent(rewritten_intent)
    canonical_snippet = canonicalize_code(snippet, slot_map)
    intent_tokens = tokenize_intent(canonical_intent)
    decanonical_snippet = decanonicalize_code(canonical_snippet, slot_map)

    reconstructed_snippet = astor.to_source(ast.parse(snippet)).strip()
    reconstructed_decanonical_snippet = astor.to_source(ast.parse(decanonical_snippet)).strip()

    assert compare_ast(ast.parse(reconstructed_snippet), ast.parse(reconstructed_decanonical_snippet))

    return {'canonical_intent': canonical_intent,
            'intent_tokens': intent_tokens,
            'slot_map': slot_map,
            'canonical_snippet': canonical_snippet}


def generate_vocab_for_paraphrase_model(vocab_path, save_path):
    from components.vocab import VocabEntry, Vocab

    vocab = pickle.load(open(vocab_path, 'rb'))
    para_vocab = VocabEntry()
    for i in range(0, 10):
        para_vocab.add('<unk_%d>' % i)
    for word in vocab.source.word2id:
        para_vocab.add(word)
    for word in vocab.code.word2id:
        para_vocab.add(word)

    pickle.dump(para_vocab, open(save_path, 'wb'))


if __name__ == '__main__':
    # the json files can be download from http://conala-corpus.github.io
    preprocess_conala_dataset(train_file='data/conala/conala-train.json',
                              test_file='data/conala/conala-test.json',
                              grammar_file='asdl/lang/py3/py3_asdl.simplified.txt', src_freq=3, code_freq=3)

    # generate_vocab_for_paraphrase_model('data/conala/vocab.src_freq3.code_freq3.bin', 'data/conala/vocab.para.src_freq3.code_freq3.bin')

