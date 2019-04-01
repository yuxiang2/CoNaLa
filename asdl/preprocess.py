import asdl
from py_transition_system import Python3TransitionSystem
from transition_system import GenTokenAction
import json
import py_utils as U
import py_asdl_helper as Helper
import ast
from action_info import get_action_infos as actions2info
from vocab import Vocab, VocabEntry
import pickle

class Preprocessor():
    def __init__(self, grammar_file):
        grammar_text = open(grammar_file).read()
        self.grammar = asdl.ASDLGrammar.from_text(grammar_text)
        self.trans_system = Python3TransitionSystem(self.grammar)

    def canonize(self, example_json):
        rewritten_intent = example_json['rewritten_intent']
        if rewritten_intent == '' or rewritten_intent == None:
            rewritten_intent = example_json['intent']
        snippet = example_json['snippet']
        
        canonical_intent, slot_map = U.canonicalize_intent(rewritten_intent)
        intent_tokens = U.tokenize_intent(canonical_intent)
        canonical_snippet = U.canonicalize_code(snippet, slot_map)
        
        return (canonical_intent, intent_tokens, slot_map, canonical_snippet)
    
    def get_action_infos(self, intent_tokens, code):
        python_ast = ast.parse(code)
        tgt_ast = Helper.python_ast_to_asdl_ast(python_ast, self.grammar)
        tgt_actions = self.trans_system.get_actions(tgt_ast)
        tgt_action_infos = actions2info(intent_tokens, tgt_actions)
        
        return tgt_action_infos
        
    def preprocess(self, dataset, mine=False):
        processed_entries = []
        for entry in dataset:
            processed_entry = {}
            intent, intent_tokens, slot_map, code = self.canonize(entry)
            try:
                action_infos = self.get_action_infos(intent_tokens, code)
                processed_entry = {
                    'intent_tokens': intent_tokens,
                    'slot_map': slot_map ,
                    'action_infos': action_infos,
                    'code': entry['snippet']
                }
                processed_entries.append(processed_entry)
            except Exception as e:
                print('one code failed to generate ast')
                print(entry['snippet'])
                print(e)
                
        return processed_entries
        
    def get_vocab(self, train_data, cut_freq=3):
        src_vocab = VocabEntry.from_corpus([e['intent_tokens'] for e in train_data], size=5000,
            freq_cutoff=cut_freq)
            
        primitive_tokens_maps = [map(lambda a: a.action.token,
                        filter(lambda a: isinstance(a.action, GenTokenAction), e['action_infos']))
                    for e in train_data]
        primitive_vocab = VocabEntry.from_corpus(primitive_tokens_maps, size=5000, freq_cutoff=cut_freq)
        
        code_tokens = [self.trans_system.tokenize_code(e['code'], mode='decoder') for e in train_data]
        code_vocab = VocabEntry.from_corpus(code_tokens, size=5000, freq_cutoff=cut_freq)
        
        return src_vocab, primitive_vocab, code_vocab
        
    
if __name__ == '__main__':
    print('load files...')
    datatrain = json.load(open('../corpus/train.json'))
    datatest = json.load(open('../corpus/test.json'))
    grammar_file = 'py_grammar.txt'
    
    print('preprocessing...')
    preprocessor = Preprocessor(grammar_file)
    processed_train = preprocessor.preprocess(datatrain)
    processed_test = preprocessor.preprocess(datatest)
    src_vocab, primitive_vocab, code_vocab = preprocessor.get_vocab(processed_train)
    
    print('store processed data...')
    pickle.dump(processed_train, open('../processed_data/processed_train.bin','wb'))
    pickle.dump(processed_test, open('../processed_data/processed_test.bin','wb'))
    pickle.dump(src_vocab, open('../processed_data/src_vocab.bin','wb'))
    pickle.dump(primitive_vocab, open('../processed_data/primitive_vocab.bin','wb'))
    pickle.dump(code_vocab, open('../processed_data/code_vocab.bin','wb'))