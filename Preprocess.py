import nltk
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# for ast and actions
import ast
import asdl
from asdl.asdl import ASDLGrammar
from asdl.lang.py.py_asdl_helper import *
from asdl.lang.py.py_transition_system import *
from asdl.hypothesis import *
import astor

# should return a list of words
## Incomplete
def process_intent(intent):
	intent = intent.lower()
	return intent.replace('?','').split()
   
# should return a list of characaters   
## Incomplete
def process_code(code):
	return code
	
# returns have two fileds: 'intent' and 'code'
def process_data(data):
	intents = []
	codes = []
	for e in data:
		intents.append(process_intent(e['intent']))
		codes.append(process_code(e['snippet']))
	return intents, codes
	
# return English vocabularies occur more than 5 times
## need to modify
def vocab_list(sentences, sos_eos=False, cut_freq=0):
	vocab = Counter()
	for sentence in sentences:
		for word in sentence:
			vocab[word] += 1

	if cut_freq > 0:
		vocab = [k for k in vocab if vocab[k] >= cut_freq]
	vocab.append('<UNK>')
	
	if sos_eos:
		vocab.append('<sos>')
		vocab.append('<eos>')
		
	vocab2num = dict(zip(vocab, range(0,len(vocab))))
	return vocab2num, vocab
	
def action_list(actions_lst, cut_freq=0):
	vocab = Counter()
	for actions in actions_lst:
		for action in actions:
			vocab[action] += 1

	vocab_filtered = []
	if cut_freq > 0:
		for action in vocab:
			if (vocab[action] > cut_freq 
			or isinstance(action, asdl.transition_system.ApplyRuleAction)):
				vocab_filtered.append(action)
	vocab_filtered.append(GenTokenAction('</UNK>'))

	vocab2num = dict(zip(vocab_filtered, range(0,len(vocab_filtered))))
	return vocab2num, vocab_filtered
	
	
#------------------------------------------------------------------------------
# the following create a ast2action and action2ast converter
class Ast_Action():
	def __init__(self, grammar_file='./asdl/lang/py3/py3_asdl.simplified.txt'):
		grammar_text = open(grammar_file).read()
		self.grammar = ASDLGrammar.from_text(grammar_text)
		self.parser = PythonTransitionSystem(self.grammar)
		
	def code2actions(self, code):
		grammar = self.grammar 
		py_ast = ast.parse(code)
		asdl_ast = python_ast_to_asdl_ast(py_ast.body[0], grammar)
		parser = self.parser
		actions = parser.get_actions(asdl_ast)
		return actions 
		
	def actions2code(self, actions):
		parser = self.parser
		grammar = self.grammar
		hypothesis = Hypothesis()
		for t, action in enumerate(actions, 1):

			# this assert the next action is a valid type
			assert action.__class__ in parser.get_valid_continuation_types(hypothesis)

			# this assert that next action has a valid field
			if isinstance(action, ApplyRuleAction) and hypothesis.frontier_node:
				assert action.production in grammar[hypothesis.frontier_field.type]
				# grammar[hypothesis.frontier_field.type] is a list of possible actions

			hypothesis.apply_action(action)
			
		ast_from_actions = asdl_ast_to_python_ast(hypothesis.tree, grammar)
		return astor.to_source(ast_from_actions).strip()
	
#------------------------------------------------------------------------------
# the following create a pytorch data loader
class code_intent_pair(Dataset): 
	def __init__(self, intents, codes, word2num, code2num):
		self.intents = []
		for intent in intents:
			num_intent = []
			for word in intent:
				if word in word2num:
					num_intent.append(word2num[word])
				else:
					num_intent.append(word2num['<UNK>'])
			self.intents.append(np.array(num_intent))

		self.labels = []
		for code in codes:
			num_code = []
			for word in code:
				if word in code2num:
					num_code.append(code2num[word])
				else:
					num_code.append(code2num[GenTokenAction('</UNK>')])
			self.labels.append(np.array(num_code))
		
	def __len__(self): 
		return len(self.labels)
	
	def __getitem__(self, idx):
		return (self.intents[idx], self.labels[idx])
		
class intent_set(Dataset):
	def __init__(self, intents, word2num):
		self.intents = []
		for intent in intents:
			num_intent = []
			for word in intent:
				if word in word2num:
					num_intent.append(word2num[word])
				else:
					num_intent.append(word2num['<UNK>'])
			self.intents.append(np.array(num_intent))
		
	def __len__(self): 
		return len(self.intents)
	
	def __getitem__(self, idx):
		return self.intents[idx]
		
def collate_lines(seq_list):
    inputs, targets = zip(*seq_list)
    lens = [len(seq) for seq in inputs]
    seq_order = sorted(range(len(lens)), key=lens.__getitem__, reverse=True)
    inputs = [torch.from_numpy(inputs[i]) for i in seq_order]
    targets = [torch.from_numpy(targets[i]) for i in seq_order]
    return inputs, targets
 
def get_train_loader(intents, labels, word2num, code2num, batch_size=16):
	dataset = code_intent_pair(intents, labels, word2num, code2num)
	return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_lines)
	
def get_test_loader(intents, word2num, batch_size=16):
	dataset = intent_set(intents, word2num)
	return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_lines)