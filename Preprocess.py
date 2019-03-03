import nltk
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

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
def vocab_list(intents, sos_eos):
	vocab = Counter()
	for intent in intents:
		for word in intent:
			vocab[word] += 1
	vocab = [k for k in vocab if vocab[k] > 5]
	vocab.append('<UNK>')
	if sos_eos:
		vocab.append('<sos>')
		vocab.append('<eos>')
	vocab2num = dict(zip(vocab, range(0,len(vocab))))
	return vocab2num
	
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
			num_code = [code2num['<sos>']]
			for word in code:
				if word in code2num:
					num_code.append(code2num[word])
				else:
					num_code.append(code2num['<UNK>'])
			num_code.append(code2num['<eos>'])
			self.labels.append(np.array(num_code))
		
	def __len__(self): 
		return len(self.labels)
	
	def __getitem__(self, idx):
		return (self.intents[idx], self.labels[idx])
		
def collate_lines(seq_list):
    inputs, targets = zip(*seq_list)
    lens = [len(seq) for seq in inputs]
    seq_order = sorted(range(len(lens)), key=lens.__getitem__, reverse=True)
    inputs = [torch.from_numpy(inputs[i]) for i in seq_order]
    targets = [torch.from_numpy(targets[i]) for i in seq_order]
    return inputs, targets
 
def get_dataloader(intents, labels, word2num, code2num, train=True, batch_size=16):
	dataset = code_intent_pair(intents, labels, word2num, code2num)
	return DataLoader(dataset, batch_size=batch_size, shuffle=train, collate_fn=collate_lines)