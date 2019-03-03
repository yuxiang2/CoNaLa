import torch
from torch import nn
import torch.nn.utils as U

class Encoder(nn.Module):
	def __init__(self, word_size):
		super(Encoder, self).__init__()
		embed_size = 128
		self.emb = nn.Embedding(word_size, embed_size)
		hidden_size = 256
		self.rnn = nn.LSTM(embed_size, hidden_size, num_layers=3, bidirectional=True)
		
	def forward(self, x):
		embed = [self.emb(datapoint.long()) for datapoint in x]
		packed = U.rnn.pack_sequence(embed)
		outputs,hidden = self.rnn(packed, None)
		hidden_h, hidden_c = hidden 
		hidden_h = hidden_h[-1] + hidden_h[-2]
		hidden_c = hidden_c[-1] + hidden_c[-2]
		return hidden_h, hidden_c
		
class Decoder(nn.Module): 
	def __init__(self, word_size, sos, eos):
		super(Decoder, self).__init__()
		embed_size = 16
		self.word_size = word_size
		self.emb = nn.Embedding(word_size, embed_size)
		hidden_size = 256
		self.cell1 = nn.LSTMCell(embed_size, hidden_size)
		self.cell2 = nn.LSTMCell(hidden_size, hidden_size)
		self.cell3 = nn.LSTMCell(hidden_size, hidden_size)
		self.linear = nn.Sequential(
			nn.Linear(hidden_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, word_size)
		)
		
		## store start of sent and eos symbols for padding
		## and generation
		self.sos = sos 
		self.eos = eos
		
	## pad code within a batch to the same length so 
	## that we can do batch rnn
	def pad_code(self, codes):
		N = len(codes)
		maxlen = 0
		eos = self.eos
		for code in codes:
			maxlen = max(maxlen, len(code))
		padded_codes = torch.LongTensor(N, maxlen)
		for i, code in enumerate(codes):
			padded_codes[i, :len(code)] = code
			padded_codes[i, len(code):] = eos
		return padded_codes
			
	def forward(self, x, hidden):
		padded_x = self.pad_code(x)
		leng = len(padded_x[0])
		embed = self.emb(padded_x)
		
		## initialize hidden states
		hidden1 = hidden 
		hidden2 = hidden 
		hidden3 = hidden 
		
		## scores is for storing logits
		scores = torch.DoubleTensor(len(x), leng, self.word_size)
		
		## for each time step
		for t in range(leng):
			embed_t = embed[:,t,:]
			hidden1 = self.cell1(embed_t, hidden1)
			hidden2 = self.cell2(hidden1[0], hidden2)
			hidden3 = self.cell3(hidden2[0], hidden3)
			
			## do linear inside for loop is inefficient, 
			## but it allows teacher forcing
			score = self.linear(hidden3[0])
			scores[:,t,:] = score
			
		## padded eos symbols are not removed, thus
		## calculated accuracy can be too high
			
		return scores.view(len(x)*leng, -1), padded_x.view(-1)
		
class Model(nn.Module):
	def __init__(self, word_size, code_size, sos, eos, lr=1e-3, best_acc=0.0):
		super(Model, self).__init__()
		self.encoder = Encoder(word_size)
		self.decoder = Decoder(code_size, sos, eos)
		self.loss = nn.CrossEntropyLoss()
		self.opt = torch.optim.Adam(self.parameters(), lr=lr)
		self.best_acc = best_acc
		
	def forward(self, intent, code, train=True):
			
		hidden = self.encoder(intent)
		scores,labels = self.decoder(code, hidden)
		
		# get statistics
		_, predicted = torch.max(scores, 1)
		num_correct = (predicted == labels).sum().item()
		acc = float(num_correct)/len(predicted)
		
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
		
	def epoch(self, train_loader, dev_loader, show_interval=15):
		
		# train
		loss = 0
		acc = 0
		for i,(x,y) in enumerate(train_loader):
			self.encoder.train()
			self.decoder.train()
			loss_i,acc_i = self.forward(x,y,train=True)
			loss += loss_i 
			acc += acc_i
			
			if (i+1) % show_interval == 0:
				loss /= show_interval
				acc /= show_interval
				print('train_loss = {}, train_acc = {}'.format(loss, acc))
				loss = 0
				acc = 0
				
		# dev
		acc = 0
		for i,(x,y) in enumerate(dev_loader):
			self.encoder.eval()
			self.decoder.eval()
			correct_i = self.forward(x,y,train=False)
			acc += acc_i
			
		acc /= i
		print('dev_acc = {}'.format(acc))
		if acc > self.best_acc:
			torch.save(self.state_dict(), 'CoNaLa_Basic.t7')
			self.best_acc = acc
		
		print('---------------------------------------------')
		
		
		