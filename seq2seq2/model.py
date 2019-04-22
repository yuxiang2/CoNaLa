import torch
import torch.nn as nn
import torch.nn.utils as U
from torch.nn.functional import softmax
import random


class Encoder(nn.Module):
    def __init__(self, word_size, hyperP):
        super(Encoder, self).__init__()

        embed_size = hyperP['encoder_embed_size']
        hidden_size = hyperP['encoder_hidden_size']
        n_layers = hyperP['encoder_layers']
        dropout = hyperP['encoder_dropout_rate']
        
        self.hidden_size = hidden_size
        self.embed = nn.Sequential(
            nn.Embedding(word_size, embed_size),
            nn.Dropout(dropout, inplace=True)
        )
        self.gru = nn.GRU(embed_size, hidden_size, n_layers,
                          dropout=dropout, bidirectional=True)

    def forward(self, src, hidden=None):
        embeddings = [self.embed(datapoint) for datapoint in src]
        packed = U.rnn.pack_sequence(embeddings)
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = U.rnn.pad_packed_sequence(outputs)
        
        # sum bidirectional outputs
        outputs = (outputs[:, :, :self.hidden_size] +
                   outputs[:, :, self.hidden_size:])
        return outputs, hidden

        
class Attention(nn.Module):
    def __init__(self, hyperP):
        super(Attention, self).__init__()
        attn_hidden_size = hyperP['attn_hidden_size']
        encoder_hidden_size = hyperP['encoder_hidden_size']
        decoder_hidden_size = hyperP['decoder_hidden_size']
        
        self.encoder_transformer = nn.Linear(encoder_hidden_size,attn_hidden_size)
        self.decoder_transformer = nn.Linear(decoder_hidden_size,attn_hidden_size)
        
    def forward(self, encoder_outputs, decoder_hidden):
        leng, bsize, _ = encoder_outputs.size()
        encoder_outputs_flatten = encoder_outputs.view(leng * bsize, -1)
        encoder_queries = self.encoder_transformer(encoder_outputs_flatten).view(leng, bsize, -1)
        encoder_queries = encoder_queries.transpose(0,1).contiguous() # [bsize, leng, -1]
        decoder_query = self.decoder_transformer(decoder_hidden).unsqueeze(-1) # [bsize, -1, 1]
        attn_energy = torch.bmm(encoder_queries, decoder_query) # [bsize, leng, 1]
        
        ## add mask here, like add [0,0,,...-inf,inf] to the energy
        
        return softmax(attn_energy, dim=1)
        

class Decoder(nn.Module):
    def __init__(self, code_size, hyperP):
        super(Decoder, self).__init__()
        
        embed_size = hyperP['decoder_embed_size']
        hidden_size = hyperP['decoder_hidden_size']
        encoder_hidden_size = hyperP['encoder_hidden_size']
        n_layers = hyperP['decoder_layers']
        dropout = hyperP['decoder_dropout_rate']

        self.embed = nn.Sequential(
            nn.Embedding(code_size, embed_size),
            nn.Dropout(dropout, inplace=True)
        )
        self.attention = Attention(hyperP)
        self.gru = nn.GRU(encoder_hidden_size + embed_size, hidden_size,
                          n_layers, dropout=dropout)
        self.linear = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.ReLU(True),
            nn.Linear(hidden_size * 2, code_size)
        )

    def forward(self, prev_token, last_hidden, encoder_outputs, encoder_outputs_reshaped, context):
        # Get the embedding of the current input word or last predicted word
        embedded = self.embed(prev_token).unsqueeze(0)  # (1,B,N)
        
        # Combine embedded input word and attended context, run through RNN
        context = context.unsqueeze(0) # (1,B,N)
        rnn_input = torch.cat([embedded, context], 2)
        outputs, hidden = self.gru(rnn_input, last_hidden)
        outputs = outputs.squeeze(0)  # (1,B,N) -> (B,N)
        
        # Calculate attention weights and context vector
        attn_weights = self.attention(encoder_outputs, hidden[-1])
        context = torch.bmm(encoder_outputs_reshaped, attn_weights)  # (B,N,1)
        context = context.squeeze(2) # (B,N)
        
        # Get output logits
        outputs = torch.cat((outputs, context), dim=1) #(B, 2N)
        logits = self.linear(outputs)
        return logits, hidden, context, attn_weights


class Seq2Seq(nn.Module):
    def __init__(self, word_size, code_size, hyperP):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(word_size, hyperP)
        self.decoder = Decoder(code_size, hyperP)
        
        self.teacher_force_rate = hyperP['teacher_force_rate']
        self.encoder_hidden_size = hyperP['encoder_hidden_size']
        self.code_size = code_size

    def forward(self, src_seq, trgt_seq, out_lens):
        batch_size = len(src_seq)
        encoder_outputs, encoder_hidden = self.encoder(src_seq)
        encoder_outputs_reshaped = encoder_outputs.permute(1, 2, 0).contiguous()
        
        ## initialize some parameters for decoder
        context = torch.zeros(batch_size, self.encoder_hidden_size)
        prev_token = trgt_seq[:,0]
        hidden = None
        
        logits_seqs = []
        for t in range(1, len(trgt_seq[0])):
            logits, hidden, context, _ = self.decoder(prev_token, hidden, 
                encoder_outputs, encoder_outputs_reshaped, context)
            if random.random() < self.teacher_force_rate:
                prev_token = trgt_seq[:,t]
            else:
                _, prev_token = torch.max(logits, 1)
            logits_seqs.append(logits)
        logits_seqs = torch.stack(logits_seqs, dim=1)
        
        ## mask here
        valid_logits_seq = []
        for i,out_len in enumerate(out_lens):
            valid_logits_seq.append(logits_seqs[i][:out_len])
        
        return torch.cat(valid_logits_seq, dim=0)
        
    def greedy_decode(self, src_seq, sos, eos, max_len=50):
        encoder_outputs, encoder_hidden = self.encoder(src_seq)
        encoder_outputs_reshaped = encoder_outputs.permute(1, 2, 0).contiguous()
        context = torch.zeros(1, self.encoder_hidden_size)
        prev_token = torch.LongTensor([sos])
        hidden = None
        
        seq = []
        for t in range(max_len):
            logits, hidden, context, _ = self.decoder(prev_token, hidden, 
                encoder_outputs, encoder_outputs_reshaped, context)
            _, prev_token = torch.max(logits, 1)
            if prev_token == eos:
                break
            seq.append(prev_token.item())
        return seq
        
    def save(self):
        torch.save(self.state_dict(), 'model.t7')
        
    def load(self):
        self.load_state_dict(torch.load('model.t7'))