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
        return outputs, output_lengths, hidden

        
class Attention(nn.Module):
    def __init__(self, hyperP):
        super(Attention, self).__init__()
        attn_hidden_size = hyperP['attn_hidden_size']
        encoder_hidden_size = hyperP['encoder_hidden_size']
        decoder_hidden_size = hyperP['decoder_hidden_size']
        
        self.encoder_transformer = nn.Linear(encoder_hidden_size,attn_hidden_size)
        self.decoder_transformer = nn.Linear(decoder_hidden_size,attn_hidden_size)
        self.encoder_queries = None
        
    def forward(self, decoder_hidden, encoder_mask=None):
        decoder_query = self.decoder_transformer(decoder_hidden).unsqueeze(-1) # [bsize, -1, 1]
        attn_energy = torch.bmm(self.encoder_queries, decoder_query) # [bsize, leng, 1]
        attn_energy = torch.tanh(attn_energy)
        if encoder_mask is None:
            return softmax(attn_energy, dim=1)
        
        attn_energy_masked = attn_energy + encoder_mask.unsqueeze(-1)
        return softmax(attn_energy_masked, dim=1)
    
    def get_encoder_queries(self, encoder_outputs):
        leng, bsize, _ = encoder_outputs.size()
        encoder_outputs_flatten = encoder_outputs.view(leng * bsize, -1)
        encoder_queries = self.encoder_transformer(encoder_outputs_flatten).view(leng, bsize, -1)
        self.encoder_queries = encoder_queries.transpose(0,1).contiguous() # [bsize, leng, -1]

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
        self.gru = nn.GRU(embed_size, hidden_size, n_layers, dropout=dropout)
        self.linear = nn.Linear(encoder_hidden_size + hidden_size, code_size)

    def forward(self, prev_token, last_hidden, encoder_outputs_reshaped,
                encoder_mask=None):
        # Get the embedding of the current input word or last predicted word
        embedded = self.embed(prev_token).unsqueeze(0)  # (1,B,N)
        
        # Combine embedded input word and attended context, run through RNN
        outputs, hidden = self.gru(embedded, last_hidden)
        outputs = outputs.squeeze(0)  # (1,B,N) -> (B,N)
        
        # Calculate attention weights and context vector
        attn_weights = self.attention(hidden[-1], encoder_mask)
        context = torch.bmm(encoder_outputs_reshaped, attn_weights)  # (B,N,1)
        context = context.squeeze(2) # (B,N)
        
        # Get output logits
        outputs = torch.cat((outputs, context), dim=1) #(B, 2N)
        logits = self.linear(outputs)
        return logits, hidden, attn_weights


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
        encoder_outputs, encoder_valid_lengths, _ = self.encoder(src_seq)
        encoder_outputs_reshaped = encoder_outputs.permute(1, 2, 0).contiguous()
        
        self.decoder.attention.get_encoder_queries(encoder_outputs)
        
        ## initialize some parameters for decoder
        prev_token = trgt_seq[:,0]
        hidden = None
        
        encoder_max_len = encoder_outputs.size(0)
        encoder_mask = torch.zeros(batch_size, encoder_max_len)
        for i,length in enumerate(encoder_valid_lengths):
            encoder_mask[i, length:] = -999.99
            
        logits_seqs = []
        for t in range(1, len(trgt_seq[0])):
            logits, hidden, _ = self.decoder(prev_token, hidden, 
                encoder_outputs_reshaped, encoder_mask)
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
        
    def change_teacher_force_rate(self, new_rate):
        old_rate = self.teacher_force_rate
        self.teacher_force_rate = new_rate
        return old_rate
        
    def greedy_decode(self, src_seq, sos, eos, unk, max_len=35):
        encoder_outputs, _, _ = self.encoder(src_seq)
        encoder_outputs_reshaped = encoder_outputs.permute(1, 2, 0).contiguous()
        self.decoder.attention.get_encoder_queries(encoder_outputs)
        
        # intialize some parameters
        prev_token = torch.LongTensor([sos])
        hidden = None
        
        seq = []
        for t in range(max_len):
            logits, hidden, _ = self.decoder(prev_token, hidden, 
                encoder_outputs_reshaped)
            _, prev_token = torch.max(logits, 1)
            if prev_token == eos:
                break
            if prev_token == unk:
                _, best_2 = logits.topk(2, 1)
                if best_2[0,1].item() != (sos, eos):
                    seq.append(best_2[0,1].item())
            else:
                seq.append(prev_token.item())
        return seq
        
    def beam_decode(self, src_seq, sos, eos, unk, beam_width=10, max_len=36):
    
        import beam
        from beam import Beam_path
    
        assert (beam_width > 2)
        encoder_outputs, _, _ = self.encoder(src_seq)
        encoder_outputs_reshaped = encoder_outputs.permute(1, 2, 0).contiguous()
        self.decoder.attention.get_encoder_queries(encoder_outputs)

        # intialize some parameters
        prev_token = torch.LongTensor([sos])
        hidden = None
        bad_tokens = (sos, unk)
        
        def get_best_token(tokens, bad_tokens):
            for token in tokens:
                if token not in bad_tokens:
                    return token
        
        # initial step
        logits, hidden, _ = self.decoder(prev_token, hidden, 
            encoder_outputs_reshaped)
        p = torch.nn.functional.softmax(logits.view(-1), dim=0)
        kp, greedy_kwords = torch.topk(p, beam_width)
        klogp = torch.log(kp).tolist()
        greedy_kwords = greedy_kwords.tolist()
        best_word = get_best_token(greedy_kwords, bad_tokens)
        
        bestk_paths = []
        for logp,init_word in zip(klogp,greedy_kwords):
            if init_word in bad_tokens:
                bestk_paths.append(Beam_path(eos, logp, init_word, hidden, None, best_word))
            else:
                bestk_paths.append(Beam_path(eos, logp, init_word, hidden, None))
                
        # steps after
        for i in range(1, max_len):
            new_paths = []
            for beam_path in bestk_paths:
                if beam_path.is_done():
                    new_paths.append(beam_path)
                    continue
                prev_hidden = beam_path.prev_hidden
                prev_word = torch.LongTensor([beam_path.prev_word])
                logits, hidden, _  = self.decoder\
                (prev_word, prev_hidden, encoder_outputs_reshaped)
                p = torch.nn.functional.softmax(logits.view(-1), dim=0)
                kp, greedy_kwords = torch.topk(p, beam_width)
                klogp = torch.log(kp).tolist()
                greedy_kwords = greedy_kwords.tolist()
                best_word = get_best_token(greedy_kwords, bad_tokens)
                new_paths.extend(beam_path.get_new_paths(greedy_kwords, bad_tokens, best_word, klogp, hidden, None))
            
            bestk_paths = Beam_path.get_bestk_paths(new_paths, beam_width)
        
        best_path = bestk_paths[-1]
        return best_path.path[:-1]
        
    def save(self):
        torch.save(self.state_dict(), 'model.t7')
        
    def load(self):
        self.load_state_dict(torch.load('model.t7'))