import ast
import copy
import torch

class Decoder():
    def __init__(self, seq2seq_model, lang_model=None, hyperP=None):
        self.model1 = seq2seq_model
        self.model2 = lang_model 
        if lang_model is not None:
            self.weight1 = hyperP['seq2seq_weight']
            self.weight2 = hyperP['lang_weight']
        else:
            self.weight1 = 1.0
            self.weight2 = 0.0
    
    def decode(self, src_seq, sos, eos, unk, beam_width=10, max_len=36):
        assert(beam_width > 2)
        model = self.model1 
        lang_model = self.model2
        
        encoder_outputs, _, _ = model.encoder(src_seq)
        encoder_outputs_reshaped = encoder_outputs.permute(1, 2, 0).contiguous()
        model.decoder.attention.get_encoder_queries(encoder_outputs)
        
        prev_token = torch.LongTensor([sos])
        hidden = None 
        bad_tokens = (sos, unk)
        
        def get_best_token(tokens, bad_tokens):
            for token in tokens:
                if token not in bad_tokens:
                    return token 
        
        # initial step for seq2seq model
        context = torch.zeros(1, model.encoder_hidden_size)
        logits, hidden, context, _ = model.decoder(prev_token, hidden,
            encoder_outputs_reshaped, context)
        p1 = torch.nn.functional.softmax(logits.view(-1), dim=0)
        logp = self.weight1 * torch.log(p1)
        
        # initial step for language model
        if lang_model is not None:
            # p2 = ....
            # logp += self.weight2 * torch.log(p2)
            pass 
            
        klogp, greedy_kwords = torch.topk(logp, beam_width)
        klogp = klogp.tolist() 
        greedy_kwords = greedy_kwords.tolist()
        best_token = get_best_token(greedy_kwords, bad_tokens)
        
        bestk_paths = []
        for logp,init_word in zip(klogp,greedy_kwords):
            if init_word in bad_tokens:
                bestk_paths.append(Beam_path(eos, logp, init_word, hidden, context, best_token))
            else:
                bestk_paths.append(Beam_path(eos, logp, init_word, hidden, context))
        
        # initial step done, steps afterwards
        for i in range(1, max_len):
            new_paths = []
            for beam_path in bestk_paths:
                if beam_path.is_done():
                    new_paths.append(beam_path)
                    continue 
                    
                prev_hidden = beam_path.prev_hidden
                prev_word = torch.LongTensor([beam_path.prev_word])
                prev_context = beam_path.prev_context
                logits, hidden, context, _  = model.decoder\
                    (prev_word, prev_hidden, encoder_outputs_reshaped, prev_context)
                p1 = torch.nn.functional.softmax(logits.view(-1), dim=0)
                logp = self.weight1 * torch.log(p1)
                
                if lang_model is not None:
                    # p2 = ....
                    # logp += self.weight2 * torch.log(p2)
                    pass 
                    
                klogp, greedy_kwords = torch.topk(logp, beam_width)
                klogp = klogp.tolist()
                greedy_kwords = greedy_kwords.tolist()
                best_token = get_best_token(greedy_kwords, bad_tokens)
                new_paths.extend(beam_path.get_new_paths(greedy_kwords, bad_tokens,\
                    best_token, klogp, hidden, context))
            
            bestk_paths = Beam_path.get_bestk_paths(new_paths, beam_width)
        
        best_path = bestk_paths[-1]
        return best_path.path[:-1]
    

class Beam_path(object):
    def __init__(self, eos=None, logp=0, cur_word=None, prev_hidden=None, prev_context=None, replace_word=None):
        self.logp = logp
        self.path = [cur_word] if replace_word == None else [replace_word]
        self.prev_word = cur_word
        self.prev_hidden = prev_hidden 
        self.prev_context = prev_context
        self.eos = eos
    
    def _copy(self):
        path = Beam_path()
        path.logp = self.logp 
        path.path = copy.copy(self.path)
        path.eos = self.eos
        return path
        
    def _update(self, cur_word, logp, hidden, context, replace_word=None):
        self.logp += logp
        self.path.append(cur_word if replace_word == None else replace_word)
        self.prev_word = cur_word 
        self.prev_hidden = hidden
        self.prev_context = context
        
    def is_done(self):
        return self.prev_word == self.eos
        
    def get_new_paths(self, branches, bad_tokens, replace_word, logps, hidden, context):
        N = len(branches)
        new_paths = []
        for i in range(N):
            new_paths.append(self._copy())
        for new_path,branch,logp in zip(new_paths,branches,logps):
            if branch in bad_tokens:
                new_path._update(branch,logp,hidden,context,replace_word)
            else:
                new_path._update(branch,logp,hidden,context)
        return new_paths
        
    def __repr__(self):
        return str(self.path) + str(self.logp)
        
    @staticmethod
    def get_bestk_paths(paths, k):
        sorted_paths = sorted(paths, key=lambda x: x.logp / len(x.path))
        return sorted_paths[-k:]