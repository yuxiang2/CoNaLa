import ast
import copy
import torch
from evaluate import get_bleu_sent, tokenize_for_bleu_eval
from nltk.tokenize import word_tokenize
from string import punctuation
import nltk
import ast

lemmatizer = nltk.wordnet.WordNetLemmatizer()

token_map = {
    'numpy' : ['np'],
    'np' : ['numpy'],
    'panda' : ['pd'],
    'pd' : ['panda'],
    'concatenate' : ['join', 'add', 'cat'],
    'average' : ['mean', 'sum', 'len'],
    'integer' : ['int'],
    'string' : ['str'],
    'contain' : ['in'],
    'append' : ['add', '\'a\''],
    'delete' : ['del', 'remove', 'pop', '!=', '=='],
    'remove' : ['del', 'delete', 'pop', '!=', '=='],
    'digits' : ['0-9'],
    'alphebet' : ['a-z', 'A-Z'],
    'url' : ['https'],
    'equal' : ['=', '==', '!='],
    'greater' : ['>', '>='],
    'smaller' : ['<', '<='],
    'regex' : ['re'],
    'sort' : ['lambda'],
    'duplicated' : ['set'],
    'read' : ['\'r\''],
    'write' : ['\'w\''],
    'add' : ['+', 'append'],
    'subtract' : ['-'],
    'minus' : ['-'],
    'multiply' : ['*'],
    'divide' : ['/'],
    'reminder' : ['mod', '%'],
    'file' : ['path', '.txt', '.csv'],
    'last' : ['-1'],
    'all' : ['for', 'in'],
    'descending' : ['reverse', 'True', '-'],
    'index' : ['i', 'idx', 'arg'],
    'json' : ['dump', 'load'],
    'column' : ['col', 'axis', 'dim'],
    'square' : ['**', '2'],
    'dictionary' : ['dict'],
    'prompt' : ['input'],
    'matplotlib' : ['plt', 'plot'],
    'type' : ['isinstance'],
    'between' : ['<', '<=', '>', '>=', 'range'],
    'lowercase' : ['lower'],
    'uppercase' : ['upper'],
    'length' : ['len'],
    'directory' : ['path'],
    'execute' : ['subprocess', 'call', 'run'],
    'dataframe' : ['df'],
    'space' : [' ', 'strip'],
    'bash' : ['subprocess']
}

def strip_punctuation(s):
    return ''.join(c for c in s if c not in punctuation)

def sub_slotmap(tokens, slot_map):
    # replace slot maps
    for i in range(len(tokens)):
        if tokens[i] in slot_map:
            value = slot_map[tokens[i]]
            tokens[i] = value

        elif len(tokens[i]) > 2 and tokens[i][1:-1] in slot_map:
            value = slot_map[tokens[i][1:-1]]
            quote = tokens[i][0]
            tokens[i] = quote + value + quote

        elif len(tokens[i]) > 6 and tokens[i][3:-3] in slot_map:
            value = slot_map[tokens[i][3:-3]]
            quote = tokens[i][0:3]
            tokens[i] = quote + value + quote
            
    return tokens

def post_process_test(intent, slot_map, beams, idx2code, code):
    print(intent)
    print(code)
    print(slot_map)
    print('before process:')
    for beam in beams:
        beam.score /= len(beam.path)
        beam.path = sub_slotmap(idx2code(beam.path)[:-1], slot_map)
        score = get_bleu_sent(' '.join(beam.path), code)
        print('b_score:' + '%.2f'%beam.score + '\tscore:' + '%.2f'%score + ':\t' + ' '.join(beam.path))
      
    # post process
    lemmatizer = nltk.wordnet.WordNetLemmatizer()
    intent_tokens = tokenize_for_bleu_eval(intent.lower())
    intent_tokens = [lemmatizer.lemmatize(token) for token in intent_tokens]
    manual_add_tokens(intent_tokens)
        
    slot_values = slot_map.values()
    slot_token_counts = {}
    for value in slot_values:
        slot_token_counts[value] = len(tokenize_for_bleu_eval(value))
    
    for i,beam in enumerate(beams):
        gen_code = ' '.join(beam.path)
        code_token = tokenize_for_bleu_eval(gen_code)
        
        # check how many tokens are in common
        for token in intent_tokens:
            if token in gen_code.lower():
                beam.score += 2.0 / len(intent_tokens)
            if token in code_token:
                beam.score += 1.0 / len(intent_tokens)
                
        # check how many slot_map tokens are used
        slotmap_used_count = 0
        for value in slot_values:
            if value in gen_code:
                slotmap_used_count += slot_token_counts[value]
        beam.score += float(slotmap_used_count) / len(intent_tokens)
    
    print('\nafter process:')
    beams = sorted(beams, key=lambda x: x.score)
    for beam in beams:
        score = get_bleu_sent(' '.join(beam.path), code)
        print('b_score:' + '%.2f'%beam.score + '\tscore:' + '%.2f'%score + ':\t' + ' '.join(beam.path))

def manual_add_tokens(tokens):
    add_tokens = []
    for token in tokens:
        if token in token_map:
            add_tokens.extend(token_map[token])
    tokens.extend(add_tokens)
        

def post_process_hand(intent, slot_map, beams, idx2code):
    for beam in beams:
        beam.score /= len(beam.path)
        beam.path = sub_slotmap(idx2code(beam.path)[:-1], slot_map)
    
    lemmatizer = nltk.wordnet.WordNetLemmatizer()
    intent_tokens = tokenize_for_bleu_eval(intent.lower())
    intent_tokens = [lemmatizer.lemmatize(token) for token in intent_tokens]
    manual_add_tokens(intent_tokens)
        
    slot_values = slot_map.values()
    slot_token_counts = {}
    for value in slot_values:
        slot_token_counts[value] = len(tokenize_for_bleu_eval(value))
    
    for i,beam in enumerate(beams):
        gen_code = ' '.join(beam.path)
        code_token = tokenize_for_bleu_eval(gen_code)
        
        # check how many tokens are in common
        for token in intent_tokens:
            if token in gen_code.lower():
                beam.score += 2.0 / len(intent_tokens)
            if token in code_token:
                beam.score += 1.0 / len(intent_tokens)
                
        # check how many slot_map tokens are used
        slotmap_used_count = 0
        for value in slot_values:
            if value in gen_code:
                slotmap_used_count += slot_token_counts[value]
        beam.score += float(slotmap_used_count) / len(intent_tokens)
    
    beams = sorted(beams, key=lambda x: x.score)
    return ' '.join(beams[-1].path)

# def post_process_hand(intent, slot_map, beams, idx2code):
#     for beam in beams:
#         beam.score /= len(beam.path)
#         beam.path = sub_slotmap(idx2code(beam.path)[:-1], slot_map)
    
#     lemmatizer = nltk.wordnet.WordNetLemmatizer()
#     intent_tokens = tokenize_for_bleu_eval(intent.lower())
#     intent_tokens = [lemmatizer.lemmatize(token) for token in intent_tokens]
#     manual_add_tokens(intent_tokens)
        
#     slot_values = slot_map.values()
#     slot_token_counts = {}
#     for value in slot_values:
#         slot_token_counts[value] = len(tokenize_for_bleu_eval(value))
    
#     for i,beam in enumerate(beams):
#         gen_code = ' '.join(beam.path)
# #         code_token = tokenize_for_bleu_eval(gen_code)
        
#         # check how many tokens are in common
#         for token in intent_tokens:
#             if token in gen_code.lower():
#                 beam.score += 2.0 / len(intent_tokens)
# #             if token in code_token:
# #                 beam.score += 1.0 / len(intent_tokens)
                
#         # check how many slot_map tokens are used
#         slotmap_used_count = 0
#         for value in slot_values:
#             if value in gen_code:
#                 slotmap_used_count += slot_token_counts[value]
#         beam.score += float(slotmap_used_count) / len(intent_tokens)
    
#     beams = sorted(beams, key=lambda x: x.score)
#     return ' '.join(beams[-1].path)
    
def post_process_dummy(slot_map, beams, idx2code):
    for beam in beams:
        beam.path = sub_slotmap(idx2code(beam.path)[:-1], slot_map)
    return ' '.join(beams[-1].path)

def post_process_model(intent, beams, idx2code, model, process_intent, intent2idx):
    intent_tokens, slot_map = process_intent(intent)
    intent_idx = intent2idx(intent_tokens)
    
    gen_code_idx = [beam.path[:-1] for beam in beams]
    gen_code = [' '.join(sub_slotmap(idx2code(idx), slot_map)) for idx in gen_code_idx]
    
    slot_values = slot_map.values()
    slot_token_counts = {}
    for value in slot_values:
        slot_token_counts[value] = len(tokenize_for_bleu_eval(value))
    
    slotmap_used_counts = []
    for code in gen_code:
        slotmap_used_count = 0
        for value in slot_values:
            if value in code:
                slotmap_used_count += slot_token_counts[value]
        slotmap_used_counts.append(slotmap_used_count)
        
    model_inputs = []
    for code_idx,count_token in zip(gen_code_idx, slotmap_used_counts):
        model_inputs.append((intent_idx, code_idx, count_token))
    
    best_idx = model.get_idx(model_inputs)
    return gen_code[best_idx]
    

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
        model.eval()
        lang_model = self.model2
        
        encoder_outputs, _, _ = model.encoder(src_seq)
        encoder_outputs_reshaped = encoder_outputs.permute(1, 2, 0).contiguous()
        model.decoder.attention.get_encoder_queries(encoder_outputs)
        
        prev_token = torch.LongTensor([sos])
        hidden = None 
        bad_tokens = (sos, unk)
        
        def get_best_token(tokens, bad_tokens):
            for token in tokens:
                if token not in bad_tokens and token != eos:
                    return token 
        
        # initial step for seq2seq model
        context = torch.zeros(1, model.encoder_hidden_size)
        logits, hidden, context, _ = model.decoder(prev_token, hidden,
            encoder_outputs_reshaped, context)
        p1 = torch.nn.functional.softmax(logits.view(-1), dim=0)
        logp = self.weight1 * torch.log(p1)
        
        # initial step for language model
        if lang_model is not None:
            logits, language_model_hidden = lang_model.calculateProb(prev_token, None)
            p2 = torch.nn.functional.softmax(logits.view(-1), dim=0)
            logp += self.weight2 * torch.log(p2)
        else:
            language_model_hidden = None
            
        klogp, greedy_kwords = torch.topk(logp, beam_width)
        klogp = klogp.tolist() 
        greedy_kwords = greedy_kwords.tolist()
        best_token = get_best_token(greedy_kwords, bad_tokens)
        
        bestk_paths = []
        for logp,init_word in zip(klogp,greedy_kwords):
            if init_word in bad_tokens:
                bestk_paths.append(Beam_path(eos, logp, init_word, hidden, context, best_token, language_model_hidden))
            else:
                bestk_paths.append(Beam_path(eos, logp, init_word, hidden, context, 
                                             prev_language_model_hidden=language_model_hidden))
        
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
                    logits, language_model_hidden = lang_model.calculateProb(prev_word, 
                                                                             beam_path.prev_language_model_hidden)
                    p2 = torch.nn.functional.softmax(logits.view(-1), dim=0)
                    logp += self.weight2 * torch.log(p2)
                else:
                    language_model_hidden = None
                    
                klogp, greedy_kwords = torch.topk(logp, beam_width)
                klogp = klogp.tolist()
                greedy_kwords = greedy_kwords.tolist()
                best_token = get_best_token(greedy_kwords, bad_tokens)
                new_paths.extend(beam_path.get_new_paths(greedy_kwords, bad_tokens,\
                    best_token, klogp, hidden, context, language_model_hidden))
            
            bestk_paths = Beam_path.get_bestk_paths(new_paths, beam_width)
        
        return bestk_paths
    


class Beam_path(object):
    def __init__(self, eos=None, logp=0, cur_word=None, prev_hidden=None, prev_context=None, replace_word=None, 
                 prev_language_model_hidden=None):
        self.score = logp
        self.path = [cur_word] if replace_word == None else [replace_word]
        self.prev_word = cur_word
        self.prev_hidden = prev_hidden 
        self.prev_language_model_hidden = prev_language_model_hidden
        self.prev_context = prev_context
        self.eos = eos
    
    def _copy(self):
        path = Beam_path()
        path.score = self.score 
        path.path = copy.copy(self.path)
        path.eos = self.eos
        return path
        
    def _update(self, cur_word, logp, hidden, context, language_model_hidden=None, replace_word=None):
        self.score += logp
        self.path.append(cur_word if replace_word == None else replace_word)
        self.prev_word = cur_word 
        self.prev_hidden = hidden
        self.prev_language_model_hidden = language_model_hidden
        self.prev_context = context
        
    def is_done(self):
        return self.prev_word == self.eos
        
    def get_new_paths(self, branches, bad_tokens, replace_word, logps, hidden, context, language_model_hidden=None):
        N = len(branches)
        new_paths = []
        for i in range(N):
            new_paths.append(self._copy())
        for new_path,branch,logp in zip(new_paths,branches,logps):
            if branch in bad_tokens:
                new_path._update(branch,logp,hidden,context,language_model_hidden,replace_word)
            else:
                new_path._update(branch,logp,hidden,context,language_model_hidden)
        return new_paths
        
    def __repr__(self):
        return str(self.path) + str(self.score)
        
    @staticmethod
    def get_bestk_paths(paths, k):
        sorted_paths = sorted(paths, key=lambda x: x.score / len(x.path))
        return sorted_paths[-k:]