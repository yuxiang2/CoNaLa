import json
import pickle
from tokenizer import tokenize_intent, tokenize_code
from collections import Counter
import nltk

lemmatizer = nltk.wordnet.WordNetLemmatizer()

def process_intent(intent):
    intent, slot_map = tokenize_intent(intent)
    intent = [lemmatizer.lemmatize(e.lower()) for e in intent]
    return intent, slot_map

def tokenize_conala_entry(entry):
    intent, slot_map = process_intent(entry['intent'])
    code = tokenize_code(entry['code'], slot_map)
    return intent, code, slot_map

def get_raw_entries(path=None):
    if path == None:
        path = '../corpus/train.json'
    with open(path) as f:
        return json.load(f)

### Handle all pre/post-processing of Code-Intent Pair
class Code_Intent_Pairs():
    def __init__(self):
        self.num2word = None
        self.num2code = None
        self.word2num = None
        self.code2num = None
        self.entries = None

    def __getitem__(self, idx: int):
        return self.entries[idx]

    def intent2idx(self, intent):
        word_dict = self.word2num
        unk = word_dict['<unk>']
        return [word_dict[word] if word in word_dict else unk for word in intent]

    def idx2intent(self, idxes):
        intent = []
        sos = self.word2num['<sos>']
        eos = self.word2num['<eos>']
        pad = self.word2num['<pad>']
        for idx in idxes:
            if idx not in (sos, eos, pad):
                intent.append(self.num2word[idx])
        return intent

    def code2idx(self, code, intent=None):
        code_dict = self.code2num
        if intent != None:
            idxes = []
            for token in code:
                if token in code_dict:
                    idxes.append(code_dict[token])
                elif token in intent:
                    idxes.append(len(code_dict) + intent.index(token))
                else:
                    idxes.append(code_dict['<unk>'])
            return idxes
        else:
            unk = code_dict['<unk>']
            return [code_dict[token] if token in code_dict else unk for token in code]

    def idx2code(self, idxes, intent=None):
        tokens = []
        sos = self.code2num['<sos>']
        eos = self.code2num['<eos>']
        pad = self.code2num['<pad>']
        num2code = self.num2code
        size = len(num2code)
        if intent != None:
            for idx in idxes:
                if idx < size:
                    if idx not in (sos, eos, pad):
                        tokens.append(num2code[idx])
                else:
                    tokens.append(intent[idx - size])
            return tokens
        else:
            return [num2code[idx] for idx in idxes]

    def get_dict_from_raw(self, path=None, word_cut_freq=5, code_cut_freq=3, copy=True, store=True):
        raw_entries = get_raw_entries(path)
        intents, codes, slot_maps = zip(*list(map(tokenize_conala_entry, raw_entries)))

        def get_vocab(sentences, cut_freq):
            vocab = Counter()
            for sentence in sentences:
                for word in sentence:
                    vocab[word] += 1

            vocab = [k for k in vocab if vocab[k] >= cut_freq]
            vocab.extend(['<unk>', '<sos>', '<eos>', '<pad>'])
            return vocab

        num2word = get_vocab(intents, word_cut_freq)
        word2num = dict(zip(num2word, range(0, len(num2word))))

        num2code = get_vocab(codes, code_cut_freq)
        code2num = dict(zip(num2code, range(0, len(num2code))))

        self.num2word = num2word
        self.word2num = word2num
        self.num2code = num2code
        self.code2num = code2num

    def load_raw_data(self, path):
        raw_entries = get_raw_entries(path)
        entries = [tokenize_conala_entry(entry) for entry in raw_entries]

        self.entries = []
        for entry in entries:
            intent, code, slot_map = entry
            intent_idx = self.intent2idx(intent)
            code_idx_copy = self.code2idx(code, intent)
            code_idx_nocopy = self.code2idx(code)
            entry_dict = {
                'intent': intent,
                'code': code,
                'slot_map': slot_map,
                'intent_indx': intent_idx,
                'code_indx_copy': code_idx_copy,
                'code_indx_nocopy': code_idx_nocopy
            }
            self.entries.append(entry_dict)
        return self.entries

    def load_raw_test_data(self, path):
        raw_entries = get_raw_entries(path)
        entries = [tokenize_conala_entry(entry) for entry in raw_entries]

        self.entries = []
        for i in range(len(entries)):
            intent, _, slot_map = entries[i]
            intent_idx = self.intent2idx(intent)
            intent = raw_entries[i]['intent']
            code = raw_entries[i]['code']
            entry_dict = {
                'intent': intent,
                'code': code,
                'slot_map': slot_map,
                'intent_indx': intent_idx,
            }
            self.entries.append(entry_dict)
        return self.entries

    def store_entries(self, path):
        with open(path, 'w') as f:
            json.dump(self.entries, f, indent=2)

    def load_entries(self, path):
        with open(path) as f:
            self.entries = json.load(f)
        return self.entries

    def store_dict(self, path=None):
        if path == None:
            path = '../vocab/'
        word_dict_path = path + 'word_dict.bin'
        pickle.dump(self.num2word, open(word_dict_path, 'wb'))
        code_dict_path = path + 'code_dict.bin'
        pickle.dump(self.num2code, open(code_dict_path, 'wb'))

    def load_dict(self, path=None):
        if path == None:
            path = '../vocab/'
        word_dict_path = path + 'word_dict.bin'
        self.num2word = pickle.load(open(word_dict_path, 'rb'))
        self.word2num = dict(zip(self.num2word, range(0, len(self.num2word))))
        code_dict_path = path + 'code_dict.bin'
        self.num2code = pickle.load(open(code_dict_path, 'rb'))
        self.code2num = dict(zip(self.num2code, range(0, len(self.num2code))))

    def pad(self, pad_code=True, pad_intent=False):
        if not pad_code and not pad_intent:
            return self.entries
        if pad_code:
            code_sos = self.code2num['<sos>']
            code_eos = self.code2num['<eos>']
        if pad_intent:
            intent_sos = self.word2num['<sos>']
            intent_eos = self.word2num['<eos>']
        for entry in self.entries:
            if pad_code:
                entry['code_indx_copy'] = [code_sos] + entry['code_indx_copy'] + [code_eos]
                entry['code_indx_nocopy'] = [code_sos] + entry['code_indx_nocopy'] + [code_eos]
            if pad_intent:
                entry['intent_indx'] = [intent_sos] + entry['intent_indx'] + [intent_eos]

    def get_special_symbols(self):
        return {
            'word_pad': self.word2num['<pad>'],
            'word_sos': self.word2num['<sos>'],
            'word_eos': self.word2num['<eos>'],
            'word_unk': self.word2num['<unk>'],
            'code_pad': self.code2num['<pad>'],
            'code_sos': self.code2num['<sos>'],
            'code_eos': self.code2num['<eos>'],
            'code_unk': self.code2num['<unk>'],
        }

    def get_word_size(self):
        return len(self.num2word)

    def get_code_size(self):
        return len(self.num2code)


### Handle code language model processing
class Code():
    def __init__(self):
        self.num2code = None
        self.code2num = None
        self.codes_indx = None

    def load_dict(self, path=None):
        if path == None:
            path = '../vocab/'
        code_dict_path = path + 'code_dict.bin'
        self.num2code = pickle.load(open(code_dict_path, 'rb'))
        self.code2num = dict(zip(self.num2code, range(0, len(self.num2code))))

    def get_special_symbols(self):
        return {
            'pad': self.code2num['<pad>'],
            'sos': self.code2num['<sos>'],
            'eos': self.code2num['<eos>'],
        }

    def code2idx(self, code):
        code_dict = self.code2num
        unk = code_dict['<unk>']
        return [code_dict[token] if token in code_dict else unk for token in code]

    def load_data(self, path):
        with open(path, 'r') as f:
            lines_token = [tokenize_code(line) for line in f]
        self.codes_indx = [self.code2idx(line_token) for line_token in lines_token]
        return self.codes_indx

    def pad(self, pad_length=1):
        if pad_length <= 0:
            return self.code_indxes
        sos = self.code2num['<sos>']
        eos = self.code2num['<eos>']
        for i in range(len(self.codes_indx)):
            self.codes_indx[i] = [sos] * pad_length \
                                 + self.codes_indx[i] + [eos] * pad_length
        return self.codes_indx
