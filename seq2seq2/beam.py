import copy

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