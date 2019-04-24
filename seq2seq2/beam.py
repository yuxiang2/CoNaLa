import copy

class Beam_path(object):
    def __init__(self, eos, logp=0, init_word=None, prev_hidden=None, prev_context=None):
        self.logp = logp
        self.path = [init_word] if init_word != None else []
        self.prev_word = init_word
        self.prev_hidden = prev_hidden 
        self.prev_context = prev_context
        self.eos = eos
    
    def _copy(self):
        path = Beam_path()
        path.logp = self.logp 
        path.path = copy.copy(self.path)
        return path
        
    def _update(self, branch, logp, hidden, context):
        self.logp += logp
        self.path.append(branch)
        self.prev_word = branch 
        self.prev_hidden = hidden
        self.prev_context = context
        
    def is_done(self):
        return self.prev_word == self.eos
        
    def get_new_paths(self, branches, logps, hidden, context):
        N = len(branches)
        new_paths = []
        for i in range(len(branches)):
            new_paths.append(self._copy())
        for new_path,branch,logp in zip(new_paths,branches,logps):
            new_path._update(branch,logp,hidden,context)
        return new_paths
        
    def __repr__(self):
        return str(self.path) + str(self.logp)
        
    @staticmethod
    def get_bestk_paths(paths, k):
        sorted_paths = sorted(paths, key=lambda x: x.logp)
        return sorted_paths[-k:]