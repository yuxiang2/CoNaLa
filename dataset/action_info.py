# coding=utf-8
from asdl.hypothesis import Hypothesis
from asdl.transition_system import ApplyRuleAction, GenTokenAction


class ActionInfo(object):
    """sufficient statistics for making a prediction of an action at a time step"""

    def __init__(self, action=None):
        self.t = 0
        self.parent_t = -1
        self.action = action
        self.frontier_prod = None
        self.frontier_field = None
        self.idx = None

        # for GenToken actions only
        self.copy_from_src = False
        self.src_token_position = -1
        self.token = None

    def __repr__(self, verbose=False):
        repr_str = '%s (idx=%d, t=%d, p_t=%d, frontier_field=%s)' % (repr(self.action),
                                                         self.idx,
                                                         self.t,
                                                         self.parent_t,
                                                         self.frontier_field.__repr__(True) if self.frontier_field else 'None')

        if verbose:
            verbose_repr = 'action_prob=%.4f, ' % self.action_prob
            if isinstance(self.action, GenTokenAction):
                verbose_repr += 'in_vocab=%s, ' \
                                'gen_copy_switch=%s, ' \
                                'p(gen)=%s, p(copy)=%s, ' \
                                'has_copy=%s, copy_pos=%s' % (self.in_vocab,
                                                              self.gen_copy_switch,
                                                              self.gen_token_prob, self.copy_token_prob,
                                                              self.copy_from_src, self.src_token_position)

            repr_str += '\n' + verbose_repr

        return repr_str


def get_action_infos(src_query, tgt_actions, action2num, token2num, force_copy=False):
    action_infos = []
    hyp = Hypothesis()
    for t, action in enumerate(tgt_actions):
        action_info = ActionInfo(action)
        action_info.t = t
        if hyp.frontier_node:
            action_info.parent_t = hyp.frontier_node.created_time
            action_info.frontier_prod = hyp.frontier_node.production
            action_info.frontier_field = hyp.frontier_field.field

        if isinstance(action, GenTokenAction):
            try:
                tok_src_idx = src_query.index(str(action.token))
                action_info.copy_from_src = True
                action_info.src_token_position = tok_src_idx
                action_info.idx = action2num[GenTokenAction('copy')]
            except ValueError:
                if force_copy: raise ValueError('cannot copy primitive token %s from source' % action.token)
                action_info.idx = action2num[GenTokenAction('token')]
                if action.token in token2num:
                    action_info.token = token2num[action.token]
                else:
                    action_info.token = token2num['<UNK>']

        else:
            action_info.idx = action2num[action]

        hyp.apply_action(action)
        action_infos.append(action_info)

    return action_infos
