# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.utils
from torch.autograd import Variable
import torch.nn.functional as F


class PointerNet(nn.Module):
    def __init__(self, hidden_size):
        super(PointerNet, self).__init__()
        self.src_encoding_linear = nn.Linear(hidden_size, hidden_size, bias=False)


def forward(self, encoding_info, src_token_len, hidden_state):
    """
    :param encoding_info: Variable(max_len, 1, hidden_size)
    :param src_token_len: int
    :param hidden_state: Variable(1, hidden_size)
    :return: Variable(tgt_action_num, batch_size, src_sent_len)
    """

    lineared_att = self.src_encoding_linear(encoding_info.squeeze(1)).unsqueeze(0)
    attn_input = torch.bmm(lineared_att, hidden_state.unsqueeze(2)).squeeze(2)

    attn_input[src_token_len:] = -float('inf')

    # attn_weights = F.softmax(attn_input, dim=1)
    # return attn_weights

    return attn_input
