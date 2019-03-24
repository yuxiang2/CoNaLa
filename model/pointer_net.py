# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.utils
from torch.autograd import Variable
import torch.nn.functional as F


class PointerNet(nn.Module):
    def __init__(self, encoder_hidden_size, hidden_size):
        super(PointerNet, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(encoder_hidden_size + hidden_size, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
            nn.Tanh()
        )

    def forward(self, encoding_info, src_token_len, hidden_state):
        """
        :param encoding_info: Variable(max_len, encoder_hidden_size)
        :param src_token_len: int
        :param hidden_state: Variable(1, hidden_size)
        :return: Variable(1, seq_len)
        """
        max_length = encoding_info.size(0)

        hidden_for_att = hidden_state.repeat(max_length, 1, 1).permute(1, 0, 2).contiguous()
        #(1, max_len, encoder_hidden_size + hidden_size)
        # print(encoding_info.size())
        # print(hidden_for_att.size())
        att_features = torch.cat((encoding_info, hidden_for_att.squeeze()), 1)

        # Variable(max_len, 1)
        lineared_att = self.attn(att_features).squeeze()
        # lineared_att[src_token_len:] = -float('inf')

        # attn_weights = F.softmax(attn_input, dim=1)
        # return attn_weights

        return lineared_att
