#*----------------------------------------------------------------------------*
#* Copyright (C) 2023 IBM Inc. All rights reserved                            *
#* SPDX-License-Identifier: GPL-3.0-only                                      *
#*----------------------------------------------------------------------------*

import torch as t
import torch.nn as nn
import torch.functional as f
from const import DIM_POSITION_2x2, DIM_NUMBER_2x2, DIM_POSITION_3x3, DIM_NUMBER_3x3, DIM_TYPE, DIM_SIZE, DIM_COLOR
from nvsa.reasoning.vsa_block_utils import block_binding3, block_unbinding2
    
class MLP(nn.Module):

    def __init__(self, d_in, d_out, d_vsa, hidden_layers=3, softmax=False):
        super(MLP, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.d_vsa = d_vsa
        self.hidden_layers = hidden_layers

        if self.d_in != self.d_vsa:
            self.learnable_vsa = nn.LazyLinear(self.d_vsa)
        
        layers = []

        for _ in range(0, self.hidden_layers):
            layers.append(nn.LazyLinear(self.d_vsa))
            layers.append(nn.LayerNorm(self.d_vsa))
            layers.append(nn.ReLU())
        
        if softmax:
            layers.append(nn.Linear(self.d_vsa, self.d_out))
            layers.append(nn.LayerNorm(self.d_out))
            layers.append(nn.Softmax(dim=-1))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        if len(x.shape) > 2:
            if self.d_in != self.d_vsa:
                x = self.learnable_vsa(x)
            x = x.reshape((x.shape[0], -1))
        return self.layers(x)


class LearnableFormula(nn.Module):
    def __init__(self, d_vsa, k, context_indeces, hardcode=None):
        super(LearnableFormula, self).__init__()
        self.d_vsa = d_vsa
        self.k = k
        self.context_len = len(context_indeces) + 1
        self.hardcode = hardcode
        
        idx_term_map = ["a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "i"]
        self.idx_term_map = [idx_term_map[i] for i in context_indeces].append("i")

        if hardcode == None:
            self.n1 = nn.Parameter(t.randn(self.context_len))
            self.n2 = nn.Parameter(t.randn(self.context_len))
            self.n3 = nn.Parameter(t.randn(self.context_len))
            self.d1 = nn.Parameter(t.randn(self.context_len))
            self.d2 = nn.Parameter(t.randn(self.context_len))
            self.d3 = nn.Parameter(t.randn(self.context_len))
        else:
            self.n1 = nn.Parameter(hardcode[0])
            self.n2 = nn.Parameter(hardcode[1])
            self.n3 = nn.Parameter(hardcode[2])
            self.d1 = nn.Parameter(hardcode[3])
            self.d2 = nn.Parameter(hardcode[4])
            self.d3 = nn.Parameter(hardcode[5])
        
        self.softmax = nn.Softmax(dim=-1)

    def __str__(self):
        n1 = self.idx_term_map[t.argmax(self.att_scores_n1).item()]
        n2 = self.idx_term_map[t.argmax(self.att_scores_n2).item()]
        n3 = self.idx_term_map[t.argmax(self.att_scores_n3).item()]
        d1 = self.idx_term_map[t.argmax(self.att_scores_d1).item()]
        d2 = self.idx_term_map[t.argmax(self.att_scores_d2).item()]
        d3 = self.idx_term_map[t.argmax(self.att_scores_d3).item()]
        n1_prob = round(t.max(self.att_scores_n1).item(), 2)
        n2_prob = round(t.max(self.att_scores_n2).item(), 2)
        n3_prob = round(t.max(self.att_scores_n3).item(), 2)
        d1_prob = round(t.max(self.att_scores_d1).item(), 2)
        d2_prob = round(t.max(self.att_scores_d2).item(), 2)
        d3_prob = round(t.max(self.att_scores_d3).item(), 2)
        probs = [n1_prob, n2_prob, n3_prob, d1_prob, d2_prob, d3_prob]
        string = "(" + n1 + "·" + n2 + "·" + n3 + ")*(" + d1 + "·" + d2 + "·" + d3 + ")" + " with confidences: " + str(probs)
        return string

    def add_identity(self, x):
        identity = t.zeros_like(x[:, 0]).unsqueeze(1)
        identity[:, :, :, 0] = 1
        x_with_identity = t.cat((x, identity), dim=1)
        return x_with_identity
        
    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], self.k, -1)
        x = self.add_identity(x)
        if self.hardcode == None:
            self.att_scores_n1 = self.softmax(self.n1.unsqueeze(0).unsqueeze(0))
            self.att_scores_n2 = self.softmax(self.n2.unsqueeze(0).unsqueeze(0))
            self.att_scores_n3 = self.softmax(self.n3.unsqueeze(0).unsqueeze(0))
            self.att_scores_d1 = self.softmax(self.d1.unsqueeze(0).unsqueeze(0))
            self.att_scores_d2 = self.softmax(self.d2.unsqueeze(0).unsqueeze(0))
            self.att_scores_d3 = self.softmax(self.d3.unsqueeze(0).unsqueeze(0))
        else:
            self.att_scores_n1 = self.n1.unsqueeze(0).unsqueeze(0)
            self.att_scores_n2 = self.n2.unsqueeze(0).unsqueeze(0)
            self.att_scores_n3 = self.n3.unsqueeze(0).unsqueeze(0)
            self.att_scores_d1 = self.d1.unsqueeze(0).unsqueeze(0)
            self.att_scores_d2 = self.d2.unsqueeze(0).unsqueeze(0)
            self.att_scores_d3 = self.d3.unsqueeze(0).unsqueeze(0)
        x = x.view(x.shape[0], x.shape[1], -1)
        n1 = t.matmul(self.att_scores_n1.repeat(x.shape[0], 1, 1), x).squeeze(1).view(x.shape[0], self.k, -1)
        n2 = t.matmul(self.att_scores_n2.repeat(x.shape[0], 1, 1), x).squeeze(1).view(x.shape[0], self.k, -1)
        n3 = t.matmul(self.att_scores_n3.repeat(x.shape[0], 1, 1), x).squeeze(1).view(x.shape[0], self.k, -1)
        d1 = t.matmul(self.att_scores_d1.repeat(x.shape[0], 1, 1), x).squeeze(1).view(x.shape[0], self.k, -1)
        d2 = t.matmul(self.att_scores_d2.repeat(x.shape[0], 1, 1), x).squeeze(1).view(x.shape[0], self.k, -1)
        d3 = t.matmul(self.att_scores_d3.repeat(x.shape[0], 1, 1), x).squeeze(1).view(x.shape[0], self.k, -1)
        n = block_binding3(n1, n2, n3)
        d = block_binding3(d1, d2, d3)
        output = block_unbinding2(n, d)
        output = output.view(output.shape[0], -1)
        return output