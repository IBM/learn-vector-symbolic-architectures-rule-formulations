#*----------------------------------------------------------------------------*
#* Copyright (C) 2023 IBM Inc. All rights reserved                            *
#* SPDX-License-Identifier: GPL-3.0-only                                      *
#*----------------------------------------------------------------------------*

from collections import namedtuple
import torch as t
import torch.nn as nn
import torch.nn.functional as F

Scene = namedtuple("Scene", ["position", "number", "type", "size", "color"])

class RuleSelector(nn.Module):

    def __init__(self, loss_fn, temperature, rule_selector='sample'):
        super(RuleSelector, self).__init__()
        self.loss_fn = loss_fn
        self.temperature = temperature
        self.train_mode = True
        self.attribute_forward = getattr(self,"attribute_forward_"+rule_selector) 
    
    def train(self):
        self.train_mode = True
    
    def eval(self):
        self.train_mode = False

    def attribute_forward_sample(self, outputs, tests, candidates=None, targets=None):
        if self.train_mode:
            # append the ground truth answer panel to the tests. 
            tests = t.cat((tests, candidates[t.arange(candidates.shape[0]), targets].unsqueeze(1)), dim=1).unsqueeze(1).expand(-1, outputs.shape[1], -1, -1)
            scores = self.loss_fn.score(outputs, tests).mean(dim=-1)
            probs = F.softmax(scores/self.temperature, dim=-1)
            dists = t.distributions.Categorical(probs=probs)
            chosen_rules = dists.sample()
        else:
            tests = tests.unsqueeze(1).expand(-1, outputs.shape[1], -1, -1)
            scores = self.loss_fn.score(outputs[:, :, :2], tests).mean(dim=-1)
            chosen_rules = t.argmax(scores, dim=-1)

        outputs = outputs[t.arange(len(chosen_rules)), chosen_rules]
        return outputs

    def attribute_forward_weight(self, outputs, tests, candidates=None, targets=None):
        if self.train_mode:
            # append the ground truth answer panel to the tests. MICHAEL: what is outputs.shape[1]
            tests = t.cat((tests, candidates[t.arange(candidates.shape[0]), targets].unsqueeze(1)), dim=1).unsqueeze(1).expand(-1, outputs.shape[1], -1, -1)
            scores = self.loss_fn.score(outputs, tests).mean(dim=-1)
            weights = F.softmax(scores/self.temperature, dim=-1)
        else:
            tests = tests.unsqueeze(1).expand(-1, outputs.shape[1], -1, -1)
            scores = self.loss_fn.score(outputs[:, :, :2], tests).mean(dim=-1)
            weights = F.softmax(scores/self.temperature, dim=-1)

        outputs = t.einsum('ijkh,ij->ikh',outputs, weights)
        return outputs
    
    def forward(self, outputs, tests, candidates=None, targets=None, use_position=True):
        if self.train_mode:
            if outputs.position != None and use_position:
                pos_num_output = self.attribute_forward(outputs.position, tests.position, candidates.position, targets)
            else:
                pos_num_output = None
            type_output = self.attribute_forward(outputs.type, tests.type, candidates.type, targets)
            size_output = self.attribute_forward(outputs.size, tests.size, candidates.size, targets)
            color_output = self.attribute_forward(outputs.color, tests.color, candidates.color, targets)
        else:
            if outputs.position != None and use_position:
                pos_num_output = self.attribute_forward(outputs.position, tests.position)
            else:
                pos_num_output = None
            type_output = self.attribute_forward(outputs.type, tests.type)
            size_output = self.attribute_forward(outputs.size, tests.size)
            color_output = self.attribute_forward(outputs.color, tests.color)
        outputs = Scene(pos_num_output, None, type_output, size_output, color_output)
        return outputs