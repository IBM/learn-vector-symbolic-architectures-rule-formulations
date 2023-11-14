#*----------------------------------------------------------------------------*
#* Copyright (C) 2023 IBM Inc. All rights reserved                            *
#* SPDX-License-Identifier: GPL-3.0-only                                      *
#*----------------------------------------------------------------------------*

from collections import namedtuple
import torch as t 
import torch.nn as nn
from const import DIM_POSITION_2x2, DIM_POSITION_3x3, DIM_NUMBER_2x2, DIM_NUMBER_3x3, DIM_TYPE, DIM_SIZE, DIM_COLOR
from nvsa.reasoning.vsa_block_utils import binding_circular, block_discrete_codebook
from models import MLP, LearnableFormula
from VSAConverter import VSAConverter

Scene = namedtuple("Scene", ["position", "number", "type", "size", "color"])

class RuleLevelReasoner(nn.Module):

    def __init__(self, device, constellation, model, hidden_layers, dictionary, vsa_conversion=False, vsa_selection=False, context_superposition=False, num_rules=5, shared_rules=False):
        super(RuleLevelReasoner, self).__init__()
        self.device = device
        self.constellation = constellation
        self.model = model
        self.compute_attribute_dims()
        self.d = dictionary.shape[1]*dictionary.shape[2]
        self.k = dictionary.shape[1]
        self.vsa_conversion = vsa_conversion
        self.vsa_selection = vsa_selection
        self.context_superposition = context_superposition
        if self.model == "LearnableFormula":
            self.vsa_converter = VSAConverter(device, self.constellation, dictionary, dictionary_type="Continuous")
        else:
            self.vsa_converter = VSAConverter(device, self.constellation, dictionary)
        self.num_rules = num_rules
        if self.context_superposition or self.model == "LearnableFormula":
            context_keys, _ = block_discrete_codebook(device=device, d=self.d, k=self.k, scene_dim=8)
            self.context_keys = nn.Parameter(context_keys)
            position_number_keys, _ = block_discrete_codebook(device=device, d=self.d, k=self.k, scene_dim=2)
            self.position_key, self.number_key = nn.Parameter(position_number_keys[0]), nn.Parameter(position_number_keys[1])
        else:
            self.context_keys = None
        self.shared_rules = shared_rules
        self.compute_attribute_rules_sets(hidden_layers)
        
    def compute_attribute_dims(self):
        if "distribute" in self.constellation:
            if "four" in self.constellation:
                DIM_POSITION = DIM_POSITION_2x2
                DIM_NUMBER = DIM_NUMBER_2x2
            else:
                DIM_POSITION = DIM_POSITION_3x3
                DIM_NUMBER = DIM_NUMBER_3x3
            self.position_dim = DIM_POSITION
            self.number_dim = DIM_NUMBER
        self.type_dim = DIM_TYPE + 1
        self.size_dim = DIM_SIZE + 1
        self.color_dim = DIM_COLOR + 1
    
    def compute_attribute_rules_sets(self, hidden_layers):
        if self.shared_rules:
            rules_set = RulesSet(self.model, hidden_layers, self.num_rules, 2*self.d, -1, self.d, self.k, self.context_superposition, self.context_keys)
            if "distribute" in self.constellation:
                self.pos_num_rules_set = rules_set
            self.type_rules_set = rules_set
            self.size_rules_set = rules_set
            self.color_rules_set = rules_set
        else:
            if self.vsa_conversion:
                if "distribute" in self.constellation:
                    self.pos_num_rules_set = RulesSet(self.model, hidden_layers, self.num_rules, 2*self.d, self.position_dim, self.d, self.k, self.context_superposition, self.context_keys)
                self.type_rules_set = RulesSet(self.model, hidden_layers, self.num_rules, self.d, self.type_dim, self.d, self.k, self.context_superposition, self.context_keys)
                self.size_rules_set = RulesSet(self.model, hidden_layers, self.num_rules, self.d, self.size_dim, self.d, self.k, self.context_superposition, self.context_keys)
                self.color_rules_set = RulesSet(self.model, hidden_layers, self.num_rules, self.d, self.color_dim, self.d, self.k, self.context_superposition, self.context_keys)
            else:
                if "distribute" in self.constellation:
                    self.pos_num_rules_set = RulesSet(self.model, hidden_layers, self.num_rules, self.position_dim + self.number_dim, self.position_dim, self.d, self.k, self.context_superposition, self.context_keys)
                self.type_rules_set = RulesSet(self.model, hidden_layers, self.num_rules, self.type_dim, self.type_dim, self.d, self.k, self.context_superposition, self.context_keys)
                self.size_rules_set = RulesSet(self.model, hidden_layers, self.num_rules, self.size_dim, self.size_dim, self.d, self.k, self.context_superposition, self.context_keys)
                self.color_rules_set = RulesSet(self.model, hidden_layers, self.num_rules, self.color_dim, self.color_dim, self.d, self.k, self.context_superposition, self.context_keys)

    def compute_vsa_outputs(self, outputs):
        if "distribute" in self.constellation:
            position_dist = outputs.position.unsqueeze(-1)
            position_dict = self.vsa_converter.position_dictionary.reshape((self.vsa_converter.position_dictionary.shape[0], -1)).unsqueeze(0).unsqueeze(0).unsqueeze(0)
            position = t.sum(position_dist*position_dict, dim=-2)
        else:
            position = None
        
        type_dist = outputs.type.unsqueeze(-1)
        type_dict = self.vsa_converter.type_dictionary.reshape((self.vsa_converter.type_dictionary.shape[0], -1)).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        type = t.sum(type_dist*type_dict, dim=-2)
        
        size_dist = outputs.size.unsqueeze(-1)
        size_dict = self.vsa_converter.size_dictionary.reshape((self.vsa_converter.size_dictionary.shape[0], -1)).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        size = t.sum(size_dist*size_dict, dim=-2)
        
        color_dist = outputs.color.unsqueeze(-1)
        color_dict = self.vsa_converter.color_dictionary.reshape((self.vsa_converter.color_dictionary.shape[0], -1)).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        color = t.sum(color_dist*color_dict, dim=-2)
        
        return Scene(position, None, type, size, color)

    def position_number_superposition(self, position, number):
        position_binded = binding_circular(position.reshape((-1, self.k, self.d//self.k)), 
                         self.position_key.unsqueeze(0).repeat(position.shape[0]*position.shape[1], 1, 1))
        number_binded = binding_circular(number.reshape((-1, self.k, self.d//self.k)), 
                         self.number_key.unsqueeze(0).repeat(number.shape[0]*number.shape[1], 1, 1))
        pos_num = (position_binded + number_binded)/2
        pos_num = pos_num.reshape((position.shape[0], position.shape[1], -1))
        return pos_num

    def forward(self, scene_prob):
        scene_vsa = self.vsa_converter(scene_prob)
        if "distribute" in self.constellation:
            scene_vsa = Scene(t.flatten(scene_vsa.position, start_dim=len(scene_vsa.position.shape)-2),
                              t.flatten(scene_vsa.number, start_dim=len(scene_vsa.number.shape)-2),
                              t.flatten(scene_vsa.type, start_dim=len(scene_vsa.type.shape)-2),
                              t.flatten(scene_vsa.size, start_dim=len(scene_vsa.size.shape)-2),
                              t.flatten(scene_vsa.color, start_dim=len(scene_vsa.color.shape)-2))
        else:
            scene_vsa = Scene(None, 
                              None,
                              t.flatten(scene_vsa.type, start_dim=len(scene_vsa.type.shape)-2),
                              t.flatten(scene_vsa.size, start_dim=len(scene_vsa.size.shape)-2),
                              t.flatten(scene_vsa.color, start_dim=len(scene_vsa.color.shape)-2))

        scene_prob = Scene(scene_prob.position_prob, scene_prob.number_prob, scene_prob.type_prob, scene_prob.size_prob, scene_prob.color_prob)

        if self.vsa_conversion:
            scene = scene_vsa
        else:
            scene = scene_prob
        
        test_indeces = [2, 5]

        if "distribute" in self.constellation:
            if self.model == "LearnableFormula" or self.context_superposition:
                pos_num = self.position_number_superposition(scene.position[:, :8], scene.number[:, :8])
                pos_num_output = self.pos_num_rules_set(pos_num)
            else:
                pos_num_output = self.pos_num_rules_set(t.cat((scene.position, scene.number), dim=-1))
        else:
            pos_num_output = None
        type_output = self.type_rules_set(scene.type)
        size_output = self.size_rules_set(scene.size)
        color_output = self.color_rules_set(scene.color)
        outputs = Scene(pos_num_output, None, type_output, size_output, color_output)
        if self.vsa_selection:
            if self.model != "LearnableFormula":
                outputs = self.compute_vsa_outputs(outputs)
            scene = scene_vsa
        else:
            scene = scene_prob
        
        if "distribute" in self.constellation:
            tests = Scene(scene.position[:, test_indeces], scene.number[:, test_indeces], scene.type[:, test_indeces], scene.size[:, test_indeces], scene.color[:, test_indeces])
            candidates = Scene(scene.position[:, 8:], scene.number[:, 8:], scene.type[:, 8:], scene.size[:, 8:], scene.color[:, 8:])
        else:
            tests = Scene(None, None, scene.type[:, test_indeces], scene.size[:, test_indeces], scene.color[:, test_indeces])
            candidates = Scene(None, None, scene.type[:, 8:], scene.size[:, 8:], scene.color[:, 8:])
        
        return outputs, candidates, tests

class RulesSet(nn.Module):

    def __init__(self, model, hidden_layers, num_rules, d_in, d_out, d_vsa, k, context_superpostion=False, context_keys=None):
        super(RulesSet, self).__init__()
        self.rules = nn.ModuleList([Rule(model, hidden_layers, d_in, d_out, d_vsa, k, context_superpostion, context_keys)
                                        for _ in range(num_rules)])
    
    def forward(self, attribute):
        output_list = [rule(attribute).reshape((attribute.shape[0], 3, -1)) for rule in self.rules]
        outputs = t.stack(output_list, dim=1)
        return outputs
    
class Rule(nn.Module):

    def __init__(self, model, hidden_layers, d_in, d_out, d_vsa, k, context_superposition=False, context_keys=None):
        super(Rule, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.d = d_vsa
        self.k = k
        self.context_superposition = context_superposition
        self.context_keys = context_keys
        self.a3_context_indeces = [0, 1, 3, 4, 5, 6, 7]
        self.a6_context_indeces = [0, 1, 2, 3, 4, 6, 7]
        self.a9_context_indeces = [0, 1, 2, 3, 4, 5, 6, 7]
        self.compute_rule(model, hidden_layers)
    
    def compute_rule(self, model, hidden_layers):
        if model == "MLP":
            self.rule_a3 = MLP(self.d_in, self.d_out, self.d, hidden_layers=hidden_layers, softmax=True)
            self.rule_a6 = MLP(self.d_in, self.d_out, self.d, hidden_layers=hidden_layers, softmax=True)
            self.rule_a9 = MLP(self.d_in, self.d_out, self.d, hidden_layers=hidden_layers, softmax=True)
        elif model == "LearnableFormula":
            self.rule_a3 = LearnableFormula(self.d, self.k, self.a3_context_indeces)
            self.rule_a6 = LearnableFormula(self.d, self.k, self.a6_context_indeces)
            self.rule_a9 = LearnableFormula(self.d, self.k, self.a9_context_indeces)

    def compute_context_superposition(self, x, context_indeces):
        if self.context_superposition:
            if x.shape[2] == self.d:
                x_with_superposition = binding_circular(x.reshape((-1, self.k, self.d//self.k)), 
                                       self.context_keys[context_indeces].repeat(x.shape[0], 1, 1))
                x_with_superposition = x_with_superposition.reshape((x.shape[0], x.shape[1], -1))
            else:
                x_pos = x[:, :, :self.d]
                x_pos_with_superposition = binding_circular(x_pos.reshape((-1, self.k, self.d//self.k)), 
                                           self.context_keys[context_indeces].repeat(x_pos.shape[0], 1, 1))
                x_pos_with_superposition = x_pos_with_superposition.reshape((x.shape[0], x.shape[1], -1))
                x_num = x[:, :, self.d:]
                x_num_with_superposition = binding_circular(x_num.reshape((-1, self.k, self.d//self.k)), 
                                           self.context_keys[context_indeces].repeat(x_num.shape[0], 1, 1))
                x_num_with_superposition = x_num_with_superposition.reshape((x.shape[0], x.shape[1], -1))
                x_with_superposition = t.cat((x_pos_with_superposition, x_num_with_superposition), dim=-1)
            x_with_superposition = x_with_superposition.sum(dim=1)/x.shape[1]
        else:
            x_with_superposition = x
        return x_with_superposition
        
    def forward(self, x):
        a3 = self.rule_a3(self.compute_context_superposition(x[:, self.a3_context_indeces], self.a3_context_indeces))
        a6 = self.rule_a6(self.compute_context_superposition(x[:, self.a6_context_indeces], self.a6_context_indeces))
        a9 = self.rule_a9(self.compute_context_superposition(x[:, self.a9_context_indeces], self.a9_context_indeces))
        return t.cat((a3, a6, a9), dim=1)