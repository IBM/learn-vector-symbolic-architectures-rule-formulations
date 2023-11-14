#*----------------------------------------------------------------------------*
#* Copyright (C) 2023 IBM Inc. All rights reserved                            *
#* SPDX-License-Identifier: GPL-3.0-only                                      *
#*----------------------------------------------------------------------------*

from collections import namedtuple
import torch as t 
import torch.nn as nn
from const import DIM_POSITION_2x2, DIM_POSITION_3x3, DIM_NUMBER_2x2, DIM_NUMBER_3x3, DIM_TYPE, DIM_SIZE, DIM_COLOR
from nvsa.reasoning.vsa_block_utils import binding_circular, block_discrete_codebook
from models import MLP
from VSAConverter import VSAConverter

Scene = namedtuple("Scene", ["position", "number", "type", "size", "color"])

class AttributeLevelReasoner(nn.Module):
        
    def __init__(self, device, constellation, model, hidden_layers, dictionary, vsa_conversion=False, vsa_selection=False, context_superposition=False):
        super(AttributeLevelReasoner, self).__init__()
        self.device = device
        self.constellation = constellation
        self.compute_attribute_dims()
        self.d = dictionary.shape[1]*dictionary.shape[2]
        self.k = dictionary.shape[1]
        self.vsa_conversion = vsa_conversion
        self.vsa_selection = vsa_selection
        self.context_superposition = context_superposition
        self.vsa_converter = VSAConverter(device, self.constellation, dictionary)
        self.compute_attribute_models(model, hidden_layers)
        if self.context_superposition:
            context_keys, _ = block_discrete_codebook(device=device, d=self.d, k=self.k, scene_dim=8)
            self.context_keys = nn.Parameter(context_keys)
            position_number_keys, _ = block_discrete_codebook(device=device, d=self.d, k=self.k, scene_dim=2)
            self.position_key, self.number_key = nn.Parameter(position_number_keys[0]), nn.Parameter(position_number_keys[1])

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
    
    def compute_attribute_models(self, model, hidden_layers):
        if self.vsa_conversion:
            if "distribute" in self.constellation:
                self.pos_num_model = MLP(2*self.d, self.position_dim, self.d, hidden_layers=hidden_layers, softmax=True)
            self.type_model = MLP(self.d, self.type_dim, self.d, softmax=True)
            self.size_model = MLP(self.d, self.size_dim, self.d, softmax=True)
            self.color_model = MLP(self.d, self.color_dim, self.d, softmax=True)
        else:
            if "distribute" in self.constellation:
                self.pos_num_model = MLP(self.position_dim + self.number_dim, self.position_dim, self.d, hidden_layers=hidden_layers, softmax=True)
            self.type_model = MLP(self.type_dim, self.type_dim, self.d, softmax=True)
            self.size_model = MLP(self.size_dim, self.size_dim, self.d, softmax=True)
            self.color_model = MLP(self.color_dim, self.color_dim, self.d, softmax=True)
    
    def compute_vsa_outputs(self, outputs):
        if "distribute" in self.constellation:
            position_dist = outputs.position.unsqueeze(-1)
            position_dict = self.vsa_converter.position_dictionary.reshape((self.vsa_converter.position_dictionary.shape[0], -1)).unsqueeze(0)
            position = t.sum(position_dist*position_dict, dim=1)
        else:
            position = None
        
        type_dist = outputs.type.unsqueeze(-1)
        type_dict = self.vsa_converter.type_dictionary.reshape((self.vsa_converter.type_dictionary.shape[0], -1)).unsqueeze(0)
        type = t.sum(type_dist*type_dict, dim=1)
        
        size_dist = outputs.size.unsqueeze(-1)
        size_dict = self.vsa_converter.size_dictionary.reshape((self.vsa_converter.size_dictionary.shape[0], -1)).unsqueeze(0)
        size = t.sum(size_dist*size_dict, dim=1)
        
        color_dist = outputs.color.unsqueeze(-1)
        color_dict = self.vsa_converter.color_dictionary.reshape((self.vsa_converter.color_dictionary.shape[0], -1)).unsqueeze(0)
        color = t.sum(color_dist*color_dict, dim=1)
        
        return Scene(position, None, type, size, color)
    
    def compute_context_superposition(self, x):
        if self.context_superposition:
            x_with_superposition = binding_circular(x.reshape((-1, self.k, self.d//self.k)), 
                                    self.context_keys.repeat(x.shape[0], 1, 1))
            x_with_superposition = x_with_superposition.reshape((x.shape[0], x.shape[1], -1))
            x_with_superposition = x_with_superposition.sum(dim=1)/x.shape[1]
        else:
            x_with_superposition = x
        return x_with_superposition

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
        
        if "distribute" in self.constellation:
            if self.context_superposition:
                pos_num = self.position_number_superposition(scene.position[:, :8], scene.number[:, :8])
                pos_num_output = self.pos_num_model(self.compute_context_superposition(pos_num))
            else:
                pos_num_output = self.pos_num_model(t.cat((self.compute_context_superposition(scene.position[:, :8]), 
                                                        self.compute_context_superposition(scene.number[:, :8])), dim=-1))
        else:
            pos_num_output = None
        type_output = self.type_model(self.compute_context_superposition(scene.type[:, :8]))
        size_output = self.size_model(self.compute_context_superposition(scene.size[:, :8]))
        color_output = self.color_model(self.compute_context_superposition(scene.color[:, :8]))
        outputs = Scene(pos_num_output, None, type_output, size_output, color_output)
        if self.vsa_selection:
            outputs = self.compute_vsa_outputs(outputs)
            scene = scene_vsa
        else:
            scene = scene_prob
        
        if "distribute" in self.constellation:
            candidates = Scene(scene.position[:, 8:], scene.number[:, 8:], scene.type[:, 8:], scene.size[:, 8:], scene.color[:, 8:])
        else:
            candidates = Scene(None, None, scene.type[:, 8:], scene.size[:, 8:], scene.color[:, 8:])
            
        return outputs, candidates