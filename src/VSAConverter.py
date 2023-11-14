#*----------------------------------------------------------------------------*
#* Copyright (C) 2023 IBM Inc. All rights reserved                            *
#* SPDX-License-Identifier: GPL-3.0-only                                      *
#*----------------------------------------------------------------------------*

from collections import namedtuple
from nvsa.reasoning.vsa_block_utils import pmf2vec, binding_circular, block_discrete_codebook
from const import DIM_POSITION_2x2, DIM_POSITION_3x3, DIM_NUMBER_2x2, DIM_NUMBER_3x3, DIM_TYPE, DIM_SIZE, DIM_COLOR
import torch.nn as nn
from nvsa.reasoning.vsa_block_utils import block_discrete_codebook, block_continuous_codebook

Scene = namedtuple("Scene", ["position", "number", "type", "size", "color"])

def generate_nvsa_codebooks(args, rng): 
    '''
    Generate the codebooks for NVSA frontend and backend. 
    The codebook can also be loaded if it is stored under args.resume/
    '''
    print("Generate new NVSA codebooks")
    backend_cb_cont, _ = block_continuous_codebook(device=args.device,scene_dim=511,d=args.nvsa_backend_d,k=args.nvsa_backend_k, rng=rng, fully_orthogonal=False)  
    backend_cb_discrete, _ = block_discrete_codebook(device=args.device, d=args.nvsa_backend_d,k=args.nvsa_backend_k, rng=rng)  
    return backend_cb_cont, backend_cb_discrete

class VSAConverter(nn.Module):

    def __init__(self, device, constellation, dictionary, dictionary_type="Discrete", context_dim=8, attributes_superposition=False):
        super(VSAConverter, self).__init__()
        self.device = device
        self.constellation = constellation
        self.d = dictionary.shape[1]*dictionary.shape[2]
        self.k = dictionary.shape[1]
        self.dictionary = dictionary
        self.dictionary_type = dictionary_type
        self.compute_attribute_dicts()
        self.context_dim = context_dim
        self.attributes_superposition = attributes_superposition
        if self.attributes_superposition:
            attribute_keys, _ = block_discrete_codebook(device=device, d=self.d, k=self.k, scene_dim=5)
            self.attribute_keys = nn.Parameter(attribute_keys)

    def compute_attribute_dicts(self):
        if "distribute" in self.constellation:
            if "four" in self.constellation:
                DIM_POSITION = DIM_POSITION_2x2
                DIM_NUMBER = DIM_NUMBER_2x2
            else:
                DIM_POSITION = DIM_POSITION_3x3
                DIM_NUMBER = DIM_NUMBER_3x3
            self.position_dictionary = self.dictionary[:DIM_POSITION]
            self.number_dictionary = self.dictionary[:DIM_NUMBER]
        if self.type == "Discrete":
            self.type_dictionary = self.dictionary[:DIM_TYPE + 1]
            self.size_dictionary = self.dictionary[:DIM_SIZE + 1]
        else:
            self.type_dictionary = self.dictionary[1:DIM_TYPE + 2]
            self.size_dictionary = self.dictionary[1:DIM_SIZE + 2]
        self.color_dictionary = self.dictionary[:DIM_COLOR + 1]

    def compute_values(self, scene_prob):
        if "distribute" in self.constellation:
            position = pmf2vec(self.position_dictionary, scene_prob.position_prob)
            number = pmf2vec(self.number_dictionary, scene_prob.number_prob)
        else:
            position = None
            number = None
        type = pmf2vec(self.type_dictionary, scene_prob.type_prob)
        size = pmf2vec(self.size_dictionary, scene_prob.size_prob)
        color = pmf2vec(self.color_dictionary, scene_prob.color_prob)
        return Scene(position, number, type, size, color)

    def compute_attributes_superposition(self, scene):
        if "distribute" in self.constellation:
            position = binding_circular(scene.position.reshape((-1, self.k, self.d//self.k)), 
                                                 self.attribute_keys[0].unsqueeze(0).expand(scene.position.shape[0]*scene.position.shape[1], -1, -1))
            number = binding_circular(scene.number.reshape((-1, self.k, self.d//self.k)), 
                                                self.attribute_keys[1].unsqueeze(0).expand(scene.number.shape[0]*scene.number.shape[1], -1, -1))
            position = position.reshape((-1, scene.position.shape[1], self.k, self.d//self.k))
            number = number.reshape((-1, scene.number.shape[1], self.k, self.d//self.k))
        else:
            position = None
            number = None
        type = binding_circular(scene.type.reshape((-1, self.k, self.d//self.k)), 
                                self.attribute_keys[2].unsqueeze(0).expand(scene.type.shape[0]*scene.type.shape[1], -1, -1))
        size = binding_circular(scene.size.reshape((-1, self.k, self.d//self.k)), 
                                self.attribute_keys[3].unsqueeze(0).expand(scene.size.shape[0]*scene.size.shape[1], -1, -1))
        color = binding_circular(scene.color.reshape((-1, self.k, self.d//self.k)), 
                                self.attribute_keys[4].unsqueeze(0).expand(scene.color.shape[0]*scene.color.shape[1], -1, -1))
        type = type.reshape((-1, scene.type.shape[1], self.k, self.d//self.k))
        size = size.reshape((-1, scene.size.shape[1], self.k, self.d//self.k))
        color = color.reshape((-1, scene.color.shape[1], self.k, self.d//self.k))

        if "distribute" in self.constellation:
            scene = (scene.position + scene.number + scene.type +scene.size + scene.color)/5
        else:
            scene = (scene.type + scene.size + scene.color)/3

        return scene
    
    def compute_context_superposition(self, scene):
        if "distribute" in self.constellation:
            position = binding_circular(scene.position.reshape((-1, self.k, self.d//self.k)), 
                                        self.context_keys.expand(scene.position.shape[0]*scene.position.shape[1], -1, -1))
            number = binding_circular(scene.number.reshape((-1, self.k, self.d//self.k)), 
                                        self.context_keys.expand(scene.number.shape[0]*scene.number.shape[1], -1, -1))
            position = position.reshape((-1, scene.position.shape[1], self.k, self.d//self.k))
            number = number.reshape((-1, scene.number.shape[1], self.k, self.d//self.k))
        else:
            position = None
            number = None
        type = binding_circular(scene.type.reshape((-1, self.k, self.d//self.k)), 
                                self.context_keys.expand(scene.type.shape[0]*scene.type.shape[1], -1, -1))
        size = binding_circular(scene.size.reshape((-1, self.k, self.d//self.k)), 
                                self.context_keys.expand(scene.size.shape[0]*scene.size.shape[1], -1, -1))
        color = binding_circular(scene.color.reshape((-1, self.k, self.d//self.k)), 
                                self.context_keys.expand(scene.color.shape[0]*scene.color.shape[1], -1, -1))
        type = type.reshape((-1, scene.type.shape[1], self.k, self.d//self.k))
        size = size.reshape((-1, scene.size.shape[1], self.k, self.d//self.k))
        color = color.reshape((-1, scene.color.shape[1], self.k, self.d//self.k))

        if "distribute" in self.constellation:
            position = position.sum(1)/self.context_dim
            number =  number.sum(1)/self.context_dim
        type = type.sum(1)/self.context_dim
        size =  size.sum(1)/self.context_dim
        color = color.sum(1)/self.context_dim
        
        return scene

    def forward(self, scene_prob):
        scene = self.compute_values(scene_prob)
        if self.attributes_superposition:
            scene = self.compute_attributes_superposition(scene)
        return scene


