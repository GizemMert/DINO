# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 12:18:09 2023

@author: MuhammedFurkanDasdel
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ViTClassifier import ViTClassifier
from transformer import Transformer

class ViTMiL(nn.Module):
    def __init__(self, class_count, multicolumn, device, max_num_embeddings = 500, model_path = None, embedding_dim=768):
        super(ViTMiL, self).__init__()
        
        self.model_path = model_path

        #self.L = embedding_dim  # Setting L to the embedding dimension.
        #self.D = 128  # hidden layer size for attention network

        self.class_count = class_count
        self.multicolumn = multicolumn
        self.device = device
        #self.max_num_embeddings = max_num_embeddings
        self.embedding_dim = embedding_dim

        modelname = 'dinov2_vitb14'

        self.vit = get_dino_finetuned_downloaded(self.model_path,modelname)
        
        # Freeze all layers
        for param in self.vit.parameters():
            param.requires_grad = False

        mlp_dim = 512
        self.transformer =Transformer(
            num_classes=64,
            input_dim=self.embedding_dim,
            dim=512,
            depth=2,
            heads=8,
            mlp_dim=mlp_dim,
            pool='cls',
            dim_head=64,
            dropout=0.1,
            emb_dropout=0.1)

            
        self.mlp = nn.Sequential(nn.LayerNorm(mlp_dim),
                                    nn.Linear(mlp_dim, 64),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.2),
                                    nn.Linear(64, self.class_count))


    def forward(self, images, return_latent = False):
        '''Forward pass of bag x through network. 
        x should have shape [batch_size, sequence_len, embedding_dim] e.g., [32, 500, 768]
        '''
        
        features = []
        for img_arr in images:
            img_arr = img_arr.to(self.device)
            out = self.vit(img_arr)
            features.append(out)
        x = torch.stack(features)

        # Pass through the transformer
        latent = self.transformer(x)
        logits = self.mlp(latent)
        
        if return_latent == True:
            return latent, logits
            
        return logits




def get_dino_finetuned_downloaded(model_path, modelname):
    model = torch.hub.load("facebookresearch/dinov2", modelname)
    # load finetuned weights

    # pos_embed has wrong shape
    if model_path is not None:
        pretrained = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False)
        # make correct state dict for loading
        new_state_dict = {}
        for key, value in pretrained["teacher"].items():
            if "dino_head" in key or "ibot_head" in key:
                pass
            else:
                new_key = key.replace("backbone.", "")
                new_state_dict[new_key] = value
        # change shape of pos_embed
        input_dims = {
            "dinov2_vits14": 384,
            "dinov2_vitb14": 768,
            "dinov2_vitl14": 1024,
            "dinov2_vitg14": 1536,
        }
        pos_embed = nn.Parameter(torch.zeros(1, 257, input_dims[modelname]))
        model.pos_embed = pos_embed
        # load state dict
        model.load_state_dict(new_state_dict, strict=True)
    return model
'''       
model = ViTMiL(7, 1, 'cuda:0').to('cuda:0')
images = torch.randn(1,400,3,224,224).to('cuda:0')
output = model(images)
'''
