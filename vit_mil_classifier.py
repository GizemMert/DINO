# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 16:17:18 2023

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

        self.L = embedding_dim  # Setting L to the embedding dimension.
        self.D = 128  # hidden layer size for attention network

        self.class_count = class_count
        self.multicolumn = multicolumn
        self.device = device
        self.max_num_embeddings = max_num_embeddings
        self.embedding_dim = embedding_dim
        
        self.vit = ViTClassifier(num_classes=21)
        if self.model_path is not None:
            pretrained = torch.load(self.model_path)
            self.vit.load_state_dict(pretrained)
        self.vit.classifier = nn.Identity()
        
        # Freeze all layers
        for param in self.vit.vit.parameters():
            param.requires_grad = False
        '''
        # Unfreeze the last two transformer layers
        for param in self.vit.vit.encoder.layer[-2:].parameters():
            param.requires_grad = True
        '''
        self.attention_multi_column = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.class_count),
        )
        # classifier (multi attention approach)
        self.classifier_multi_column = nn.ModuleList()
        for a in range(self.class_count):
            self.classifier_multi_column.append(nn.Sequential(
                nn.Linear(self.L, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            ))


    def forward(self, images, return_attention = False):
        '''Forward pass of bag x through network. 
        x should have shape [batch_size, sequence_len, embedding_dim] e.g., [32, 500, 768]
        '''
        
        features = []
        for img_arr in images:
            img_arr = img_arr.to(self.device)
            out = self.vit(img_arr)
            features.append(out)
        x = torch.stack(features)
        x = x.squeeze(0)

        # Pass through the transformer
        if(self.multicolumn):
            prediction = []
            bag_feature_stack = []
            attention_stack = []
            # calculate attention
            att_raw = self.attention_multi_column(x)
            #print(att_raw.shape)
            att_raw = torch.transpose(att_raw, 1, 0)

            # for every possible class, repeat
            for a in range(self.class_count):
                # softmax + Matrix multiplication
                att_softmax = F.softmax(att_raw[a, ...][None, ...], dim=1)
                bag_features = torch.mm(att_softmax, x)
                bag_feature_stack.append(bag_features)
                # final classification with one output value (value indicating
                # this specific class to be predicted)
                pred = self.classifier_multi_column[a](bag_features)
                prediction.append(pred)

            prediction = torch.stack(prediction).view(1, self.class_count)
            bag_feature_stack = torch.stack(bag_feature_stack).squeeze()
            # final softmax to obtain probabilities over all classes
            # prediction = F.softmax(prediction, dim=1)
            
            if return_attention == True:
                return prediction, att_raw, F.softmax(
                att_raw, dim=1), bag_feature_stack
            
            return prediction

'''   
model = ViTMiL(7, 1, 'cpu')
images = torch.randn(1,400,3,224,224)
output = model(images)
'''
