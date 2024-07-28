# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 11:35:32 2023

@author: MuhammedFurkanDasdel
"""

import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig

class ViTClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ViTClassifier, self).__init__()
        
        # Load the pre-trained ViT model
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        
        # Get the hidden size from the config (usually 768 for base models)
        hidden_size = self.vit.config.hidden_size
        
        # Define a classifier (fully connected layer) to output the desired number of classes
        self.classifier = nn.Linear(hidden_size, num_classes)
        
    def forward(self, images):
        # Pass images through the ViT model
        outputs = self.vit(torch.squeeze(images,1))
        
        # Get the [CLS] embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        # Pass the [CLS] embedding through the classifier
        logits = self.classifier(cls_embedding)
        
        return logits

model2 = ViTClassifier(21)