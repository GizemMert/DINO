import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import ViTImageProcessor, ViTFeatureExtractor
from PIL import Image

import sys
import os
import pandas as pd


class MllDataset(Dataset):
    '''MLL mil dataset class. Can be used by pytorch DataLoader '''

    def __init__(self, path_to_data, data_files, class_count):
        '''
        Args:
            path_to_data (str): raw images path
            data_files (dataframe): name of patients in csv file, has two columns: patient_files and labels
            class_count (int): number of classes
        '''

        # Dictionary to store loaded features
        self.features_loaded = {}
        self.path_to_data = path_to_data
        self.data_files = data_files
        self.class_count = class_count
        self.process = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

    def __len__(self):
        '''returns amount of images contained in object'''
        return len(self.data_files)
    
    def preprocess(self, image):
        return self.process(image, return_tensors="pt")

    def __getitem__(self, idx):
        '''returns specific item from this dataset'''
        
        label = ''

        # get the file path
        file_name = self.data_files.patient_files[idx]
        
        
        file_path = os.path.join(self.path_to_data, file_name)
        
        # check if data has been loaded before
        if file_name in self.features_loaded:
            data = self.features_loaded[file_name]
            
            
        else:
            # load data
            images = []
            img_paths = []
            for image in os.listdir(file_path):
                if image.endswith(".TIF"):
                    #print(os.path.join(file_path, image))
                    img_path = os.path.join(file_path, image)
                    img = Image.open(img_path)
                    img_tensor = self.preprocess(img)
                    img_tensor = img_tensor['pixel_values']
                    images.append(img_tensor)
                    img_paths.append(img_path)  
            
            #print(images.shape)
            if len(images) == 0:
                data = torch.zeros(500,3,224,224)
                print(file_path)
                
            else:
                data = torch.stack(images).squeeze()
            
            if len(data.shape) == 3:
                data = data.unsqueeze(0)


            #print(data.shape)
        
                
        # extract the id from the filename
        #data_id = file_name.replace("_features","")
        
        # get the label
        label = self.data_files.labels[idx]
        # convert the label to one-hot
        label_onehot = np.zeros(self.class_count, dtype=np.float32)
        label_onehot[label] = 1
        label_onehot = torch.from_numpy(label_onehot)
        


        return data, label_onehot, img_paths


