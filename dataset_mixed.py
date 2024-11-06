import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import ViTImageProcessor, ViTFeatureExtractor
from PIL import Image

import sys
import os
import pandas as pd


class MllDataset_train(Dataset):
    '''MLL dataset class for mixed real and artificial data. Can be used by PyTorch DataLoader'''

    def __init__(self, mixed_data_filepaths, data_files, class_count, data_path, artificial_data_path):
        '''
        Args:
            mixed_data_filepaths (dict): dictionary containing file paths (from file_paths.pkl).
            data_files (DataFrame): dataframe with patient files and labels (from mixed_train.csv).
            class_count (int): number of classes.
            data_path (str): raw images path for real patients.
            artificial_data_path (str): path for artificial patient folders.
        '''
        # Dictionary to store loaded features
        self.features_loaded = {}
        self.mixed_data_filepaths = mixed_data_filepaths
        self.data_files = data_files
        self.class_count = class_count
        self.data_path = data_path
        self.artificial_data_path = artificial_data_path
        self.process = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

    def __len__(self):
        '''Returns the amount of images contained in the dataset'''
        return len(self.data_files)

    def preprocess(self, image):
        '''Preprocess the image for ViTFeatureExtractor'''
        return self.process(image, return_tensors="pt")

    def load_images_from_folder(self, folder_path):
        '''Load images from a folder (real patient)'''
        images = []
        img_paths = []
        for image in os.listdir(folder_path):
            if image.endswith(".TIF"):
                img_path = os.path.join(folder_path, image)
                img = Image.open(img_path).convert("RGB")
                img_tensor = self.preprocess(img)
                img_tensor = img_tensor['pixel_values']
                images.append(img_tensor)
                img_paths.append(img_path)
        return images, img_paths

    def load_images_from_txt(self, txt_file_path):
        '''Load images from a text file (artificial patient)'''
        images = []
        img_paths = []
        if os.path.exists(txt_file_path):
            with open(txt_file_path, 'r') as f:
                image_paths = f.read().splitlines()
                for img_path in image_paths:
                    if os.path.exists(img_path):
                        img = Image.open(img_path).convert("RGB")
                        img_tensor = self.preprocess(img)
                        img_tensor = img_tensor['pixel_values']
                        images.append(img_tensor)
                        img_paths.append(img_path)
        return images, img_paths

    def __getitem__(self, idx):
        '''Returns a specific item from the dataset'''

        label = ''

        # Get the patient file and label from data_files (from mixed_train.csv)
        patient_id = self.data_files.patient_files[idx]
        label = self.data_files.labels[idx]

        # Get the folder path from mixed_data_filepaths
        folder_path = self.mixed_data_filepaths[patient_id]

        # Check if the folder path corresponds to a real or artificial patient
        if os.path.exists(os.path.join(self.artificial_data_path, folder_path)):
            # It's an artificial patient; load image paths from images.txt
            images, img_paths = self.load_images_from_txt(
                os.path.join(self.artificial_data_path, folder_path, 'images.txt'))
        elif os.path.exists(os.path.join(self.data_path, folder_path)):
            # It's a real patient; load images directly from the folder
            images, img_paths = self.load_images_from_folder(os.path.join(self.data_path, folder_path))
        else:
            raise ValueError(f"Path not found for patient {patient_id}")

        # If no images were loaded, create a default zero tensor
        if len(images) == 0:
            data = torch.zeros(500, 3, 224, 224)  # Adjust the dimensions based on your model's requirements
            print(f"No images found for patient {patient_id}, returning zero tensor.")
        else:
            # Stack all image tensors for this patient
            data = torch.stack(images).squeeze()

        # Ensure the data has the correct shape
        if len(data.shape) == 3:
            data = data.unsqueeze(0)

        # Convert label to one-hot
        label_onehot = np.zeros(self.class_count, dtype=np.float32)
        label_onehot[label] = 1
        label_onehot = torch.from_numpy(label_onehot)

        # Return the data (images), one-hot label, and image paths
        return data, label_onehot, img_paths


