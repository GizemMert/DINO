import os
import re
import torch
import torch.nn.functional as F
import sys
import torch.nn as nn
sys.path.append(os.path.abspath("/home/aih/gizem.mert/Dino/DINO"))
from transformer import Transformer
from classifier_dropout import ViTMiL
# from dataset_mixed import *  # dataset
# from model import *  # actual MIL model
from sklearn import metrics as metrics
import csv
import shutil
import pandas as pd
import numpy as np
import pickle
from transformers import ViTFeatureExtractor
from PIL import Image
import types


feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

def preprocess_image(image):
    """
    Preprocess the image using ViTFeatureExtractor, ensuring it is resized to 224x224.
    """
    # The feature extractor will handle resizing and normalization
    processed_image = feature_extractor(images=image, return_tensors="pt")
    return processed_image['pixel_values'].squeeze(0)  # Remove batch dimension


def load_images_from_txt(txt_file_path):
    """
    Load and preprocess images listed in a text file (for artificial patients).
    """
    images = []
    img_paths = []

    if os.path.exists(txt_file_path):
        with open(txt_file_path, 'r') as f:
            image_paths = f.read().splitlines()
            for img_path in image_paths:
                if os.path.exists(img_path):
                    img = Image.open(img_path).convert("RGB")
                    img_tensor = preprocess_image(img)  # Preprocess the image using ViTFeatureExtractor
                    images.append(img_tensor)
                    img_paths.append(img_path)
    return images, img_paths

label_to_diagnose_dict = {
    'Acute leukaemia': 0,
    'Lymphoma': 1,
    'MDS': 2,
    'MDS / MPN': 3,  # Assuming MDSMPN maps to this
    'MPN': 4,
    'No malignancy': 5,
    'Plasma cell neoplasm': 6
}

num_classes = 7
# Load class label information from CSV
label_to_diagnose_path = '/home/aih/gizem.mert/Dino/DINO/data_cross_val/label_to_diagnose.csv'
label_to_diagnose = pd.read_csv(label_to_diagnose_path)
label_to_diagnose_dict_2 = dict(zip(label_to_diagnose['label'], label_to_diagnose['diagnose']))
class_labels = label_to_diagnose['diagnose'].tolist()
n_classes = len(class_labels)

# seed = 42
experiment_source = 'experiment_3'
real_data_source = '/lustre/groups/labs/marr/qscd01/datasets/230824_MLL_BELUGA/RawImages'
SOURCE_FOLDER = f'/home/aih/gizem.mert/Dino/DINO/fold1/artificial_data/{experiment_source}/data'
TARGET_FOLDER = '/home/aih/gizem.mert/Dino/DINO/Results_fold16'
output_folder = f'/home/aih/gizem.mert/Dino/DINO/fold1/mixed_uncertain'

train_csv_path = '/home/aih/gizem.mert/Dino/DINO/data_cross_val/data_fold_1/train.csv'
train_patients_df = pd.read_csv(train_csv_path)
selected_patients = train_patients_df
patient_to_label = dict(zip(selected_patients['patient_files'], selected_patients['labels']))

def get_patient_name(path):
    return os.path.basename(path)

def get_class_name(patient_name):
    label = patient_to_label.get(patient_name)
    if label is not None:
        return label_to_diagnose_dict.get(label, "Unknown")
    return "Unknown"


def parse_patient_folder(folder_name):
    """
    Parse the folder name to get diagnosis and patient ID.
    """
    parts = folder_name[len('patient_'):].rsplit('_', 1)  # Remove the 'patient_' prefix
    diagnosis = parts[0]
    patient_id = parts[1]

    # Handle special case: 'MDSMPN' in folder name, but 'MDS / MPN' in diagnosis CSV
    if diagnosis == 'MDSMPN':
        diagnosis = 'MDS / MPN'

    return diagnosis, patient_id

def get_image_path_list(folder_patient_path): #For real patients
    """
    Given a folder path, return a list of all valid image paths.
    Assumes that image files have specific extensions (e.g., .jpg, .png, .tif).
    """
    valid_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')  # Add or modify based on your data
    image_paths = [
        os.path.join(folder_patient_path, f)
        for f in os.listdir(folder_patient_path)
        if f.endswith(valid_extensions)
    ]
    return image_paths


def load_image_paths(patient_folder):
    images_path_file = os.path.join(patient_folder, 'images.txt')
    with open(images_path_file, 'r') as file:
        image_paths = file.readlines()
    return [img_path.strip() for img_path in image_paths]





# Function to update misclassification count
def update_misclassification_count(probability_vector, one_hot_target, current_misclassification_count):
    one_hot_prediction = torch.zeros_like(probability_vector)
    one_hot_prediction[0, torch.argmax(probability_vector).item()] = 1
    if torch.argmax(one_hot_prediction).item() != torch.argmax(one_hot_target).item():
        current_misclassification_count += 1
    return current_misclassification_count

# Number of Monte Carlo samples
num_samples = 10
model_path = "/home/aih/gizem.mert/Dino/DINO/DinoBloom-B.pth"
# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ViTMiL(
    class_count=num_classes,
    multicolumn=1,
    device=device,
    model_path=model_path
)
# Load model
state_dict_path = os.path.join(TARGET_FOLDER, "state_dictmodel.pt")
pretrained_weights = torch.load(state_dict_path, map_location=device)


state_dict = {k.replace("module.", ""): v for k, v in pretrained_weights.items()}

# Load the state dict into the model
model.load_state_dict(state_dict, strict=False)

model = model.to(device)


model.train()  # Set the model to training mode to keep dropout active

all_uncertainties = {}
missclassification_counts = {}
max_uncertainties = {}
sum_uncertainties = {}

# No gradient calculation for uncertainty estimation, but model.train() keeps dropout active
with torch.no_grad():
    for folder_name in os.listdir(SOURCE_FOLDER):
        if folder_name.startswith('patient_'):
            print(f"Processing folder: {folder_name}")

            diagnosis, patient_id = parse_patient_folder(folder_name)
            patient_folder = os.path.join(SOURCE_FOLDER, folder_name)

            if diagnosis in label_to_diagnose_dict:
                label_index = label_to_diagnose_dict[diagnosis]
            else:
                print(f"Warning: Diagnosis '{diagnosis}' not found in label_to_diagnose_dict")
                continue

            lbl = np.zeros(num_classes)
            lbl[label_index] = 1

            # Load images and paths
            images, image_paths = load_images_from_txt(os.path.join(patient_folder, 'images.txt'))

            if len(images) == 0:
                print(f"No images found for patient {folder_name}")
                continue

            pred = []
            missclassification_count = 0
            # Perform Monte Carlo Dropout sampling
            for j in range(num_samples):
                bag = torch.stack(images).to(device)
                bag = torch.unsqueeze(bag, 0)

                # Forward pass and softmax
                prediction = model(bag)
                softmax_pred = torch.softmax(prediction, dim=1)
                pred.append(softmax_pred.cpu().numpy())

                missclassification_count = update_misclassification_count(
                    softmax_pred,
                    torch.tensor(lbl).to(device),
                    missclassification_count
                )

            if len(pred) == 0:
                print(f"No predictions collected for patient {folder_name}")
                continue

            # Calculate mean and uncertainty of predictions
            pred_tensor = torch.stack([torch.from_numpy(arr).to(device) for arr in pred])
            mean_prediction = pred_tensor.mean(dim=0)
            uncertainty = pred_tensor.std(dim=0)

            # Store max and sum uncertainties
            uncertainty_value_max = torch.max(uncertainty).item()
            uncertainty_value_sum = torch.sum(uncertainty).item()

            max_uncertainties[folder_name] = {
                'path': patient_folder,
                'data': uncertainty.cpu().numpy().squeeze(),
                'uncertainty': uncertainty_value_max
            }
            sum_uncertainties[folder_name] = {
                'path': patient_folder,
                'data': uncertainty.cpu().numpy().squeeze(),
                'uncertainty': uncertainty_value_sum
            }
            missclassification_counts[folder_name] = {
                'path': patient_folder,
                'uncertainty': missclassification_count / num_samples
            }
print("Total patients with recorded max uncertainties:", len(max_uncertainties))

def sort_and_print(uncertainties):
    sorted_uncertainties = dict(sorted(uncertainties.items(), key=lambda item: item[1]['uncertainty'], reverse=True))
    for p, data in sorted_uncertainties.items():
        print(f"Patient {p}: Uncertainty - {data['uncertainty']:.4}")

sort_and_print(max_uncertainties)

print(f"Total patients in max uncertainties: {len(max_uncertainties.keys())}")
print(f"Unique patients in max uncertainties: {len(set(max_uncertainties.keys()))}")

def select_paths(uncertainties, percentage):
    sorted_uncertainties = dict(sorted(uncertainties.items(), key=lambda item: item[1]['uncertainty'], reverse=True))
    num_paths = int(len(sorted_uncertainties) * (percentage / 100.0))
    selected_paths = {p: data['path'] for p, data in list(sorted_uncertainties.items())[:num_paths]}
    return selected_paths




def get_realpatients_filepaths_dictionary(data_directory, patient_to_label):

    paths = {}

    # Iterate over all patient folders in the raw images directory
    for folder_patient in os.listdir(data_directory):

        if folder_patient not in patient_to_label:
            continue

        # Get the full path to the patient folder
        folder_patient_path = os.path.join(data_directory, folder_patient)


        if os.path.isdir(folder_patient_path):
            # Gather all image files from the patient folder
            images = get_image_path_list(folder_patient_path)

            paths[folder_patient] = images

    return paths


import copy

def save_patient_filepaths(selected_paths, new_folder, paths_real_patients):
    """
    Save file paths for both uncertain patients and real patients.
    """
    print(f"Saving file paths for mixed patients in {new_folder}")
    os.makedirs(new_folder, exist_ok=True)

    # Combine uncertain and real patient paths
    paths_mixed_patients = copy.deepcopy(paths_real_patients)

    # Add uncertain patients to the paths
    for p, path in selected_paths.items():
        if p in paths_mixed_patients:
            paths_mixed_patients[p] += path
        else:
            paths_mixed_patients[p] = path

    # Remove duplicates
    for key in paths_mixed_patients.keys():
        len_before = len(paths_mixed_patients[key])
        paths_mixed_patients[key] = list(set(paths_mixed_patients[key]))  # Remove duplicate paths
        print(f"Removed {len_before - len(paths_mixed_patients[key])} duplicates for patient {key}")

    # Save the mixed paths to a pickle file
    with open(os.path.join(new_folder, 'file_paths.pkl'), 'wb') as f:
        pickle.dump(paths_mixed_patients, f)

    print(f"File paths saved to {new_folder}/file_paths.pkl")


def update_train_files_with_artificial(new_folder, selected_paths, train_csv_path, label_to_diagnose_dict):


    train_files = pd.read_csv(train_csv_path)

    # Create a new DataFrame for artificial patients
    artificial_patients = pd.DataFrame(columns=['patient_files', 'labels'])

    # Iterate over uncertain patients and get their labels from the folder name
    for p in selected_paths.keys():
        diagnosis, patient_id = parse_patient_folder(p)  # Assuming parse_patient_folder extracts diagnosis and patient ID

        # Get label index from diagnosis
        label = label_to_diagnose_dict.get(diagnosis)

        if label is not None:
            # Add the artificial patient to the DataFrame
            artificial_patients = artificial_patients.append({'patient_files': p, 'labels': label}, ignore_index=True)

    # Combine real and artificial patients into a new DataFrame
    mixed_train_files = pd.concat([train_files, artificial_patients], ignore_index=True)
    mixed_train_files = pd.concat([train_files, artificial_patients], ignore_index=True)

    # Save the new train.csv as mixed_train.csv
    mixed_train_csv_path = os.path.join(new_folder, "mixed_train.csv")
    mixed_train_files.to_csv(mixed_train_csv_path, index=False)
    print(f"Mixed train.csv saved to {mixed_train_csv_path}")



# Iterate over different percentages from 10 to 50 and save uncertain patients
paths_real_patients = get_realpatients_filepaths_dictionary(real_data_source, patient_to_label)

for percentage in [10, 20, 30, 50]:
    new_folder_max = os.path.join(output_folder, f'max_{percentage}_percent')

    # Select the paths for uncertain patients based on the percentage
    selected_max_paths = select_paths(max_uncertainties, percentage)

    # Save the mixed file paths to pickle
    save_patient_filepaths(selected_max_paths, new_folder_max, paths_real_patients)

    # Update the train.csv with the artificial patients and save
    update_train_files_with_artificial(new_folder_max, selected_max_paths, train_csv_path, label_to_diagnose_dict)