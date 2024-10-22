import os
import re
import torch
import torch.nn.functional as F
import sys
sys.path.append(os.path.abspath("/home/aih/gizem.mert/Dino/DINO"))
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
label_to_diagnose_dict = dict(zip(label_to_diagnose['label'], label_to_diagnose['diagnose']))
class_labels = label_to_diagnose['diagnose'].tolist()
n_classes = len(class_labels)

# seed = 42
experiment_source = 'experiment_3'
real_data_source = '/lustre/groups/labs/marr/qscd01/datasets/230824_MLL_BELUGA/RawImages'
SOURCE_FOLDER = f'/home/aih/gizem.mert/Dino/DINO/fold4/artificial_data/'+experiment_source
TARGET_FOLDER = '/home/aih/gizem.mert/Dino/DINO/Results_fold46'
output_folder = f'/home/aih/gizem.mert/Dino/DINO/fold4/mixed_uncertain'

train_csv_path = '/home/aih/gizem.mert/Dino/DINO/data_cross_val/data_fold_4/train.csv'
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



# Now `patients` will have all the patient keys (e.g., "patient_MDS_1") with the associated folder paths and diagnoses.
# print(f"Total patients loaded: {len(patients)}")

# Function to update misclassification count
def update_misclassification_count(probability_vector, one_hot_target, current_misclassification_count):
    one_hot_prediction = torch.zeros_like(probability_vector)
    one_hot_prediction[0, torch.argmax(probability_vector).item()] = 1
    if torch.argmax(one_hot_prediction).item() != torch.argmax(one_hot_target).item():
        current_misclassification_count += 1
    return current_misclassification_count

# Number of Monte Carlo samples
num_samples = 10

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load model
model = torch.load(os.path.join(TARGET_FOLDER, "model.pt"), map_location="cpu")
model = model.to(device)
model.train()

# Initialize arrays to store uncertainties
all_uncertainties = {}
missclassification_counts = {}
max_uncertainties = {}
sum_uncertainties = {}

# Iterate over all patient folders in the source folder
for folder_name in os.listdir(SOURCE_FOLDER):
    if folder_name.startswith('patient_'):
        # Parse folder name to get diagnosis and patient ID
        diagnosis, patient_id = parse_patient_folder(folder_name)
        patient_folder = os.path.join(SOURCE_FOLDER, folder_name)

        # Get the label from the CSV using the diagnosis
        if diagnosis in label_to_diagnose_dict:
            label_index = label_to_diagnose_dict[diagnosis]
        else:
            print(f"Warning: Diagnosis '{diagnosis}' not found in label_to_diagnose_dict")
            continue

        # Create label as a one-hot vector
        lbl = np.zeros(num_classes)
        lbl[label_index] = 1

        # Load and preprocess images once (this handles preprocessing)
        images, image_paths = load_images_from_txt(os.path.join(patient_folder, 'images.txt'))

        pred = []
        missclassification_count = 0

        # Perform Monte Carlo Dropout
        with torch.no_grad():
            for j in range(num_samples):
                # Use preprocessed images (no need to preprocess again)
                bag = torch.stack(images).to(device)
                bag = torch.unsqueeze(bag, 0)  # Add batch dimension

                # Forward pass with model
                prediction = model(bag)
                pred.append(torch.softmax(prediction, dim=1).cpu().detach().numpy())

                # Update misclassification count
                missclassification_count = update_misclassification_count(
                    torch.softmax(prediction, dim=1),
                    torch.tensor(lbl),
                    missclassification_count
                )

        # Convert predictions to tensor and compute mean and uncertainty
        pred_tensor = torch.stack([torch.from_numpy(arr) for arr in pred])
        mean_prediction = pred_tensor.mean(dim=0)
        uncertainty = pred_tensor.std(dim=0)

        uncertainty_value_max = torch.max(uncertainty).item()
        uncertainty_value_sum = torch.sum(uncertainty).item()

        # Store uncertainty information
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

def sort_and_print(uncertainties):
    sorted_uncertainties = dict(sorted(uncertainties.items(), key=lambda item: item[1]['uncertainty'], reverse=True))
    for p, data in sorted_uncertainties.items():
        print(f"Patient {p}: Uncertainty - {data['uncertainty']:.4}")

sort_and_print(max_uncertainties)

print(len(max_uncertainties.keys()))
print(len(set(max_uncertainties.keys())))

def select_paths(uncertainties, percentage):
    sorted_uncertainties = dict(sorted(uncertainties.items(), key=lambda item: item[1]['uncertainty'], reverse=True))
    num_paths = int(len(sorted_uncertainties) * (percentage / 100.0))
    selected_paths = {p: data['path'] for p, data in list(sorted_uncertainties.items())[:num_paths]}
    return selected_paths




def get_realpatients_filepaths_dictionary(data_directory, patient_to_label):

    paths = {}

    # Iterate over all patient folders in the raw images directory
    for folder_patient in os.listdir(data_directory):
        # Check if the folder is in the patient_to_label mapping (i.e., selected in train.csv)
        if folder_patient not in patient_to_label:
            continue  # Skip if this patient is not in the selected patients

        # Get the full path to the patient folder
        folder_patient_path = os.path.join(data_directory, folder_patient)

        # Ensure it's a directory
        if os.path.isdir(folder_patient_path):
            # Gather all image files from the patient folder
            images = get_image_path_list(folder_patient_path)  # Assume this gets all image files in the folder

            # Store the image paths under the patient folder name
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
            paths_mixed_patients[p] += path  # Append uncertain paths if patient exists
        else:
            paths_mixed_patients[p] = path  # Create new entry for uncertain patient

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
    """
    Update the training files (train.csv) with both real and artificial patient IDs and labels, and save it as mixed_train.csv.
    - `new_folder`: The folder where the new mixed_train.csv will be saved.
    - `selected_paths`: Dictionary of selected uncertain patient paths.
    - `train_csv_path`: Path to the original train.csv for real patients.
    - `label_to_diagnose_dict`: Dictionary mapping diagnoses to labels.
    """
    # Load the real patient train.csv
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