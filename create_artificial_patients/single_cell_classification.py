import torch
import os
from PIL import Image
# import label_converter  # make sure the label_converter.py is in the folder with this script
import numpy as np
from torchvision import transforms
import torch.nn as nn

# Define paths
PATH_TO_IMAGES = '/lustre/groups/labs/marr/qscd01/datasets/230824_MLL_BELUGA/RawImages'
PATH_TO_MODEL = os.path.join(os.getcwd(), "/home/aih/gizem.mert/SCEMILA_5K/SCEMILA_Patient_Generation-5Fold_CV/Single_Cell_Classifier/class_conversion-csv/model.pt")

# Load model and print architecture
model = torch.load(PATH_TO_MODEL, map_location=torch.device('cpu'))


def create_dataset(root_dirs):
    # Create dataset
    data = []

    for sgl_dir in root_dirs:
        for file_sgl in os.listdir(sgl_dir):
            # Check for .RGB.TIF files (case insensitive)
            if not file_sgl.lower().endswith('.rgb.tif'):
                continue
            data.append(os.path.join(sgl_dir, file_sgl))

    # Convert the list to a NumPy array
    data = np.array(data)

    # Extract numerical part from filenames for sorting
    # Assuming filenames are like "Gal-000001.RGB.TIF"
    numeric_part = np.array([int(name.split('-')[1].split('.')[0]) for name in data])

    # Get the indices that would sort the numeric part
    sorted_indices = np.argsort(numeric_part)

    # Use the sorted indices to rearrange the file names array
    sorted_images = data[sorted_indices]

    return sorted_images


def get_image(idx, data):
    '''returns specific item from this dataset'''
    # Load image, remove alpha channel, transform
    image = Image.open(data[idx])
    image_arr = np.asarray(image)[:, :, :3]
    image = Image.fromarray(image_arr)
    return torch.tensor(image_arr)


def save_single_cell_probabilities(data, folder_patient):
    array_list = []
    for idx in range(len(data)):
        input = get_image(idx, data)
        input = input.permute(2, 0, 1).unsqueeze(0)

        # Convert input to float
        input = input.float()
        input = input / 255.

        # Normalize the input
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        input = normalize(input)

        model.eval()
        pred = model(input)
        softmax = nn.Softmax(dim=1)
        pred_probability = softmax(pred)

        # Save probabilities in a file
        pred_vect = pred_probability.detach().numpy().flatten()
        array_list.append([pred_vect])

    # Concatenate all features for one artificial patient
    single_cell_probs = np.concatenate(array_list, axis=0)
    output_npy_file = folder_patient + '/single_cell_probabilities.npy'
    # Save the array to the .npy file
    np.save(output_npy_file, single_cell_probs)

patient_folders = [os.path.join(PATH_TO_IMAGES, patient_folder) for patient_folder in os.listdir(PATH_TO_IMAGES)]
# Save class probabilities for each patient
print("Starting processing of image folders...")
# Process each patient folder
for folder_patient in patient_folders:
    print(f"Processing patient folder: {folder_patient}")

    # Create dataset from current patient folder
    data = create_dataset([folder_patient])

    # Check if any .tif files were found
    if len(data) == 0:
        print("Skipping patient folder without .RGB.TIF files:", folder_patient)
        continue

    print(f"Found {len(data)} .RGB.TIF files in patient folder: {folder_patient}")
    # Call your function to process and save data
    save_single_cell_probabilities(data, folder_patient)
    print(f"Finished processing patient folder: {folder_patient}")