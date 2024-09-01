import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D
import pandas as pd
import re
import seaborn as sns


def get_counts_vector(labels_vector):
    unique_labels, label_counts = np.unique(labels_vector, return_counts=True)
    counts_vector = np.zeros(21, dtype=int)
    counts_vector[unique_labels] = label_counts
    return counts_vector, unique_labels

sc_class_labels= ['eosinophil granulocyte', 'reactive lymphocyte',
       'neutrophil granulocyte (segmented)', 'typical lymphocyte',
       'other', 'neutrophil granulocyte (band)', 'monocyte',
       'large granulated lymphocyte', 'atypical promyelocyte',
       'basophil granulocyte', 'smudge cell', 'neoplastic lymphocyte',
       'promyelocyte', 'myelocyte', 'myeloblast', 'metamyelocyte',
       'normo', 'plasma cell', 'hair cell', 'bilobed M3v',
       'mononucleosis']

aml_class_labels = ["Acute leukaemia","Lymphoma","MDS","MDS / MPN","MPN", "No malignancy", "Plasma cell neoplas"]
# Path to the folder containing your files

# Paths to the folders and files
data_path = '/lustre/groups/labs/marr/qscd01/datasets/230824_MLL_BELUGA/RawImages'
result_path = '/home/aih/gizem.mert/Dino/DINO/fold2/train'
train_csv_path = '/home/aih/gizem.mert/Dino/DINO/data_cross_val/data_fold_2/train.csv'
val_csv_path = '/home/aih/gizem.mert/Dino/DINO/data_cross_val/data_fold_2/val.csv'
label_to_diagnose_path = '/home/aih/gizem.mert/Dino/DINO/data_cross_val/label_to_diagnose.csv'

# Read the patient IDs from the CSV files
train_patients_df = pd.read_csv(train_csv_path)
val_patients_df = pd.read_csv(val_csv_path)
selected_patients = pd.concat([train_patients_df, val_patients_df], ignore_index=True)

# Read the label to diagnose mapping
label_to_diagnose = pd.read_csv(label_to_diagnose_path)
# Ensure labels are integers for consistency
label_to_diagnose['label'] = label_to_diagnose['label'].astype(int)
label_to_diagnose_dict = dict(zip(label_to_diagnose['label'], label_to_diagnose['diagnose']))

# Convert the patient files and labels to a dictionary for lookup
patient_to_label = dict(zip(selected_patients['patient_files'], selected_patients['labels']))


def get_patient_name(path):
    return os.path.basename(path)
"""
def get_class_name(path):
    return re.search(r"/data/(\w+)", path).group(1)

def get_image_number(path):
    return re.search(r"image_(\d).tif", path).group(1)
"""


def get_class_name(patient_name):
    label = patient_to_label.get(patient_name)
    print(f"Patient: {patient_name}, Label from CSV: {label}")

    if label is not None:
        label = int(label)
        # Return the corresponding class name (diagnose)
        class_name = label_to_diagnose_dict.get(label, "Unknown")
        print(f"Lookup class name for label {label}: {class_name}")
        return class_name
    return "Unknown"
def get_classification_patient(patient_folder):
    probs_path = patient_folder + '/single_cell_probabilities.npy'
    sc_probs = np.load(probs_path)
    sc_class= np.argmax(sc_probs, axis=1)
    return sc_class

df = pd.DataFrame(columns=["patient", "AML_subtype"] + sc_class_labels)
# Process only the selected patients
print("Starting processing of selected patient folders...")
for folder_patient in os.listdir(data_path):
    if folder_patient not in patient_to_label:
        continue  # Skip patients not in the selected list

    folder_patient_path = os.path.join(data_path, folder_patient)

    if os.path.isdir(folder_patient_path):
        print(f"Processing patient folder: {folder_patient_path}")

        # Check if the .npy file exists in the folder
        if "single_cell_probabilities.npy" not in os.listdir(folder_patient_path):
            print("Skipping patient folder without single_cell_probabilities.npy:", folder_patient_path)
            continue

        # Get classifications from the .npy file
        sc_class = get_classification_patient(folder_patient_path)

        # Get count vectors for each class
        counts_vector, unique_labels = get_counts_vector(sc_class)

        # Get patient name and corresponding class name (diagnose)
        patient_name = get_patient_name(folder_patient_path)
        class_name = get_class_name(patient_name)
        df.loc[len(df)] = np.array([patient_name, class_name] + counts_vector.tolist())

df[sc_class_labels] = df[sc_class_labels].astype(int)
df[["patient", "AML_subtype"]] = df[["patient", "AML_subtype"]].astype(str)

# Save the results to a CSV file
df.to_csv(os.path.join(result_path, "single_cell_results.csv"), index=False)
print("Finished processing and saved results to CSV.")
