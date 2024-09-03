import numpy as np
import os
import re
import glob
import pandas as pd

# Load class label information from CSV
label_to_diagnose_path = '/home/aih/gizem.mert/Dino/DINO/data_cross_val/label_to_diagnose.csv'
label_to_diagnose = pd.read_csv(label_to_diagnose_path)
label_to_diagnose_dict = dict(zip(label_to_diagnose['label'], label_to_diagnose['diagnose']))
class_labels = label_to_diagnose['diagnose'].tolist()
n_classes = len(class_labels)

# Function to get list of image_paths in one folder
def get_image_path_list(folder_path):
    # List only .TIF files and ignore other types of files like .npy
    tif_files = [f for f in glob.glob(f"{folder_path}/*.TIF") if f.lower().endswith('.tif')]
    return sorted(tif_files)  # Ensure files are sorted consistently

# Extracts the number of image in the file_path e.g. "Gal-000123.RGB.TIF"
def extract_number_image(file_path):
    match = re.search(r'Gal-(\d{6})\.RGB\.TIF', file_path, re.IGNORECASE)
    if match:
        return int(match.group(1))
    else:
        print(f"Warning: Image file name does not match expected pattern: {file_path}")
        return None

def get_patient_name(path):
    return os.path.basename(path)

def get_class_name(patient_name):
    label = patient_to_label.get(patient_name)
    if label is not None:
        return label_to_diagnose_dict.get(label, "Unknown")
    return "Unknown"

def get_classification_patient(patient_folder):
    probs_path = os.path.join(patient_folder, 'single_cell_probabilities.npy')
    sc_probs = np.load(probs_path)
    sc_class = np.argmax(sc_probs, axis=1)
    return sc_class

data_directory = '/lustre/groups/labs/marr/qscd01/datasets/230824_MLL_BELUGA/RawImages'
n_patients = 200
experiment_name = "experiment_3"
output_folder = f'/home/aih/gizem.mert/Dino/DINO/fold1/artificial_data/{experiment_name}/data'
output_folder_csv = f'/home/aih/gizem.mert/Dino/DINO/fold1/artificial_data/{experiment_name}'

# Load patient data from train and validation csv
train_csv_path = '/home/aih/gizem.mert/Dino/DINO/data_cross_val/data_fold_1/train.csv'
val_csv_path = '/home/aih/gizem.mert/Dino/DINO/data_cross_val/data_fold_1/val.csv'
train_patients_df = pd.read_csv(train_csv_path)
val_patients_df = pd.read_csv(val_csv_path)
selected_patients = pd.concat([train_patients_df, val_patients_df], ignore_index=True)
patient_to_label = dict(zip(selected_patients['patient_files'], selected_patients['labels']))

artificial_patients_metadata = []
# Iterate over real dataset and store image paths in a dataframe df
df = pd.DataFrame(columns=["patient", "AML_subtype", "SC_Label", "image_path"])
for folder_patient in os.listdir(data_directory):
    if folder_patient not in patient_to_label:
        continue  # Skip if the patient is not in the selected list
    folder_patient_path = os.path.join(data_directory, folder_patient)
    if os.path.isdir(folder_patient_path):
        AML_subtype = get_class_name(folder_patient)
        images = get_image_path_list(folder_patient_path)  # Only get image files
        sc_classes = get_classification_patient(folder_patient_path)

        # Match images with classification results by filename number
        image_numbers = [extract_number_image(image) for image in images]
        image_classes = [(num, img) for num, img in zip(image_numbers, images) if num is not None]

        # Sort both lists by the numeric part
        sorted_image_classes = sorted(image_classes, key=lambda x: x[0])
        sorted_classes = sorted(zip([num for num in image_numbers if num is not None], sc_classes), key=lambda x: x[0])

        # Only keep entries that match by image number
        matched_images_classes = [(img_path, cls) for (num_img, img_path), (num_cls, cls) in zip(sorted_image_classes, sorted_classes) if num_img == num_cls]

        # Process matched images and classifications
        for image_path, classification in matched_images_classes:
            df.loc[len(df)] = [get_patient_name(folder_patient_path), AML_subtype, classification, image_path]



# Calculate mean and std for each cell type that will be later used to sample data with normal distribution
sc_class_labels = ['eosinophil granulocyte', 'reactive lymphocyte',
                   'neutrophil granulocyte (segmented)', 'typical lymphocyte',
                   'other', 'neutrophil granulocyte (band)', 'monocyte',
                   'large granulated lymphocyte', 'atypical promyelocyte',
                   'basophil granulocyte', 'smudge cell', 'neoplastic lymphocyte',
                   'promyelocyte', 'myelocyte', 'myeloblast', 'metamyelocyte',
                   'normo', 'plasma cell', 'hair cell', 'bilobed M3v',
                   'mononucleosis']

df_sc_res = pd.read_csv("/home/aih/gizem.mert/Dino/DINO/fold1/train/single_cell_results.csv")
df_meanstd = df_sc_res.groupby(["AML_subtype"]).agg(["mean", "std"])

# This cell creates artificial patients and stores the single cell counts per patient in cell_type_counts_dict
cell_type_counts_dict = {}
selected_images_per_patient = {}

# Iterate over all AML subtypes
for aml_subtype in class_labels:
    class_means = df_meanstd.loc[aml_subtype, :].loc[:, "mean"].values
    class_variances = df_meanstd.loc[aml_subtype, :].loc[:, "std"].values
    for patient_number in range(n_patients):
        print(f"Generating data for patient {patient_number + 1} of subtype {aml_subtype}...")
        generated_data = np.random.normal(class_means, class_variances, 21).astype(int)
        generated_data = generated_data * (generated_data > 0)
        image_file_paths = []
        selected_images_count = {}
        for cell_type_number, cell_type in enumerate(sc_class_labels):
            df_cell_type = df[df["SC_Label"] == cell_type_number]
            file_path = df_cell_type["image_path"].values
            image_paths = np.random.choice(file_path, size=generated_data[cell_type_number]).tolist()
            print(f"\t\tSelected {len(image_paths)} images for {cell_type}")
            image_file_paths += image_paths
            selected_images_count[cell_type] = len(image_paths)
        patient_id = f"patient_{aml_subtype}_{patient_number + 1}"
        selected_images_per_patient[patient_id] = selected_images_count
        new_patient_folder = os.path.join(output_folder, patient_id)
        os.makedirs(new_patient_folder, exist_ok=True)
        image_file_paths.sort()
        txt_file_path = os.path.join(new_patient_folder, 'images.txt')
        with open(txt_file_path, 'w') as txt_file:
            for image_path in image_file_paths:
                txt_file.write(image_path + '\n')
        with open(os.path.join(new_patient_folder, "image_file_paths"), 'wb') as fp:
            pickle.dump(image_file_paths, fp)
        cell_type_count = {cell_type: image_file_paths.count(cell_type) for cell_type in set(image_file_paths)}
        print(f"\tCell type count for patient {patient_number + 1}: {cell_type_count}")
        cell_type_counts_dict[(aml_subtype, patient_id)] = cell_type_count
        print(f"\tCell type counts dictionary for patient {patient_id}: {cell_type_count}")

        # Add artificial patient metadata
        artificial_patients_metadata.append({"patient_files": patient_id, "labels": aml_subtype})

# Print selected images count per SC class per patient
print("\nSelected Images Count per SC class per Patient:")
for patient_id, sc_counts in selected_images_per_patient.items():
    print(patient_id)
    for sc_class, count in sc_counts.items():
        print(f"\t{sc_class}: {count}")

# Save artificial patient metadata to CSV
artificial_patients_df = pd.DataFrame(artificial_patients_metadata)
artificial_patients_csv_path = os.path.join(output_folder_csv, 'artificial_patients.csv')
artificial_patients_df.to_csv(artificial_patients_csv_path, index=False)
print(f"Artificial patient metadata saved to {artificial_patients_csv_path}")

# Save the DataFrame with image paths and related information
df.to_csv(os.path.join(output_folder_csv, 'image_data.csv'), index=False)
print(f"Image data saved to {os.path.join(output_folder_csv, 'image_data.csv')}")

# Create metadata including single cell types
rows = []
for patient_id, cell_counts in selected_images_per_patient.items():
    myeloblast_count = cell_counts.get('myeloblast', 0)
    promyelocyte_count = cell_counts.get('promyelocyte', 0)
    myelocyte_count = cell_counts.get('myelocyte', 0)
    metamyelocyte_count = cell_counts.get('metamyelocyte', 0)
    neutrophil_band_count = cell_counts.get('neutrophil granulocyte (band)', 0)
    neutrophil_segmented_count = cell_counts.get('neutrophil granulocyte (segmented)', 0)
    eosinophil_count = cell_counts.get('eosinophil granulocyte', 0)
    basophil_count = cell_counts.get('basophil granulocyte', 0)
    monocyte_count = cell_counts.get('monocyte', 0)
    lymph_typ_count = cell_counts.get('typical lymphocyte', 0)
    lymph_atyp_react_count = cell_counts.get('reactive lymphocyte', 0)
    lymph_atyp_neopl_count = cell_counts.get('neoplastic lymphocyte', 0)
    other_count = cell_counts.get('other', 0)
    total_count = sum(cell_counts.values())
    index = patient_id.find("_")
    bag = patient_id[:index] if index != -1 else patient_id
    id = patient_id.find("_")
    patient = patient_id[id + 1:] if index != -1 else ""
    age = 0  # dummy value for now
    row = {'patient_id': patient,
           'sex_1f_2m': None,
           'age': age,
           'bag_label': bag,
           'instance_count': total_count,
           'leucocytes_per_µl': None,
           'pb_myeloblast': round((myeloblast_count / total_count) * 100, 1),
           'pb_promyelocyte': round((promyelocyte_count / total_count) * 100, 1),
           'pb_myelocyte': round((myelocyte_count / total_count) * 100, 1),
           'pb_metamyelocyte': round((metamyelocyte_count / total_count) * 100, 1),
           'pb_neutrophil_band': round((neutrophil_band_count / total_count) * 100, 1),
           'pb_neutrophil_segmented': round((neutrophil_segmented_count / total_count) * 100, 1),
           'pb_eosinophil': round((eosinophil_count / total_count) * 100, 1),
           'pb_basophil': round((basophil_count / total_count) * 100, 1),
           'pb_monocyte': round((monocyte_count / total_count) * 100, 1),
           'pb_lymph_typ': round((lymph_typ_count / total_count) * 100, 1),
           'pb_lymph_atyp_react': round((lymph_atyp_react_count / total_count) * 100, 1),
           'pb_lymph_atyp_neopl': round((lymph_atyp_neopl_count / total_count) * 100, 1),
           'pb_other': round((other_count / total_count) * 100, 1),
           'pb_total': round((total_count / total_count) * 100, 1)}
    rows.append(row)

# Create DataFrame from the list of rows
artificial_metadata = pd.DataFrame(rows)

# Save the DataFrame to a CSV file
metadata_csv_path = os.path.join(output_folder_csv, 'metadata.csv')
artificial_metadata.to_csv(metadata_csv_path, index=False)
print(f"Metadata saved to {metadata_csv_path}")