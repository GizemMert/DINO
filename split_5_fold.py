# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 17:59:45 2023

@author: MuhammedFurkanDasdel
"""

    
    
import os
from sklearn.model_selection import StratifiedKFold
import pandas as pd

LABEL_CSV_PATH = "/lustre/groups/labs/marr/qscd01/datasets/230824_MLL_BELUGA/patient_list_Christian_coarse_noMGUS.csv"
data_path = "/lustre/groups/labs/marr/qscd01/datasets/230824_MLL_BELUGA/RawImages"
output_path = "data_cross_val"

meta_data = pd.read_csv(LABEL_CSV_PATH)
label_to_index = pd.read_csv("label_to_diagnose.csv")
classes = sorted(meta_data['diagnose'].unique())
class_to_idx = {cl: i for cl,i in zip(label_to_index["diagnose"],label_to_index["label"])}

data_files = os.listdir(data_path)
common_files = []
labels = []

for file in data_files:
    # replace in case there is features extesnion in the file name
    patient_id = file.replace('_features', '')
    temp = os.path.join(data_path,file)
    if os.path.isdir(temp) and any(f.endswith('.TIF') for f in os.listdir(temp)):
        if patient_id in meta_data['patient_id'].values:
            common_files.append(patient_id)
            diagnosis = meta_data[meta_data['patient_id'] == patient_id]['diagnose'].values[0]
            label = class_to_idx[diagnosis]
            labels.append(label)
            
df = pd.DataFrame({
    'patient_files': common_files,
    'labels': labels
})
skf = StratifiedKFold(5, shuffle=True, random_state=38)
splits = list(skf.split(df, df['labels']))


for i in range(5):
    train_indices = splits[i][0]
    test_indices = splits[i][1]
    k = (i+1) % 5
    val_indices = splits[k][1]
    train_indices = [n for n in train_indices if n not in val_indices]
    
    train = set(train_indices)
    val = set(val_indices)
    test =  set(test_indices)

    print("Train size:", len(train))
    print("Val size:", len(val))
    print("Test size:", len(test))
    
    # Create dataframes for train, val, and test
    train_df = df.iloc[train_indices]
    val_df = df.iloc[val_indices]
    test_df = df.iloc[test_indices]
    safe = 0
    # Check if there's any overlap
    if train & val:  # Intersection of train and val
        print("Overlap between train and val")
    if train & test:  # Intersection of train and test
        print("Overlap between train and test")
    if val & test:  # Intersection of val and test
        print("Overlap between val and test")
    if not (train & val or train & test or val & test):
        safe = 1
        print("Safe")
    print("-----")

    if safe == 1:
    # Create a new directory for the fold if it doesn't exist
        fold_dir = os.path.join(output_path, f"data_fold_{i}")
        if not os.path.exists(fold_dir):
            os.makedirs(fold_dir)
    
        # Save the dataframes to CSV
        train_df.to_csv(os.path.join(fold_dir, "train.csv"), index=False)
        val_df.to_csv(os.path.join(fold_dir, "val.csv"), index=False)
        test_df.to_csv(os.path.join(fold_dir, "test.csv"), index=False)


