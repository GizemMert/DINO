# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 19:37:36 2023

@author: zehra
"""

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split


LABEL_CSV_PATH = "/lustre/groups/labs/marr/qscd01/datasets/230824_MLL_BELUGA/patient_list_Christian_coarse_noMGUS.csv"
data_path = "/lustre/groups/labs/marr/qscd01/datasets/230824_MLL_BELUGA/RawImages"


meta_data = pd.read_csv(LABEL_CSV_PATH)
classes = sorted(meta_data['diagnose'].unique())
class_to_idx = {cl: i for i, cl in enumerate(classes)}
class_count = len(class_to_idx)


data_files = os.listdir(data_path)

common_files = []
labels = []
#diagnosiss = []

for file in data_files:
    patient_id = file.replace('_features', '')
    temp = os.path.join(data_path,file)
    if os.path.isdir(temp):
        if len(os.listdir(temp)) != 0 and any(f.endswith('.TIF') for f in os.listdir(temp)):
            # Check if the patient_id exists in the dataframe
            if patient_id in meta_data['patient_id'].values:
                common_files.append(patient_id)
                # Find the diagnosis for the patient_id
                diagnosis = meta_data[meta_data['patient_id'] == patient_id]['diagnose'].values[0]
                label = class_to_idx[diagnosis]
                labels.append(label)

# 0.6 0.2 0.2 distribution on dataset
train_data, val_data, train_label, val_label = train_test_split(common_files, labels, test_size=0.4, random_state=42)
val_data, test_data, val_label, test_label = train_test_split(val_data, val_label, test_size=0.5, random_state=42)


#train_data = np.array(train_data)
#train_label = np.array(train_label)

train = pd.DataFrame([train_data, train_label]).T
train.columns = ['patient_files', 'labels']

val = pd.DataFrame([val_data, val_label]).T
val.columns = ['patient_files', 'labels']

test = pd.DataFrame([test_data, test_label]).T
test.columns = ['patient_files', 'labels']
print('train:', len(train))
print('val:', len(val))
print('test:', len(test))

train.to_csv('data/train_raw.csv',index = False)
val.to_csv('data/val_raw.csv',index = False)
test.to_csv('data/test_raw.csv',index = False)


label_to_diganose = pd.DataFrame.from_dict(class_to_idx,orient='index')
label_to_diganose.columns = ['label']
label_to_diganose.to_csv('data/label_to_diganose.csv')
