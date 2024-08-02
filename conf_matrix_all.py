import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plot_confusion import plot_confusion_matrix

RESULT_FOLDER = "/home/aih/gizem.mert/Dino/DINO/confusion_all_test"

fold_0_confusion_path = "/home/aih/gizem.mert/Dino/DINO/Results_fold02/test_conf_matrix.npy"
fold_1_confusion_path = "/home/aih/gizem.mert/Dino/DINO/Results_fold12/test_conf_matrix.npy"
fold_2_confusion_path = "/home/aih/gizem.mert/Dino/DINO/Results_fold22/test_conf_matrix.npy"
fold_3_confusion_path = "/home/aih/gizem.mert/Dino/DINO/Results_fold32/test_conf_matrix.npy"
fold_4_confusion_path = "/home/aih/gizem.mert/Dino/DINO/Results_fold45/test_conf_matrix.npy"

csv_root = "data_cross_val"

label_to_diagnose = pd.read_csv(os.path.join(csv_root, "label_to_diagnose.csv"))

# Load the confusion matrices
conf_matrix_0 = np.load(fold_0_confusion_path)
conf_matrix_1 = np.load(fold_1_confusion_path)
conf_matrix_2 = np.load(fold_2_confusion_path)
conf_matrix_3 = np.load(fold_3_confusion_path)
conf_matrix_4 = np.load(fold_4_confusion_path)

conf_matrix = conf_matrix_0 + conf_matrix_1 + conf_matrix_2 + conf_matrix_3 + conf_matrix_4
plot_confusion_matrix(conf_matrix, RESULT_FOLDER, label_to_diagnose)

