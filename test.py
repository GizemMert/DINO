# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 14:10:27 2023

@author: MuhammedFurkanDasdel
"""
from infer import ModelInfer
#from model_eval import *  # model training function
from classifier import ViTMiL     
from dataset2 import MllDataset       # dataset
from plot_confusion import plot_confusion_matrix

import torch.optim as optim
#from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, TensorDataset
import torch.multiprocessing
import torch
import sys
import os
import time
import argparse as ap
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import get_linear_schedule_with_warmup

torch.multiprocessing.set_sharing_strategy('file_system')

import pandas as pd


def main():
    
    # store results in target folder
    
    start = time.time()
    
    RESULT_FOLDER = f"Results_fold4_test_all_data"
    def get_unique_folder(base_folder):
        counter = 1
        new_folder = base_folder
        
        while os.path.exists(new_folder):
            new_folder = f"{base_folder}-{counter}"
            counter += 1
        
        return new_folder

    if not os.path.exists(RESULT_FOLDER):
        os.makedirs(RESULT_FOLDER)
    else:
        unique_folder = get_unique_folder(RESULT_FOLDER)
        os.makedirs(unique_folder)
        RESULT_FOLDER = unique_folder
    
    # 2: Dataset
    # Initialize datasets, dataloaders, ...
    print("")
    print('Initialize datasets...')
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ngpu = torch.cuda.device_count()
    print("Found device: ", ngpu, "x ", device)
    
    ncpu = os.cpu_count()
    print("ncpu="+str(ncpu))
    
    datasets = {}


    data_path = "/lustre/groups/labs/marr/qscd01/datasets/230824_MLL_BELUGA/RawImages"
    
    csv_root = "data"


    def get_label(df, diagnosis):
        return df[df['diagnose'] == diagnosis]['label'].values[0]
    
    label_to_diagnose = pd.read_csv(os.path.join(csv_root,"label_to_diagnose.csv"))

    class_count = len(label_to_diagnose)
    
    test_files = pd.read_csv(os.path.join(csv_root,"all_data.csv"))
    #test_files = pd.read_csv('train_raw_mock.csv')

    datasets['test'] = MllDataset(data_path, test_files, class_count)
    
    
    # Initialize dataloaders
    print("Initialize dataloaders...")
    dataloaders = {}
    

    
    #batch_size = 1
    num_workers = 4
     
    dataloaders['test'] = DataLoader(datasets['test'],batch_size=1, shuffle = True, num_workers=num_workers)
    
    print("Datoladers are ready..")
    
    model_path = "Results_fold4/scemila_transformer_best.pth"
    
    model = ViTMiL(
        class_count=class_count,
        multicolumn=1,
        device=device,
        model_path = "DinoBloom-B.pth")
    print(model.eval())
    #print(model.state_dict().keys())
    
    pre = torch.load(model_path)
    #vit_state_dict = {k.replace("module.", ""): v for k, v in pre.items()}
    model.load_state_dict(pre, strict=True)
    
    if(ngpu > 1):
        model = torch.nn.DataParallel(model)
    model = model.to(device)
        
    
    
    # launch training
    infer_obj = ModelInfer(
        model=model,
        dataloaders=dataloaders,
        class_count=class_count,
        device=device,
        save_path = RESULT_FOLDER)
    print("Starting inferring")
    model, conf_matrix, data_obj = infer_obj.launch_infering()
    
    
    # 4: aftermath
    # save confusion matrix from test set, all the data , model, print parameters
    
    np.save(os.path.join(RESULT_FOLDER, 'test_conf_matrix.npy'), conf_matrix)
    plot_confusion_matrix(conf_matrix, RESULT_FOLDER, label_to_diagnose)
    
    pickle.dump(
        data_obj,
        open(
            os.path.join(
                RESULT_FOLDER,
                'inferring_data.pkl'),
            "wb"))
    

    end = time.time()
    runtime = end - start
    time_str = str(int(runtime // 3600)) + "h" + str(int((runtime %
                                                          3600) // 60)) + "min" + str(int(runtime % 60)) + "s"
    
    # other parameters
    print("")
    print("------------------------Final report--------------------------")
    print('Runtime', time_str)
    

if __name__ == "__main__":
    main()
    
