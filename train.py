# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 14:10:27 2023

@author: MuhammedFurkanDasdel
"""
from model_train import ModelTrainer
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
    

    
    # import from other, own modules
    
    
    # 1: Setup. Source Folder is parent folder for both mll_data_master and
    
    # get arguments from parser, set up folder
    # parse arguments
    parser = ap.ArgumentParser()
    
    # Algorithm / training parameters
    parser.add_argument(
        '--fold',
        help='offset for cross-validation (1-5). Change to cross-validate',
        required=False,
        type=int,
        default=0)  # shift folds for cross validation. Increasing by 1 moves all folds by 1.
    parser.add_argument(
        '--lr',
        help='used learning rate',
        required=False,
        type=float,
        default=2e-5)                                     # learning rate
    parser.add_argument(
        '--scheduler',
        help='scheduler',
        required=False,
        default='ReduceLROnPlateau')  
    parser.add_argument(
        '--ep',
        help='max. amount after which training should stop',
        required=False,
        type=int,
        default=50)               # epochs to train
    parser.add_argument(
        '--es',
        help='early stopping if no decrease in loss for x epochs',
        required=False,
        type=int,
        default=10)          # epochs without improvement, after which training should stop.
    parser.add_argument(
        '--multi_att',
        help='use multi-attention approach',
        required=False,
        type=int,
        default=1)                          # use multiple attention values if 1
    parser.add_argument(
        '--checkpoint',
        help='checkpoint',
        required=False,
        default=None)
    parser.add_argument(
        '--metric',
        help='loss or f1',
        required=False,
        default='loss')      
    
    # Data parameters: Modify the dataset
    parser.add_argument(
        '--prefix',
        help='define which set of features shall be used',
        required=False,
        default='fnl34_')        # define feature source to use (from different CNNs)
    # pass -1, if no filtering acc to peripheral blood differential count
    # should be done
    parser.add_argument(
        '--filter_diff',
        help='Filters AML patients with less than this perc. of MYB.',
        default=20)
    # Leave out some more samples, if we have enough without them. Quality of
    # these is not good, but if data is short, still ok.
    parser.add_argument(
        '--filter_mediocre_quality',
        help='Filters patients with sub-standard sample quality',
        default=0)
    parser.add_argument(
        '--bootstrap_idx',
        help='Remove one specific patient at pos X',
        default=-
        1)                             # Remove specific patient to see effect on classification
    
    # Output parameters
    parser.add_argument(
        '--result_folder',
        help='store folder with custom name',
        required=False)                                 # custom output folder name
    parser.add_argument(
        '--save_model',
        help='choose wether model should be saved',
        required=False,
        default=1)                  # store model parameters if 1
    args = parser.parse_args()
    
    # store results in target folder
    checkpoint = args.checkpoint
    start = time.time()
    
    RESULT_FOLDER = f"Results_fold{args.fold}"
    def get_unique_folder(base_folder):
        counter = 1
        new_folder = base_folder
        
        while os.path.exists(new_folder):
            new_folder = f"{base_folder}{counter}"
            counter += 1
        
        return new_folder

    if not os.path.exists(RESULT_FOLDER):
        os.makedirs(RESULT_FOLDER)
    else:
        if checkpoint is None:
            unique_folder = get_unique_folder(RESULT_FOLDER)
            os.makedirs(unique_folder)
            RESULT_FOLDER = unique_folder
    
    print('Results are saved in: ',RESULT_FOLDER)
    # 2: Dataset
    # Initialize datasets, dataloaders, ...
    print("")
    print('Initialize datasets...')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ngpu = torch.cuda.device_count()
    print("Found device: ", ngpu, "x ", device)
    
    ncpu = os.cpu_count()
    print("ncpu="+str(ncpu))
    
    datasets = {}


    data_path = "/lustre/groups/labs/marr/qscd01/datasets/230824_MLL_BELUGA/RawImages"
    
    csv_root = "data_cross_val"


    def get_label(df, diagnosis):
        return df[df['diagnose'] == diagnosis]['label'].values[0]
    
    label_to_diagnose = pd.read_csv(os.path.join(csv_root,"label_to_diagnose.csv"))

    class_count = len(label_to_diagnose)
    
    print('Reading files from: ',os.path.join(csv_root,f'data_fold_{args.fold}'))
    train_files = pd.read_csv(os.path.join(csv_root,f'data_fold_{args.fold}',"train.csv"))
    val_files = pd.read_csv(os.path.join(csv_root,f'data_fold_{args.fold}',"val.csv"))
    test_files = pd.read_csv(os.path.join(csv_root,f'data_fold_{args.fold}',"test.csv"))


    datasets['train'] = MllDataset(data_path, train_files, class_count)
    datasets['val'] = MllDataset(data_path, val_files, class_count)
    datasets['test'] = MllDataset(data_path, test_files, class_count)
    
    
    # Initialize dataloaders
    print("Initialize dataloaders...")
    dataloaders = {}
    
    def create_balanced_sampler(dataset, class_count):
        # compute the count of each class in the data
        count = [0] * class_count
        for item in dataset:
            label = item[1].nonzero(as_tuple=True)[0].item()  # get the class label from one-hot vector
            count[label] += 1
    
        # calculate weights for each sample in the dataset
        weights = [1.0 / count[item[1].nonzero(as_tuple=True)[0].item()] for item in dataset]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    
        return sampler
    
    #batch_size = 1
    num_workers = 4
    
    for split in ['train', 'val', 'test']:
        dataset = datasets[split]
        #sampler = None
        #if split != 'test':  # balanced sampling for train and val sets
            #sampler = create_balanced_sampler(dataset,class_count)
    
        dataloaders[split] = DataLoader(dataset, batch_size=1, shuffle = True, num_workers=num_workers, pin_memory=True)
    
    print("Datoladers are ready..")
    
    # 3: Model
    # initialize model, GPU link, training
    
    # set up GPU link and model (check for multi GPU setup)
    if checkpoint is None:
        model_path = "/home/aih/gizem.mert/Dino/DINO/DinoBloom-B.pth"
    else:
        model_path = None

    seed = 38
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    model = ViTMiL(
        class_count=class_count,
        multicolumn=int(
            args.multi_att),
        device=device,
        model_path = model_path)

    if checkpoint is not None:
        pre = torch.load(checkpoint)
        vit_state_dict = {k.replace("module.", ""): v for k, v in pre.items()}
        model.load_state_dict(vit_state_dict, strict=False)
        print(f"Using weights from {checkpoint}")
    
    if(ngpu > 1):
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    print(model.eval())
    print("Setup complete.")
    print("")
    
    
    optimizer = optim.AdamW(model.parameters(), lr=float(args.lr))
    '''
    optimizer = optim.SGD(
        model.parameters(),
        lr=float(
            args.lr),
        momentum=0.9,
        nesterov=True)
    '''

    warmup_steps = 200
    total_steps = int(args.ep) * len(dataloaders['train'])/40
    if args.scheduler == 'get_linear_schedule_with_warmup':
        scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        print('Using get_linear_schedule_with_warmup')
    elif args.scheduler == 'ReduceLROnPlateau':
        if args.metric == 'f1':
            mode = 'max'
        else:
            mode = 'min'
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode, patience=5,min_lr=2e-6,verbose=True)
        print(f'Using ReduceLROnPlateau with {args.metric}')
    elif args.scheduler == 'OneCycleLR':
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=float(args.lr)*10, steps_per_epoch=len(dataloaders['train'])/20, epochs=int(args.ep))
        print('Using OneCycleLR')
    else:
        scheduler = None
    
    
    
    # launch training
    train_obj = ModelTrainer(
        model=model,
        dataloaders=dataloaders,
        epochs=int(args.ep),
        optimizer=optimizer,
        scheduler_name=args.scheduler,
        scheduler=scheduler,
        class_count=class_count,
        early_stop=int(args.es),
        device=device,
        save_path = RESULT_FOLDER)
    print("Starting training")
    model, conf_matrix, data_obj = train_obj.launch_training()

    
    
    # 4: aftermath
    # save confusion matrix from test set, all the data , model, print parameters
    
    np.save(os.path.join(RESULT_FOLDER, 'test_conf_matrix.npy'), conf_matrix)
    plot_confusion_matrix(conf_matrix, RESULT_FOLDER, label_to_diagnose)
    
    pickle.dump(
        data_obj,
        open(
            os.path.join(
                RESULT_FOLDER,
                'testing_data.pkl'),
            "wb"))
    
    if(int(args.save_model)):
        torch.save(model, os.path.join(RESULT_FOLDER, 'model.pt'))
        #torch.save(model.state_dict(), os.path.join(RESULT_FOLDER, 'state_dictmodel.pt'))
    
    end = time.time()
    runtime = end - start
    time_str = str(int(runtime // 3600)) + "h" + str(int((runtime %
                                                          3600) // 60)) + "min" + str(int(runtime % 60)) + "s"
    
    # other parameters
    print("")
    print("------------------------Final report--------------------------")
    print('prefix', args.prefix)
    print('Runtime', time_str)
    print('max. Epochs', args.ep)
    print('Learning rate', args.lr)
    

if __name__ == "__main__":
    main()
    
