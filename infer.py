import torch
import torch.utils.data as data_utils
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import transformers

import sys
import time
import copy
import numpy as np
from tqdm import tqdm
import os
import pandas as pd


class ModelInfer:
    '''class containing all the info about the training process and handling the actual
    training function'''

    def __init__(
            self,
            model,
            dataloaders,
            class_count,
            device,
            save_path):
        self.model = model
        self.dataloaders = dataloaders
        self.class_count = class_count
        self.device = device
        self.save_path = save_path
        self.data_obj = DataMatrix()

    def launch_infering(self):
        '''initializes training process.'''
        #Save logits for test
        test_loss, accuracy, conf_matrix, pred_stack, label_stack, file_paths = self.infer_test('test')
        np.save(os.path.join(self.save_path,'test_prediction.npy'),pred_stack)
        np.save(os.path.join(self.save_path,'test_labels.npy'),label_stack)
        df = pd.DataFrame({
            'patients': file_paths,
            'label': label_stack,
            'prediction': [list(p) for p in pred_stack]
        })
        csv_path = os.path.join(self.save_path, 'metadata_results_test.csv')
        df.to_csv(csv_path, index=False)
        np.save(os.path.join(self.save_path,'test_metadata.npy'), df.to_dict('records'))


        return self.model, conf_matrix, self.data_obj
    

    def infer_test(self,split):
        self.model.eval()
        pred_stack = []
        label_stack =[]
        file_paths = []
        confusion_matrix = np.zeros((self.class_count, self.class_count), np.int16)
        corrects = 0
        test_loss = 0.
        

        with torch.no_grad():
            for (bag, label, img_paths) in tqdm(self.dataloaders[split]):
                
                patient_id = img_paths[0][0].split('/')[-2]
                # send to gpu
                label = label.to(self.device)
                bag = bag.to(self.device)
                num_of_images = bag.shape[1]
                print('Processing patient: ',patient_id)

                # forward pass
                latent, prediction = self.model(bag,return_latent = True)
                cls_attention_scores = generate_rollout(self.model,bag,start_layer=0)
                cls_attention_scores = cls_attention_scores.squeeze(0)

                loss_func = nn.CrossEntropyLoss()
                label_indices = torch.argmax(label,dim=1)
                loss_out = loss_func(prediction, label_indices)
                test_loss += loss_out.item()

                label_prediction = torch.argmax(prediction, dim=1).item()
                label_groundtruth = label_indices.item()

                if(label_prediction == label_groundtruth):
                    corrects += 1
                confusion_matrix[label_groundtruth, label_prediction] += int(1)
                
                
                self.data_obj.add_patient(
                    label_groundtruth,
                    patient_id,
                    latent,
                    cls_attention_scores,
                    img_paths,
                    label_prediction,
                    prediction,
                    loss_out)
                
                pred_stack.append(prediction)
                label_stack.append(label_groundtruth)
                file_paths.append(os.path.basename(patient_id))

        samples = len(self.dataloaders[split])
        test_loss /= samples
        accuracy = corrects / samples
        
        pred_stack = torch.cat(pred_stack, dim=0).detach().cpu().numpy()
        #label_stack = torch.cat(label_stack, dim=0).detach().cpu().numpy()

        print('- loss: {:.3f}, acc: {:.3f}, {}'.format(test_loss, accuracy, split))
            
        return test_loss, accuracy, confusion_matrix, pred_stack, label_stack, file_paths

class DataMatrix():
    '''DataMatrix contains all information about patient classification for later storage.
    Data is stored within a dictionary:

    self.data_dict[true entity] contains another dictionary with all patient paths for
                                the patients of one entity (e.g. AML-PML-RARA, SCD, ...)

    --> In this dictionary, the paths form the keys to all the data of that patient
        and it's classification, stored as a tuple:

        - attention_raw:    attention for all single cell images before softmax transform
        - attention:        attention after softmax transform
        - prediction:       Numeric position of predicted label in the prediction vector
        - prediction_vector:Prediction vector containing the softmax-transformed activations
                            of the last AMiL layer
        - loss:             Loss for that patients' classification
        - out_features:     Aggregated bag feature vectors after attention calculation and
                            softmax transform. '''

    def __init__(self):
        self.data_dict = dict()

    def add_patient(
            self,
            entity,
            patient_id,
            latent,
            cls_attention_scores,
            img_paths,
            prediction,
            prediction_vector,
            loss):
        
        '''Add a new patient into the data dictionary. Enter all the data packed into a tuple into the dictionary as:
        self.data_dict[entity][path_full] = (cls_attention_scores, attention, prediction, prediction_vector, loss, out_features)

        accepts:
        - entity: true patient label
        - path_full: path to patient folder
        - cls_attention_scores: attention before softmax transform
        - img_paths: img_paths after softmax transform
        - prediction: numeric bag label
        - prediction_vector: output activations of AMiL model
        - loss: loss calculated from output actiations
        - out_features: bag features after attention calculation and matrix multiplication

        returns: Nothing
        '''

        if not (entity in self.data_dict):
            self.data_dict[entity] = dict()
        self.data_dict[entity][patient_id] = (
            (cls_attention_scores.detach().cpu().numpy(),img_paths),
            prediction,
            prediction_vector.data.cpu().numpy()[0],
            latent.detach().cpu().numpy(),
            float(loss.data.cpu()))

    def return_data(self):
        return self.data_dict
    
def compute_rollout_attention(all_layer_matrices, start_layer=0):
    # adding residual consideration- code adapted from https://github.com/samiraabnar/attention_flow
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    matrices_aug = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
                          for i in range(len(all_layer_matrices))]
    joint_attention = matrices_aug[start_layer]
    for i in range(start_layer+1, len(matrices_aug)):
	    joint_attention = matrices_aug[i].bmm(joint_attention)
    return joint_attention

def generate_rollout(model, input, start_layer=0):
    model(input)
    blocks = model.transformer.transformer.layers
    all_layer_attentions = []
    for blk in blocks:
	    attn_heads = blk[0].fn.get_attention_map()
	    avg_heads = (attn_heads.sum(dim=1) / attn_heads.shape[1]).detach()
	    all_layer_attentions.append(avg_heads)
    rollout = compute_rollout_attention(all_layer_attentions, start_layer=start_layer)
    return rollout[:,0, 1:]
