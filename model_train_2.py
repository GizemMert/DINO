import torch
import torch.utils.data as data_utils
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import transformers
from model_utils_2 import calculate_f1_from_confusion_matrix

import sys
import time
import copy
import numpy as np
from tqdm import tqdm
import os
import pandas as pd


class ModelTrainer:
    '''class containing all the info about the training process and handling the actual
    training function'''

    def __init__(
            self,
            model,
            dataloaders,
            epochs,
            optimizer,
            scheduler_name,
            scheduler,
            class_count,
            device,
            save_path,
            early_stop=20):
        self.model = model
        self.dataloaders = dataloaders
        self.epochs = epochs
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.class_count = class_count
        self.early_stop = early_stop
        self.device = device
        self.save_path = save_path
        self.scheduler_name = scheduler_name
        self.data_obj = DataMatrix()

        self.mode = self.scheduler.mode

    def launch_training(self):
        '''initializes training process.'''
        #print("Initialized")
        best_loss = 10  # high value, so that future loss values will always be lower
        best_f1 = 0 # low value, so that future f1 values will always be higher
        no_improvement_for = 0
        best_model = copy.deepcopy(self.model.state_dict())

        for ep in range(self.epochs):
            # perform train/val iteration
            loss, acc, f1, conf_matrix = self.dataset_to_model(ep, 'train')
            torch.cuda.empty_cache()
            loss, acc, f1, conf_matrix = self.dataset_to_model(ep, 'val')
            torch.cuda.empty_cache()
            no_improvement_for += 1

            # if improvement, reset counter
            if self.mode == 'min':
                if(loss < best_loss):
                    best_model = copy.deepcopy(self.model.state_dict())
                    best_loss = loss
                    best_f1 = f1
                    no_improvement_for = 0
                    print("Best Loss!")
                    torch.save(self.model.state_dict(),os.path.join(self.save_path,"scemila_transformer_best.pth"))
                elif(loss == best_loss):
                    if (f1 > best_f1):
                        best_model = copy.deepcopy(self.model.state_dict())
                        best_loss = loss
                        best_f1 = f1
                        no_improvement_for = 0
                        print("Best Loss and F1!")
                        torch.save(self.model.state_dict(),os.path.join(self.save_path,"scemila_transformer_best.pth"))
            elif self.mode == 'max':
                if(f1 > best_f1):
                    best_model = copy.deepcopy(self.model.state_dict())
                    best_loss = loss
                    best_f1 = f1
                    no_improvement_for = 0
                    print("Best F1!")
                    torch.save(self.model.state_dict(),os.path.join(self.save_path,"scemila_transformer_best.pth"))               
                elif (f1==best_f1):
                    if(loss < best_loss):
                        best_model = copy.deepcopy(self.model.state_dict())
                        best_loss = loss
                        best_f1 = f1
                        no_improvement_for = 0
                        print("Best F1 and Loss!")
                        torch.save(self.model.state_dict(),os.path.join(self.save_path,"scemila_transformer_best.pth"))

            if ep%5 == 0:
                torch.save(self.model.state_dict(),os.path.join(self.save_path,f"scemila_transformer_{ep}.pth"))

            # break if X times no improvement
            if(no_improvement_for == self.early_stop):
                break

            if self.scheduler_name == 'ReduceLROnPlateau':
                if self.mode == 'max':
                    self.scheduler.step(f1)
                else:
                    self.scheduler.step(loss)
            elif self.scheduler_name == 'OneCycleLR':
                self.scheduler.step()

            '''
            # scheduler (optional)
            if not (self.scheduler is None):
                if isinstance(
                        self.scheduler,
                        optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(loss)
                elif isinstance(self.scheduler,optim.lr_scheduler.OneCycleLR):
                    self.scheduler.step()
            '''
        torch.save(self.model.state_dict(),os.path.join(self.save_path,"scemila_transformer_last.pth"))
        # load best performing model, and launch on test set
        self.model.load_state_dict(best_model)

        #Save logits for val
        test_loss, accuracy, conf_matrix, pred_stack, label_stack, file_paths = self.infer_test('val')
        np.save(os.path.join(self.save_path,'val_prediction.npy'),pred_stack)
        np.save(os.path.join(self.save_path,'val_labels.npy'),label_stack)
        np.save(os.path.join(self.save_path, 'val_conf_matrix.npy'), conf_matrix)
        df = pd.DataFrame({
            'patients': file_paths,
            'label': label_stack,
            'prediction': [list(p) for p in pred_stack]
        })
        csv_path = os.path.join(self.save_path, 'metadata_results_val.csv')
        df.to_csv(csv_path, index=False)
        np.save(os.path.join(self.save_path,'val_metadata.npy'), df.to_dict('records'))

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
    

    def dataset_to_model(self, epoch, split, backprop_every=20):
        '''launch iteration for 1 epoch on specific dataset object, with backprop being optional
        - epoch: epoch count, is only printed and thus not super important
        - split: if equal to 'train', apply backpropagation. Otherwise, don`t.
        - backprop_every: only apply backpropagation every n patients. Allows for gradient accumulation over
          multiple patients, like in batch processing in a regular neural network.'''

        if(split == 'train'):
            backpropagation = True
            self.model.train()
            #print("burasi train basladi")
        else:
            backpropagation = False
            self.model.eval()

        # initialize data structures to store results
        corrects = 0
        train_loss = 0.
        
        
        
        time_pre_epoch = time.time()
        confusion_matrix = np.zeros(
            (self.class_count, self.class_count), np.int16)
        

        self.optimizer.zero_grad()
        backprop_counter = 0

        gradient_context = torch.enable_grad if backpropagation else torch.no_grad
        with gradient_context():
            for batch_idx, (bag, label, img_paths) in enumerate(tqdm(self.dataloaders[split])):
                patient_id = img_paths[0][0].split('/')[-2]
                #print(backprop_counter) 
                # send to gpu
                label = label.to(self.device)
                bag = bag.to(self.device)

                # forward pass
                prediction = self.model(bag)

                # calculate and store loss
                # loss_func = nn.BCELoss()
                # loss_out = loss_func(prediction, label)

                # pred_log = torch.log(prediction)
                loss_func = nn.CrossEntropyLoss()
                label_indices = torch.argmax(label,dim=1)

                #loss_out = loss_func(prediction, label[0])
                loss_out = loss_func(prediction, label_indices)
                train_loss += loss_out.item()
                

                # apply backpropagation if indicated
                if(backpropagation):
                    loss_out.backward()
                    backprop_counter += 1
                    # counter makes sure only every X samples backpropagation is
                    # excluded (resembling a training batch)
                    if(backprop_counter % backprop_every == 0):
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.optimizer.step()
                        '''
                        if self.scheduler is not None:
                            if isinstance(self.scheduler, transformers.get_linear_schedule_with_warmup):
                                self.scheduler.step()
                        '''
                        if self.scheduler_name == 'get_linear_schedule_with_warmup':
                            self.scheduler.step()

                        # print_grads(self.model)
                        self.optimizer.zero_grad()

                # transforms prediction tensor into index of position with highest
                # value
                label_prediction = torch.argmax(prediction, dim=1).item()
                label_groundtruth = label_indices.item()

                # store predictions accordingly in confusion matrix
                if(label_prediction == label_groundtruth):
                    corrects += 1
                confusion_matrix[label_groundtruth, label_prediction] += int(1)

            #print('----- loss: {:.3f}, gt: {} , pred: {}, prob: {}'.format(loss_out, label_groundtruth, label_prediction, prediction.detach().cpu().numpy()))

        samples = len(self.dataloaders[split])
        train_loss /= samples

        accuracy = corrects / samples

        f1_scores_per_class, f1_macro, f1_micro, f1_weighted = calculate_f1_from_confusion_matrix(confusion_matrix)

        print('- ep: {}/{}, loss: {:.3f}, acc: {:.3f}, f1_macro: {:.3f}, {}s, {}'.format(
            epoch + 1, self.epochs, train_loss,
            accuracy, f1_macro, int(time.time() - time_pre_epoch), split))

        return train_loss, accuracy, f1_macro, confusion_matrix

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
