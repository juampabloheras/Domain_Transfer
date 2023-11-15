#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 21:42:21 2023

@author: juampablo
Email: jehr@uw.edu

"""

# Import packages
import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt

from utils import make_data_loader, make_train_val_data_loader
from model import domain_classifier

from checkpoint import load_checkpoint, save_checkpoint

# # Local
TRAIN_DIR_GLI='/Users/juampablo/Desktop/Kurtlab_A23/BraTS_images' #string
TRAIN_DIR_SSA='/Users/juampablo/Desktop/Kurtlab_23/Africa_files' #string
# VAL_DIR_GLI = '/Users/juampablo/Desktop/Kurtlab_A23/gli_val_data'
# VAL_DIR_SSA = '/Users/juampablo/Desktop/Kurtlab_A23/ssa_val_data'


# # Klone
# TRAIN_DIR_GLI='/gscratch/kurtlab/brats2023/data/brats-gli/NEW_SplitData_final/trainning' #string
# TRAIN_DIR_SSA='/gscratch/kurtlab/brats2023/data/brats-ssa/Processed-TrainingData_V2' #string

# VAL_DIR_GLI = '/gscratch/kurtlab/brats2023/data/brats-gli/NEW_SplitData_final/validation'
# VAL_DIR_SSA = '/gscratch/kurtlab/brats2023/data/brats-ssa/Processed-ValidationData'

if __name__ == '__main__':
    
    model = domain_classifier() 
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10
    save_interval = 5
    
    epoch_start, model, optimizer, latest_ckpt_path, saved_ckpts_dir, out_dir, epoch_loss, epoch_loss_val = load_checkpoint(model, optimizer, out_dir = None)
    
    
    # Define device and move model to device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    
    model = model.to(device)
    
    # Data loading
    
    # in_dir = [TRAIN_DIR_GLI, TRAIN_DIR_SSA]
    # train_loader = make_data_loader(in_dir,shuffle=True, batch_size = 10, train_val_split=0.5)
    # print("data loaded")
    
    # in_dir_val = [VAL_DIR_GLI, VAL_DIR_SSA]
    # val_loader = make_data_loader(in_dir_val, shuffle=False, batch_size=1)
    
    in_dir = [TRAIN_DIR_GLI, TRAIN_DIR_SSA]
    train_val_split  = 0.5
    train_loader, val_loader = make_train_val_data_loader(in_dir, train_val_split, shuffle=True, batch_size=1)
    

    for epoch in range(epoch_start, num_epochs):
        model.train()
        running_loss = []
        batch_count = 0
        for _, data, labels in train_loader:
      
            batch_count +=1
            optimizer.zero_grad()
            
            # Reformat data
            data = data[:-1] # remove segmentation map
            print(np.shape(labels))
            
            data = np.transpose(data, (1, 0, 3, 4, 5, 2))
            data = np.squeeze(data, axis=-1)
            data = torch.tensor(data)
            
            
            print("datashape", np.shape(data))
            
            # Move data to device
            data = data.to(device)
            labels = labels.unsqueeze(1).to(device)
            
            # Forward pass
            outputs = model(data)

            print(outputs)
            print(labels)
            print("Images shape", np.shape(data))
            print("Labels shape", np.shape(labels))
            print("Outputs shape", np.shape(outputs))


            loss = criterion(outputs, labels.float())
            
            loss.backward()
            optimizer.step()
            
            running_loss.append(loss.detach().to('cpu'))
            
            # print("Batch loss: ", loss.item())
        
        
        # Validation
        model.eval()
        running_loss_val = []
        for _, data, labels in val_loader:
            data = data[:-1] # remove segmentation map
            print(np.shape(labels))
            
            data = np.transpose(data, (1, 0, 3, 4, 5, 2))
            data = np.squeeze(data, axis=-1)
            data = torch.tensor(data)
            
            
            print("datashape", np.shape(data))
            
            # Move data to device
            data = data.to(device)
            labels = labels.unsqueeze(1).to(device)
            
            outputs = model(data)
            
            loss = criterion(outputs, labels.float())

            running_loss_val.append(loss.detach().to('cpu'))
            



        epoch_loss.append(np.mean(running_loss))
        epoch_loss_val.append(np.mean(running_loss_val))
        
        
        

        
        checkpoint = {
            'epoch': epoch,
            'model_sd': model.state_dict(),
            'optim_sd': optimizer.state_dict(),
            'epoch_loss': epoch_loss,
            'epoch_loss_val': epoch_loss_val
            }
        
        torch.save(checkpoint, latest_ckpt_path)
        
        with open(os.path.join(out_dir, 'loss_values.dat'), 'a') as f:
            f.write(f'{epoch}, {epoch_loss}\n')
        print(f'Epoch {epoch} completed. Average loss = {np.mean(running_loss):.4f}.')
        
        # Save checkpoint
        save_checkpoint(epoch, save_interval, checkpoint, saved_ckpts_dir)
            
    plt.figure()
    plt.plot(epoch_loss, label = 'train')
    plt.plot(epoch_loss_val, label='val')
    plt.legend()
    plt.title("Loss")
    plt.savefig('ssa_gli_loss.png')
        
