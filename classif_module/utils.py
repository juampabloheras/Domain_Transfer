#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 21:42:21 2023

@author: juampablo
Email: jehr@uw.edu

"""

# File to store string to function dictionaries for arg parsing
# Update whenever new loss or model functions are needed

import torch.nn as nn
import new_losses as lf # in same dir
import losses as lf2 # in same directory

import torch
import numpy as np
import glob, os

from torch.utils.data import DataLoader, ConcatDataset
from data import datasets
from data import trans
from torchvision import transforms

from torch.utils.data import random_split
from math import ceil



LOSS_STR_TO_FUNC = {
    # 'mse': nn.MSELoss(),
    # 'cross-entropy': nn.CrossEntropyLoss(),
    # 'mask-regulizer': lf2.Maskregulizer(),
    # 'edge-loss': EdgeLoss3D.GMELoss3D(),
    # 'dice': lf.DiceLoss(),
    # 'focal': lf.FocalLoss(),
    # 'BCE': nn.BCELoss()
    # 'hd'
}

# MODEL_STR_TO_FUNC = {
#     'attention-unet': attention_unet.attention_unet()
# }

AFFINE = np.array([[ -1.,   0.,   0.,  -0.],
                   [  0.,  -1.,   0., 239.],
                   [  0.,   0.,   1.,   0.],
                   [  0.,   0.,   0. ,  1.]])

# New function
def split_seg_labels(seg):
    # Split the segmentation labels into 3 channels
    seg3 = torch.zeros(seg.shape + (3,), dtype=torch.float)
    seg3[:, 0, :, :, :] = (seg == 1).float()
    seg3[:, 1, :, :, :] = (seg == 2).float()
    seg3[:, 2, :, :, :] = (seg == 3).float()
    return seg3


def make_data_loader(in_dir, shuffle, batch_size=1):
    # Process and load the training data
    # WARNING - be careful with transforms as they are applied to ground truth labels as well as the 4 modalities
    train_composed = transforms.Compose([trans.CenterCropBySize([128,192,128]), 
                                              trans.NumpyType((np.float32, np.float32,np.float32, np.float32,np.float32)),
                                              ])                   
    
    #  Load datasets and make dataloader               
    data_paths = in_dir if isinstance(in_dir, list) else [in_dir] # Makes sure in_dir is a list
    datasets_list = [datasets.MEN_SSA_Dataset(glob.glob(os.path.join(path, '*.pkl')), transforms=train_composed) for path in data_paths] # Compiles datasets into a list
    compiled_data_set = ConcatDataset(datasets_list)         # Turns list of compiled datasets to a single dataset      
                    
    data_loader = DataLoader(compiled_data_set, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True) # Make dataloader

    return data_loader 


def make_train_val_data_loader(in_dir,  train_val_split, shuffle=True, batch_size =1):
    train_composed = transforms.Compose([trans.CenterCropBySize([128,192,128]), 
                                                  trans.NumpyType((np.float32, np.float32,np.float32, np.float32,np.float32)),
                                                  ])     
    data_paths = in_dir if isinstance(in_dir, list) else [in_dir] # Makes sure in_dir is a list
    
    ##  Stratified sampling procedure to ensure class balance in train-val splits ##
    datasets_list = [{'dataset_'+path:datasets.MEN_SSA_Dataset(glob.glob(os.path.join(path, '*.pkl')), transforms=train_composed)} for path in data_paths] # Compiles datasets into a list
    
    # Make two lists containing dictionaries containing train and val datasets respectively. Each dataset will be split individually according to train_val_split.
    train_dsets = []
    val_dsets = []
    for dataset in datasets_list:
        length = len(dataset)
        train_count = ceil(train_val_split*length) # calculates number of files in training set
        train_split, val_split =  random_split(dataset.values(), [train_count, length-train_count]) # train_val_split is len(train)/len(data)
        train_dsets.append({'train_from_'+next(iter(dataset)):train_split}) # makes a dictionary where key is train_from_path_to_dataset, value is dataset 
        val_dsets.append({'val_from_'+next(iter(dataset)):val_split}) # similar to above but for val
    compiled_train = ConcatDataset(train_dsets)
    compiled_val = ConcatDataset(val_dsets)
    
    train_data_loader = DataLoader(compiled_train, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True) # Make train dataloader from compiled train set
    val_data_loader = DataLoader(compiled_val, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True) # Make val dataloader from compiled val set
    
    return train_data_loader, val_data_loader
        
        
        
        
        
