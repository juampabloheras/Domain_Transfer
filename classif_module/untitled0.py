#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 15:54:38 2023

@author: juampablo
Email: jehr@uw.edu

"""

import matplotlib.pyplot as plt
import torch
import numpy as np

# Example: dataloader containing MRI images

from utils import make_data_loader
TRAIN_DIR_MEN='/Users/juampablo/Desktop/Kurtlab_A23/BraTS_images/' #string
TRAIN_DIR_SSA='/Users/juampablo/Desktop/Kurtlab_A23/Africa_files' #string
in_dir = [TRAIN_DIR_MEN, TRAIN_DIR_SSA]

data_loader = make_data_loader(in_dir,shuffle=True,batch_size =2)

# Assuming data_loader is an iterable
batch_count = 0
for _, batch, labels in data_loader:
    batch_count +=1
    images = batch  # Assuming the batch contains MRI images
    print("Images shape", np.shape(images))
    print(np.shape(labels))
    # Select the first MRI image from the batch
    sample_image1 = images[3]  # Assuming shape is [1, C, H, W, D]
    print("shape sample_image", np.shape(sample_image1))
    
    # Choose the index for the slice
    z_slice = 100  # Change this value to select a different slice
    
    # Extract and plot a slice from the MRI image
    for i in range(np.shape(sample_image1)[0]):
        plt.imshow(sample_image1[i, 0, :, :, z_slice], cmap='gray')  # Display a slice using grayscale colormap
        plt.title(f"Slice {z_slice} in Z-axis (batch {batch_count}, label {labels[i]})")
        plt.colorbar()
        plt.show()

print(batch_count)


#%%%
import pickle
import os
import torch.nn as nn
from model import domain_classifier


in_dir = [TRAIN_DIR_MEN, TRAIN_DIR_SSA]
train_loader = make_data_loader(in_dir,shuffle=True, batch_size = 2)
print("data loaded")
epoch_start = 0
num_epochs = 20
device = 'cpu'

model = domain_classifier() 
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


epoch_loss =[]
for epoch in range(epoch_start, num_epochs):
    model.train()
    running_loss = []
    batch_count = 0
    for _, data, labels in train_loader:
        batch_count +=1
        
        optimizer.zero_grad()
        
        orig = data
        print("Images shape", np.shape(orig))

        
        
        # Reformat data
        data = data[:-1] # remove segmentation map
        print(np.shape(labels))
        
        data = np.transpose(data, (1, 0, 3, 4, 5, 2))
        data = np.squeeze(data, axis=-1)
        data = torch.tensor(data)
        
        print("datashape:", np.shape(data))
        
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
        
        # loss = criterion(outputs, label.float())
        loss.backward()
        optimizer.step()
        
        running_loss.append(loss.detach().to('cpu'))
                
    print(f'Epoch {epoch} completed...')   
    epoch_loss.append(np.mean(running_loss))


plt.plot(epoch_loss)




    
    
