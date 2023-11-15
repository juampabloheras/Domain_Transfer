#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 21:42:21 2023

@author: juampablo
Email: jehr@uw.edu

"""
# Import packages

import torch.nn as nn
import numpy as np




# Define domain classifier model

class domain_classifier(nn.Module):
    def __init__(self):
        super(domain_classifier, self).__init__()
        
        # Define the 3D convolution layers
        self.conv1 = nn.Conv3d(4, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # Define max-pooling layers
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Define fully connected layers
        self.fc1 = nn.Linear(49152, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 1)  # 2 output classes (class 0 and class 1)
        
        # Define activation functions
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Convolutional layers

        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = self.relu(self.conv4(x))
        x = self.pool(x)
        
        # Flatten the feature maps
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x




    




