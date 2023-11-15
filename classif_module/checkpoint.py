#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 12:39:14 2023

@author: juampablo
Email: jehr@uw.edu

"""

import os
import torch
import glob

# Before training - set up directories and paths.
# def load_checkpoint(model, optimizer, out_dir = None):
    
#     if out_dir is None:
#         out_dir = os.getcwd()
    
    
#     latest_ckpt_path = os.path.join(out_dir, 'latest_ckpt.pth.tar')  # Latest checkpoints directory
#     saved_ckpts_dir = os.path.join(out_dir, 'saved_ckpts')   # Saved checkpoints directory

#     if not os.path.exists(saved_ckpts_dir):
#         os.makedirs(saved_ckpts_dir)
#         os.system(f'chmod a+rwx {saved_ckpts_dir}')
    
#     if not os.path.exists(latest_ckpt_path):
#         epoch_start = 0
#         print('No training checkpoint found. Will train from beginning.')         
#     else:
#         print('Training checkpoint found. Loading checkpoint...')
#         print(latest_ckpt_path)
#         checkpoint = torch.load(latest_ckpt_path)
#         epoch_start = checkpoint['epoch'] + 1
#         model.load_state_dict(checkpoint['model_sd'])
#         optimizer.load_state_dict(checkpoint['optim_sd'])
#         print(f'Checkpoint loaded. Will continue training from epoch {epoch_start}.')
        
#     return epoch_start, model, optimizer, latest_ckpt_path, saved_ckpts_dir, out_dir


def load_checkpoint(model, optimizer, out_dir=None):
    
    if out_dir is None:
        out_dir = os.getcwd()
    
    latest_ckpt_path = os.path.join(out_dir, 'latest_ckpt.pth.tar')
    saved_ckpts_dir = os.path.join(out_dir, 'saved_ckpts')
    
    if not os.path.exists(saved_ckpts_dir):
        os.makedirs(saved_ckpts_dir)
        os.system(f'chmod a+rwx {saved_ckpts_dir}')
    
    if not os.path.exists(latest_ckpt_path):
        epoch_start = 0
        epoch_loss = []
        epoch_loss_val = []
        print('No training checkpoint found. Will train from the beginning.')
        return epoch_start, model, optimizer, latest_ckpt_path, saved_ckpts_dir, out_dir, epoch_loss, epoch_loss_val
    else:
        print('Training checkpoint found. Loading checkpoint...')
        print(latest_ckpt_path)
        try:
            checkpoint = torch.load(latest_ckpt_path)
            epoch_start = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model_sd'])
            optimizer.load_state_dict(checkpoint['optim_sd'])
            epoch_loss = checkpoint['epoch_loss']
            epoch_loss_val = checkpoint['epoch_loss_val']
            print(f'Checkpoint loaded. Will continue training from epoch {epoch_start}.')
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Attempting to load the second-latest checkpoint...")
            ckpt_list = sorted(glob.glob(os.path.join(saved_ckpts_dir, '*.pth.tar')))
            if len(ckpt_list) < 2:
                print("No additional checkpoint available to load.")
                epoch_start = 0
                epoch_loss = []
                epoch_loss_val = []
                return epoch_start, model, optimizer, latest_ckpt_path, saved_ckpts_dir, out_dir, epoch_loss, epoch_loss_val
            second_latest_ckpt_path = ckpt_list[-2]
            print(f"Loading second-latest checkpoint: {second_latest_ckpt_path}")
            checkpoint = torch.load(second_latest_ckpt_path)
            epoch_start = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model_sd'])
            optimizer.load_state_dict(checkpoint['optim_sd'])
            epoch_loss = checkpoint['epoch_loss']
            epoch_loss_val = checkpoint['epoch_loss_val']
            print(f"Second-latest checkpoint loaded. Will continue training from epoch {epoch_start}.")
        
    return epoch_start, model, optimizer, latest_ckpt_path, saved_ckpts_dir, out_dir, epoch_loss, epoch_loss_val


def save_checkpoint(epoch, save_interval, checkpoint, saved_ckpts_dir):
    if epoch % save_interval == 0:
        torch.save(checkpoint, os.path.join(saved_ckpts_dir, f'epoch{epoch}.pth.tar'))
    print('Checkpoint saved successfully.')