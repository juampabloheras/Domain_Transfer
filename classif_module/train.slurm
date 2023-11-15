#!/bin/bash

#SBATCH --job-name=BraTS_test
#SBATCH --mail-user=jehr@uw.edu ## CHANGE TO YOUR USERNAME
#SBATCH --mail-type=ALL

#SBATCH --gpus-per-node=a40:1
#SBATCH --account=kurtlab
#SBATCH --partition=ckpt ## CHANGE TO CORRECT PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=13
#SBATCH --mem=238G
#SBATCH --time=50:00:00 ## Max runtime in DD-HH:MM:SS format.

## Change to your own repo base folder
#SBATCH --chdir=/gscratch/kurtlab/juampablo/classif_module
#SBATCH --export=all
## Log files should be inside OUT_DIR - need to add this manually

#SBATCH --output=/gscratch/kurtlab/juampablo/classif_module/outputs/output_train.txt
#SBATCH --error=/gscratch/kurtlab/juampablo/classif_module/outputs/error_train.txt

source ~/.bashrc
source activate undergraddl

# Your programs to run.

# WARNING: do not leave spaces around equal sign


python3 train.py 