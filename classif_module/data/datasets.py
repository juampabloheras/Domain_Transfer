from torch.utils.data import Dataset, ConcatDataset
import os
import nibabel as nib
from preprocess import znorm_rescale
import numpy as np
import torch
from .data_utils import pkload
from torchvision import transforms
import pickle

from data import trans
import numpy as np


class MEN_SSA_Dataset(Dataset):
    def __init__(self, data_path, transforms, normalized=True, gt_provided=True):
        self.paths = data_path
        self.transforms = transforms
        self.normalized = normalized
        self.gt_provided = gt_provided

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        x1, x2, x3, x4, y1 = pkload(path) if self.gt_provided else pkload(path)[:4]

        classification = 0 if '-GLI-' in path else (1 if '-SSA-' in path else None)
        print("classification: ", classification)

        # Add an extra dimension to variables    
        x1, x2, x3, x4 = [x[None, ...] for x in (x1, x2, x3, x4)]
        y1 = y1[None, ...] if self.gt_provided else None

        # Transform variables
        data_list = self.transforms([x1, x2, x3, x4, y1]) if self.gt_provided else self.transforms([x1, x2, x3, x4])

        # Normalize ONLY the modalities (if needed)
        if self.normalized:
            norm_tf = transforms.Compose([trans.Normalize0_1()])
            data_list = norm_tf(data_list)

        # Convert to Torch tensors
        data_list = [torch.from_numpy(np.ascontiguousarray(item)) for item in data_list if item is not None]

        # Extract case information from the filename
        filename = path.split('/')[-1]
        case_info = tuple(filename.split('.')[0].split('-')[2:4])  # (case_id, timepoint)

        return case_info, data_list, classification



