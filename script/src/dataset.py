import os
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import numpy as np
import torch

relabel = {"other":0, "cd4":1, "cd8":2, "macrophage":3,"bcell":4}

class ImmData(Dataset):
    def __init__(self, meta, transform=None,use_dec = True):
        self.transform = transform
        self.meta = meta.reset_index(drop=True)
        self.use_dec = use_dec
    def __len__(self):
        return len(self.meta)
    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        dirs = f"/home/data/work_zzx/cell_dect/Immuno_AI/{row['patient']}/HE/"
        img_path = dirs+row["mask"]
        img_rgb = Image.open(img_path)
        img_rgb = img_rgb.convert('RGB')
        label_name = row["class"]
        label = relabel[label_name]
        if self.transform is not None:
            img_rgb = self.transform(img_rgb)
            if self.use_dec:
                mask_dir = f"/home/xieyj/work_zzx/cell_dect/Immuno_AI/{row['patient']}/mask/"
                mask_path = mask_dir+row["mask"]
                mask_res = np.array(Image.open(mask_path)) 
                mask_res = torch.from_numpy(mask_res).float()
                mask_res = mask_res.unsqueeze(0)
                img_rgb = img_rgb * mask_res      
        return img_rgb, label




#6848454  


# import pandas as pd
# import numpy as np

# meta = pd.read_csv("/home/data/work_zzx/cell_dect/Immuno_AI/curated_ImmunoctoV1_refinement2.csv")


import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from torchvision import transforms
from Utils import ColourAugment  # Assumed you have this utility



class CellSamdataset(Dataset):
    def __init__(self, tile_dataset, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.meta = tile_dataset
        self.inference = False

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        img_dirs = f"/gluster-data/home/xieyj/data/work_zzx/cell_dect/Immuno_AI/{row['patient']}/HE/"
        img_path = img_dirs + row["mask"]

        mask_dir = f"/gluster-data/home/xieyj/data/work_zzx/cell_dect/Immuno_AI/{row['patient']}/mask/"
        mask_path = mask_dir + row["mask"]

        label_name = row["class"]
        label = relabel[label_name]

        patch_HE = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
        patch_mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.float32)
        patch_mask = patch_mask[..., None]

        if np.max(patch_mask) > 0:
            patch_mask = patch_mask / np.max(patch_mask)

        patches = np.concatenate((patch_HE, patch_mask), axis=2).astype(np.float32)  # HWC
        patches = torch.from_numpy(patches).permute(2, 0, 1)  # CHW -> [4, H, W]

        if self.transform is not None:
            patches = self.transform(patches)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return patches, label
    

class CellSamdataset_our(Dataset):
    def __init__(self, tile_dataset, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.meta = tile_dataset
        self.inference = False

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        mask_path = row["cell_mask_crop_path"]
        img_path = row["cell_image_crop_path"]
        sample = row["global_instance_id"]
        patch_HE = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
        patch_mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.float32)
        patch_mask = patch_mask[..., None]

        if np.max(patch_mask) > 0:
            patch_mask = patch_mask / np.max(patch_mask)

        patches = np.concatenate((patch_HE, patch_mask), axis=2).astype(np.float32)  # HWC
        patches = torch.from_numpy(patches).permute(2, 0, 1)  # CHW -> [4, H, W]

        if self.transform is not None:
            patches = self.transform(patches)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return patches,sample
