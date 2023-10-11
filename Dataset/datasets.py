import torch
import numpy as np
import skimage.io as io
import skimage.transform as trans
import os
from torch.utils.data import Dataset
import random
import pandas as pd
from PIL import Image
from glob import glob
from tqdm import trange
# import cv2
from skimage.io import imread, imsave

random.seed(2333)

class CelebA(Dataset):
    def __init__(self, mode, gt_root, img_root, mask_root, file_list, sizes, mask_type, transform=None, mask_transform=None):
        """
        Lite way, 
        only for 256 256 input imgs. 
        rect: if True, __getitem__ return the rect value and gt
        """
        self.mode = mode
        self.files = file_list
        self.gt_root = gt_root
        self.img_root = img_root
        self.mask_root = mask_root
        # self.augmentation = augmentation
        self.transform = transform
        self.mask_transform = mask_transform
        self.sizes = sizes # saving the size information.
        # self.rect = rect   # 
        self.mask_type = mask_type
        # self.N_mask = len(mask_root)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        filename = self.files[index]

        if self.mask_type == "cnt_mask":
            gt_path = os.path.join(self.gt_root, filename)

            gt_data = io.imread(gt_path)
            gt_data = trans.resize(gt_data, self.sizes, order=0)
            if self.mode == "train":
                degree = 90 * np.random.choice([0, 1, 2, 3], 1)[0]
                gt_data = trans.rotate(gt_data, degree)
            gt_data = self.transform(gt_data)
            mask_data = torch.rand(1, self.sizes[0], self.sizes[1])
            img_data = torch.rand(3, self.sizes[0], self.sizes[1])
            

        elif self.mask_type == "face_mask":  
# TODO: pay attention to the invalid images
            img_path = os.path.join(self.img_root, filename)
            
            gt_path = os.path.join(self.gt_root, filename.split('.')[0]+'.jpg')
            mask_path = os.path.join(self.mask_root, filename)

            gt_data = io.imread(gt_path)
            gt_data = trans.resize(gt_data, self.sizes, order=0)

            img_data = io.imread(img_path)
            img_data = trans.resize(img_data, self.sizes, order=0)

            mask_data = io.imread(mask_path, as_gray=True)
            mask_data = trans.resize(mask_data, self.sizes, order=0)
            mask_data = np.expand_dims(mask_data, axis=2)

            # transform:
            comp_data = np.concatenate([img_data, mask_data, gt_data], axis=-1)
            comp_data = trans.resize(comp_data, self.sizes, order=0)
            if self.mode == "train":
                degree = 90 * np.random.choice([0, 1, 2, 3], 1)[0]
                comp_data = trans.rotate(comp_data, degree)

            img_data, mask_data, gt_data = np.split(comp_data, [3, 4], axis=-1)

            mask_data = self.mask_transform(mask_data)
            gt_data = self.transform(gt_data)
            img_data = self.transform(img_data)
        
        elif self.mask_type == "irr_mask":
            

            gt_path = os.path.join(self.gt_root, filename)

            gt_data = io.imread(gt_path)
            gt_data = trans.resize(gt_data, self.sizes, order=0)

            img_data = torch.rand(self.sizes[0], self.sizes[1], 3) # [256, 256, 3]


            mask_ratio_list = ["10_20", "20_30", "30_40", "40_50"]
            mask_ratio = random.choice(mask_ratio_list)
            self.mask_ratio_root = os.path.join(self.mask_root, mask_ratio)

            self.mask_paths = glob('{:s}/*.png'.format(self.mask_ratio_root))
            self.N_mask = len(self.mask_paths)

            mask_data = io.imread(self.mask_paths[random.randint(0, self.N_mask - 1)], as_gray=True)
            mask_data = trans.resize(mask_data, self.sizes)
            mask_data = np.expand_dims(mask_data, axis=2)
            # mask_data = self.mask_transform(mask_data)
            
            comp_data = np.concatenate([img_data, mask_data, gt_data], axis=-1)
            comp_data = trans.resize(comp_data, self.sizes, order=0)
            if self.mode == "train":
                degree = 90 * np.random.choice([0, 1, 2, 3], 1)[0]
                comp_data = trans.rotate(comp_data, degree)

            img_data, mask_data, gt_data = np.split(comp_data, [3, 4], axis=-1)

            mask_data = self.mask_transform(mask_data)
            gt_data = self.transform(gt_data)
            img_data = self.transform(img_data)
        
        else:
            raise ValueError("Mask_type [%s] not recognized. Please choose among ['face_mask', 'cnt_mask', 'irr_mask']  " % self.mask_type)

       
        return gt_data, img_data, mask_data

    def __rect_mask(self):
        low, high, full = self.sizes
        rect_size = np.random.choice(high-low+1) + low
        assert rect_size >= low and rect_size <= high, "value error"

        top_l_x = np.random.choice(full - rect_size)
        top_l_y = np.random.choice(full - rect_size)

        mask = np.zeros((full, full, 1), dtype=float)
        mask[top_l_y:top_l_y+rect_size, top_l_x:top_l_x+rect_size, :] = 1.0

        return mask

