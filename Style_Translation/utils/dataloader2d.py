import torch.utils.data as data
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import random
import cv2
import os
import matplotlib.pyplot as plt

class I2IDataset(data.Dataset):
    def __init__(self, train=True,A_root='datasets/CT_2d',B_root='datasets/MR_2d'):
        self.is_train=train
        self.A_root=A_root
        self.B_root=B_root
        self.A_imgs, self.B_imgs = self.load_train_data()
        self.gan_aug = A.Compose([
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5,
                               border_mode=cv2.BORDER_CONSTANT),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=(0.5), std=(0.5), max_pixel_value=1.0),
            ToTensorV2()
        ])

    def load_train_data(self):
        A_imgs = []
        print(self.A_root)
        for f in os.listdir(self.A_root):
            A_imgs.append(os.path.join(self.A_root, f))
            if len(A_imgs) == 2000:
                break
        
        B_imgs = []
        for f in os.listdir(self.B_root):
            B_imgs.append(os.path.join(self.B_root, f))
        return A_imgs,B_imgs

    def __getitem__(self, index):
        A_img = np.load(self.A_imgs[index])
        B_index = random.randint(0, len(self.B_imgs) - 1)
        
        B_img = np.load(self.B_imgs[B_index])
        print(A_img.shape)
        print(B_img.shape)
        A_img = self.gan_aug(image=A_img)["image"]
        B_img = self.gan_aug(image=B_img)["image"]

        
        return {'A_img': A_img, 'B_img': B_img}

    def __len__(self):
        return len(self.A_imgs)
    