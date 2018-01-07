# -*- coding: utf-8 -*-
# Created on Mon Jan 01 2018 9:12:15
# Author: WuLC
# EMail: liangchaowu5@gmail.com

import os
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms as T

class FaceExpression(data.Dataset):
    def __init__(self, root, transforms = None, train=True, test=False):
        """get path for all images and split for training and evaluate"""
        self.test = test
        self.transforms = transforms

        if root[-1] != '/':
            root += '/'
        imgs = [root + img for img in os.listdir(root)]
        num_imgs = len(imgs)

        # shuffle images
        np.random.seed(435)
        np.random.shuffle(imgs)

        if self.test:
            self.imgs = imgs
        elif train:
            self.imgs = imgs[:int(num_imgs*0.8)]
        else:
            self.imgs = imgs[int(num_imgs*0.8):]
        
        if self.transforms == None:
            normalize = T.Normalize(mean = [0.485, 0.456, 0.406],
                                    std = [0.229, 0.224, 0.225])
            if train:
                self.transforms = T.Compose([
                T.Resize(256),
                T.RandomHorizontalFlip(),
                T.CenterCrop(227),
                T.RandomCrop(227),
                T.ToTensor(),
                normalize
                ])
            else:
                self.transforms = T.Compose([
                T.Resize(256),
                T.CenterCrop(227),
                T.ToTensor(),
                normalize
                ])
        
    def __getitem__(self, index):
        mapping =  {'NE':0, 'AN':1, 'SU':2, 'DI':3, 'FE':4, 'HA':5, 'SA':6}
        img_path = self.imgs[index]
        label = mapping[img_path.split('/')[-1].split('_')[0]]
        data = Image.open(img_path)
        data = self.transforms(data)
        return data, label

    
    def __len__(self):
        return len(self.imgs)




