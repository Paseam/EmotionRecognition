# -*- coding: utf-8 -*-
# Created on Fri Jan 19 2018 21:54:33
# Author: WuLC
# EMail: liangchaowu5@gmail.com

import os
import random
from PIL import Image

import torchvision.transforms as T 
import visdom


train_img_dir = '/home/lc/FaceExpression/train/CKDEF/'
val_img_dir = '/home/lc/FaceExpression/validation/face/'

def random_img(img_dir):
    return img_dir + random.choice(os.listdir(img_dir))


normalize = T.Normalize(mean = [.5, .5, .5],
                        std = [.5, .5, .5])

trans = T.Compose([
    T.Resize(256),
    T.RandomHorizontalFlip(),
    # T.CenterCrop(227),
    T.RandomCrop(227),
    T.Grayscale(1),
    T.ToTensor(),
    #normalize
])

img_path = random_img(train_img_dir)
img = Image.open(img_path)
vis = visdom.Visdom()
for _ in range(10):
    vis.image(trans(img))



