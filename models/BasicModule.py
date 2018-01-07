# -*- coding: utf-8 -*-
# Created on Tue Jan 02 2018 9:1:13
# Author: WuLC
# EMail: liangchaowu5@gmail.com

import torch
import time

class BasicModule(torch.nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self)) # 默认名字
    
    
    def load(self, path):
        self.load_state_dict(torch.load(path)) # just load the weight

    
    def save(self, model_path = None):
        if not model_path:
            model_path = './checkpoints/{0}_{1}.pth'.format(self.model_name, time.strftime('%m%d_%H%M%S'))
        torch.save(self.state_dict(), model_path)
        return model_path


class Flat(torch.nn.Module):
    '''
    把输入reshape成（batch_size,dim_length）
    '''

    def __init__(self):
        super(Flat, self).__init__()
        #self.size = size

    def forward(self, x):
        return x.view(x.size(0), -1)    
    