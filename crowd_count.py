import torch
import torch.nn as nn
import torch.nn.functional as F

import network
from models import CMTL



# import os
# import numpy as np
# import sys
# from datetime import datetime
# from data_loader import ImageDataLoader
# from timer import Timer
# import utils
# from termcolor import cprint
# from tensorboardX import SummaryWriter



class CrowdCounter(nn.Module):
    def __init__(self, ce_weights=None):
        super(CrowdCounter, self).__init__()        
        self.CCN = CMTL()
        if ce_weights is not None:
            ce_weights = torch.Tensor(ce_weights)
            ce_weights = ce_weights.cuda()
        self.loss_mse_fn = nn.MSELoss()
        self.loss_bce_fn = nn.BCELoss(weight=ce_weights)
        
    @property
    def loss(self):
        return self.loss_mse + 0.0001*self.cross_entropy
    
    def forward(self,  im_data, gt_data=None, gt_cls_label=None, ce_weights=None):        
        im_data = network.np_to_variable(im_data, is_cuda=True, is_training=self.training)
        density_map, density_cls_score = self.CCN(im_data)
        density_cls_prob = F.softmax(density_cls_score)
        
        if self.training:
            gt_data = network.np_to_variable(gt_data, is_cuda=True, is_training=self.training)
            gt_cls_label = network.np_to_variable(gt_cls_label, is_cuda=True, is_training=self.training,dtype=torch.FloatTensor)
            self.loss_mse, self.cross_entropy = self.build_loss(density_map, density_cls_prob, gt_data, gt_cls_label, ce_weights)

            
        return density_map
    
    def build_loss(self, density_map, density_cls_score, gt_data, gt_cls_label, ce_weights):
        loss_mse = self.loss_mse_fn(density_map, gt_data)        
        ce_weights = torch.Tensor(ce_weights)
        ce_weights = ce_weights.cuda()
        cross_entropy = self.loss_bce_fn(density_cls_score, gt_cls_label)
        return loss_mse, cross_entropy

# train_path = 'dataset/Shanghai/formatted_trainval/shanghaitech_part_A_patches_9/train'
# train_gt_path = 'dataset/Shanghai/formatted_trainval/shanghaitech_part_A_patches_9/train_den'
# data_loader = ImageDataLoader(train_path, train_gt_path, shuffle=True, gt_downsample=False, pre_load=True)
# class_wts = data_loader.get_classifier_weights()
# model = CrowdCounter(ce_weights=class_wts)
# dummy_input = torch.rand(13, 1, 28, 28)
# with SummaryWriter(comment='cascaded') as w:
#     w.add_graph(model, (dummy_input, ))