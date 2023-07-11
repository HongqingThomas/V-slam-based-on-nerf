# ---------------------------------------------------------------------------
# DeepPruner: Learning Efficient Stereo Matching via Differentiable PatchMatch
#
# Copyright (c) 2019 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project. 
#
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Written by Shivam Duggal
# ---------------------------------------------------------------------------
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import skimage.io
import numpy as np
from .dataloader import kitti_submission_collector as ls
from .dataloader import preprocess
from PIL import Image
from .models.deeppruner import DeepPruner
from .models.config import config as config_args


class deeppredict():
    def __init__(self,
                 loadmodel,
                 ):
        self.loadmodel = loadmodel
        self.downsample_scale = config_args.cost_aggregator_scale * 8.0
        self.seed = 1 # 'random seed (default: 1)'
    
        torch.backends.cudnn.benchmark = True
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True

        self.model = DeepPruner()
        self.model = nn.DataParallel(self.model)
        self.model.cuda()
        self.state_dict = torch.load(self.loadmodel)
        self.model.load_state_dict(self.state_dict['state_dict'], strict=True)

    def run(self, imgL, imgR):
        # print("DeepPrunner imgL: ", imgL.shape)
        # Preprocess images
        imgL = imgL.detach().cpu().numpy()
        imgR = imgR.detach().cpu().numpy()
        processed = preprocess.get_transform()
        imgL = processed(imgL).numpy()
        imgR = processed(imgR).numpy()
        imgL = np.reshape(imgL, [1, 3, imgL.shape[1], imgL.shape[2]])
        imgR = np.reshape(imgR, [1, 3, imgR.shape[1], imgR.shape[2]])
        # Padding Image
        w = imgL.shape[3]
        h = imgL.shape[2]
        dw = int(self.downsample_scale - (w%self.downsample_scale + (w%self.downsample_scale==0)*self.downsample_scale))
        dh = int(self.downsample_scale - (h%self.downsample_scale + (h%self.downsample_scale==0)*self.downsample_scale))
        top_pad = dh
        left_pad = dw
        imgL = np.lib.pad(imgL, ((0, 0), (0, 0), (top_pad, 0), (0, left_pad)), mode='constant', constant_values=0)
        imgR = np.lib.pad(imgR, ((0, 0), (0, 0), (top_pad, 0), (0, left_pad)), mode='constant', constant_values=0)

        # Predict
        self.model.eval()
        with torch.no_grad():
            imgL = Variable(torch.FloatTensor(imgL))
            imgR = Variable(torch.FloatTensor(imgR))
            imgL, imgR = imgL.cuda(), imgR.cuda()
            disparity = self.model(imgL, imgR)
            
        # disparity = disparity[:, top_pad:, :-left_pad].squeeze()
        # # disparity=disparity[0].cpu().numpy().squeeze()
        # disparity = disparity * 1000.0

        # _, _, dis_w  = disparity.shape
        # disparity1 = disparity[0, top_pad:, : dis_w-left_pad].squeeze()
        # disparity1=disparity1.cpu().numpy()
        # disparity1 = (disparity1 * 1000.0).astype('uint16')
        # # disparity = torch.from_numpy(disparity).cuda()
        # print("disparity 1: ",np.min(disparity1), np.max(disparity1))

        # For NiceSlam
        _, _, dis_w = disparity.shape
        disparity2 = disparity[0, top_pad:, : dis_w-left_pad].squeeze()
        disparity2 = (disparity2 * 1000.0)
        # print("disparity 2: ",torch.min(disparity2), torch.max(disparity2))

        return disparity2

# if __name__ == '__main__':
#     # Input
#     datapath = "/data/parkfusion/training_directory_stereo_2015"
#     current_directory = os.getcwd()
#     test_left_img, test_right_img = ls.datacollector(current_directory+datapath)
#     left_image_path = test_left_img[0]
#     right_image_path = test_right_img[0]
#     imgL = torch.from_numpy(np.asarray(Image.open(left_image_path))).cuda()
#     imgR = torch.from_numpy(np.asarray(Image.open(right_image_path))).cuda()


#     # Predict
#     loadmodel = "./model/kiti_testfinetune_119.tar"
#     deep_predict = deeppredict(loadmodel)
#     disparity = deep_predict.run(imgL, imgR)

#     # print("disparity: ",disparity,  disparity.shape)
#     print("disparity 2: ",np.min(disparity), np.max(disparity))

#     # Save
#     save_dir = "./save_dirct"
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#     depth_name = left_image_path.split('/')[-1].split('.')[0]+'.png'
#     skimage.io.imsave(os.path.join(save_dir, depth_name), disparity)

