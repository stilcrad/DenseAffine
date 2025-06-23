from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append(".")
import os
import time

from PIL import Image
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tqdm import tqdm
import math
import torch.nn.functional as F

from copy import deepcopy

from hesaffnet.SparseImgRepresenter import ScaleSpaceAffinePatchExtractor
from hesaffnet.LAF import normalizeLAFs, denormalizeLAFs, LAFs2ell, abc2A, convertLAFs_to_A23format
from hesaffnet.Utils import line_prepender, batched_forward
from hesaffnet.architectures import AffNetFast
# from hesaffnet.HardNet import HardNet
# from library import opt


USE_CUDA = True
# USE_CUDA = torch.cuda.is_available()
WRITE_IMGS_DEBUG = False

AffNetPix = AffNetFast(PS = 32)
weightd_fname = "/home/xxx/project/python/DKM-main/hesaffnet/pretrained/AffNet.pth"
checkpoint = torch.load(weightd_fname)
AffNetPix.load_state_dict(checkpoint['state_dict'])
AffNetPix.eval()

# HardNetDescriptor = HardNet()
# # model_weights = opt.bindir+'pretrained/HardNet++.pth'
# model_weights = "/home/xxx/project/python/locate-master/hesaffnet/pretrained/HardNet++.pth"
# if USE_CUDA:
#     hncheckpoint = torch.load(model_weights, map_location='cuda:0')
# else:
#     hncheckpoint = torch.load(model_weights, map_location='cpu')
# HardNetDescriptor.load_state_dict(hncheckpoint['state_dict'])
# HardNetDescriptor.eval()

if USE_CUDA:
    AffNetPix = AffNetPix.cuda()
    # HardNetDescriptor = HardNetDescriptor.cuda()
else:
    AffNetPix = AffNetPix.cpu()
    # HardNetDescriptor = HardNetDescriptor.cpu()



from hesaffnet.library import SaveImageWithKeys, packSIFTOctave
import cv2



def AffNetHardNet_describeFromKeys(img_np, KPlist,patch_list):
    img = torch.autograd.Variable(torch.from_numpy(np.array(img_np).astype(np.float32)), volatile = True)
    img = img.view(1, 1, img.size(0),img.size(1))
    # HessianAffine = ScaleSpaceAffinePatchExtractor( mrSize = 5.192, num_features = 0, border = 0, num_Baum_iters = 0)
    # if USE_CUDA:
    #     HessianAffine = HessianAffine.cuda()
    #     img = img.cuda()
    #     HessianAffine.createScaleSpace(img) # to generate scale pyramids and stuff
    # descriptors = []
    Alist = []
    n=0
    # for patch_np in patches:
    for i , kp in enumerate(KPlist):
        x = torch.tensor(kp[0])
        y = torch.tensor(kp[1])
        LAFs = normalizeLAFs( torch.tensor([[AffNetPix.PS/2, 0, x], [0, AffNetPix.PS/2, y]]).reshape(1,2,3), img.size(3), img.size(2) )
        patch = patch_list[i].reshape(1,1,32,32).cpu()
        if WRITE_IMGS_DEBUG:
            SaveImageWithKeys(patch.detach().cpu().numpy().reshape([32,32]), [], 'p2/'+str(n)+'.png' )
        if USE_CUDA:
            # or ---> A = AffNetPix(subpatches.cuda()).cpu()
                A = batched_forward(AffNetPix, patch.cuda(), 256).cpu()
        else:
                A = AffNetPix(patch)
        new_LAFs = torch.cat([torch.bmm(A,LAFs[:,:,0:2]), LAFs[:,:,2:] ], dim =2)
        dLAFs = denormalizeLAFs(new_LAFs, img.size(3), img.size(2))
        # patchaff = HessianAffine.extract_patches_from_pyr(dLAFs, PS = 32)
        if WRITE_IMGS_DEBUG:
            # SaveImageWithKeys(patchaff.detach().cpu().numpy().reshape([32,32]), [], 'p1/'+str(n)+'.png' )
            SaveImageWithKeys(img_np, [kp], 'im1/'+str(n)+'.png' )
        Alist.append( convertLAFs_to_A23format( dLAFs.detach().cpu().numpy().astype(np.float32) ) )
    n=n+1
    return  Alist


def AffNetHardNet_describeFromKeys_justAFFnetoutput(img_np, KPlist,patch_list):
    img = torch.autograd.Variable(torch.from_numpy(np.array(img_np).astype(np.float32)), volatile = True)
    img = img.view(1, 1, img.size(0),img.size(1))
    # HessianAffine = ScaleSpaceAffinePatchExtractor( mrSize = 5.192, num_features = 0, border = 0, num_Baum_iters = 0)
    # if USE_CUDA:
    #     HessianAffine = HessianAffine.cuda()
    #     img = img.cuda()
    #     HessianAffine.createScaleSpace(img) # to generate scale pyramids and stuff
    Alist = []
    n=0
    # for patch_np in patches:
    for i , kp in enumerate(KPlist):
        # x = torch.tensor(kp[0])
        # y = torch.tensor(kp[1])
        # LAFs = normalizeLAFs( torch.tensor([[AffNetPix.PS/2, 0, x], [0, AffNetPix.PS/2, y]]).reshape(1,2,3), img.size(3), img.size(2) )
        patch = patch_list[i].reshape(1,1,32,32).cpu()
        if USE_CUDA:
            # or ---> A = AffNetPix(subpatches.cuda()).cpu()
                A = batched_forward(AffNetPix, patch.cuda(), 256).cpu()
        else:
                A = AffNetPix(patch)
        if i == 0:
            A_ = A
        else:
            A_ = torch.cat((A_,A),dim=0)

    # return    torch.tensor(A_,dtype=torch.double)
    return    torch.tensor(A_,dtype=torch.double)


