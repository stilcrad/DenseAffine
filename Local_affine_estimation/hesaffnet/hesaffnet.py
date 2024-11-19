from __future__ import print_function
import torch
import numpy as np
import sys
sys.path.append("")
from Local_affine_estimation.hesaffnet.LAF import normalizeLAFs, denormalizeLAFs, convertLAFs_to_A23format
from Local_affine_estimation.hesaffnet.Utils import batched_forward
from Local_affine_estimation.hesaffnet.architectures import AffNetFast

USE_CUDA = True
# USE_CUDA = torch.cuda.is_available()
WRITE_IMGS_DEBUG = False

AffNetPix = AffNetFast(PS = 32)
weightd_fname = "/home/xxx/project/python/DKM-main/hesaffnet/pretrained/AffNet.pth"
checkpoint = torch.load(weightd_fname)
AffNetPix.load_state_dict(checkpoint['state_dict'])
AffNetPix.eval()
if USE_CUDA:
    AffNetPix = AffNetPix.cuda()
    # HardNetDescriptor = HardNetDescriptor.cuda()
else:
    AffNetPix = AffNetPix.cpu()
    # HardNetDescriptor = HardNetDescriptor.cpu()



from Local_affine_estimation.hesaffnet.library import SaveImageWithKeys


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


