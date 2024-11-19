# When default GPU is being used... prepare to use a second one
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import cv2
from matplotlib import pyplot as plt
from library import *
import time
from tensorflow.compat.v1.keras import layers
from tensorflow.compat.v1.keras.models import Model
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.keras.backend import set_session
from models import *

config = tf.ConfigProto(allow_soft_placement=True
#, device_count = {'CPU' : 1, 'GPU' : 1}
# , intra_op_parallelism_threads=4
)
config.gpu_options.per_process_gpu_memory_fraction = 0.1
tfsession = tf.Session(config=config)
set_session(tfsession)

#  VGG like network
vgg_input_shape = tuple([60,60]) + tuple([1])
MODEL_NAME = 'AID_simCos_BigDesc_dropout'
weights2load = '../model-data/model.'+MODEL_NAME+'_75.hdf5'
BigAIDmodel, sim_type = create_model(vgg_input_shape, None, model_name = MODEL_NAME, Norm=None, resume = True, ResumeFile = weights2load)
BigAIDmodel.trainable = False

geoestiMODEL_NAME = 'DA_Pts_dropout'
geoestiweights2load ='../model-data/model.'+geoestiMODEL_NAME+'_L1_75.hdf5'
LOCATEmodel = create_model(tuple([60,60,2]), tuple([16]), model_name = geoestiMODEL_NAME, Norm='L1', resume = True, ResumeFile = geoestiweights2load)
LOCATEmodel.trainable = False

import sys
# sys.path.append("hesaffnet")
sys.path.append(opt.bindir+"hesaffnet")
from hesaffnet import *


def RefineKplistsPyrPos(Qkp0, Tkp0, Tkplist, lambda_Q_to_T):
    ''' Refine all keypoints in Tkplist (keypoint list in the target image) in 
        terms of their position in the Gaussian Pyramid.
        That refining is done by the means of the estimated scale and the Gauss Pos of kp.
    '''        
    Q_o, Q_l, _, Q_xi = unpackSIFTOctave(Qkp0, XI=True)
    T_o, T_l, _, T_xi = unpackSIFTOctave(Tkp0, XI=True)
    params = [[T_o,T_l,o2,l2] for o2 in range(-1,3) for l2 in range(0, siftparams.nOctaveLayers+1)]
    mindiff = math.inf
    qmin = [T_o,T_l,0.0,0.0]
    for q in params:
        temp = abs( lambda_Q_to_T - (pow(2.0,q[2])*pow(2.0, (q[3]+T_xi)/siftparams.nOctaveLayers)) )
        if mindiff>temp:
            mindiff = temp
            qmin = q
    q = qmin
    rTkplist = []
    for kp in Tkplist:
        kp.size = Tkp0.size*(pow(2.0,q[2])*pow(2.0, (q[3]+T_xi)/siftparams.nOctaveLayers))/(pow(2.0,T_o)*pow(2.0, (T_l+T_xi)/siftparams.nOctaveLayers))
        kp.octave = packSIFTOctave(q[2],q[3], xi = T_xi)
        rTkplist.append(kp)        
    return rTkplist


def GrowMatches(KPlist1, pyr1, KPlist2, pyr2, Qmap, growed_matches, growing_matches, img1, img2, LocalGeoEsti='8pts'):
    if LocalGeoEsti=='8pts':
        useAffnet = False
        useIdentity = False
    elif LocalGeoEsti=='affnet':
        useAffnet=True
        useIdentity = False
    else:
        useAffnet=True
        useIdentity = True
    patches1, _, _ = ComputePatches(KPlist1,pyr1, border_mode=cv2.BORDER_REFLECT )
    patches2, _, _ = ComputePatches(KPlist2,pyr2, border_mode=cv2.BORDER_REFLECT )
    pxl_radius = 20
    w, h = 60, 60
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    Identity = np.float32([[1, 0, 0], [0, 1, 0]])
    if not useAffnet:
        bP = np.zeros(shape=tuple([len(growing_matches),60,60,2]), dtype = np.float32)
    Akp1, Akp2 = [], []
    keys2seek = []
    for n in range(0,len(growing_matches)):
        idx1 = growing_matches[n].queryIdx
        idx2 = growing_matches[n].trainIdx        
        if not useAffnet:
            bP[n,:,:,:] = np.dstack((patches1[idx1]/255.0, patches2[idx2]/255.0))
        Akp1.append(kp2LocalAffine(KPlist1[idx1]))
        Akp2.append(kp2LocalAffine(KPlist2[idx2]))
        InPatchKeys2seek = [cv2.KeyPoint(x = w/2+pxl_radius*i - pxl_radius/2, y = h/2 +pxl_radius*j - pxl_radius/2,
            size =  2.0, 
            angle =  0.0,
            response =  KPlist1[idx1].response, octave =  packSIFTOctave(-1,0),
            class_id =  i*2+j) for i in range(0,2) for j in range(0,2)]
        InPatchKeys2seek.append(cv2.KeyPoint(x = w/2, y = h/2,
            size =  2.0, 
            angle =  0.0,
            response =  KPlist1[idx1].response, octave =  packSIFTOctave(-1,0),
            class_id =  -1))
        temp = AffineKPcoor(InPatchKeys2seek, cv2.invertAffineTransform(Akp1[n]))
        temp, _ = Filter_Affine_In_Rect(temp,Identity,[0,0],[w1,h1]) 
        keys2seek.append( temp )        


    if useAffnet:
        if not useIdentity:
            emb_1, bP_Alist1 = AffNetHardNet_describe(np.expand_dims(np.array(patches1),axis=3))
            emb_2, bP_Alist2 = AffNetHardNet_describe(np.expand_dims(np.array(patches2),axis=3))
    else:    
        bEsti =LOCATEmodel.layers[2].predict(bP)
    GA = GenAffine("", DryRun=True)
    lda = CPPbridge(opt.bindir+'libDA.so')

    def WriteImgKeys(img, keys, pathname):
        colors=( (0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 255, 255) )
        patch = cv2.cvtColor( img.astype(np.uint8), cv2.COLOR_GRAY2RGB )
        for n in range(np.min([len(colors),len(keys)])):
            patch=cv2.drawKeypoints(patch,keys[n:n+1],patch, color=colors[n] ,flags=0)
        cv2.imwrite(opt.workdir+''+pathname,patch)

    rgrowing_Matches = []
    for n in range(0,len(growing_matches)):
        if not useAffnet:
            evec = bEsti[n,:]
        idx1 = growing_matches[n].queryIdx
        x, y = np.round(KPlist1[idx1].pt).astype(int)
        if Qmap[x,y]:
            continue

        idx2 = growing_matches[n].trainIdx
        if useAffnet:
            if useIdentity:
                A_p1_to_p2 = Identity
            else:
                A_p1_to_p2 = ComposeAffineMaps( bP_Alist2[idx2], cv2.invertAffineTransform(bP_Alist1[idx1]) )
        else:
            A_p1_to_p2 = cv2.invertAffineTransform( GA.AffineFromNormalizedVector(evec) )
        A_query_to_p2 = ComposeAffineMaps(A_p1_to_p2,Akp1[n])
        A_p1_to_target = ComposeAffineMaps( cv2.invertAffineTransform(Akp2[n]), A_p1_to_p2 )
        A = ComposeAffineMaps( cv2.invertAffineTransform(Akp2[n]), A_query_to_p2 )

        Adecomp = affine_decomp(A_p1_to_target,doAssert=False)
        Aidecomp = affine_decomp( cv2.invertAffineTransform(A_query_to_p2),doAssert=False)
        
        keys4p1 = AffineKPcoor(keys2seek[n], Akp1[n])
        keys4p2 = AffineKPcoor(keys2seek[n], A_query_to_p2)
        keys4target = AffineKPcoor(keys2seek[n], A)
        for m in range(len(keys4target)):
            keys4target[m].size = KPlist2[idx2].size
            keys4target[m].octave = KPlist2[idx2].octave
        
        if Adecomp[0]<=1.0:
            keys2seek[n] = RefineKplistsPyrPos(KPlist2[idx2], KPlist1[idx1], keys2seek[n], Aidecomp[0])
            keys2seek[n], _ = Filter_Affine_In_Rect(keys2seek[n],Identity,[0,0],[w1,h1])             
            classids = [kp.class_id for kp in keys2seek[n]]
            keys4target = [kp for kp in keys4target if kp.class_id in classids]

        if Adecomp[0]>1.0:
            keys4target = RefineKplistsPyrPos(KPlist1[idx1],KPlist2[idx2], keys4target, Adecomp[0])
            keys4target, _ = Filter_Affine_In_Rect(keys4target,Identity,[0,0],[w2,h2])             
            classids = [kp.class_id for kp in keys4target]
            keys2seek[n] = [kp for kp in keys2seek[n] if kp.class_id in classids]

        if len(keys4target)==0:
            continue

        newpatches1, _, _ = ComputePatches(keys2seek[n],pyr1,border_mode=cv2.BORDER_REFLECT)
        newpatches2, _, _ = ComputePatches(keys4target,pyr2,border_mode=cv2.BORDER_REFLECT)

        bP = np.zeros( shape = tuple([len(newpatches1)])+tuple(np.shape(newpatches1[0]))+tuple([1]), dtype=np.float32)
        for k in range(0,len(newpatches1)):
            bP[k,:,:,0] = newpatches1[k][:,:]/255.0
        emb_1 = BigAIDmodel.get_layer("aff_desc").predict(bP)

        bP = np.zeros( shape=tuple([len(newpatches2)])+tuple(np.shape(newpatches2[0]))+tuple([1]), dtype=np.float32)
        for k in range(0,len(newpatches2)):
            bP[k,:,:,0] = newpatches2[k][:,:]/255.0
        emb_2 = BigAIDmodel.get_layer("aff_desc").predict(bP)

        simis = BigAIDmodel.get_layer("sim").predict([emb_1, emb_2])
        
        for m in range(0,len(keys2seek[n])):
            # WriteImgKeys(patches1[idx1], keys4p1, 'p1init/'+str(n)+'.'+str(m)+'.png' )
            # WriteImgKeys(patches2[idx2], keys4p2, 'p2init/'+str(n)+'.'+str(m)+'.png' )
            # WriteImgKeys(newpatches1[m], [], 'p1/'+str(n)+'.'+str(m)+'.png' )
            # WriteImgKeys(newpatches2[m], [], 'p2/'+str(n)+'.'+str(m)+'.png' )
            # WriteImgKeys(img1, keys2seek[n], 'im1/'+str(n)+'.'+str(m)+'.png' )
            # WriteImgKeys(img2, keys4target, 'im2/'+str(n)+'.'+str(m)+'.png' )
            if simis[m]>0.6:
                KPlist1.append(keys2seek[n][m])
                KPlist2.append(keys4target[m])
                if keys4target[m].class_id>=0:
                    rgrowing_Matches.append( cv2.DMatch(len(KPlist1)-1,len(KPlist2)-1,1) )        
                else:
                    growed_matches.append( cv2.DMatch(len(KPlist1)-1,len(KPlist2)-1,1) )        
                    pQ = 2
                    idx1 = growed_matches[len(growed_matches)-1].queryIdx
                    x, y = np.round(KPlist1[idx1].pt).astype(int)
                    if x>pQ and x<w1-pQ and y>pQ and y<h1-pQ:
                        Qmap[(x-pQ):(x+pQ+1) , (y-pQ):(y+pQ+1)] = True 

    return KPlist1, KPlist2, growed_matches, rgrowing_Matches, Qmap


import sklearn.preprocessing
def RootSIFT(img1,img2, MatchingThres = opt.rootsift_thres, knn_num = 2, Rooted = True, GFilter=opt.gfilter, Visual=opt.visual, EuPrecision=24):
    start_time = time.time()
    KPlist1, sift_des1 = ComputeSIFTKeypoints(img1, Desc = True)
    KPlist2, sift_des2 = ComputeSIFTKeypoints(img2, Desc = True)
    Identity = np.float32([[1, 0, 0], [0, 1, 0]])
    h, w = img1.shape[:2]
    KPlist1, sift_des1, temp = Filter_Affine_In_Rect(KPlist1,Identity,[0,0],[w,h], desc_list = sift_des1)
    h, w = img2.shape[:2]
    KPlist2, sift_des2, temp = Filter_Affine_In_Rect(KPlist2,Identity,[0,0],[w,h], desc_list = sift_des2)
    if Rooted:
        sift_des1 = np.sqrt(sklearn.preprocessing.normalize(sift_des1, norm='l2',axis=1))
        sift_des2 = np.sqrt(sklearn.preprocessing.normalize(sift_des2, norm='l2',axis=1))

    ET_KP = time.time() - start_time


    bf = cv2.BFMatcher()
    start_time = time.time()
    sift_matches = bf.knnMatch(sift_des1,sift_des2, k=knn_num)
    ET_M = time.time() - start_time

    # Apply ratio test
    lda = CPPbridge(opt.bindir+'libDA.so')
    sift_all = []
    if knn_num==2:
        for m,n in sift_matches:
            if m.distance < MatchingThres*n.distance:
                sift_all.append(m)
    elif knn_num==1:
        for m in sift_matches:
            if m[0].distance <= MatchingThres:
                sift_all.append(m[0])

    sift_all = OnlyUniqueMatches(sift_all,KPlist1,KPlist2,SpatialThres=5)

    if GFilter[0:5] == 'Aff_H':
        from AffRANSAC import Aff_RANSAC_H
        _, H_sift, sift_consensus = Aff_RANSAC_H(img1, KPlist1, img2, KPlist2, sift_all, AffInfo=int(GFilter[6]), precision=EuPrecision)
    else:
        sift_consensus, H_sift = lda.GeometricFilter(KPlist1, img1, KPlist2, img2, sift_all, Filter=GFilter, precision=EuPrecision)

    if Visual:
        # img4 = cv2.drawMatches(img1,KPlist1,img2,KPlist2,sift_all, None,flags=2)
        # cv2.imwrite(opt.workdir+'SIFTmatches.png',img4)
        img4 = cv2.drawMatches(img1,KPlist1,img2,KPlist2,sift_consensus, None,flags=2)
        cv2.imwrite(opt.workdir+'matches.png',img4)

    return sift_all, sift_consensus, KPlist1, KPlist2, H_sift, ET_KP, ET_M


def HessAff_HardNet(img1,img2, MatchingThres = opt.hardnet_thres, Ndesc=500, GFilter=opt.gfilter, Visual=opt.visual, EuPrecision=24, HessAffNet=True, affmaps_method='locate', ransac_iters = 1000):
    start_time = time.time()
    if HessAffNet:
        KPlist1, Patches1, descriptors1, Alist1, Score1 = HessAffNetHardNet_DetectAndDescribe(img1, Nfeatures=Ndesc)
        KPlist2, Patches2, descriptors2, Alist2, Score2 = HessAffNetHardNet_DetectAndDescribe(img2, Nfeatures=Ndesc)
    else:
        KPlist1, Patches1, descriptors1, Alist1, Score1 = HessAffNetHardNet_DetectAndDescribe(img1, Nfeatures=Ndesc, useAffnet=None)
        KPlist2, Patches2, descriptors2, Alist2, Score2 = HessAffNetHardNet_DetectAndDescribe(img2, Nfeatures=Ndesc, useAffnet=None)
    ET_KP = time.time() - start_time

    #Bruteforce matching with SNN threshold    
    start_time = time.time()
    tent_matches_in_1, tent_matches_in_2 = BruteForce4HardNet(descriptors1,descriptors2, SNN_threshold=MatchingThres)
    ET_M = time.time() - start_time

    sift_all = OnlyUniqueMatches( [cv2.DMatch(int(i), int(j), 1.0) for i,j in zip(tent_matches_in_1,tent_matches_in_2)], KPlist1, KPlist2 )

    if GFilter[0:5] in ['Aff_H','Aff_O']:
        useORSA = GFilter[4] == 'O'
        from AffRANSAC import Aff_RANSAC_H
        Alist1 = [cv2.invertAffineTransform(A) for A in Alist1]
        Alist2 = [cv2.invertAffineTransform(A) for A in Alist2]
        Aq2t, A_p1_to_target = get_Aq2t(Alist1, Patches1[:,0,:,:], Alist2, Patches2[:,0,:,:], sift_all, method=affmaps_method, noTranslation=True)
        _, H_sift, sift_consensus = Aff_RANSAC_H(img1, KPlist1, img2, KPlist2, sift_all, AffInfo=int(GFilter[6]), precision=EuPrecision, Aq2t=Aq2t, ORSAlike=useORSA, Niter=ransac_iters )
        if Visual:
            save_randomAffineInliers(img1,img2,KPlist1,KPlist2,Alist1,Alist2,sift_consensus, Aq2t, A_p1_to_target,patch_width=5, noTranslation=True)
    else:
        lda = CPPbridge(opt.bindir+'libDA.so')
        sift_consensus, H_sift = lda.GeometricFilter(KPlist1, img1, KPlist2, img2, sift_all, Filter=GFilter, precision=EuPrecision)
        lda.DestroyMatcher()

    # # Affine maps from query to target
    # pxl_radius = 20
    # n=0
    # for m in sift_consensus:
    #     Aq2t = ComposeAffineMaps( Alist2[m.trainIdx], cv2.invertAffineTransform(Alist1[m.queryIdx]) )
    #     kps1 = [cv2.KeyPoint(x = KPlist1[m.queryIdx].pt[0]+pxl_radius*i - pxl_radius/2, 
    #         y = KPlist1[m.queryIdx].pt[1] +pxl_radius*j - pxl_radius/2,
    #         _size =  KPlist1[m.queryIdx].size, 
    #         _angle =  KPlist1[m.queryIdx].angle,
    #         _response =  KPlist1[m.queryIdx].response, _octave = KPlist1[m.queryIdx].octave,
    #         _class_id =  i*2+j) for i in range(0,2) for j in range(0,2)]
    #     kps1.append(cv2.KeyPoint(x = KPlist1[m.queryIdx].pt[0], 
    #         y = KPlist1[m.queryIdx].pt[1],
    #         _size =  KPlist1[m.queryIdx].size, 
    #         _angle =  KPlist1[m.queryIdx].angle,
    #         _response =  KPlist1[m.queryIdx].response, _octave =  KPlist1[m.queryIdx].octave,
    #         _class_id =  -1))
    #     kps2 = AffineKPcoor( kps1, Aq2t)
    #     # SaveImageWithKeys(img1, kps1, 'im1/'+str(n)+'.png' )
    #     # SaveImageWithKeys(img2, kps2, 'im2/'+str(n)+'.png' )
    #     n=n+1

    if Visual:
        # img4 = cv2.drawMatches(img1,KPlist1,img2,KPlist2,sift_all, None,flags=2)
        # cv2.imwrite(opt.workdir+'Affnet_matches.png',img4)
        img4 = cv2.drawMatches(img1,KPlist1,img2,KPlist2,sift_consensus, None,flags=2)
        cv2.imwrite(opt.workdir+'matches.png',img4)

    return sift_all, sift_consensus, KPlist1, KPlist2, H_sift, ET_KP, ET_M


def SIFT_AffNet_HardNet(img1,img2, MatchingThres = opt.hardnet_thres, knn_num = 2, GFilter=opt.gfilter, Visual=opt.visual, EuPrecision=24, AffNetBeforeDesc=True, affmaps_method='locate',ransac_iters = 1000):
    # find the keypoints with SIFT
    start_time = time.time()
    KPlist1, sift_des1 = ComputeSIFTKeypoints(img1, Desc = True)
    KPlist2, sift_des2 = ComputeSIFTKeypoints(img2, Desc = True)
    Identity = np.float32([[1, 0, 0], [0, 1, 0]])
    h, w = img1.shape[:2]
    KPlist1, sift_des1, temp = Filter_Affine_In_Rect(KPlist1,Identity,[0,0],[w,h], desc_list = sift_des1)
    h, w = img2.shape[:2]
    KPlist2, sift_des2, temp = Filter_Affine_In_Rect(KPlist2,Identity,[0,0],[w,h], desc_list = sift_des2)

    maxoctaves = 4
    pyr1 = buildGaussianPyramid( img1, maxoctaves + 2 )
    pyr2 = buildGaussianPyramid( img2, maxoctaves + 2 )

    patches1, A_list1, Ai_list1 = ComputePatches(KPlist1,pyr1)
    patches2, A_list2, Ai_list2 = ComputePatches(KPlist2,pyr2)

    bP = np.zeros( shape = tuple([len(patches1)])+tuple(np.shape(patches1[0]))+tuple([1]), dtype=np.float32)
    for k in range(0,len(patches1)):
        bP[k,:,:,0] = patches1[k][:,:]
    if AffNetBeforeDesc:
        emb_1, bP_Alist1 = AffNetHardNet_describe(bP)
    else:
        emb_1, bP_Alist1 = HardNet_describe(bP)

    bP = np.zeros( shape=tuple([len(patches2)])+tuple(np.shape(patches2[0]))+tuple([1]), dtype=np.float32)
    for k in range(0,len(patches2)):
        bP[k,:,:,0] = patches2[k][:,:]
    if AffNetBeforeDesc:
        emb_2, bP_Alist2 = AffNetHardNet_describe(bP)
    else:
        emb_2, bP_Alist2 = HardNet_describe(bP)
    ET_KP = time.time() - start_time

    bf = cv2.BFMatcher()
    start_time = time.time()
    sift_matches = bf.knnMatch(emb_1,emb_2, k=knn_num)
    ET_M = time.time() - start_time


    lda = CPPbridge(opt.bindir+'libDA.so')
    # Apply ratio test
    sift_all = []
    if knn_num==2:
        for m,n in sift_matches:
            if m.distance < MatchingThres*n.distance:
                sift_all.append(m)
    elif knn_num==1:
        for m in sift_matches:
            if m[0].distance <= MatchingThres:
                sift_all.append(m[0])

    sift_all = OnlyUniqueMatches(sift_all,KPlist1,KPlist2,SpatialThres=5)      

    if GFilter[0:5] in ['Aff_H','Aff_O']:
        useORSA = GFilter[4] == 'O'
        from AffRANSAC import Aff_RANSAC_H
        Aq2t, A_p1_to_target = get_Aq2t(A_list1, patches1, A_list2, patches2, sift_all, method=affmaps_method)
        _, H_sift, sift_consensus = Aff_RANSAC_H(img1, KPlist1, img2, KPlist2, sift_all, AffInfo=int(GFilter[6]), precision=EuPrecision, Aq2t=Aq2t, ORSAlike=useORSA, Niter=ransac_iters)
        if Visual:
            save_randomAffineInliers(img1,img2,KPlist1,KPlist2,A_list1,A_list2,sift_consensus, Aq2t, A_p1_to_target)
    else:
        sift_consensus, H_sift = lda.GeometricFilter(KPlist1, img1, KPlist2, img2, sift_all, Filter=GFilter, precision=EuPrecision)

    # # Visualize affine maps from query to target
    # pxl_radius = 20
    # n=0
    # for m in sift_consensus:
    #     Aq2t = ComposeAffineMaps( bP_Alist2[m.trainIdx], cv2.invertAffineTransform(bP_Alist1[m.queryIdx]) )
    #     Aq2t = ComposeAffineMaps( Ai_list2[m.trainIdx], Aq2t )
    #     Aq2t = ComposeAffineMaps( Aq2t, A_list1[m.queryIdx] )
    #     kps1 = [cv2.KeyPoint(x = KPlist1[m.queryIdx].pt[0]+pxl_radius*i - pxl_radius/2, 
    #         y = KPlist1[m.queryIdx].pt[1] +pxl_radius*j - pxl_radius/2,
    #         _size =  KPlist1[m.queryIdx].size, 
    #         _angle =  KPlist1[m.queryIdx].angle,
    #         _response =  KPlist1[m.queryIdx].response, _octave = KPlist1[m.queryIdx].octave,
    #         _class_id =  i*2+j) for i in range(0,2) for j in range(0,2)]
    #     kps1.append(cv2.KeyPoint(x = KPlist1[m.queryIdx].pt[0], 
    #         y = KPlist1[m.queryIdx].pt[1],
    #         _size =  KPlist1[m.queryIdx].size, 
    #         _angle =  KPlist1[m.queryIdx].angle,
    #         _response =  KPlist1[m.queryIdx].response, _octave =  KPlist1[m.queryIdx].octave,
    #         _class_id =  -1))
    #     kps2 = AffineKPcoor( kps1, Aq2t)
    #     SaveImageWithKeys(img1, kps1, 'im1/'+str(n)+'.png' )
    #     SaveImageWithKeys(img2, kps2, 'im2/'+str(n)+'.png' )
    #     n=n+1

    if Visual:
        # img4 = cv2.drawMatches(img1,KPlist1,img2,KPlist2,sift_all, None,flags=2)
        # cv2.imwrite(opt.workdir+'SIFT_Affnet_matches.png',img4)
        img4 = cv2.drawMatches(img1,KPlist1,img2,KPlist2,sift_consensus, None,flags=2)
        cv2.imwrite(opt.workdir+'matches.png',img4)
    lda.DestroyMatcher()
    return sift_all, sift_consensus, KPlist1, KPlist2, H_sift, ET_KP, ET_M


def HessianLaplaceRootSIFT(img1,img2, MatchingThres = opt.rootsift_thres, Ndesc=500, knn_num = 2, Rooted = True, GFilter=opt.gfilter, Visual=opt.visual, EuPrecision=24):
    lda = CPPbridge(opt.bindir+'libDA.so')
    start_time = time.time()
    lda.CreateMatcher(128, k = knn_num, sim_thres = MatchingThres)
    
    KPlist1, Patches1, sift_des1, Alist1, Scores1 = lda.call_vlfeat(img1,0)
    KPlist2, Patches2, sift_des2, Alist2, Scores2 = lda.call_vlfeat(img2,0)

    idx1 = np.argsort(Scores1[:,0])
    idx2 = np.argsort(Scores2[:,0])
    # idx1 = idx1[:np.min([Ndesc,len(idx1)])]
    # idx2 = idx2[:np.min([Ndesc,len(idx2)])]
    idx1 = idx1[np.max([0, len(idx1)-Ndesc]):len(idx1)]
    idx2 = idx2[np.max([0, len(idx2)-Ndesc]):len(idx2)]

    KPlist1 = [KPlist1[i] for i in idx1]
    KPlist2 = [KPlist2[i] for i in idx2]
    Patches1 = [Patches1[i] for i in idx1]
    Patches2 = [Patches2[i] for i in idx2]
    sift_des1 = sift_des1[idx1,:]
    sift_des2 = sift_des2[idx2,:]
    Alist1 = [Alist1[i] for i in idx1]
    Alist2 = [Alist2[i] for i in idx2]
    # for i in range(len(Patches1)):
    #     SaveImageWithKeys(Patches1[i], [], '/p1/'+str(i)+'.png' )
    #     SaveImageWithKeys(img1,[KPlist1[i]],'/p2/'+str(i)+'.png')


    if Rooted:
        sift_des1 = np.sqrt(sklearn.preprocessing.normalize(sift_des1, norm='l2',axis=1))
        sift_des2 = np.sqrt(sklearn.preprocessing.normalize(sift_des2, norm='l2',axis=1))

    ET_KP = time.time() - start_time


    bf = cv2.BFMatcher()
    start_time = time.time()
    sift_matches = bf.knnMatch(sift_des1,sift_des2, k=knn_num)
    ET_M = time.time() - start_time

    # Apply ratio test
    sift_all = []
    if knn_num==2:
        for m,n in sift_matches:
            if m.distance < MatchingThres*n.distance:
                sift_all.append(m)
    elif knn_num==1:
        for m in sift_matches:
            if m[0].distance <= MatchingThres:
                sift_all.append(m[0])

    sift_all = OnlyUniqueMatches(sift_all,KPlist1,KPlist2,SpatialThres=5) 
    
    # Affine maps from query to target
    Aq2t_list = []
    for m in sift_all:
        Aq2t = ComposeAffineMaps( Alist2[m.trainIdx], cv2.invertAffineTransform(Alist1[m.queryIdx]) )
        Aq2t_list.append( Aq2t )

    if GFilter[0:5] == 'Aff_H':
        from AffRANSAC import Aff_RANSAC_H
        _, H_sift, sift_consensus = Aff_RANSAC_H(img1, KPlist1, img2, KPlist2, sift_all, AffInfo=int(GFilter[6]), precision=EuPrecision, Aq2t=Aq2t_list)
    else:
        sift_consensus, H_sift = lda.GeometricFilter(KPlist1, img1, KPlist2, img2, sift_all, Filter=GFilter, precision=EuPrecision)

    # # Visualize affine maps from query to target
    # pxl_radius = 20
    # n=0
    # for m in sift_consensus:
    #     Aq2t = ComposeAffineMaps( Alist2[m.trainIdx], cv2.invertAffineTransform(Alist1[m.queryIdx]) )
    #     kps1 = [cv2.KeyPoint(x = KPlist1[m.queryIdx].pt[0]+pxl_radius*i - pxl_radius/2, 
    #         y = KPlist1[m.queryIdx].pt[1] +pxl_radius*j - pxl_radius/2,
    #         _size =  KPlist1[m.queryIdx].size, 
    #         _angle =  KPlist1[m.queryIdx].angle,
    #         _response =  KPlist1[m.queryIdx].response, _octave = KPlist1[m.queryIdx].octave,
    #         _class_id =  i*2+j) for i in range(0,2) for j in range(0,2)]
    #     kps1.append(cv2.KeyPoint(x = KPlist1[m.queryIdx].pt[0], 
    #         y = KPlist1[m.queryIdx].pt[1],
    #         _size =  KPlist1[m.queryIdx].size, 
    #         _angle =  KPlist1[m.queryIdx].angle,
    #         _response =  KPlist1[m.queryIdx].response, _octave =  KPlist1[m.queryIdx].octave,
    #         _class_id =  -1))
    #     kps2 = AffineKPcoor( kps1, Aq2t)
    #     SaveImageWithKeys(img1, kps1, 'im1/'+str(n)+'.png' )
    #     SaveImageWithKeys(img2, kps2, 'im2/'+str(n)+'.png' )
    #     n=n+1

    if Visual:
        # img4 = cv2.drawMatches(img1,KPlist1,img2,KPlist2,sift_all, None,flags=2)
        # cv2.imwrite(opt.workdir+'HessLaplaceSIFTmatches.png',img4)
        img4 = cv2.drawMatches(img1,KPlist1,img2,KPlist2,sift_consensus, None,flags=2)
        cv2.imwrite(opt.workdir+'matches.png',img4)
    lda.DestroyMatcher()
    return sift_all, sift_consensus, KPlist1, KPlist2, H_sift, ET_KP, ET_M

    
def HessAffAID(img1,img2, Ndesc=500, MatchingThres = opt.aid_thres, Simi='SignProx', knn_num = 1, GFilter=opt.gfilter, Visual=opt.visual, safe_sim_thres_pos = 0.8, safe_sim_thres_neg = 0.2, GetAllMatches=False, EuPrecision=24, descRadius=math.inf, affmaps_method = "locate", ransac_iters = 1000):
    if Simi=='CosProx':
        FastCode = 0
    elif Simi=='SignProx':
        FastCode = 1
    elif Simi=='SignProxKNN':
        FastCode = 2
    else:
        print('Wrong similarity choice for HessAffSIFT !!!')
        exit()

    # find the keypoints with SIFT
    start_time = time.time()
    KPlist1, patches1, Alist1, responses1 = HessAff_Detect(img1, PatchSize=60, Nfeatures=Ndesc)
    KPlist2, patches2, Alist2, responses2 = HessAff_Detect(img2, PatchSize=60, Nfeatures=Ndesc)    
    
    bP = np.zeros( shape = tuple([patches1.shape[0], patches1.shape[2], patches1.shape[3], 1]), dtype=np.float32)
    for k in range(0,len(patches1)):
        bP[k,:,:,0] = patches1[k,0,:,:]/255.0
    emb_1 = BigAIDmodel.get_layer("aff_desc").predict(bP)
    emb_1 = CreateSubDesc(emb_1, coef=1.0, NewDescRadius=descRadius)

    bP = np.zeros( shape = tuple([patches2.shape[0], patches2.shape[2], patches2.shape[3], 1]), dtype=np.float32)
    for k in range(0,len(patches2)):
        bP[k,:,:,0] = patches2[k,0,:,:]/255.0
    emb_2 = BigAIDmodel.get_layer("aff_desc").predict(bP)
    emb_2 = CreateSubDesc(emb_2, coef=-1.0, NewDescRadius=descRadius)
    
    ET_KP = time.time() - start_time

    desc_dim = np.shape(emb_1)[1]
    lda = CPPbridge(opt.bindir+'libDA.so')
    lda.CreateMatcher(desc_dim, k = knn_num, sim_thres = MatchingThres)
    start_time = time.time()
    lda.KnnMatch(KPlist1,emb_1, KPlist2,emb_2,FastCode)    
    ET_M = time.time() - start_time

    AID_all = []
    if GFilter[0:5] in ['Aff_H','Aff_O']:
        useORSA = GFilter[4] == 'O'
        from AffRANSAC import Aff_RANSAC_H
        AID_all = lda.GetAllMatches()
        AID_all = OnlyUniqueMatches(AID_all,KPlist1,KPlist2,SpatialThres=10)
        Alist1 = [cv2.invertAffineTransform(A) for A in Alist1]
        Alist2 = [cv2.invertAffineTransform(A) for A in Alist2]
        Aq2t, A_p1_to_target = get_Aq2t(Alist1, patches1[:,0,:,:], Alist2, patches2[:,0,:,:], AID_all, method=affmaps_method, noTranslation=True)
        _, H_AID, AID_consensus = Aff_RANSAC_H(img1, KPlist1, img2, KPlist2, AID_all, AffInfo=int(GFilter[6]), precision=EuPrecision, Aq2t=Aq2t, ORSAlike=useORSA, Niter=ransac_iters)
        if Visual:
            save_randomAffineInliers(img1,img2,KPlist1,KPlist2,Alist1,Alist2,AID_consensus, Aq2t, A_p1_to_target,patch_width=5,noTranslation=True)
    else:
        if GetAllMatches:
            AID_all = lda.GetAllMatches()
            AID_all = OnlyUniqueMatches(AID_all,KPlist1,KPlist2,SpatialThres=5)
        AID_consensus, H_AID = lda.GeometricFilterFromMatcher(img1, img2, Filter=GFilter, precision=EuPrecision,verb=False)

    if Visual:
        # img3 = cv2.drawMatches(img1,KPlist1,img2,KPlist2,AID_good, None,flags=2)
        # cv2.imwrite(opt.workdir+'AID_total_matches.png',img3)
        img3 = cv2.drawMatches(img1,KPlist1,img2,KPlist2,AID_consensus, None,flags=2)
        cv2.imwrite(opt.workdir+'matches.png',img3)
        # h, w = img2.shape[:2]
        # warp_AID = cv2.warpPerspective(img1, H_AID,(w, h))
        # warp_AID = warp_AID
        # cv2.imwrite(opt.workdir+'HessAffAID_panorama.png',warp_AID)

    lda.DestroyMatcher()
    return AID_all, AID_consensus, KPlist1, KPlist2, H_AID, ET_KP, ET_M


def siftAID(img1,img2, MatchingThres = opt.aid_thres, Simi='SignProx', knn_num = 1, GFilter=opt.gfilter, Visual=opt.visual, safe_sim_thres_pos = 0.8, safe_sim_thres_neg = 0.2, GetAllMatches=False, EuPrecision=24, descRadius=math.inf, RegionGrowingIters=-1, affmaps_method = "locate", ransac_iters = 1000):
    if Simi=='CosProx':
        FastCode = 0
    elif Simi=='SignProx':
        FastCode = 1
    elif Simi=='SignProxKNN':
        FastCode = 2
    else:
        print('Wrong similarity choice for AI-SIFT !!!')
        exit()

    # find the keypoints with SIFT
    start_time = time.time()
    KPlist1, sift_des1 = ComputeSIFTKeypoints(img1, Desc = True)
    KPlist2, sift_des2 = ComputeSIFTKeypoints(img2, Desc = True)
    Identity = np.float32([[1, 0, 0], [0, 1, 0]])
    h, w = img1.shape[:2]
    KPlist1, sift_des1, temp = Filter_Affine_In_Rect(KPlist1,Identity,[0,0],[w,h], desc_list = sift_des1)
    h, w = img2.shape[:2]
    KPlist2, sift_des2, temp = Filter_Affine_In_Rect(KPlist2,Identity,[0,0],[w,h], desc_list = sift_des2)

    maxoctaves = 4
    pyr1 = buildGaussianPyramid( img1, maxoctaves + 2 )
    pyr2 = buildGaussianPyramid( img2, maxoctaves + 2 )

    patches1, A_list1, Ai_list1 = ComputePatches(KPlist1,pyr1)
    patches2, A_list2, Ai_list2 = ComputePatches(KPlist2,pyr2)

    bP = np.zeros( shape = tuple([len(patches1)])+tuple(np.shape(patches1[0]))+tuple([1]), dtype=np.float32)
    for k in range(0,len(patches1)):
        bP[k,:,:,0] = patches1[k][:,:]/255.0
    
    emb_1 = BigAIDmodel.get_layer("aff_desc").predict(bP)
    emb_1 = CreateSubDesc(emb_1, coef=1.0, NewDescRadius=descRadius)

    bP = np.zeros( shape=tuple([len(patches2)])+tuple(np.shape(patches2[0]))+tuple([1]), dtype=np.float32)
    for k in range(0,len(patches2)):
        bP[k,:,:,0] = patches2[k][:,:]/255.0
    emb_2 = BigAIDmodel.get_layer("aff_desc").predict(bP)
    emb_2 = CreateSubDesc(emb_2, coef=-1.0, NewDescRadius=descRadius)
    
    ET_KP = time.time() - start_time

    desc_dim = np.shape(emb_1)[1]
    lda = CPPbridge("/home/xxx/project/python/locate-master/libDA.so")
    lda.CreateMatcher(desc_dim, k = knn_num, sim_thres = MatchingThres)
    start_time = time.time()
    lda.KnnMatch(KPlist1,emb_1, KPlist2,emb_2,FastCode)    
    ET_M = time.time() - start_time

    AID_all = []
    if GetAllMatches or RegionGrowingIters>=0 or GFilter[0:5] in ['Aff_H','Aff_O']:
        AID_all = lda.GetAllMatches()
        AID_all = OnlyUniqueMatches(AID_all,KPlist1,KPlist2,SpatialThres=5)

    start_time = time.time()
    if RegionGrowingIters>=0:
        h1, w1 = img1.shape[:2]
        Qmap = np.zeros(shape=[w1,h1],dtype=bool)
        KPlist1, KPlist2, growed_matches, growing_matches, Qmap = GrowMatches(KPlist1, pyr1, KPlist2, pyr2, Qmap, [], AID_all, img1, img2, LocalGeoEsti='8pts')
        for i in range(RegionGrowingIters):
            KPlist1, KPlist2, growed_matches, growing_matches, Qmap = GrowMatches(KPlist1, pyr1, KPlist2, pyr2, Qmap, growed_matches, growing_matches, img1, img2, LocalGeoEsti='8pts')
            if len(growing_matches)==0:
                break
        AID_all = growed_matches 
        patches1, A_list1, Ai_list1 = ComputePatches(KPlist1,pyr1)
        patches2, A_list2, Ai_list2 = ComputePatches(KPlist2,pyr2)
    ET_G = time.time() - start_time 
    
    if GFilter[0:5] in ['Aff_H','Aff_O']:
        useORSA = GFilter[4] == 'O'
        from AffRANSAC import Aff_RANSAC_H
        Aq2t, A_p1_to_target = get_Aq2t(A_list1, patches1, A_list2, patches2, AID_all, method=affmaps_method) 
        _, H_AID, AID_consensus = Aff_RANSAC_H(img1, KPlist1, img2, KPlist2, AID_all, AffInfo=int(GFilter[6]), precision=EuPrecision, Aq2t=Aq2t, ORSAlike=useORSA, Niter=ransac_iters)
        if Visual:
            save_randomAffineInliers(img1,img2,KPlist1,KPlist2,A_list1,A_list2,AID_consensus, Aq2t, A_p1_to_target)       
    else:
        if RegionGrowingIters>=0:
            AID_consensus, H_AID = lda.GeometricFilter(KPlist1, img1, KPlist2, img2, AID_all, Filter=GFilter, precision=EuPrecision)
        else:
            AID_consensus, H_AID = lda.GeometricFilterFromMatcher(img1, img2, Filter=GFilter, precision=EuPrecision,verb=False)

    if Visual:
        # img3 = cv2.drawMatches(img1,KPlist1,img2,KPlist2,AID_all, None,flags=2)
        # cv2.imwrite(opt.workdir+'AID_total_matches.png',img3)
        img3 = cv2.drawMatches(img1,KPlist1,img2,KPlist2,AID_consensus, None,flags=2)
        cv2.imwrite(opt.workdir+'matches.png',img3)
        # h, w = img2.shape[:2]
        # warp_AID = cv2.warpPerspective(img1, H_AID,(w, h))
        # warp_AID = warp_AID
        # cv2.imwrite(opt.workdir+'siftAID_panorama.png',warp_AID)

    lda.DestroyMatcher()
    return AID_all, AID_consensus, KPlist1, KPlist2, H_AID, ET_KP, ET_M+ET_G
