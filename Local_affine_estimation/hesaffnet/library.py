import numpy as np
import cv2
import math
import time
import glob, os
import random
import psutil
import ctypes
from datetime import datetime
import argparse

MaxSameKP_dist = 5 # pixels
MaxSameKP_angle = 10 # degrees

class BaseOptions():
    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False
        self.save_dir = '/'

    def initialize(self, parser):
        """Define the common options."""
        # basic parameters
        parser.add_argument('--im1', type=str, default='./acc-test/adam.1.png', required=False, help='Path to query image')
        parser.add_argument('--im2', type=str, default='./acc-test/adam.2.png', required=False, help='Path to target image')
        
        parser.add_argument('--gfilter', type=str, default="Aff_H_2", help='Geometric filter to apply: [USAC_H, ORSA_H, USAC_F, ORSA_F, Aff_H_0, Aff_H_1, Aff_H_2, Aff_O_0, Aff_O_1, Aff_O_2]')
        parser.add_argument('--detector', type=str, default='SIFT', required=False, help='Detector: [ HessAff, SIFT]')
        parser.add_argument('--descriptor', type=str, default="AID", required=False, help='Descriptor code: [ AID, RootSIFT, HardNet]')
        parser.add_argument('--affmaps', type=str, default='locate', required=False, help='Affine maps provided by: [ locate, affnet, simple]')
        parser.add_argument('--aid_thres', type=float, default=4000, help='AID matching threshold')
        parser.add_argument('--hardnet_thres', type=float, default=0.8, help='Hardnet matching threshold')
        parser.add_argument('--rootsift_thres', type=float, default=0.8, help='AID matching threshold')
        parser.add_argument('--precision', type=float, default=24, help='Precision of the symmetric transfer error')
        parser.add_argument('--ransac_iters', type=int, default=1000, help='The number of RANSAC iterations')
        

        parser.add_argument('--visual', default=False, action='store_true', help='Visualize output images')
        parser.add_argument('--verbose', default=False, action='store_true', help='Verbose mode')

        parser.add_argument('--workdir', type=str, default='./temp/', required=False, help='Work dir for output images.')
        parser.add_argument('--bindir', type=str, default='./', required=False, help='Binaries directory (for IPOL demo).')

        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options (only once).
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        if opt.verbose:
            print(message)

        # save to the disk
        # expr_dir = self.save_dir
        # file_name = os.path.join(expr_dir, 'opt.txt')
        # with open(file_name, 'wt') as opt_file:
        #     opt_file.write(message)
        #     opt_file.write('\n')

    def parse(self):
        """Parse our options."""
        opt = self.gather_options()
        self.print_options(opt)
        self.opt = opt
        return self.opt

opt = BaseOptions().parse()


class ClassSIFTparams():
    def __init__(self, nfeatures = 0, nOctaveLayers = 3, contrastThreshold = 0.04, edgeThreshold = 10, sigma = 1.6, firstOctave = -1, sift_init_sigma = 0.5, graydesc = True):
        self.nOctaves = 4
        self.nfeatures = nfeatures
        self.nOctaveLayers = nOctaveLayers
        self.contrastThreshold = contrastThreshold
        self.edgeThreshold = edgeThreshold
        self.sigma = sigma
        self.firstOctave = firstOctave
        self.sift_init_sigma = sift_init_sigma
        self.graydesc = graydesc
        self.flt_epsilon = 1.19209e-07
        self.lambda_descr = 6
        self.new_radius_descr = 29.5


siftparams = ClassSIFTparams(graydesc = True)


class GenAffine():
    def __init__(self, path_to_imgs, tmax = 75, zmax = 1.6, save_path = "/tmp", saveMem = True, normalizeIMG=True, ActiveRadius=60, ReducedRadius=0.25, DoBigEpochs=False, DryRun = False, inImageNegative=False):
        self.path_to_imgs = path_to_imgs
        self.save_path = save_path
        TouchDir(save_path)
        self.max_simu_zoom = zmax
        self.max_simu_theta = np.deg2rad(tmax)
        self.imgs_path = ()
        self.imgs = []
        self.imgs_gray = []
        self.KPlists = []
        self.saveMem = saveMem
        self.imgdivfactor = 1.0
        if normalizeIMG==True:
            self.imgdivfactor = 255.0
        ar = np.float32(ActiveRadius)
        rr = np.float32(ReducedRadius)
        self.vecdivfactor = ar/rr
        self.vectraslation = (1.0 - rr)/2.0 #+rr/2.0
        self.gen_P_list = []
        self.gen_GT_list = []
        self.OutputNegativePatch = inImageNegative
        self.gen_GTn_list = []
        self.gen_dirs = []
        # self.Avec_tras = np.float32(  [0.0, -4*math.pi,    -7.0,    -4*math.pi, -4.0*siftparams.new_radius_descr,   -4.0*siftparams.new_radius_descr])
        # self.Avec_factor = np.float32([2.0, 8.0*math.pi,   16.0,    8.0*math.pi, 8.0*siftparams.new_radius_descr,    8.0*siftparams.new_radius_descr])
        self.Avec_tras = np.float32(  [0.0, 0.0,    1.0,    0.0, -4.0*siftparams.new_radius_descr,   -4.0*siftparams.new_radius_descr])
        self.Avec_factor = np.float32([2.0, 2.0*math.pi, 8.0, math.pi, 2.0*4.0*siftparams.new_radius_descr, 2.0*4.0*siftparams.new_radius_descr])
        
        self.A_tras = np.float32(  [-8.0, -8.0, -1.0*siftparams.new_radius_descr, -8.0, -8.0, -1.0*siftparams.new_radius_descr])
        self.A_factor = np.float32([16.0, 16.0,  2.0*siftparams.new_radius_descr, 16.0, 16.0,  2.0*siftparams.new_radius_descr])
        
        self.BigAffineVec = False
        self.DoBigEpochs = DoBigEpochs
        self.LastTimeDataChecked = time.time()
        self.GAid = random.randint(0,1000)
        set_big_epoch_number(self,0)

        if not DryRun:            
            if self.OutputNegativePatch:
                for dir in glob.glob(self.save_path+"/inImageNegatives/*.npz"):
                    self.gen_dirs.append(dir)
            else:
                for dir in glob.glob(self.save_path+"/*.npz"):
                    self.gen_dirs.append(dir)

            imgspaths = (self.path_to_imgs+"/*.png", self.path_to_imgs+"/*.jpg")
            for ips in imgspaths:
                for file in glob.glob(ips):
                    self.imgs_path += tuple([file])
                    if saveMem==False:
                        self.imgs.append( cv2.imread(file) )
                        self.imgs_gray.append( cv2.cvtColor(self.imgs[len(self.imgs)-1],cv2.COLOR_BGR2GRAY) )
                        self.KPlists.append( ComputeSIFTKeypoints(self.imgs[len(self.imgs)-1]) )
            assert (len(self.imgs_path)>0), 'We need at least one image in folder '+self.path_to_imgs

    def NormalizeVector(self,vec):
        ''' For stability reasons, the network should be trained with a normalized vector.
        Use this function to normalize a vector in patch coordinates and make it compatile
        with the output of the network.
        '''
        return vec/self.vecdivfactor + self.vectraslation

    def UnNormalizeVector(self,vec):
        ''' For stability reasons, the network should be trained with a normalized vector.
        Use this function to unnormalize an output vector of the network.
        The resulting vector info will be now in patch coordinates.
        '''
        return (vec - self.vectraslation)*self.vecdivfactor

    def Nvec2Avec(self, normalizedvec):
        ''' Computes the passage from a normalized vector to the affine_decomp vector
        normalizedvec has the flatten normalized info of points x1,...,x8, such that
              A(ci) = xi  for  i=1,...,4
           A^-1(ci) = xi  for  i=5,...,8
           where ci are the corners of a patch
        '''
        A = np.array(self.AffineFromNormalizedVector(normalizedvec))
        avec = (affine_decomp(A)-self.Avec_tras)/self.Avec_factor
        assert np.greater_equal(avec,np.zeros(np.shape(avec))).all() and np.less_equal(avec,np.ones(np.shape(avec))).all(), 'Failed attempt to Normalize affine parameters in Nvec2Avec \n ' + str(avec)
        if self.BigAffineVec:
            Ai = np.array(cv2.invertAffineTransform(A))
            aivec = (affine_decomp(Ai)-self.Avec_tras)/self.Avec_factor
            assert np.greater_equal(aivec,np.zeros(np.shape(aivec))).all() and np.less_equal(aivec,np.ones(np.shape(aivec))).all(), 'Failed attempt to Normalize inverse affine parameters in Nvec2Avec \n ' + str(aivec)
            return np.concatenate((avec,aivec))
        return avec


    def Avec2Nvec(self, affinevec, d = np.int32(siftparams.new_radius_descr*2)+1):
        ''' Computes the passage from an affine_decomp vector to a normalized vector which
        has the flatten normalized info of points x1,...,x8, such that
              A(ci) = xi  for  i=1,...,4
           A^-1(ci) = xi  for  i=5,...,8
           where ci are the corners of a patch
        '''
        SquarePatch = SquareOrderedPts(d,d,CV=False)
        avec = affine_decomp2affine(affinevec[0:6]*self.Avec_factor + self.Avec_tras)
        A = np.reshape(avec,(2,3))
        evec = np.zeros((16),np.float32)
        evec[0:8] = self.NormalizeVector( Pts2Flatten(AffineArrayCoor(SquarePatch,A)) )

        if self.BigAffineVec:
            aivec = affine_decomp2affine(affinevec[6:12]*self.Avec_factor + self.Avec_tras)
            Ai = np.reshape(aivec,(2,3))
            evec[8:16] = self.NormalizeVector( Pts2Flatten(AffineArrayCoor(SquarePatch,Ai)) )
        else:
            evec[8:16] = self.NormalizeVector( Pts2Flatten(AffineArrayCoor(SquarePatch,cv2.invertAffineTransform(A))) )
        return evec

    def AffineFromNormalizedVector(self,vec0, d = np.int32(siftparams.new_radius_descr*2)+1, use_inv_pts=True):
        ''' Computes the affine map fitting vec0.
         vec0 has the flatten normalized info of points x1,...,x8, such that
               A(ci) = xi  for  i=1,...,4
            A^-1(ci) = xi  for  i=5,...,8
         where ci are the corners of a patch
        '''
        vec = self.UnNormalizeVector(vec0.copy())
        X = SquareOrderedPts(d,d,CV=False)
        Y1 = Flatten2Pts(vec[0:8])
        if use_inv_pts:
            Y2 = Flatten2Pts(vec[8:16])
            return AffineFit(np.concatenate((X, Y2)),np.concatenate((Y1, X)))
        else:
            return AffineFit(X,Y1)

    def MountGenData(self, MaxData = 31500):
        start_time = time.time()
        if len(self.gen_dirs)>0:
            i = random.randint(0,len(self.gen_dirs)-1)
            path = self.gen_dirs.pop(i)
            print("\n Loading Gen Data (MaxData = "+str(MaxData)+") from "+path+" \n", end="")
            npzfile = np.load(path)
            vec_list = npzfile['vec_list']
            p1_list = npzfile['p1_list']
            p2_list = npzfile['p2_list']
            if self.OutputNegativePatch:
                pN_list = npzfile['pN_list']
                vecN_list = npzfile['vecN_list']
            
            assert len(vec_list)==len(p1_list) and len(vec_list)==len(p2_list)
            for i in range(0,len(vec_list)):
                if self.OutputNegativePatch:
                    self.gen_P_list.append( np.dstack((p1_list[i].astype(np.float32)/self.imgdivfactor, p2_list[i].astype(np.float32)/self.imgdivfactor, pN_list[i].astype(np.float32)/self.imgdivfactor)) )
                    self.gen_GTn_list.append( self.NormalizeVector(vecN_list[i]) )
                else:
                    self.gen_P_list.append( np.dstack((p1_list[i].astype(np.float32)/self.imgdivfactor, p2_list[i].astype(np.float32)/self.imgdivfactor)) )
                self.gen_GT_list.append( self.NormalizeVector(vec_list[i]) )

                if np.int32( len(self.gen_P_list) % np.int32(MaxData/10) )==0:
                    elapsed_time = time.time() - start_time
                    tstr = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
                    start_time = time.time()
                    print("\n "+str(MaxData/10) +" items loaded in "+ tstr, end="")
                    # print("\r "+ str(MaxData/10) +" items loaded in "+ tstr, end="")

                if len(self.gen_P_list)>=MaxData:
                    print("\n Maximal data items attained !")
                    break

    def ScatteredGenData_2_BlockData(self, BlockItems = 31000):
        hours, minutes, seconds = HumanElapsedTime(self.LastTimeDataChecked,time.time())
        if minutes>30 or hours>0:
            self.LastTimeDataChecked = time.time()
            globtxt_list = []
            SaveData = False
            if self.OutputNegativePatch:
                toSearch = "/*.vectorN.txt"
            else:
                toSearch = "/*.vector.txt"
            for file in glob.glob(self.save_path+toSearch):
                if self.OutputNegativePatch:
                    globtxt_list.append(file[0:(len(file)-11)]+"vector.txt")
                else:
                    globtxt_list.append(file)
                if len(globtxt_list)==BlockItems:
                    SaveData = True
                    break
            if SaveData:
                start_time = time.time()
                vec_list = []
                vecN_list = []
                p1_list = []
                p2_list = []
                pN_list = []
                for file in globtxt_list:
                    try:
                        vec = np.loadtxt(file)
                        if len(vec)!=16:
                            print("There was an error loading generated vector. That pair will be skipped !")
                            continue
                        os.remove(file)
                        file = file[0:(len(file)-10)]
                        p1 = cv2.imread(file+"p1.png")
                        p2 = cv2.imread(file+"p2.png")
                        if self.OutputNegativePatch:
                            vecN = np.loadtxt(file+"vectorN.txt")
                            pN = cv2.imread(file+"pN.png")
                        if (siftparams.graydesc):
                            p1 = cv2.cvtColor(p1, cv2.COLOR_BGR2GRAY).astype(np.uint8)
                            p2 = cv2.cvtColor(p2, cv2.COLOR_BGR2GRAY).astype(np.uint8)
                            if self.OutputNegativePatch:
                                pN = cv2.cvtColor(pN, cv2.COLOR_BGR2GRAY).astype(np.uint8)
                        vec_list.append(vec)
                        p1_list.append(p1)
                        p2_list.append(p2)
                        os.remove(file+"p1.png")
                        os.remove(file+"p2.png")
                        if self.OutputNegativePatch:
                            pN_list.append(pN)
                            vecN_list.append(vecN)
                            os.remove(file+"pN.png")
                            os.remove(file+"vectorN.txt")
                    except:
                        print("Error loading data. That pair will be skipped !")
                ts = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
                if self.OutputNegativePatch:
                    np.savez(self.save_path+'/inImageNegatives/block_'+ts,vec_list=vec_list,p1_list=p1_list,p2_list=p2_list, pN_list=pN_list, vecN_list=vecN_list)
                else:
                    np.savez(self.save_path+'/block_'+ts,vec_list=vec_list,p1_list=p1_list,p2_list=p2_list)
                elapsed_time = time.time() - start_time
                tstr = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
                print("A block of data was created in "+ tstr)

    def AvailableGenData(self):
        return len(self.gen_P_list)+len(self.gen_dirs)

    def Fast_gen_affine_patches(self, AllOutput=True):
        if self.DoBigEpochs and len(self.gen_P_list)<=10 and len(self.gen_dirs)==0:
            set_big_epoch_number(self,get_big_epoch_number(self)+1)
            if self.OutputNegativePatch:
                for dir in glob.glob(self.save_path+"/inImageNegatives/*.npz"):
                    self.gen_dirs.append(dir)
            else:
                for dir in glob.glob(self.save_path+"/*.npz"):
                    self.gen_dirs.append(dir)
        while len(self.gen_P_list)<=10 and len(self.gen_dirs)>0:
            self.MountGenData()
        
        if (len(self.gen_P_list)==0):
            return self.gen_affine_patches(AllOutput=AllOutput)
        else:
            i = random.randint(0,len(self.gen_P_list)-1)
            if self.OutputNegativePatch and AllOutput:
                return self.gen_P_list.pop(i), self.gen_GT_list.pop(i), self.gen_GTn_list.pop(i)
            else:
                return self.gen_P_list.pop(i), self.gen_GT_list.pop(i)

    def gen_affine_patches(self, AllOutput=True):
        im1_zoom = random.uniform(1.0, self.max_simu_zoom)
        im2_zoom = random.uniform(1.0, self.max_simu_zoom)

        theta1 = random.uniform(0.0, self.max_simu_theta)
        theta2 = random.uniform(0.0, self.max_simu_theta)

        im1_t = 1.0/np.cos(theta1)
        im2_t = 1.0/np.cos(theta2)

        im1_phi1 = np.rad2deg( random.uniform(0.0, math.pi) )
        im2_phi1 = np.rad2deg( random.uniform(0.0, math.pi) )

        im1_phi2 = np.rad2deg( random.uniform(0.0, 2*math.pi) )
        im2_phi2 = np.rad2deg( random.uniform(0.0, 2*math.pi) )
        
        while True:
            idx_img = random.randint(0,len(self.imgs_path)-1)

            if self.saveMem==True:
                img = cv2.cvtColor(cv2.imread(self.imgs_path[idx_img]), cv2.COLOR_BGR2GRAY)
                KPlist = ComputeSIFTKeypoints(img)
            else:
                img = self.imgs[idx_img]
                KPlist = self.KPlists[idx_img]

            h, w = img.shape[:2]

            img1, mask1, A1, Ai1 = SimulateAffineMap(im1_zoom, im1_phi2, im1_t, im1_phi1, img)
            KPlist1 = ComputeSIFTKeypoints(img1)
            KPlist1, temp = Filter_Affine_In_Rect(KPlist1,A1,[0,0],[w,h])
            KPlist1_affine = AffineKPcoor(KPlist1,Ai1)

            img2, mask2, A2, Ai2 = SimulateAffineMap(im2_zoom, im2_phi2, im2_t, im2_phi1, img)
            KPlist2 = ComputeSIFTKeypoints(img2)
            KPlist2, temp = Filter_Affine_In_Rect(KPlist2,A2,[0,0],[w,h])
            KPlist2_affine = AffineKPcoor(KPlist2,Ai2)

            for i in np.random.permutation( range(0,len(KPlist)) ):
                idx1 = FilterKPsinList(KPlist[i], KPlist1_affine)
                idx2 = FilterKPsinList(KPlist[i], KPlist2_affine)
                if self.OutputNegativePatch:
                    idxN = FilterKPsinList(KPlist[i], KPlist2_affine, maxdist = 10*MaxSameKP_dist)
                    idxN = list( set(idxN) - set(idx2) )
                    if len(idxN)==0:
                        idxN = FilterKPsinList(KPlist[i], KPlist2_affine, maxdist = 100*MaxSameKP_dist)
                        idxN = list( set(idxN) - set(idx2) )
                    if len(idxN)==0:
                        idxN = range(len(KPlist2))
                        idxN = list( set(idxN) - set(idx2) )                        
                    if len(idxN)>0:                      
                        idxN = idxN[ random.randint(0,len(idxN)-1) ]
                        oN, lN, sN = unpackSIFTOctave(KPlist2[idxN])
                    else:
                        idxN = None
                sidx1, sidx2 = FindBestKPinLists( im1_zoom, im2_zoom, [KPlist1_affine[i] for i in idx1],[KPlist2_affine[i] for i in idx2])
                if np.size(idx1)>0 and np.size(idx2)>0 and sidx1 !=None and sidx2 !=None and (self.OutputNegativePatch==False or idxN!=None):
                    idx1 = idx1[sidx1:sidx1+1]
                    idx2 = idx2[sidx2:sidx2+1]

                    o, l, s = unpackSIFTOctave(KPlist1[idx1[0]])
                    pyr1 = buildGaussianPyramid( img1, o+2 )
                    o, l, s = unpackSIFTOctave(KPlist2[idx2[0]])
                    if self.OutputNegativePatch:
                        o = np.max([o,oN])
                    pyr2 = buildGaussianPyramid( img2, o+2 )

                    patches1, A_list1, Ai_list1 = ComputePatches(KPlist1[idx1[0]:idx1[0]+1],pyr1)
                    patches2, A_list2, Ai_list2 = ComputePatches(KPlist2[idx2[0]:idx2[0]+1],pyr2)

                    hs, ws = patches1[0].shape[:2]
                    p = np.zeros((hs, ws), np.uint8)
                    p[:] = 1

                    AS1 = ComposeAffineMaps(A_list1[0],A1)
                    AS2 = ComposeAffineMaps(A_list2[0],A2)
                    ASi1 = ComposeAffineMaps(Ai1,Ai_list1[0])
                    ASi2 = ComposeAffineMaps(Ai2,Ai_list2[0])


                    A_from_1_to_2 = ComposeAffineMaps(AS2,ASi1)
                    A_from_2_to_1 = ComposeAffineMaps(AS1,ASi2)

                    kp_sq = SquareOrderedPts(hs,ws)

                    kp_sq1 = AffineKPcoor(kp_sq,A_from_2_to_1)
                    kp_sq2 = AffineKPcoor(kp_sq,A_from_1_to_2)
                    kpin1 = [pt for k in kp_sq1 for pt in k.pt]
                    kpin2 = [pt for k in kp_sq2 for pt in k.pt]

                    stamp = str(time.time())+'.'+ str(np.random.randint(0,9999))
                    cv2.imwrite(self.save_path+"/"+stamp+".p1.png",patches1[0])
                    cv2.imwrite(self.save_path+"/"+stamp+".p2.png",patches2[0])
                    np.savetxt(self.save_path+"/"+stamp+".vector.txt", np.concatenate((kpin1,kpin2)))

                    if self.OutputNegativePatch:
                        patchesN, A_listN, Ai_listN = ComputePatches([KPlist2[idxN]],pyr2)
                        ASN = ComposeAffineMaps(A_listN[0],A2)
                        ASiN = ComposeAffineMaps(Ai2,Ai_listN[0])

                        A_from_1_to_N = ComposeAffineMaps(ASN,ASi1)
                        A_from_N_to_1 = ComposeAffineMaps(AS1,ASiN)

                        kp_sq1 = AffineKPcoor(kp_sq,A_from_N_to_1)
                        kp_sq2 = AffineKPcoor(kp_sq,A_from_1_to_N)
                        kpin1N = [pt for k in kp_sq1 for pt in k.pt]
                        kpin2N = [pt for k in kp_sq2 for pt in k.pt]

                        cv2.imwrite(self.save_path+"/"+stamp+".pN.png",patchesN[0])
                        np.savetxt(self.save_path+"/"+stamp+".vectorN.txt", np.concatenate((kpin1N,kpin2N)))
                        if AllOutput:
                            return np.dstack((patches1[0]/self.imgdivfactor, patches2[0]/self.imgdivfactor, patchesN[0]/self.imgdivfactor)), self.NormalizeVector(np.concatenate((kpin1,kpin2))), self.NormalizeVector(np.concatenate((kpin1N,kpin2N)))
                        else:
                            return np.dstack((patches1[0]/self.imgdivfactor, patches2[0]/self.imgdivfactor, patchesN[0]/self.imgdivfactor)), self.NormalizeVector(np.concatenate((kpin1,kpin2)))
                    else:
                        # to retreive info do:
                        # np.concatenate((kpin1,kpin2)) = np.loadtxt(self.save_path+"/"+stamp+".vector.txt")
                        return np.dstack((patches1[0]/self.imgdivfactor, patches2[0]/self.imgdivfactor)), self.NormalizeVector(np.concatenate((kpin1,kpin2)))




def SimulateAffineMap(zoom_step,psi,t1_step,phi,img0,mask=None, CenteredAt=None, t2_step = 1.0, inter_flag = cv2.INTER_CUBIC, border_flag = cv2.BORDER_CONSTANT, SimuBlur = True):
    '''
    Computing affine deformations of images as in [https://rdguez-mariano.github.io/pages/imas]
    Let A = R_psi0 * diag(t1,t2) * R_phi0    with t1>t2
          = lambda * R_psi0 * diag(t1/t2,1) * R_phi0

    Parameters given should be as:
    zoom_step = 1/lambda
    t1_step = 1/t1
    t2_step = 1/t2
    psi = -psi0 (in degrees)
    phi = -phi0 (in degrees)

    ASIFT proposed params:
    inter_flag = cv2.INTER_LINEAR
    SimuBlur = True

    Also, another kind of exterior could be:
    border_flag = cv2.BORDER_REPLICATE
    '''

    tx = zoom_step*t1_step
    ty = zoom_step*t2_step
    assert tx>=1 and ty>=1, 'Either scale or t are defining a zoom-in operation. If you want to zoom-in do it manually. tx = '+str(tx)+', ty = '+str(ty)

    img = img0.copy()
    arr = []
    DoCenter = False
    if type(CenteredAt) is list:
        DoCenter = True
        arr = np.array(CenteredAt).reshape(-1,2)

    h, w = img.shape[:2]
    tcorners = SquareOrderedPts(h,w,CV=False)
    if mask is None:
        mask = np.zeros((h, w), np.uint8)
        mask[:] = 255
    A1 = np.float32([[1, 0, 0], [0, 1, 0]])

    if phi != 0.0:
        phi = np.deg2rad(phi)
        s, c = np.sin(phi), np.cos(phi)
        A1 = np.float32([[c,-s], [ s, c]])
        tcorners = np.dot(tcorners, A1.T)
        x, y, w, h = cv2.boundingRect(np.int32(tcorners).reshape(1,-1,2))
        A1 = np.hstack([A1, [[-x], [-y]]])
        if DoCenter and tx == 1.0 and ty == 1.0 and psi == 0.0:
            arr = AffineArrayCoor(arr,A1)[0].ravel()
            h0, w0 = img0.shape[:2]
            A1[0][2] += h0/2.0 - arr[0]
            A1[1][2] += w0/2.0 - arr[1]
            w, h = w0, h0
            img = cv2.warpAffine(img, A1, (w, h), flags=inter_flag, borderMode=border_flag)
        else:
            img = cv2.warpAffine(img, A1, (w, h), flags=inter_flag, borderMode=border_flag)


    h, w = img.shape[:2]
    A2 = np.float32([[1, 0, 0], [0, 1, 0]])
    tcorners = SquareOrderedPts(h,w,CV=False)
    if tx != 1.0 or ty != 1.0:
        sx = 0.8*np.sqrt(tx*tx-1)
        sy = 0.8*np.sqrt(ty*ty-1)
        if SimuBlur:
            img = cv2.GaussianBlur(img, (0, 0), sigmaX=sx, sigmaY=sy)
        A2[0] /= tx
        A2[1] /= ty

    if psi != 0.0:
        psi = np.deg2rad(psi)
        s, c = np.sin(psi), np.cos(psi)
        Apsi = np.float32([[c,-s], [ s, c]])
        Apsi = np.matmul(Apsi,A2[0:2,0:2])
        tcorners = np.dot(tcorners, Apsi.T)
        x, y, w, h = cv2.boundingRect(np.int32(tcorners).reshape(1,-1,2))
        A2[0:2,0:2] = Apsi
        A2[0][2] -= x
        A2[1][2] -= y



    if tx != 1.0 or ty != 1.0 or psi != 0.0:
        if DoCenter:
            A = ComposeAffineMaps(A2,A1)
            arr = AffineArrayCoor(arr,A)[0].ravel()
            h0, w0 = img0.shape[:2]
            A2[0][2] += h0/2.0 - arr[0]
            A2[1][2] += w0/2.0 - arr[1]
            w, h = w0, h0
        img = cv2.warpAffine(img, A2, (w, h), flags=inter_flag, borderMode=border_flag)

    A = ComposeAffineMaps(A2,A1)

    if psi!=0 or phi != 0.0 or tx != 1.0 or ty != 1.0:
        if DoCenter:
            h, w = img0.shape[:2]
        else:
            h, w = img.shape[:2]
        mask = cv2.warpAffine(mask, A, (w, h), flags=inter_flag)
    Ai = cv2.invertAffineTransform(A)
    return img, mask, A, Ai


def unpackSIFTOctave(kp, XI=False):
    ''' Opencv packs the true octave, scale and layer inside kp.octave.
    This function computes the unpacking of that information.
    '''
    _octave = kp.octave
    octave = _octave&0xFF
    layer  = (_octave>>8)&0xFF
    if octave>=128:
        octave |= -128
    if octave>=0:
        scale = float(1/(1<<octave))
    else:
        scale = float(1<<-octave)

    if XI:
        yi = (_octave>>16)&0xFF
        xi = yi/255.0 - 0.5
        return octave, layer, scale, xi
    else:
        return octave, layer, scale

def packSIFTOctave(octave, layer, xi=0.0):
    po = octave&0xFF
    pl = (layer&0xFF)<<8
    pxi = round((xi + 0.5)*255.0)&0xFF
    pxi = pxi<<16
    return  po + pl + pxi

def DescRadius(kp, InPyr=False, SIFT=False):
    ''' Computes the Descriptor radius with respect to either an image
        in the pyramid or to the original image.
    '''
    factor = siftparams.new_radius_descr
    if SIFT:
        factor = siftparams.lambda_descr
    if InPyr:
        o, l, s = unpackSIFTOctave(kp)
        return( np.float32(kp.size*s*factor*0.5) )
    else:
        return( np.float32(kp.size*factor*0.5) )


def AngleDiff(a,b, InRad=False):
    ''' Computes the Angle Difference between a and b.
        0<=a,b<=360
    '''
    if InRad:
        a = np.rad2deg(a) % 360
        b = np.rad2deg(b) % 360
    assert a>=0 and a<=360 and b>=0 and b<=360, 'a = '+str(a)+', b = '+str(b)
    anglediff = abs(a-b)% 360
    if anglediff > 180:
        anglediff = 360 - anglediff
    
    if InRad:
        return np.deg2rad(anglediff)
    else:
        return anglediff



def FilterKPsinList(kp0,kp_list,maxdist = MaxSameKP_dist, maxangle = MaxSameKP_angle):
    ''' Filters out all keypoints in kp_list having angle differences and distances above some thresholds.
     Those comparisons should be made with restpect to the groundtruth image.
    '''
    idx = () # void tuple
    for i in range(0,np.size(kp_list)):
        dist = cv2.norm(kp0.pt,kp_list[i].pt)
        anglediff = AngleDiff( kp0.angle , kp_list[i].angle )
        if dist<maxdist and anglediff<maxangle:
            idx += tuple([i])
    return idx


def FilterOutUselessKPs(kplist,patches, TensorThres = 10.0):
    ''' The Structure Tensor is used here to filter out unidimensional patches, i.e,
    patches for which there is only information in one dimension.
    For that we demand l_max / l_min less than a threshold (where l_i are 
    the eigenvalues of the structure matrix)
    
    In practice if A = k B,     A*B = k B^2
    			(A + B)^2 = (k+1)^2 * B^2
    			k B^2 >  t * (k+1)^2 * B^2 sii   k  / (k+1)^2 > t
    This is a decreasing function for k > 1 and value 0.3 at k=1.
          f(k) = k  / (k+1)^2
    Setting t = 0.08, means k<=10
    '''
    tensor_thres = TensorThres / pow( 1 + TensorThres ,2)
    rkplist, rpatches = [], []
    for n in range(len(kplist)):
        dx = cv2.Sobel(patches[n],cv2.CV_64F,1,0,ksize = 1)
        dy = cv2.Sobel(patches[n],cv2.CV_64F,0,1,ksize = 1)
        ts_xy = (dx*dy).sum()
        ts_xx = (dx*dx).sum()
        ts_yy = (dy*dy).sum()
        det = ts_xx * ts_yy - ts_xy * ts_xy # \prod l_i
        trace = ts_xx + ts_yy # \sum l_i
        if ((det > tensor_thres * trace * trace)):
            rkplist.append(kplist[n])
            rpatches.append(patches[n])
    # print(len(kplist),len(rkplist))
    return rkplist, rpatches
        

def FindBestKPinLists(lambda1,lambda2, kp_list1, kp_list2):
    ''' Finds the best pair (i,j) such that kp_list1[i] equals kp_list2[j] in
        the groundtruth image. The groundtruth image was zoom-out by a factor lambda1
        for the image corresponding to kp_list1, and same goes for lambda2 and kp_list2.
    '''
    idx1 = None
    idx2 = None
    mindist = MaxSameKP_dist
    mindiffsizes = 10.0
    for i in range(0,np.size(kp_list1)):
        kp1 = kp_list1[i]
        size1 = DescRadius(kp1)*lambda1
        for j in range(0,np.size(kp_list2)):
            kp2 = kp_list2[j]
            size2 = DescRadius(kp2)*lambda2
            dist = cv2.norm(kp1.pt,kp2.pt)
            diffsizes = abs(size1 - size2)
            if diffsizes<mindiffsizes or (dist<mindist and diffsizes==mindiffsizes) :
                mindist = dist
                mindiffsizes = diffsizes
                # print(size1, size2, diffsizes)
                idx1 = i
                idx2 = j
    return idx1, idx2


def features_deepcopy(f):
    return [cv2.KeyPoint(x = k.pt[0], y = k.pt[1],
            size = k.size, angle = k.angle,
            response = k.response, octave = k.octave,
            class_id = k.class_id) for k in f]

def matches_deepcopy(f):
    return [cv2.DMatch(k.queryIdx, k.trainIdx, k.distance) for k in f]

def Filter_Affine_In_Rect(kp_list, A, p_min, p_max, desc_list = None, isSIFT=False):
    ''' Filters out all descriptors in kp_list that do not lay inside the
    the parallelogram defined by the image of a rectangle by the affine transform A.
    The rectangle is defined by (p_min,p_max).
    '''
    desc_listing = False
    desc_list_in = []
    desc_pos = 0
    if type(desc_list) is np.ndarray:
        desc_listing = True
        desc_list_in = desc_list.copy()
    x1, y1 = p_min[:2]
    x2, y2 = p_max[:2]
    Ai = cv2.invertAffineTransform(A)
    kp_back = AffineKPcoor(kp_list,Ai)
    kp_list_in = []
    kp_list_out = []
    cyclic_corners = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]])
    cyclic_corners = AffineArrayCoor(cyclic_corners,A)
    for i in range(0,np.size(kp_back)):
        if kp_back[i].pt[0]>=x1 and kp_back[i].pt[0]<x2 and kp_back[i].pt[1]>=y1 and kp_back[i].pt[1]<y2:
            In = True
            r = DescRadius(kp_list[i],SIFT=isSIFT)*1.4142
            for j in range(0,4):
                if r > dist_pt_to_line(kp_list[i].pt,cyclic_corners[j],cyclic_corners[j+1]):
                    In = False
            if In == True:
                if desc_listing:
                    desc_list_in[desc_pos,:]= desc_list[i,:]
                    desc_pos +=1
                kp_list_in.append(kp_list[i])
            else:
                kp_list_out.append(kp_list[i])
        else:
            kp_list_out.append(kp_list[i])
    if desc_listing:
        return kp_list_in, desc_list_in[:desc_pos,:], kp_list_out
    else:
        return kp_list_in, kp_list_out




def dist_pt_to_line(p,p1,p2):
    ''' Computes the distance of a point (p) to a line defined by two points (p1, p2). '''
    x0, y0 = np.float32(p[:2])
    x1, y1 = np.float32(p1[:2])
    x2, y2 = np.float32(p2[:2])
    dist = abs( (y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1 ) / np.sqrt( pow(y2-y1,2) + pow(x2-x1,2) )
    return dist


def PolarCoor_from_vector(p_source,p_arrow):
    ''' It computes the \rho and \theta such that
        \rho * exp( i * \theta ) = p_arrow-p_source
    '''
    p = np.array(p_arrow)- np.array(p_source)
    rho = np.linalg.norm(p)
    theta = 0
    if rho>0:
        theta = np.arctan2(p[1],p[0])
        theta = np.rad2deg(theta % (2 * np.pi))
    return  rho, theta


def ComposeAffineMaps(A_lhs,A_rhs):
    ''' Comutes the composition of affine maps:
        A = A_lhs ∘ A_rhs
    '''
    A = np.matmul(A_lhs[0:2,0:2],A_rhs)
    A[:,2] += A_lhs[:,2]
    return A

def kp2LocalAffine(kp, w=60,h=60):
    ''' Computes the affine map A such that: for any x 
    living in the image coordinates A(x) is the 
    corresponding coordinates of x in the patch computed 
    from the keypoint kp.
    '''
    scale = siftparams.new_radius_descr/DescRadius(kp)
    x, y= kp.pt[0], kp.pt[1]
    angle = 360.0 - kp.angle
    if(np.abs(angle - 360.0) < siftparams.flt_epsilon):
        angle = 0.0
    phi = np.deg2rad(angle)
    s, c = np.sin(phi), np.cos(phi)
    R = np.float32([[c,-s], [ s, c]])
    A = scale * np.float32([[1, 0, -x], [0, 1, -y]])
    A = np.matmul(R,A)
    A[:,2] += np.array([w/2, h/2])
    return A    


def AffineArrayCoor(arr,A):
    if type(arr) is list:
        arr = np.array(arr).reshape(-1,2)
    AA = A[0:2,0:2]
    Ab = A[:,2]
    arr_out = []
    for j in range(0,arr.shape[0]):
        arr_out.append(np.matmul(AA,np.array(arr[j,:])) + Ab )
    return np.array(arr_out)

def AffineKPcoor(kp_list,A, Pt_mod = True, Angle_mod = True):
    ''' Transforms information details on each kp_list keypoints by following
        the affine map A.
    '''
    kp_affine = features_deepcopy(kp_list)
    AA = A[0:2,0:2]
    Ab = A[:,2]
    for j in range(0,np.size(kp_affine)):
        newpt = tuple( np.matmul(AA,np.array(kp_list[j].pt)) + Ab)
        if Pt_mod:
            kp_affine[j].pt = newpt
        if Angle_mod:
            phi = np.deg2rad(kp_list[j].angle)
            s, c = np.sin(phi), np.cos(phi)
            R = np.float32([[c,-s], [ s, c]])
            p_arrow = np.matmul( R , [50.0, 0.0] ) + np.array(kp_list[j].pt)
            p_arrow = tuple( np.matmul(AA,p_arrow) + Ab)
            rho, kp_affine[j].angle =  PolarCoor_from_vector(newpt, p_arrow)
    return kp_affine


def ProposeOpticallyOverlapedPatches(P1,A_p1_to_p2,P2, w=60,h=60):
    A = cv2.invertAffineTransform(A_p1_to_p2)
    Avec = affine_decomp(A)
    Ai = A_p1_to_p2
    Aivec = affine_decomp(Ai)
    center = [h/2.0, w/2.0]
    t1 = Avec[0]*Avec[2]
    t2 = Avec[0]
    if t1>=1 and t2>=1: # Only Patch 1 loose information
        lambdastar = 1.0
        prop_p2, mask2, sA2, sAi2 = SimulateAffineMap(lambdastar,0.0,1.0,0.0,P2,mask=None,CenteredAt=center)
        center = list( AffineArrayCoor(center,A).ravel() )
        prop_p1, mask1, sA1, sAi1 = SimulateAffineMap(lambdastar*Avec[0],-np.rad2deg(Aivec[1]),1.0,-np.rad2deg(Aivec[3]),P1,mask=None,CenteredAt=center,t2_step = Aivec[2])
        # Why the above line is really computing A^-1 ?
        # Read the following:
        # Let   A  =  R_phi * diag(t1,t2) * R_psi  =
        #          =  t2 * R_phi * diag(t1/t2,1) * R_psi
        # which implies A^-1 = 1/t1 * R_(-pi/2-psi) * diag(t1/t2) * R_(pi/2 - phi)
        # So
        # t1_step_{A^-1} = 1/t1_{A^-1} = t2
        # t2_step_{A^-1} = 1/t2_{A^-1} = t1
        # So, as Avec[0] = t2,
        # t2*diag(1,t1/t2) = diag(t2,t1) = diag(t1_step_{A^-1}, t2_step_{A^-1})
    elif t1<1 and t2<1: # Only Patch 2 loose information
        lambdastar = 1.0
        prop_p1, mask1, sA1, sAi1 = SimulateAffineMap(lambdastar,0.0,1.0,0.0,P1,mask=None,CenteredAt=center)
        center = list( AffineArrayCoor(center,Ai).ravel() )
        prop_p2, mask2, sA2, sAi2 = SimulateAffineMap(lambdastar*Aivec[0],-np.rad2deg(Avec[1]),1.0,-np.rad2deg(Avec[3]),P2,mask=None,CenteredAt=center,t2_step = Avec[2])
    else: # Both Patches loose information
        if t2 < 1.0/t1: # Avec[0] < Aivec[0]
            lambdastar = 1.0/Avec[0] + 0.000001
            prop_p1, mask1, sA1, sAi1 = SimulateAffineMap(lambdastar,0.0,1.0,0.0,P1,mask=None,CenteredAt=center)
            center = list( AffineArrayCoor(center,Ai).ravel() )
            prop_p2, mask2, sA2, sAi2 = SimulateAffineMap(lambdastar*Aivec[0],-np.rad2deg(Avec[1]),1.0,-np.rad2deg(Avec[3]),P2,mask=None,CenteredAt=center,t2_step = Avec[2])
        else:
            lambdastar = 1.0/Aivec[0] + 0.000001
            prop_p2, mask2, sA2, sAi2 = SimulateAffineMap(lambdastar,0.0,1.0,0.0,P2,mask=None,CenteredAt=center)
            center = list( AffineArrayCoor(center,A).ravel() )
            prop_p1, mask1, sA1, sAi1 = SimulateAffineMap(lambdastar*Avec[0],-np.rad2deg(Aivec[1]),1.0,-np.rad2deg(Aivec[3]),P1,mask=None,CenteredAt=center,t2_step = Aivec[2])
    prop_p1[prop_p1<0] = 0
    prop_p1[prop_p1>255] = 255
    prop_p2[prop_p2<0] = 0
    prop_p2[prop_p2>255] = 255
    mask = mask1*mask2
    return prop_p1, prop_p2, mask


def affine_decomp2affine(vec):
    lambda_scale = vec[0]
    phi2 = vec[1]
    t = vec[2]
    phi1 = vec[3]

    s, c = np.sin(phi1), np.cos(phi1)
    R_phi1 = np.float32([[c,s], [ -s, c]])
    s, c = np.sin(phi2), np.cos(phi2)
    R_phi2 = np.float32([[c,s], [ -s, c]])

    A = lambda_scale * np.matmul(R_phi2, np.matmul(np.diag([t,1.0]),R_phi1) )
    if np.shape(vec)[0]==6:
        A = np.concatenate(( A, [[vec[4]], [vec[5]]] ), axis=1)
    return A


def affine_decomp(A0,doAssert=True, ModRots=False):
    '''Decomposition of a 2x2 matrix A (whose det(A)>0) satisfying
        A = lambda*R_phi2*diag(t,1)*R_phi1.
        where lambda and t are scalars, and R_phi1, R_phi2 are rotations.
    '''
    epsilon = 0.0001
    A = A0[0:2,0:2]
    Adet = np.linalg.det(A)
    if doAssert:
        assert Adet>0

    if Adet>0:
        #   A = U * np.diag(s) * V
        U, s, V = np.linalg.svd(A, full_matrices=True)
        T = np.diag(s)
        K = np.float32([[-1, 0], [0, 1]])

        # K*D*K = D
        if ((np.linalg.norm(np.linalg.det(U)+1)<=epsilon) and (np.linalg.norm(np.linalg.det(V)+1)<=epsilon)):
            U = np.matmul(U,K)
            V = np.matmul(K,V)

        phi2_drift = 0.0
        # Computing First Rotation
        phi1 = np.arctan2( V[0,1], V[0,0] )
        if ModRots and phi1<0:
            phi1 = phi1 + np.pi
            phi2_drift = -np.pi

        # Computing Second Rotation
        phi2 = np.mod(  np.arctan2( U[0,1],U[0,0]) + phi2_drift  , 2.0*np.pi)

        # Computing Tilt and Lambda
        lambda_scale = T[1,1]
        T[0,0]=T[0,0]/T[1,1]
        T[1,1]=1.0

        if T[0,0]-1.0<=epsilon:
            phi2 = np.mod(phi1+phi2,2.0*np.pi)
            phi1 = 0.0
        
        s, c = np.sin(phi1), np.cos(phi1)
        R_phi1 = np.float32([[c,s], [ -s, c]])
        s, c = np.sin(phi2), np.cos(phi2)
        R_phi2 = np.float32([[c,s], [ -s, c]])
        
        temp = lambda_scale*np.matmul( R_phi2 ,np.matmul(T,R_phi1) )

        # Couldnt decompose A
        if doAssert and np.linalg.norm(A - temp,'fro')>epsilon:
            print('Error: affine_decomp couldnt really decompose A')
            print(A0)
            print('----- end of A')

        rvec = [lambda_scale, phi2, T[0,0], phi1]
        if np.shape(A0)[1]==3:
            rvec = np.concatenate(( rvec, [A0[0,2], A0[1,2]] ))
    else:
        rvec = [1.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    return rvec


def transition_tilt( Avec, Bvec ):
    ''' Computes the transition tilt between two affine maps as in [https://rdguez-mariano.github.io/pages/imas]
    Let
    A = lambda1 * R_phi1 * diag(t,1) * psi1
    B = lambda2 * R_phi2 * diag(s,1) * psi2
    then Avec and Bvec are respectively the affine_decomp of A and B
    '''
    t = Avec[2]
    psi1 = Avec[3]
    s = Bvec[2]
    psi2 = Bvec[3]
    cos_2 = pow( np.cos(psi1-psi2), 2.0)
    g = ( pow(t/s, 2.0) + 1.0 )*cos_2 + ( 1.0/pow(s, 2.0) + pow(t,2.0) )*( 1.0 - cos_2 )
    G = (s/t)*g/2.0
    tau = G + np.sqrt( pow(G,2.0) - 1.0 )
    return tau

def ComputeSIFTKeypoints(img, Desc = False, MaxNum = -1):
    gray = []
    if len(img.shape)!=2:
        gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    else:
        gray = img.view()

    sift = cv2.xfeatures2d.SIFT_create(
    nfeatures = siftparams.nfeatures,
    nOctaveLayers = siftparams.nOctaveLayers, contrastThreshold = siftparams.contrastThreshold,
    edgeThreshold = siftparams.edgeThreshold, sigma = siftparams.sigma
    )
    if Desc:
        kp, des = sift.detectAndCompute(gray,None)
        if MaxNum>0 and len(kp)>MaxNum:
            responses = [k.response for k in kp]
            idxs = np.fliplr( np.reshape(np.argsort(responses),(1,-1)) ).reshape(-1) 
            kpF = []
            desF =np.zeros(shape=(MaxNum,des.shape[1]), dtype=des.dtype)
            for n in range(MaxNum):
                kpF.append( kp[idxs[n]] )
                desF[n,:] = des[idxs[n],:]
            return kpF, desF
        else:
            return kp, des
    else:
        kp = sift.detect(gray,None)
        if MaxNum>0 and len(kp)>MaxNum:
            responses = [k.response for k in kp]
            idxs = np.fliplr( np.reshape(np.argsort(responses),(1,-1)) ).reshape(-1) 
            kpF = []
            for n in range(MaxNum):
                kpF.append( kp[idxs[n]] )
            return kpF
        else:            
            return kp


def ComputePatches(kp_list,gpyr, border_mode = cv2.BORDER_CONSTANT):
    ''' Computes the associated patch to each keypoint in kp_list.
        Returns:
        img_list - list of patches.
        A_list - lists of affine maps A such that A(BackgroundImage)*1_{[0,2r]x[0,2r]} = patch.
        Ai_list - list of the inverse of the above affine maps.

    '''
    img_list = []
    A_list = []
    Ai_list = []
    for i in range(0,np.size(kp_list)):
        kpt = kp_list[i]
        octave, layer, scale = unpackSIFTOctave(kpt)
        assert octave >= siftparams.firstOctave and layer <= siftparams.nOctaveLayers+2, 'octave = '+str(octave)+', layer = '+str(layer)
        # formula in opencv:  kpt.size = sigma*powf(2.f, (layer + xi) / nOctaveLayers)*(1 << octv)*2
        step = kpt.size*scale*0.5 # sigma*powf(2.f, (layer + xi) / nOctaveLayers)
        ptf = np.array(kpt.pt)*scale
        angle = 360.0 - kpt.angle
        if(np.abs(angle - 360.0) < siftparams.flt_epsilon):
            angle = 0.0

        img = gpyr[(octave - siftparams.firstOctave)*(siftparams.nOctaveLayers + 3) + layer]

        r = siftparams.new_radius_descr

        phi = np.deg2rad(angle)
        s, c = np.sin(phi), np.cos(phi)
        A = np.float32([[c,-s], [ s, c]]) / step
        Rptf = np.matmul(A,ptf)
        x = Rptf[0]-r
        y = Rptf[1]-r
        A = np.hstack([A, [[-x], [-y]]])

        dim = np.int32(2*r+1)
        img = cv2.warpAffine(img, A, (dim, dim), flags=cv2.INTER_CUBIC, borderMode=border_mode)
        #print('Octave =', octave,'; Layer =', layer, '; Scale =', scale,'; Angle =',angle)

        oA = np.float32([[1, 0, 0], [0, 1, 0]]) * scale
        A = ComposeAffineMaps(A,oA)
        Ai = cv2.invertAffineTransform(A)
        img_list.append(img.astype(np.float32))
        A_list.append(A)
        Ai_list.append(Ai)
    return img_list, A_list, Ai_list


def ComputeSimilaritiesFromKPs(kp_list):
    A_list = []
    Ai_list = []
    for i in range(0,np.size(kp_list)):
        kpt = kp_list[i]
        octave, layer, scale = unpackSIFTOctave(kpt)
        assert octave >= siftparams.firstOctave and layer <= siftparams.nOctaveLayers+2, 'octave = '+str(octave)+', layer = '+str(layer)
        step = kpt.size*scale*0.5 # sigma*powf(2.f, (layer + xi) / nOctaveLayers)
        ptf = np.array(kpt.pt)*scale
        angle = 360.0 - kpt.angle
        if(np.abs(angle - 360.0) < siftparams.flt_epsilon):
            angle = 0.0

        r = siftparams.new_radius_descr

        phi = np.deg2rad(angle)
        s, c = np.sin(phi), np.cos(phi)
        A = np.float32([[c,-s], [ s, c]]) / step
        Rptf = np.matmul(A,ptf)
        x = Rptf[0]-r
        y = Rptf[1]-r
        A = np.hstack([A, [[-x], [-y]]])

        oA = np.float32([[1, 0, 0], [0, 1, 0]]) * scale
        A = ComposeAffineMaps(A,oA)
        Ai = cv2.invertAffineTransform(A)
        A_list.append(A)
        Ai_list.append(Ai)
    return A_list, Ai_list

def SaveImageWithKeys(img, keys, pathname, rootfolder='temp/', Flag=2):
        colors=( (0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 255, 255) )
        if len(np.shape(img))==2:
            patch = cv2.cvtColor( img.astype(np.uint8), cv2.COLOR_GRAY2RGB )
        else:
            patch = img.astype(np.uint8)
        if len(keys)>len(colors):
            patch=cv2.drawKeypoints(patch,keys,patch, flags=Flag)
        else:
            for n in range(np.min([len(colors),len(keys)])):
                patch=cv2.drawKeypoints(patch,keys[n:n+1],patch, color=colors[n] ,flags=Flag)
        cv2.imwrite(rootfolder+pathname,patch)


def buildGaussianPyramid( base, LastOctave ):
    '''
    Computing the Gaussian Pyramid as in opencv SIFT
    '''
    if siftparams.graydesc and len(base.shape)!=2:
        base = cv2.cvtColor(base,cv2.COLOR_BGR2GRAY)
    else:
        base = base.copy()

    if siftparams.firstOctave<0:
        sig_diff = np.sqrt( max(siftparams.sigma * siftparams.sigma - siftparams.sift_init_sigma * siftparams.sift_init_sigma * 4, 0.01) )
        base = cv2.resize(base, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR_EXACT)

    rows, cols = base.shape[:2]

    nOctaves = np.round(np.log( np.float32(min( cols, rows ))) / np.log(2.0) - 2) - siftparams.firstOctave
    nOctaves = min(nOctaves,LastOctave)
    nOctaves = np.int32(nOctaves)

    sig = ([siftparams.sigma])
    k = np.float32(pow( 2.0 , 1.0 / np.float32(siftparams.nOctaveLayers) ))

    for i in range(1,siftparams.nOctaveLayers + 3):
        sig_prev = pow(k, np.float32(i-1)) * siftparams.sigma
        sig_total = sig_prev*k
        sig += ([ np.sqrt(sig_total*sig_total - sig_prev*sig_prev) ])

    assert np.size(sig) == siftparams.nOctaveLayers + 3

    pyr = []
    for o in range(nOctaves):
        for i in range(siftparams.nOctaveLayers + 3):
            pyr.append([])

    assert len(pyr) == nOctaves*(siftparams.nOctaveLayers + 3)


    for o in range(nOctaves):
        for i in range(siftparams.nOctaveLayers + 3):
            if o == 0  and  i == 0:
                pyr[o*(siftparams.nOctaveLayers + 3) + i] = base.copy()
            elif i == 0:
                src = pyr[(o-1)*(siftparams.nOctaveLayers + 3) + siftparams.nOctaveLayers]
                srcrows, srccols = src.shape[:2]
                pyr[o*(siftparams.nOctaveLayers + 3) + i] = cv2.resize(src, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
            else:
                src = pyr[o*(siftparams.nOctaveLayers + 3) + i-1]
                pyr[o*(siftparams.nOctaveLayers + 3) + i] = cv2.GaussianBlur(src, (0, 0), sigmaX=sig[i], sigmaY=sig[i])
    return(pyr)

def load_image(path):
    img = cv2.imread(path, 1)
    # OpenCV loads images with color channels
    # in BGR order. So we need to reverse them
    return img[...,::-1]

def FirstOrderApprox_Homography(H0, X0=np.array([[0],[0],[1]])):
    ''' Computes the first order Taylor approximation (which is an affine map)
    of the Homography H0 centered at X0 (X0 is in homogeneous coordinates).
    '''
    X0 = np.array( X0 ).reshape(-1,1)
    H = H0.copy()
    col3 = np.matmul(H,X0)
    H[:,2] = col3.reshape(3)
    A = np.zeros((2,3), dtype = np.float32)
    A[0:2,0:2] = H[0:2,0:2]/H[2,2] - np.array([ H[0,2]*H[2,0:2], H[1,2]*H[2,0:2] ])/pow(H[2,2],2)
    A[:,2] = H[0:2,2]/H[2,2] - np.matmul( A[0:2,0:2], X0[0:2,0]/X0[2,0] )
    return A


def AffineFit(Xi,Yi):
    assert np.shape(Xi)[0]==np.shape(Yi)[0] and np.shape(Xi)[1]==2 and np.shape(Yi)[1]==2
    n = np.shape(Xi)[0]
    A = np.zeros((2*n,6),dtype=np.float32)
    b = np.zeros((2*n,1),dtype=np.float32)
    for i in range(0,n):
        A[2*i,0] = Xi[i,0]
        A[2*i,1] = Xi[i,1]
        A[2*i,2] = 1.0
        A[2*i+1,3] = Xi[i,0]
        A[2*i+1,4] = Xi[i,1]
        A[2*i+1,5] = 1.0

        b[2*i,0] = Yi[i,0]
        b[2*i+1,0] = Yi[i,1]
    result = np.linalg.lstsq(A,b,rcond=None)
    return result[0].reshape((2, 3))


def SquareOrderedPts(hs,ws,CV=True):
    # Patch starts from the origin
    ws = ws - 1
    hs = hs - 1
    if CV:
        return [
            cv2.KeyPoint(x = 0,  y =0, size = 10, angle = 0, response = 1.0, octave = 0, class_id = 0),
            cv2.KeyPoint(x = ws, y =0, size = 10, angle = 0, response = 1.0, octave = 0, class_id = 0),
            cv2.KeyPoint(x = ws, y =hs, size = 10, angle = 0, response = 1.0, octave = 0, class_id = 0),
            cv2.KeyPoint(x = 0,  y =hs, size = 10, angle = 0, response = 1.0, octave = 0, class_id = 0)
            ]
    else:
        # return np.float32([ [0,0], [ws+1,0], [ws+1, hs+1], [0,hs+1] ])
        return np.float32([ [0,0], [ws,0], [ws, hs], [0,hs] ])

def Flatten2Pts(vec):
    X = np.zeros( (np.int32(len(vec)/2), 2), np.float32)
    X[:,0] = vec[0::2]
    X[:,1] = vec[1::2]
    return X

def Pts2Flatten(X):
    h,w= np.shape(X)[:2]
    vec = np.zeros( (h*w), np.float32)
    vec[0::2] = X[:,0]
    vec[1::2] = X[:,1]
    return vec

def close_per(vec):
    return( np.array(tuple(vec)+tuple([vec[0]])) )

def Check_FirstThreadTouch(GA):
    for file in glob.glob(GA.save_path+"/"+str(GA.GAid)+".threadsdata"):
        if np.loadtxt(file)>0.5:
            return True
        else:
            return False
    Set_FirstThreadTouch(GA,False)
    return False


def Set_FirstThreadTouch(GA,value):
    np.savetxt(GA.save_path+"/"+str(GA.GAid)+".threadsdata", [value])

def get_big_epoch_number(GA):
    return np.loadtxt(GA.save_path+"/"+str(GA.GAid)+".big_epoch")

def set_big_epoch_number(GA,value):
    # print(GA.save_path+"/big_epoch  -> "+ str(value))
    np.savetxt(GA.save_path+"/"+str(GA.GAid)+".big_epoch", [value])


def get_process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss

def HumanElapsedTime(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    # print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
    return hours, minutes, seconds

def TouchDir(directory):
    ''' Creates a directory if it doesn't exist
    '''
    if not os.path.exists(directory):
        os.makedirs(directory)

def OnlyUniqueMatches(goodM, KPlistQ, KPlistT, SpatialThres=4, simidist=True):
    ''' Filter out non unique matches with less similarity score
    '''
    uniqueM = []
    doubleM = np.zeros(len(goodM),dtype=bool)
    for i in range(0,len(goodM)):
        if doubleM[i]:
            continue
        bestsim = goodM[i].distance
        bestidx = i
        for j in range(i+1,len(goodM)):
            if  ( cv2.norm(KPlistQ[goodM[i].queryIdx].pt, KPlistQ[goodM[j].queryIdx].pt) < SpatialThres \
            and   cv2.norm(KPlistT[goodM[i].trainIdx].pt, KPlistT[goodM[j].trainIdx].pt) < SpatialThres ):
                doubleM[j] = True
                if (simidist and bestsim<goodM[j].distance) or ((not simidist) and bestsim>goodM[j].distance):
                    bestidx = j
                    bestsim = goodM[j].distance
        uniqueM.append(goodM[bestidx])
    return uniqueM


def CreateSubDesc(emb, coef=0, NewDescRadius=2.1):
    NewDescRadius = NewDescRadius**2
    n = 0
    for i in range(7):
        for j in range(7):
            if (i-3)**2+(j-3)**2<NewDescRadius:
                n=n+1   
    if emb.shape[0]>n:
        if coef==0:
            subemb = np.zeros(shape = tuple([emb.shape[0], 128*n]), dtype = emb.dtype)
        else:
            subemb = coef*np.ones(shape = emb.shape, dtype=emb.dtype)
        n = 0
        for i in range(7):
            for j in range(7):
                if (i-3)**2+(j-3)**2<NewDescRadius:
                    m = i*7 + j
                    subemb[:,n*128:(n+1)*128] = emb[:, m*128:(m+1)*128]
                    n=n+1
        return subemb
    else:
        return emb


class CPPbridge(object):
    def __init__(self,libDApath):
        self.libDA = ctypes.cdll.LoadLibrary(libDApath)
        self.MatcherPtr = 0
        self.last_i1_list = []
        self.last_i2_list = []

        self.libDA.GeometricFilter.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_bool]
        self.libDA.GeometricFilter.restype = None

        self.libDA.vlfeat.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.libDA.vlfeat.restype = ctypes.c_int
        self.libDA.get_vlfeatData.argtypes = [ctypes.c_void_p,ctypes.c_void_p]
        self.libDA.get_vlfeatData.restype = None
        self.libDA.get_NvlfeatData.argtypes = [ctypes.c_void_p]
        self.libDA.get_NvlfeatData.restype = ctypes.c_int
        self.libDA.get_vlfeatPatches.argtypes = [ctypes.c_void_p,ctypes.c_void_p]
        self.libDA.get_vlfeatPatches.restype = None
        self.libDA.get_PatchWidth.argtypes = [ctypes.c_void_p]
        self.libDA.get_PatchWidth.restype = ctypes.c_int
        self.libDA.get_vldescriptors.argtypes = [ctypes.c_void_p,ctypes.c_void_p]
        self.libDA.get_vldescriptors.restype = None
        self.libDA.get_vldescDim.argtypes = [ctypes.c_void_p]
        self.libDA.get_vldescDim.restype = ctypes.c_int

        self.libDA.GeometricFilterFromNodes.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_bool]
        self.libDA.GeometricFilterFromNodes.restype = None
        self.libDA.ArrayOfFilteredMatches.argtypes = [ctypes.c_void_p,ctypes.c_void_p]
        self.libDA.ArrayOfFilteredMatches.restype = None
        self.libDA.NumberOfFilteredMatches.argtypes = [ctypes.c_void_p]
        self.libDA.NumberOfFilteredMatches.restype = ctypes.c_int

        self.libDA.newMatcher.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_float]
        self.libDA.newMatcher.restype = ctypes.c_void_p
        self.libDA.destroyMatcher.argtypes = [ctypes.c_void_p]
        self.libDA.destroyMatcher.restype = None
        self.libDA.KnnMatcher.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        self.libDA.KnnMatcher.restype = None

        self.libDA.BypassGeoFilter.argtypes = [ctypes.c_void_p]
        self.libDA.BypassGeoFilter.restype = None

        self.libDA.ComparePatchesAC.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.libDA.ComparePatchesAC.restype = ctypes.c_bool

        self.libDA.GetData_from_QueryNode.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        self.libDA.GetData_from_QueryNode.restype = None
        self.libDA.GetQueryNodeLength.argtypes = [ctypes.c_void_p]
        self.libDA.GetQueryNodeLength.restype = ctypes.c_int

        self.libDA.LastQueryNode.argtypes = [ctypes.c_void_p]
        self.libDA.LastQueryNode.restype = ctypes.c_void_p
        self.libDA.FirstQueryNode.argtypes = [ctypes.c_void_p]
        self.libDA.FirstQueryNode.restype = ctypes.c_void_p
        self.libDA.NextQueryNode.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.libDA.NextQueryNode.restype = ctypes.c_void_p
        self.libDA.PrevQueryNode.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.libDA.PrevQueryNode.restype = ctypes.c_void_p

        self.libDA.FastMatCombi.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
        self.libDA.FastMatCombi.restype = None

    def ComparePatchesAC(self, P1, P2,mask, img1, img2):
        P1[mask<1] = -1025.0
        P2[mask<1] = -1025.0
        
        h, w = P1.shape[:2]
        CP1 = np.zeros(h*w, dtype = ctypes.c_float)
        CP2 = np.zeros(h*w, dtype = ctypes.c_float)
        CP1[:]= P1.flatten()[:]
        CP2[:]= P2.flatten()[:]

        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        floatp = ctypes.POINTER(ctypes.c_float)
        return self.libDA.ComparePatchesAC(CP1.ctypes.data_as(floatp), 
                                           CP2.ctypes.data_as(floatp),
                                           w,h, w1,h1,w2,h2)

    def GeometricFilter(self, KPlist1, im1, KPlist2, im2, Matches, Filter = 'ORSA_H', precision = 24, verb=False):
        filtercode=0
        if Filter=='ORSA_F':
            filtercode=1
        elif Filter=='USAC_H':
            filtercode=2
        elif Filter=='USAC_F':
            filtercode=3
        
        if len(Matches)==0:
            return [], None
                
        src_pts = np.float32([ KPlist1[m.queryIdx].pt for m in Matches ]).ravel()
        dst_pts = np.float32([ KPlist2[m.trainIdx].pt for m in Matches ]).ravel()
        src_pts = src_pts.astype(ctypes.c_float)
        dst_pts = dst_pts.astype(ctypes.c_float)
        N = int(len(src_pts)/2)
        MatchMask = np.zeros(N, dtype = ctypes.c_bool)
        T = np.zeros(9, dtype = ctypes.c_float)
        floatp = ctypes.POINTER(ctypes.c_float)
        boolp = ctypes.POINTER(ctypes.c_bool)
        h1, w1 = im1.shape[:2]
        h2, w2 = im2.shape[:2]
        self.libDA.GeometricFilter(src_pts.ctypes.data_as(floatp), dst_pts.ctypes.data_as(floatp),
                                    MatchMask.ctypes.data_as(boolp), T.ctypes.data_as(floatp),
                                    N, w1, h1, w2, h2, filtercode, ctypes.c_float(precision), verb)
        Consensus = []
        for i in range(0,len(MatchMask)):
            if MatchMask[i]==True:
                Consensus.append(Matches[i])
        return Consensus, T.astype(np.float).reshape(3,3)

    def call_vlfeat(self, image, method):
        gray = []
        dataDim = 10
        if len(image.shape)!=2:
            gray= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        else:
            gray = image.view()
        h, w = gray.shape[:2]
        im = np.zeros(h*w, dtype = ctypes.c_float)
        im[:] = gray.reshape(-1)[:]
        floatp = ctypes.POINTER(ctypes.c_float)
        ucharp = ctypes.POINTER(ctypes.c_ubyte)

        self.libDA.vlfeat(self.MatcherPtr, im.ctypes.data_as(floatp), ctypes.c_int(w), ctypes.c_int(h), ctypes.c_int(method))
        NoD = self.libDA.get_NvlfeatData(self.MatcherPtr)
        Data = np.zeros(dataDim*NoD, dtype = ctypes.c_float)
        floatp = ctypes.POINTER(ctypes.c_float)
        self.libDA.get_vlfeatData(self.MatcherPtr, Data.ctypes.data_as(floatp))           

        pw = self.libDA.get_PatchWidth(self.MatcherPtr)
        PatchesData = np.zeros(pw*pw*NoD,dtype=ctypes.c_ubyte)
        self.libDA.get_vlfeatPatches(self.MatcherPtr, PatchesData.ctypes.data_as(ucharp))
        NormilizedPatches = [ PatchesData[pw*pw*i:pw*pw*(i+1)].reshape(pw,pw) for i in range(NoD)]

        descdim = self.libDA.get_vldescDim(self.MatcherPtr)
        descs = np.zeros(descdim*NoD,dtype=ctypes.c_float)
        self.libDA.get_vldescriptors(self.MatcherPtr, descs.ctypes.data_as(floatp))
        desc_list = descs.reshape(-1,128).astype(np.float32)
        # desc_list = [ descs[descdim*i:descdim*(i+1)] for i in range(NoD) ]

        KPlist = [cv2.KeyPoint(x=Data[dataDim*i+4], y=Data[dataDim*i+5], size=10, angle=0.0,
                               response=1, octave=packSIFTOctave(0,0),class_id=1)
                                for i in range(0,NoD)]

        Alist = [ np.reshape([Data[dataDim*i], Data[dataDim*i+1], Data[dataDim*i+4] , Data[dataDim*i+2], Data[dataDim*i+3], Data[dataDim*i+5] ], [2,3])
                                for i in range(0,NoD)]

        # peakScore, edgeScore, orientationScore, laplacianScaleScore
        Data = Data.reshape(-1,10)
        Scores = Data[:,6:]
        return KPlist, NormilizedPatches, desc_list, Alist, Scores


    def GeometricFilterFromMatcher(self, im1, im2, Filter = 'ORSA_H', precision=24, verb=False):
        filtercode=0
        if Filter=='ORSA_F':
            filtercode=1
        elif Filter=='USAC_H':
            filtercode=2
        elif Filter=='USAC_F':
            filtercode=3
        T = np.zeros(9, dtype = ctypes.c_float)
        floatp = ctypes.POINTER(ctypes.c_float)
        intp = ctypes.POINTER(ctypes.c_int)
        # boolp = ctypes.POINTER(ctypes.c_bool)
        h1, w1 = im1.shape[:2]
        h2, w2 = im2.shape[:2]
        self.libDA.GeometricFilterFromNodes(self.MatcherPtr, T.ctypes.data_as(floatp),
                                    w1, h1, w2, h2, filtercode, ctypes.c_float(precision), ctypes.c_bool(verb))
              
        NFM = self.libDA.NumberOfFilteredMatches(self.MatcherPtr)
        if NFM>0:
            FM = np.zeros(3*NFM, dtype = ctypes.c_int)
            self.libDA.ArrayOfFilteredMatches(self.MatcherPtr,FM.ctypes.data_as(intp))
            # print(NFM,FM)                
            Matches = [cv2.DMatch(FM[3*i], FM[3*i+1], float(FM[3*i+2])) for i in range(0,NFM)]
        else:
            Matches = []
        return Matches, T.astype(np.float).reshape(3,3)
    
    def GetAllMatches(self):
        self.libDA.BypassGeoFilter(self.MatcherPtr)
        NFM = self.libDA.NumberOfFilteredMatches(self.MatcherPtr)
        if NFM>0:
            FM = np.zeros(3*NFM, dtype = ctypes.c_int)
            intp = ctypes.POINTER(ctypes.c_int)
            self.libDA.ArrayOfFilteredMatches(self.MatcherPtr,FM.ctypes.data_as(intp))           
            Matches = [cv2.DMatch(FM[3*i],FM[3*i+1],float(FM[3*i+2])) for i in range(0,NFM)]
        else:
            Matches = []
        return Matches

    def GetMatches_from_QueryNode(self, qn):
        N = self.libDA.GetQueryNodeLength(qn)
        if N>0:
            Query_idx = np.zeros(1, dtype = ctypes.c_int)
            Target_idxs = np.zeros(N, dtype = ctypes.c_int)
            simis = np.zeros(N, dtype = ctypes.c_float)
            floatp = ctypes.POINTER(ctypes.c_float)
            intp = ctypes.POINTER(ctypes.c_int)
            self.libDA.GetData_from_QueryNode(qn, Query_idx.ctypes.data_as(intp), Target_idxs.ctypes.data_as(intp), simis.ctypes.data_as(floatp))
            return [cv2.DMatch(Query_idx[0], Target_idxs[i], float(simis[i])) for i in range(0,N)]
        else:
            return []

    def FirstLast_QueryNodes(self):
        return self.libDA.FirstQueryNode(self.MatcherPtr), self.libDA.LastQueryNode(self.MatcherPtr)

    def NextQueryNode(self, qn):
        return self.libDA.NextQueryNode(self.MatcherPtr, qn)

    def PrevQueryNode(self, qn):
        return self.libDA.PrevQueryNode(self.MatcherPtr, qn)

    def KnnMatch(self,QKPlist,Qdesc, TKPlist, Tdesc, FastCode):
        Nq = ctypes.c_int(np.shape(Qdesc)[0])
        Nt = ctypes.c_int(np.shape(Tdesc)[0])
        Qkps = np.array([x for kp in QKPlist for x in kp.pt],dtype=ctypes.c_float)
        Tkps = np.array([x for kp in TKPlist for x in kp.pt],dtype=ctypes.c_float)        
        floatp = ctypes.POINTER(ctypes.c_float)
        Qdesc = Qdesc.ravel().astype(ctypes.c_float)
        Tdesc = Tdesc.ravel().astype(ctypes.c_float)
        QdescPtr = Qdesc.ctypes.data_as(floatp)
        TdescPtr = Tdesc.ctypes.data_as(floatp)
        QkpsPtr = Qkps.ctypes.data_as(floatp)
        TkpsPtr = Tkps.ctypes.data_as(floatp)
        
        self.libDA.KnnMatcher(self.MatcherPtr,QkpsPtr, QdescPtr, Nq, TkpsPtr, TdescPtr, Nt, ctypes.c_int(FastCode))

    def CreateMatcher(self,desc_dim, k=1, sim_thres=0.7):
        self.MatcherPtr = self.libDA.newMatcher(k,desc_dim,sim_thres)
    
    def DestroyMatcher(self):
        if self.MatcherPtr!=0:
            self.libDA.destroyMatcher(self.MatcherPtr)

    def PrepareForFastMatCombi(self,len_i_list):
        self.last_i1_list = -1*np.ones(shape=(len_i_list), dtype = ctypes.c_int)
        self.last_i2_list = -1*np.ones(shape=(len_i_list), dtype = ctypes.c_int)

    def FastMatCombi(self,bP, i_list, ps1, j_list, ps2, MemStepImg, MemStepBlock):
        intp = ctypes.POINTER(ctypes.c_int)
        floatp = ctypes.POINTER(ctypes.c_float)
        i1_list = i_list.ctypes.data_as(intp)
        i2_list = j_list.ctypes.data_as(intp)
        ps1p = ps1.ctypes.data_as(floatp)
        ps2p = ps2.ctypes.data_as(floatp)
        bPp = bP.ctypes.data_as(floatp)

        last_i1_listp = self.last_i1_list.ctypes.data_as(intp)
        last_i2_listp = self.last_i2_list.ctypes.data_as(intp)

        self.libDA.FastMatCombi(  ctypes.c_int(len(self.last_i1_list)), bPp,
         i1_list, i2_list, ps1p, ps2p, ctypes.c_int(MemStepImg), last_i1_listp, last_i2_listp )

        self.last_i1_list = i_list.copy()
        self.last_i2_list = j_list.copy()

def get_Aq2t_from_SIFTkeys(cvkeys1, cvkeys2, cvMatches, A_p1_to_p2_list=None):
    Aq2t = []
    assert (A_p1_to_p2_list is None or len(cvMatches)==len(A_p1_to_p2_list))
    for i,m in enumerate(cvMatches):
        Akp1 = kp2LocalAffine(cvkeys1[m.queryIdx])
        Akp2 = kp2LocalAffine(cvkeys2[m.trainIdx])
        if A_p1_to_p2_list is None:
            A_p1_to_p2 = np.float32([[1, 0, 0], [0, 1, 0]])
        else:
            A_p1_to_p2 = A_p1_to_p2_list[i]
        A_query_to_p2 = ComposeAffineMaps(A_p1_to_p2, Akp1)
        A_query_to_target = ComposeAffineMaps( cv2.invertAffineTransform(Akp2), A_query_to_p2 )
        Aq2t.append( A_query_to_target )
    return Aq2t

def get_Aq2t_from_NormAffmaps(Affmaps1, Affmaps2, cvMatches, A_p1_to_p2_list=None):
    Aq2t = []
    assert (A_p1_to_p2_list is None or len(cvMatches)==len(A_p1_to_p2_list))
    for i,m in enumerate(cvMatches):
        Akp1 = Affmaps1[m.queryIdx]
        Akp2 = Affmaps2[m.trainIdx]
        if A_p1_to_p2_list is None:
            A_p1_to_p2 = np.float32([[1, 0, 0], [0, 1, 0]])
        else:
            A_p1_to_p2 = A_p1_to_p2_list[i]
        A_query_to_p2 = ComposeAffineMaps(A_p1_to_p2,Akp1)
        A_query_to_target = ComposeAffineMaps( cv2.invertAffineTransform(Akp2), A_query_to_p2 )
        Aq2t.append( A_query_to_target )
    return Aq2t

def get_Aq2t(Alist_q_2_p1, patches1, Alist_t_to_p2, patches2, cvMatches, method='locate', noTranslation=False):
    assert len(Alist_q_2_p1)==len(patches1) and len(Alist_q_2_p1)==len(patches1)
    Aq2t = []
    if len(cvMatches)==0:
        return Aq2t
    A_p1_to_p2_list = []
    if method=="simple":
            A_p1_to_p2 = np.float32([[1, 0, 0], [0, 1, 0]])
    elif method=="locate":
        from libLocalDesc import LOCATEmodel
        GA = GenAffine("", DryRun=True)
        bP = np.zeros(shape=tuple([len(cvMatches),60,60,2]), dtype = np.float32)
        for i,m in enumerate(cvMatches):
            bP[i,:,:,:] = np.dstack((patches1[m.queryIdx]/255.0, patches2[m.trainIdx]/255.0))
        bEsti = LOCATEmodel.layers[2].predict(bP)
        for i,m in enumerate(cvMatches):
            evec = bEsti[i,:]
            A_p1_to_p2 = cv2.invertAffineTransform( GA.AffineFromNormalizedVector(evec) )
            A_p1_to_p2_list.append( A_p1_to_p2 )
    elif method=="affnet":
        from Utils import batched_forward
        from hesaffnet import AffNetPix, USE_CUDA
        from LAF import normalizeLAFs, denormalizeLAFs, convertLAFs_to_A23format
        import torch
        pw, ph = np.shape(patches1[0])
        x, y = pw/2.0 + 2, ph/2.0 + 2     
        baseLAFs1 = normalizeLAFs( torch.tensor([[AffNetPix.PS/2, 0, x], [0, AffNetPix.PS/2, y]]).reshape(1,2,3), pw, ph ).repeat(len(patches1),1,1)
        baseLAFs2 = normalizeLAFs( torch.tensor([[AffNetPix.PS/2, 0, x], [0, AffNetPix.PS/2, y]]).reshape(1,2,3), pw, ph ).repeat(len(patches2),1,1)
        subpatches1 = torch.from_numpy(np.array(patches1)[:,16:48,16:48]).unsqueeze(1)
        subpatches2 = torch.from_numpy(np.array(patches2)[:,16:48,16:48]).unsqueeze(1)
        if USE_CUDA:
            with torch.no_grad():
                A1 = batched_forward(AffNetPix, subpatches1.cuda(), 256).cpu()
                A2 = batched_forward(AffNetPix, subpatches2.cuda(), 256).cpu()
        else:
            with torch.no_grad():
                A1 = AffNetPix(subpatches1)
                A2 = AffNetPix(subpatches2)
        LAFs1 = torch.cat([torch.bmm(A1,baseLAFs1[:,:,0:2]), baseLAFs1[:,:,2:] ], dim =2)
        LAFs2 = torch.cat([torch.bmm(A2,baseLAFs2[:,:,0:2]), baseLAFs2[:,:,2:] ], dim =2)
        dLAFs1 = denormalizeLAFs(LAFs1, pw, ph)
        dLAFs2 = denormalizeLAFs(LAFs2, pw, ph)
        Alist1 = convertLAFs_to_A23format( dLAFs1.detach().cpu().numpy().astype(np.float32) )
        Alist2 = convertLAFs_to_A23format( dLAFs2.detach().cpu().numpy().astype(np.float32) )
        for m in cvMatches:
            A_p1_to_p2_list.append( ComposeAffineMaps( Alist2[m.trainIdx], cv2.invertAffineTransform(Alist1[m.queryIdx]) ) )
    else:
        print("ERROR: "+method+" is not yet implemented")
        exit()

    assert method=="simple" or len(cvMatches)==len(A_p1_to_p2_list)
    A_p1_to_target = []
    for i,m in enumerate(cvMatches):
        Akp1 = Alist_q_2_p1[m.queryIdx]
        Akp2 = Alist_t_to_p2[m.trainIdx]
        if not method=="simple":
            A_p1_to_p2 = A_p1_to_p2_list[i]
        if noTranslation:
            A_p1_to_p2[:,2] = 0
        A_query_to_p2 = ComposeAffineMaps(A_p1_to_p2,Akp1)
        A_query_to_target = ComposeAffineMaps( cv2.invertAffineTransform(Akp2), A_query_to_p2 )
        Aq2t.append( A_query_to_target )
        A_p1_to_target.append( ComposeAffineMaps( cv2.invertAffineTransform(Akp2), A_p1_to_p2 ) )

    return Aq2t, A_p1_to_target


def AffineGridCoor(size,A):
    w2, h2 = size
    xvec = [x for x in np.arange(0, h2)]
    yvec = [y for y in np.arange(0, w2)]
    X,Y = np.meshgrid(xvec, yvec)
    XA = A[0,0]*X + A[0,1] * Y + A[0,2]
    YA = A[1,0]*X + A[1,1] * Y + A[1,2]
    return XA, YA


import copy
def save_randomAffineInliers(img1,img2,KPlist1,KPlist2,A_list1,A_list2, consensus, Aq2t, A_p1_to_target, Nshow =10, patch_width = 60, noTranslation = False):
    """ 
    Remark: consensus (subset of M_T) is a list of cvMatches where each m.imgIdx is pointing at the position in the parent 
    list of all matches (M_T). Aff_RANSAC_H is creating that indexing automatically. If not it needs to be done manually.
    """
    img1 = copy.deepcopy(img1)
    img2 = copy.deepcopy(img2)
    consensus = np.random.permutation(consensus)[0:np.min([len(consensus),Nshow])]
    SquarePatch = SquareOrderedPts(patch_width,patch_width,CV=True)
    mask = np.zeros((patch_width-1, patch_width-1), np.uint8)
    mask[:] = 1.0
    cornersQ, cornersT = [], []
    imgs = []
    masks = []
    kps1, kps2 = [], []
    h2, w2 = img2.shape[:2]
    
    flags = cv2.INTER_LINEAR
    for m in consensus:
        Akp1 = A_list1[m.queryIdx] 
        Akp2 = A_list2[m.trainIdx]
        this_A_p1_to_target = A_p1_to_target[m.imgIdx]
        if noTranslation:
            patchcenter = np.array([(patch_width-1)/2,(patch_width-1)/2])
            Akp1[:,2] = patchcenter - np.matmul(Akp1[0:2,0:2],KPlist1[m.queryIdx].pt)
            Akp2[:,2] = patchcenter - np.matmul(Akp2[0:2,0:2],KPlist2[m.trainIdx].pt)
            this_A_p1_to_target[:,2] = KPlist2[m.trainIdx].pt - np.matmul(this_A_p1_to_target[0:2,0:2],patchcenter)            

        kps1.append( KPlist1[m.queryIdx] )
        kps2.append( KPlist2[m.trainIdx] )
        A = Aq2t[m.imgIdx]
        cornersQ.append( AffineKPcoor(SquarePatch, cv2.invertAffineTransform(Akp1)) )
        cornersT.append( AffineKPcoor(SquarePatch, this_A_p1_to_target) )
        img = cv2.warpAffine(img1, A, (w2, h2), flags=flags, borderMode=cv2.BORDER_CONSTANT)
        
        affinemask = np.zeros((h2,w2))
        X,Y = AffineGridCoor(img2.shape[:2],cv2.invertAffineTransform(this_A_p1_to_target))
        logical_map = (X>=0) * (X < patch_width-1) * (Y>=0) * (Y < patch_width-1)
        affinemask[logical_map] = 1.0
        img = img*affinemask
        imgs.append( img )
        masks.append( affinemask )
    
    masks = np.sum(masks,axis=0)
    backimg = 0.5*img2
    backimg[masks>0] = 0.0
    masks[masks==0] = 1.0

    img = np.sum(imgs,axis=0)/masks

    img = np.sum(imgs,axis=0)/masks + backimg
    for n in range(len(cornersT)):
        pts = np.array([np.array(kp.pt) for kp in cornersT[n]], np.int32)
        pts = pts.reshape((-1,1,2))
        img = cv2.polylines(img,[pts],True,(255,255,255))

        pts = np.array([np.array(kp.pt) for kp in cornersQ[n]], np.int32)
        pts = pts.reshape((-1,1,2))
        img1 = cv2.polylines(img1,[pts],True,(255,255,255))

    img3 = cv2.drawMatches(img1,kps1,img.astype(np.uint8),kps2,[cv2.DMatch(i,i,1.0) for i in range(len(kps1))], None,flags=2)
    cv2.imwrite(opt.workdir+'randomAffineInliers.png',img3)    
