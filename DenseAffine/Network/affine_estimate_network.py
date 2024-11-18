from PIL import Image
import PIL
import torch
from DenseMatch.dkm import DKMv3_outdoor
from Local_affine_estimation.S3Esti.abso_esti_net import EstiNet

from Local_affine_estimation.hesaffnet.hesaffnet import AffNetHardNet_describeFromKeys_justAFFnetoutput


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import os
os.chdir("..")
from  dataloader.megadepth import *

import torch

# The default config uses dual-softmax.
# The outdoor and indoor models share the same config.
# You can change the default values like thr and coarse_match_type.
# 这是一个示例 Python 脚本。
import cv2
import numpy as np

# from megadepth import *


def Compose_Essential(R,t):

    E_all = []
    for i in range(R.shape[0]):
        W = np.array([[0, -t[i,2], t[i,1]],
                      [t[i,2], 0, -t[i,0]],
                      [-t[i,1], t[i,0], 0]])

        E = W @ R[i]
        E_all.append((E))
    E_ = np.array(E_all)
    return E_


def E2F(K1,K2,E):
    F_= []
    for i in range(E.shape[0]):

        F= np.linalg.inv(K2.T) @ E[i] @ np.linalg.inv(K1)
        F_.append(F)
    F_all =np.array(F_)
    return F_all

def F2E(K1,K2,F):
    E_ = []
    for i in range(F.shape[0]):
        E = K2.T @ F @K1
        E_.append(E)
    E_all = np.array(E_)
    return E_all


def read_img_cam(root):
    images = {}
    Image = collections.namedtuple(
        "Image", ["name", "w", "h", "fx", "fy", "cx", "cy","E_gt","F_gt","rvec", "tvec"])
    for scene_id in os.listdir(root):
        densefs = [f for f in os.listdir(os.path.join(root, scene_id))]
        folder = os.path.join(root, scene_id,"image_0")
        img_cam_txt_path = os.path.join(root, scene_id, 'calib.txt')

        # K1 = np.loadtxt('/home/xxx/project/C++/graph-cut-ransac/build/data/kitti/kitti_{}.K'.format(scene_id))
        # # K2 = np.loadtxt('/home/xxx/project/C++/graph-cut-ransac/build/data/kitti/kitti_04.K')
        # K2 = K1

        with open("/home/xxx/project/matlab/SPJ_1/kitti_features_extraction/relativepose/relativepose{}.txt".format(scene_id),"r") as f:
            posetxt = f.readlines()
            GTpose = np.array(
                [[float(i) for i in j.replace("   ", ",").replace("  ", ",").split(",")[1:]] for j in posetxt])
            GTpose = GTpose.reshape(GTpose.shape[0], 3, 4)
        R_gt_all  = GTpose[:, :3, :3]
        t_ge_all  = GTpose[:, :3, 3]
        E_gt_all = Compose_Essential(R_gt_all, t_ge_all)
        # F_gt_all = E2F(K1, K1, E_gt_all)

        with open(img_cam_txt_path, "r") as fid:
            line = fid.readline()
            if not line:
                break
            line = line.strip("P0:")
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                K1 = np.array([float(i)  for i in elems]).reshape(3,4)[:,:3]
                F_gt_all = E2F(K1, K1, E_gt_all)

                for i in range(len(os.listdir(folder))-1):

                    R = R_gt_all[i]
                    T = t_ge_all[i]

                    E_gt = E_gt_all[i]
                    F_gt = F_gt_all[i]

                    img_path = os.path.join(folder, "{}.png".format(str(i).rjust(6,"0")))
                    img = cv2.imread(img_path)
                    h, w ,c = img.shape
                    fx, fy = float(elems[0]), float(elems[2])
                    cx, cy = float(elems[5]), float(elems[6])

                    images[img_path] = Image(
                        name=img_path, w=w, h=h, fx=fx, fy=fy, cx=cx, E_gt = E_gt,F_gt= F_gt,cy=cy ,rvec=R, tvec=T
                    )
    return images


def read_pairs(root):
    imf1s, imf2s = [], []
    print('reading image pairs from {}...'.format(root))
    for scene_id in tqdm(os.listdir(root), desc='# loading data from scene folders'):
        imf1s_ = []
        imf2s_ = []
        folder = os.path.join(root, scene_id, 'image_0')
        for num in range(len(os.listdir(folder))-1):
            imf1s.append(os.path.join(folder,"{}.png".format(str(num).rjust(6,"0"))))
            imf2s.append(os.path.join(folder,"{}.png".format(str(num+1).rjust(6,"0"))))

    return imf1s, imf2s



def get_intrinsics(im_meta):
    return np.array([[im_meta.fx, 0, im_meta.cx],
                     [0, im_meta.fy, im_meta.cy],
                     [0, 0, 1]])


def get_extrinsics(im_meta):
    R = im_meta.rvec.reshape(3, 3)
    t = im_meta.tvec
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = t
    return extrinsic

def data_loader(imf1s,imf2s,images,item):
    imf1 = imf1s[item]
    imf2 = imf2s[item]
    im1_meta = images[imf1]
    im2_meta = images[imf2]
    im1 = cv2.imread(imf1, cv2.IMREAD_GRAYSCALE)
    im2 = cv2.imread(imf2, cv2.IMREAD_GRAYSCALE)

    h, w = im1.shape[:2]

    intrinsic1 = get_intrinsics(im1_meta)
    intrinsic2 = get_intrinsics(im2_meta)

    extrinsic1 = get_extrinsics(im1_meta)
    extrinsic2 = get_extrinsics(im2_meta)

    relative = extrinsic2.dot(np.linalg.inv(extrinsic1))
    R = relative[:3, :3]

    T = relative[:3, 3]
    tx = data_utils.skew(T)
    # E_gt = np.dot(tx, R)
    # F_gt = np.linalg.inv(intrinsic2).T.dot(E_gt).dot(np.linalg.inv(intrinsic1))
    F_gt = im1_meta.F_gt
    E_gt = im1_meta.E_gt

    F_gt = torch.from_numpy(F_gt).float() / (F_gt[-1, -1] + 1e-10)
    intrinsic1 = torch.from_numpy(intrinsic1).float()
    intrinsic2 = torch.from_numpy(intrinsic2).float()
    pose = torch.from_numpy(relative[:3, :]).float()


    out = {'im1': im1,
           'im2': im2,
           'pose': pose,
           'F_gt': F_gt,
           'E_gt': E_gt,
           'K1': intrinsic1,
           'K2': intrinsic2,}

    return out


def DKM_matching(imf1s,imf2s,path_to_weights,images,n,draw_img=False):

    out = data_loader(imf1s,imf2s,images,n)
    # 读取图片

    F_gt = out["F_gt"]
    im1_path=imf1s[n]
    im2_path=imf2s[n]

    # Create model
    dkm_model = DKMv3_outdoor(path_to_weights,
                              device=device)

    W_A, H_A = Image.open(im1_path).size
    W_B, H_B = Image.open(im2_path).size

    # Match
    warp, certainty = dkm_model.match(im1_path, im2_path,device=device)
    # Sample matches for estimation
    matches, certainty = dkm_model.sample(warp, certainty)
    kpts1, kpts2 = dkm_model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)
    pt1 = kpts1.cpu().numpy()[:2000]
    pt2 = kpts2.cpu().numpy()[:2000]
    good_matches = []
    # print("筛选前匹配数量", len(pt1))

    kp1_arr=np.array([pt1[0]])
    kp2_arr=np.array([pt2[0]])

    for i, kpt1, kpt2 in zip(range(len(pt1)), pt1, pt2 ):
        pt1_ = np.array([kpt1[0], kpt1[1], 1])
        pt2_ = np.array([kpt2[0], kpt2[1], 1])

        # 计算极线
        line = np.dot(F_gt, pt1_)

        # 计算点到极线的距离
        distance = np.abs(np.dot(pt2_, line)) / np.sqrt(line[0] ** 2 + line[1] ** 2)

        # 如果距离小于10，那么保存这个匹配点
        if distance < 1:
            # print("点到极线距离是",distance)
            kp1_arr = np.concatenate((kp1_arr, [kpt1]), axis=0)
            kp2_arr = np.concatenate((kp2_arr, [kpt2]), axis=0)


    # print("筛选后匹配数量", len(kp1_arr))

    torch.cuda.empty_cache()
    return kp1_arr , kp2_arr,  out

def parse_matrix(file_path):
    matrices = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            matrix = np.array([float(num) for num in line.split()]).reshape(3, 3)
            matrices.append(matrix)
    return matrices

def generate_patch( img1, img2, coord1, coord2, patch_size,):

    # img1 = img1.mean(dim=1, keepdim=True).squeeze(1)
    # img2 = img2.mean(dim=1, keepdim=True).squeeze(1)
    coord1 = torch.tensor(coord1)
    coord2 = torch.tensor(coord2)

    tl1=torch.clone(coord1)
    tr1=torch.clone(coord1)
    bl1=torch.clone(coord1)
    br1=torch.clone(coord1)
    tl2 = torch.clone(coord2)
    tr2 = torch.clone(coord2)
    bl2 = torch.clone(coord2)
    br2 = torch.clone(coord2)
    tl1[:, 0] -= patch_size // 2
    tl1[:, 1] -= patch_size // 2
    tr1[:, 0] -= patch_size // 2
    tr1[:, 1] += patch_size // 2
    bl1[:, 0] += patch_size // 2
    bl1[:, 1] -= patch_size // 2
    br1[:, 0] += patch_size // 2
    br1[:, 1] += patch_size // 2

    tl2[:, 0] -= patch_size // 2
    tl2[:, 1] -= patch_size // 2
    tr2[:, 0] -= patch_size // 2
    tr2[:, 1] += patch_size // 2
    bl2[:, 0] += patch_size // 2
    bl2[:, 1] -= patch_size // 2
    br2[:, 0] += patch_size // 2
    br2[:, 1] += patch_size // 2


    # patch1_coord = torch.tensor(np.array([item.cpu().detach().numpy() for item in [tl1, tr1, bl1, br1]])).cuda().permute(1, 2, 0)
    # patch1_coord = torch.tensor(np.array([item.cpu().detach().numpy() for item in [tl1, tr1, bl1, br1]])).cuda().permute(1,0,2)
    patch1_coord =torch.cat([tl1.unsqueeze(1), tr1.unsqueeze(1), bl1.unsqueeze(1), br1.unsqueeze(1)], dim=1)

    # patch2_coord = torch.tensor(np.array([item.cpu().detach().numpy() for item in [tl2, tr2, bl2, br2]])).cuda().permute(1, 2, 0)
    # patch2_coord = torch.tensor(np.array([item.cpu().detach().numpy() for item in [tl2, tr2, bl2, br2]])).cuda().permute(1,0,2)
    patch2_coord = torch.cat([tl2.unsqueeze(1), tr2.unsqueeze(1), bl2.unsqueeze(1), br2.unsqueeze(1)], dim=1)

    patch_tensor1 = point2patch(patch1_coord, img1,patch_size)
    patch_tensor2 = point2patch(patch2_coord, img2,patch_size)


    return patch_tensor1 , patch_tensor2


def point2patch(coord ,img , patch_size):

    coord= coord.cuda()
    if type(img)== PIL.Image.Image :
        img= torch.tensor(np.array(img)[:,:,:,0]).cuda()
    else:
        img = torch.tensor(img).cuda()

    index=[]
    if len(coord.shape)==3:
        coord = coord.unsqueeze(0)
        img = img.unsqueeze(0)
    for i,p1co, img1_ in zip(range(coord.shape[0]),coord, img):
        if i ==0:
            for j,p1co1 in enumerate(p1co):
                xmin = torch.min(p1co1[:, 1])
                xmax = torch.max(p1co1[:, 1])
                ymin = torch.min(p1co1[:, 0])
                ymax = torch.max(p1co1[:, 0])
                patch = img1_[xmin:xmax , ymin :ymax ].unsqueeze(0)
                if j == 0:
                    if patch.shape[1] != patch_size or patch.shape[2] != patch_size:
                        index.append((i, j))
                        zero_patch = torch.zeros(1, patch_size, patch_size).cuda()
                        n_patch = zero_patch
                    else :
                        n_patch = patch
                    b_n_patch = n_patch.unsqueeze(0)
                else:
                    if patch.shape[1]!= patch_size or patch.shape[2]!= patch_size:
                        index.append((i,j))
                        zero_patch= torch.zeros(1,patch_size,patch_size).cuda()
                        n_patch = torch.cat((n_patch, zero_patch), dim=0)
                    else:
                        n_patch = torch.cat((n_patch, patch), dim=0)
                    b_n_patch = n_patch.unsqueeze(0)

        else:
            for j,p1co1 in enumerate(p1co):
                xmin = torch.min(p1co1[:, 1])
                xmax = torch.max(p1co1[:, 1])
                ymin = torch.min(p1co1[:, 0])
                ymax = torch.max(p1co1[:, 0])
                patch = img1_[xmin:xmax , ymin :ymax ].unsqueeze(0)
                if j == 0:
                    if patch.shape[1] != patch_size or patch.shape[2] != patch_size:
                        index.append((i, j))
                        zero_patch = torch.zeros(1, patch_size, patch_size).cuda()
                        n_patch = zero_patch
                    else :
                        n_patch = patch
                    b_n_patch = torch.cat((b_n_patch, n_patch.unsqueeze(0)), dim=1)
                else:
                    if patch.shape[1]!= patch_size or patch.shape[2]!= patch_size:
                        index.append((i,j))
                        zero_patch= torch.zeros(1,patch_size,patch_size).cuda()
                        n_patch = torch.cat((n_patch, zero_patch), dim=0)
                    else:
                        n_patch = torch.cat((n_patch, patch), dim=0)
                    b_n_patch = torch.cat((b_n_patch, n_patch.unsqueeze(0)), dim=1)

    return b_n_patch



def ComposeAffineMaps(A_lhs,A_rhs):
    ''' Comutes the composition of affine maps:
        A = A_lhs ∘ A_rhs
    '''
    A = np.matmul(A_lhs[0:2,0:2],A_rhs)
    A[:,2] += A_lhs[:,2]
    return A

def network_forward(co1,co2 ,out):

    im1 = out["im1"]
    im2 = out["im2"]
    F_gt = out["F_gt"]
    E_gt = out["E_gt"]
    K1 = out["K1"]

    patch1, patch2 = generate_patch(im1, im2, co1.round().astype(int), co2.round().astype(int), patch_size=32)
    co1 = torch.tensor(co1).cuda()
    co2 = torch.tensor(co2).cuda()
    co1 = co1.unsqueeze(0).cuda()
    co2 = co2.unsqueeze(0).cuda()

    patch1_ = torch.empty(1, patch1.shape[2], patch1.shape[3]).cuda()
    patch2_ = torch.empty(1, patch1.shape[2], patch1.shape[3]).cuda()
    coord1_A = torch.empty(1, co1.shape[2]).cuda()
    coord2_A = torch.empty(1, co2.shape[2]).cuda()

    _index = []
    for i, b1, b2, c1, c2 in zip(range(len(patch1)), patch1, patch2, co1, co2):
        for j, p1, p2, d1, d2 in zip(range(len(b1)), b1, b2, c1, c2):
            if not ((p1 == 0).all() or (p2 == 0).all()):
                _index.append((i, j))
                patch1_ = torch.cat((patch1_, p1.unsqueeze(0)), dim=0)
                patch2_ = torch.cat((patch2_, p2.unsqueeze(0)), dim=0)
                coord1_A = torch.cat((coord1_A, d1.unsqueeze(0)), dim=0)
                coord2_A = torch.cat((coord2_A, d2.unsqueeze(0)), dim=0)

    patch1_ = patch1_[1:]
    patch2_ = patch2_[1:]
    coord1_A = coord1_A[1:]
    coord2_A = coord2_A[1:]

    print(patch1_.shape[0])
    if patch1_.shape[0] == 0:
        return 0

    device ="cuda"
    esti_scale_ratio_list = [0.5, 1, 2]
    patch_size =32
    scale_num = 300
    angle_num = 360



    model_scale = EstiNet(need_bn=True, device=device, out_channels=scale_num,
                          patch_size=patch_size, scale_ratio=esti_scale_ratio_list)
    model_angle = EstiNet(need_bn=True, device=device, out_channels=angle_num,
                          patch_size=patch_size, scale_ratio=esti_scale_ratio_list)


    model_scale.cuda()
    model_angle.cuda()


    patch1_r = patch1_.unsqueeze(1).expand(-1, 9, -1, -1)
    patch2_r = patch2_.unsqueeze(1).expand(-1, 9, -1, -1)

    scale1_resp = model_scale(patch1_r)
    angle1_resp = model_angle(patch1_r)
    scale2_resp = model_scale(patch2_r)
    angle2_resp = model_angle(patch2_r)

    scale1_ind_pred = torch.argmax(scale1_resp, dim=1).cpu().numpy()
    angle1_ind_pred = torch.argmax(angle1_resp, dim=1).cpu().numpy()
    scale2_ind_pred = torch.argmax(scale2_resp, dim=1).cpu().numpy()
    angle2_ind_pred = torch.argmax(angle2_resp, dim=1).cpu().numpy()

    scale1 = model_scale.map_id_to_scale(scale1_ind_pred).cpu().numpy()
    scale2 = model_scale.map_id_to_scale(scale2_ind_pred).cpu().numpy()
    angle1 = model_angle.map_id_to_angle(angle1_ind_pred).cpu().numpy()
    angle2 = model_angle.map_id_to_angle(angle2_ind_pred).cpu().numpy()


    # A = get_affine_from_just_sca_rot(scale1[0],angle1[0],scale2[0] ,angle2[0])# the A decompose as s and alph

    def angles2A(angles):
        cos_a = torch.cos(angles).view(-1, 1, 1)
        sin_a = torch.sin(angles).view(-1, 1, 1)
        A1_ang = torch.cat([cos_a, sin_a], dim = 2)
        A2_ang = torch.cat([-sin_a, cos_a], dim = 2)
        return  torch.cat([A1_ang, A2_ang], dim = 1)

    lamda1 = torch.tensor(scale1,dtype=torch.double).view(scale1.shape[0], 1, 1)*torch.eye(2).unsqueeze(0).expand(scale1.shape[0],2,2)[:,]
    lamda2 = torch.tensor(scale2,dtype=torch.double).view(scale2.shape[0], 1, 1)*torch.eye(2).unsqueeze(0).expand(scale2.shape[0],2,2)[:,]
    R1 = angles2A(torch.tensor(angle1,dtype=torch.double))
    R2 = angles2A(torch.tensor(angle2,dtype=torch.double))


    A1_list = AffNetHardNet_describeFromKeys_justAFFnetoutput(im1, coord1_A, patch1_)
    A2_list = AffNetHardNet_describeFromKeys_justAFFnetoutput(im2, coord2_A, patch1_)

    diagonal_matrix = torch.eye(2)

    A1_list += diagonal_matrix
    A2_list += diagonal_matrix

    A1 = torch.bmm(torch.bmm(lamda1, R1), A1_list)
    A2 = torch.bmm(torch.bmm(lamda2, R2), A2_list)

    A = (A1 @ torch.inverse(A2)).cuda()

    the_affine_correspondences = torch.cat((coord1_A, coord2_A, A.view(A.shape[0], 4)), dim=1).cpu().numpy()

    return {
        "coor1":coord1_A,
        "coor2":coord2_A,
        "pred_A": A,
        "F_gt" : F_gt,
        "E_gt" : E_gt,
        "K1" : K1,
        "_index": _index,
        "out":out
    }

def train_forward(dkm_ckpt,root,i):

    images = read_img_cam(root)
    imf1s, imf2s = read_pairs(root)
    torch.cuda.empty_cache()
    kp1_arr , kp2_arr, out= DKM_matching(imf1s,imf2s,dkm_ckpt,images,i)
    data  = network_forward(kp1_arr, kp1_arr, out)

    return data





