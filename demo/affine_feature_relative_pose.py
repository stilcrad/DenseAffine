import warnings
import sys
warnings.filterwarnings("ignore")
from affine.HardNet import HardNet
from Network.affine_estimate_network import  *
import matplotlib
matplotlib.use('Agg')
from PIL import Image
from DenseMatch.dkm import DKMv3_outdoor
from Network.essential_matrix_affine_without_kornia import *


def distance_matrix_vector(anchor, positive):
    """Given batch of anchor descriptors and positive descriptors calculate distance matrix"""

    d1_sq = torch.sum(anchor * anchor, dim=1).unsqueeze(-1)
    d2_sq = torch.sum(positive * positive, dim=1).unsqueeze(-1)

    eps = 1e-6
    return torch.sqrt((d1_sq.repeat(1, positive.size(0)) + torch.t(d2_sq.repeat(1, anchor.size(0)))
                       - 2.0 * torch.bmm(anchor.unsqueeze(0), torch.t(positive).unsqueeze(0)).squeeze(0)) + eps)


def euclidean_distance(matrix1, matrix2):
    return np.linalg.norm(matrix1 - matrix2)


def verify_cv2_ess(kps1, kps2, tentatives, K1, K2, h1, w1, h2, w2):
    src_pts = np.float32(kps1).reshape(-1, 2)
    dst_pts = np.float32(kps2).reshape(-1, 2)

    # Normalize the threshold
    threshold = 0.75
    avgDiagonal = (K1[0][0] + K1[1][1] + K2[0][0] + K2[1][1]) / 4;
    normalizedThreshold = threshold / avgDiagonal

    # Normalize the point coordinates
    normalizedSourcePoints = cv2.undistortPoints(np.expand_dims(src_pts, axis=1), cameraMatrix=K1, distCoeffs=None)
    normalizedDestinationPoints = cv2.undistortPoints(np.expand_dims(dst_pts, axis=1), cameraMatrix=K2, distCoeffs=None)

    # Estimate the essential matrix from the normalized coordinates
    # using the normalized threshold.
    E, mask = cv2.findEssentialMat(normalizedSourcePoints,
                                   normalizedDestinationPoints,
                                   focal=1.0,
                                   pp=(0., 0.),
                                   method=cv2.USAC_ACCURATE,
                                   prob=0.99,
                                   threshold=normalizedThreshold)
    mask_ = [i[0] for i in mask ]
    return E, mask_

def calculate_error_new(Rotation_rel_GroundTruth, E_estimate):
    Rc1toc2 = Rotation_rel_GroundTruth[:3, :3]
    Tc1toc2 = Rotation_rel_GroundTruth[:, 3]
    E_est = E_estimate
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    U, S, Vt = np.linalg.svd(E_est)
    V = Vt.T
    t1_our = np.dot(U, np.array([0, 0, 1]))
    t2_our = -t1_our
    R1_our = np.dot(np.dot(U, W), V.T)
    R2_our = np.dot(np.dot(U, W.T), V.T)

    assert np.abs(np.linalg.det(R1_our)) - 1 < 1e-12
    assert np.abs(np.linalg.det(R2_our)) - 1 < 1e-12

    if np.linalg.det(R1_our) < 0:
        R1_our = -R1_our
    if np.linalg.det(R2_our) < 0:
        R2_our = -R2_our

    error_R1_Our = (np.arccos(np.clip(((np.trace(np.dot(R1_our, Rc1toc2.T)) - 1) / 2), -1, 1)) * 180 / np.pi).real
    error_R2_Our = (np.arccos(np.clip(((np.trace(np.dot(R2_our, Rc1toc2.T)) - 1) / 2), -1, 1)) * 180 / np.pi).real

    error_t1_Our = np.arccos(np.clip((
            np.dot(t1_our, Tc1toc2) / (np.linalg.norm(t1_our) * np.linalg.norm(Tc1toc2))), -1, 1)) * 180 / np.pi
    error_t2_Our = np.arccos(np.clip((
            np.dot(t2_our, Tc1toc2) / (np.linalg.norm(t2_our) * np.linalg.norm(Tc1toc2))), -1, 1)) * 180 / np.pi

    if error_R1_Our < error_R2_Our:
        error_R_Our = error_R1_Our
    else:
        error_R_Our = error_R2_Our

    if error_t1_Our < error_t2_Our:
        error_t_Our = error_t1_Our
    else:
        error_t_Our = error_t2_Our

    return error_R_Our, error_t_Our


def multicompute(path1 ,path2):
    from argparse import ArgumentParser
    parser = ArgumentParser()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--im_A_path", default=path1 , type=str)
    parser.add_argument("--im_B_path", default=path2, type=str)

    args, _ = parser.parse_known_args()
    im1_path = args.im_A_path
    im2_path = args.im_B_path
    with torch.no_grad():
        # Create model
        dkm_model = DKMv3_outdoor(path_to_weights="weights/outdoor/Outdoor_matching.pth",device=device)

        H, W =Image.open(im1_path).size

        W_A, H_A = Image.open(im1_path).size
        W_B, H_B = Image.open(im2_path).size

        im1 = Image.open(im1_path).convert('RGB').resize((H, W))
        im2 = Image.open(im2_path).convert('RGB').resize((H, W))

        # Match
        # Sampling not needed, but can be done with model.sample(warp, certainty)

        warp, certainty = dkm_model.match(im1, im2, device=device)
        # Sample matches for estimation
        matches, certainty = dkm_model.sample(warp, certainty)
        kpts1, kpts2 = dkm_model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)
        pt1 = kpts1.cpu().numpy()
        pt2 = kpts2.cpu().numpy()

        kp1_arr=np.array(pt1)
        kp2_arr=np.array(pt2)

        co1 =kp1_arr
        co2 =kp2_arr

        im1 = cv2.imread( im1_path)[:,:,0]
        im2 = cv2.imread( im2_path)[:,:,0]

        patch1, patch2 = generate_patch(im1, im2, co1.round().astype(int), co2.round().astype(int), patch_size=32)
        co1 = torch.tensor(co1).cuda()
        co2 = torch.tensor(co2).cuda()
        co1 = co1.unsqueeze(0)
        co2 = co2.unsqueeze(0)

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

        patch_size =32
        scale_num = 300
        angle_num = 360

        esti_checkpoint_path = 'weights/outdoor/ori_scal.pth'

        device ="cuda"
        esti_scale_ratio_list = [0.5, 1, 2]

        model_scale = EstiNet(need_bn=True, device=device, out_channels=scale_num,
                              patch_size=patch_size, scale_ratio=esti_scale_ratio_list)
        model_angle = EstiNet(need_bn=True, device=device, out_channels=angle_num,
                              patch_size=patch_size, scale_ratio=esti_scale_ratio_list)
        checkpoint = torch.load(esti_checkpoint_path, map_location=device)
        model_scale.load_state_dict(checkpoint['model_scale'], strict=False)
        model_scale.eval()

        model_angle.load_state_dict(checkpoint['model_angle'], strict=False)
        model_angle.eval()

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

        descriptor = HardNet(pretrained = True)
        model_weights = 'weights/HardNet++.pth'
        hncheckpoint = torch.load(model_weights)
        descriptor.load_state_dict(hncheckpoint['state_dict'])
        descriptor.eval()

        im1 = load_torch_image(im1_path)
        im2 = load_torch_image(im2_path)
        descriptor = LAFDescriptor(HardNet(True))
        weight_path = "weights/outdoor/Aff_res_shape.pth"
        affnet = LAFAffNetShapeEstimator(weight_path).cuda()


        lafs1, r1 = laf_from_ours(coord1_A, scale1, angle1, mrSize=6, with_resp=True, device=device)
        ori1 = get_laf_orientation(lafs1)
        lafs1 = affnet(lafs1.cuda(), im1.mean(dim=1, keepdim=True).cuda())
        lafs1 = set_laf_orientation(lafs1, ori1)
        lafs2, r2 = laf_from_ours(coord2_A, scale2, angle2, mrSize=6, with_resp=True, device=device)
        ori2 = get_laf_orientation(lafs2)
        lafs2 = affnet(lafs2.cuda(), im2.mean(dim=1, keepdim=True).cuda())
        lafs2 = set_laf_orientation(lafs2, ori2)

        lafs1 = scale_laf(lafs1, torch.tensor(scale1))
        descs1 = descriptor(im1.cuda(), lafs1.cuda())
        lafs2 = scale_laf(lafs2,  torch.tensor(scale2))
        descs2 = descriptor(im2.cuda(), lafs2.cuda())

        # Send the detected keypoints and other variables to the CPU and convert them to numpy array
        lafs1np = np.squeeze(lafs1.cpu().detach().numpy())
        lafs2np = np.squeeze(lafs2.cpu().detach().numpy())

        distance = distance_matrix_vector(descs1.squeeze(0), descs2.squeeze(0))
        distance_list = torch.tensor([distance[i][i] for i in range(len(distance))])
        sorted_indices = torch.argsort(distance_list)
        distance_list = distance_list[sorted_indices]
        threshold = np.percentile(distance_list,75)

        mask = [index for index, value in enumerate(distance_list) if value < threshold]

        kps1, kps2 = get_coordinates(lafs1np, lafs2np)
        kps1 = np.array([kps1[i] for i in mask])
        lafs1np = np.array([lafs1np[i]  for i in mask])
        lafs2np = np.array([lafs2np[i]  for i in mask])
        tentatives = [i for i in range(len(kps1))]
        ACs = get_affine_correspondences(lafs1np, lafs2np, tentatives)
        torch.cuda.empty_cache()

    return pt1 , pt2 , ACs


if __name__ == '__main__':

    # squ_list = ["00","01","02","03","04","05","06","07","08","09","10"]
    squ_list = [ "04" ]
    for squ in squ_list:
        print(squ)
        n = 300
        root_path = "data/kitti/sequences/{}/".format(
            squ)
        img_list = os.listdir(os.path.join(root_path, "image_0"))
        calibtxt = "data/kitti/sequences/{}/calib.txt".format(
            squ)
        groundtruth_posepath = "data/kitti/relativepose/relativepose{}.txt".format(
            squ)

        K1 = np.loadtxt('data/kitti/cali/kitti_{}.K'.format(squ))
        K2 = np.loadtxt('data/kitti/cali/kitti_{}.K'.format(squ))

        with open(groundtruth_posepath) as f:
            posetxt = f.readlines()
            GTpose = np.array(
                [[float(i) for i in j.replace("   ", ",").replace("  ", ",").split(",")[1:]] for j in posetxt])
            GTpose = GTpose.reshape(GTpose.shape[0], 3, 4)
        R = GTpose[:, :3, :3]
        t = GTpose[:, :3, 3]
        E_gt_all = Compose_Essential(R, t)
        F_gt_all = E2F(K1, K1, E_gt_all)

        for i in range(len(img_list) - 1):
            path1 = os.path.join(os.path.join(root_path, "image_0"), "{}.png".format(str(i).rjust(6, '0')))
            path2 = os.path.join(os.path.join(root_path, "image_0"), "{}.png".format(str(i + 1).rjust(6, '0')))
            img1 = load_torch_image(path1)
            img2 = load_torch_image(path2)
            img1.to(device)
            img2.to(device)
            E_gt = E_gt_all[i]
            F_gt = F_gt_all[i]

            DKM_5pc_errorlist = []
            DKM_5pc_pc_list = []
            DenseAffine_2ac_errorlist = []

            with open("/home/xxx/project/ECCV/DKM/kitti_0/pc_{}/out_PC{}.txt".format(squ,i), "r") as f1:
                the_pc = f1.readlines()
                PCs_DKM = np.array([np.array([float(i) for i in j.split(",")[:4]]) for j in the_pc])[:n]
                DKM_pt1 = PCs_DKM[:, 0:2]
                DKM_pt2 = PCs_DKM[:, 2:4]

            pt1, pt2, ACs = multicompute(path1 ,path2)



            pt1 = pt1[:n]
            pt2 = pt2[:n]
            ACs = ACs[:n]

            tentatives_dkm = [0.5] * PCs_DKM.shape[0]

            tentatives = [i for i in range(len(DKM_pt1))]

            DKM5pcE, _mask = verify_cv2_ess(DKM_pt1, DKM_pt2, tentatives, K1, K2, img1.shape[2],
                                                          # The height of the source image
                                                          img1.shape[3],  # The width of the source image
                                                          img2.shape[2],  # The height of the destination image
                                                          img2.shape[3],  # The width of the destination image
                                            )

            DKM_5pc_R_error, DKM_5pc_T_error = calculate_error_new(GTpose[i].reshape(3, 4), DKM5pcE)
            DKM_5pc_errorlist.append([DKM_5pc_R_error, DKM_5pc_T_error])





            ACs_ours = ACs
            ours_pt1 = ACs_ours[:, 0:2]
            ours_pt2 = ACs_ours[:, 2:4]
            ours_affines = ACs_ours[:, 4:8]
            tentatives_ours = [0.5] * ACs_ours.shape[0]
            t = time()
            ours_E, ours_mask = verify_affine_pygcransac_essential_matrix(ACs_ours,  # The input affine correspondences
                                                                          tentatives_ours,
                                                                          # The matches containing the indices of the matched keypoints
                                                                          img1.shape[2],
                                                                          # The height of the source image
                                                                          img1.shape[3],
                                                                          # The width of the source image
                                                                          img2.shape[2],
                                                                          # The height of the destination image
                                                                          img2.shape[3],
                                                                          # The width of the destination image
                                                                          K1,
                                                                          K1,
                                                                          Sampler.ARSampler.value)  # The id of the used sampler. 0 - uniform, 1 - PROSAC, 3 - NG-RANSAC's sampler, 4 - AR-Sampler

            ours_R_error, ours_T_error = calculate_error_new(GTpose[i].reshape(3, 4), ours_E)
            DenseAffine_2ac_errorlist.append([ours_R_error, ours_T_error])
    ours_R_errorlist = [i[0] for i in DenseAffine_2ac_errorlist]
    ours_T_errorlist = [i[1] for i in DenseAffine_2ac_errorlist]
    dkm_5pc_R_errorlist = [i[0] for i in DKM_5pc_errorlist]
    dkm_5pc_T_errorlist = [i[1] for i in DKM_5pc_errorlist]
    ours_R_median_error = np.median(ours_R_errorlist)
    dkm_5pc_R_median_error = np.median(dkm_5pc_R_errorlist)
    ours_T_median_error = np.median(ours_T_errorlist)
    dkm_5pc_T_median_error = np.median(dkm_5pc_T_errorlist)
    print("ours_R_median_error:", ours_R_median_error)
    print("dkm_5pc_R_median_error:", dkm_5pc_R_median_error)
    print("ours_T_median_error:", ours_T_median_error)
    print("dkm_5pc_T_median_error:", dkm_5pc_T_median_error)





