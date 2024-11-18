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
from matching import match_image,drawMatches


def distance_matrix_vector(anchor, positive):
    """Given batch of anchor descriptors and positive descriptors calculate distance matrix"""

    d1_sq = torch.sum(anchor * anchor, dim=1).unsqueeze(-1)
    d2_sq = torch.sum(positive * positive, dim=1).unsqueeze(-1)

    eps = 1e-6
    return torch.sqrt((d1_sq.repeat(1, positive.size(0)) + torch.t(d2_sq.repeat(1, anchor.size(0)))
                       - 2.0 * torch.bmm(anchor.unsqueeze(0), torch.t(positive).unsqueeze(0)).squeeze(0)) + eps)


def euclidean_distance(matrix1, matrix2):
    return np.linalg.norm(matrix1 - matrix2)

def multicompute(path1 ,path2,output_dir):
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

        im1 = load_torch_image(im1_path,W_A,H_A)
        im2 = load_torch_image(im2_path,W_B,H_B)
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

        with open(output_dir, "w") as f:
            for x in range(len(ACs)):
                f.write(str(ACs[x][0]) + "," + str(ACs[x][1]) + "," + str(ACs[x][2]) + "," + str(ACs[x][3]) + "," + str(
                    ACs[x][4]) + "," + str(ACs[x][5]) + "," + str(ACs[x][6]) + "," + str(ACs[x][7]) + "\n")

if __name__ == '__main__':
    imgpath1 = "data/3424741608_87876e2909_b.jpg"
    imgpath2 = "data/3424775484_984abc7347_b.jpg"
    output_dir = "results/ACs.txt"
    out_path = "results/matches.jpg"
    multicompute(imgpath1, imgpath2, output_dir)
    match_image(imgpath1 ,imgpath2,output_dir , out_path)


