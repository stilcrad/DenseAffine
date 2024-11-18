from torch.autograd import Variable
import affine_trans_loss
from ffine_network import *
from op_localaffinetransmation import *
from Local_affine_estimation.hesaffnet.architectures import AffNetFast
from Local_affine_estimation.S3Esti.abso_esti_net import EstiNet
from Local_affine_estimation.S3Esti.vgg import *

esti_scale_ratio_list = [0.5, 1, 2]
scale_num = 300
angle_num = 360
patch_size = 32

class Extracaffine(nn.Module):
    def __init__(self, ):
        super(Extracaffine, self).__init__()

        checkpoint_name ="/home/xxx/project/python/S3Esti-master/S3Esti-master/checkpoint_kitti/checkpoint_end_ep_20657.pth"
        self.device = device ="cuda"
        self.model_scale = EstiNet(need_bn=True, device=device, out_channels=scale_num,
                              patch_size=patch_size, scale_ratio=esti_scale_ratio_list)
        self.model_angle = EstiNet(need_bn=True, device=device, out_channels=angle_num,
                              patch_size=patch_size, scale_ratio=esti_scale_ratio_list)
        use_pretrain = (checkpoint_name is not None)
        if use_pretrain:
            checkpoint = torch.load(checkpoint_name, map_location=device)
            model_scale.load_state_dict(checkpoint['model_scale'], strict=True)
            model_scale.train()
            model_angle.load_state_dict(checkpoint['model_angle'], strict=True)
            model_angle.train()
        model_scale.to(device)
        model_angle.to(device)

        CE_loss = torch.nn.CrossEntropyLoss()
        if use_pretrain:
            optimizer_scale.load_state_dict(checkpoint['optimizer_scale'])
            optimizer_angle.load_state_dict(checkpoint['optimizer_angle'])


        self.model_scale.cuda()
        self.model_angle.cuda()


        self.affnet = AffNetFast().to(self.device)

        # self.net.eval()
        # for p in self.net.parameters():
        #     p.requires_grad = False


class AffineModel(nn.Module):

    def name(self):
        return 'Affine Model'

    def __init__(self, args):
        super(AffineModel, self).__init__()
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        weightd_fname = self.args.ckpt_path
        checkpoint = torch.load(weightd_fname)

        # init model, optimizer, scheduler
        self.affnet = AffNetFast().to(self.device)

        self.model_scale = EstiNet(need_bn=True, device=device, out_channels=scale_num,
                                   patch_size=patch_size, scale_ratio=esti_scale_ratio_list)
        self.model_angle = EstiNet(need_bn=True, device=device, out_channels=angle_num,
                                   patch_size=patch_size, scale_ratio=esti_scale_ratio_list)

        self.model_scale.load_state_dict(checkpoint['model_scale'], )
        self.model_scale.train()
        self.model_angle.load_state_dict(checkpoint['model_angle'], strict=True)
        self.model_angle.train()
        self.affnet.load_state_dict(checkpoint['model_affnet'],strict=True)
        self.affnet.train()

        self.optimizer_scale = torch.optim.SGD(self.model_scale.parameters(), lr=0.00001, momentum=0.9)
        self.optimizer_angle = torch.optim.SGD(self.model_angle.parameters(), lr=0.00001, momentum=0.9)
        self.optimizer_affnet = torch.optim.Adam(self.affnet.parameters(), lr=0.00001)

        CE_loss = torch.nn.CrossEntropyLoss()

        self.optimizer_scale.load_state_dict(checkpoint['optimizer_scale'])
        self.optimizer_angle.load_state_dict(checkpoint['optimizer_angle'])

        self.model_scale.cuda()
        self.model_angle.cuda()

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer_affnet,
                                                         step_size=args.lrate_decay_steps,
                                                         gamma=args.lrate_decay_factor)



    def val(self,imf1s,imf2s,images,path_to_weights,step):
        # images = read_img_cam(root)
        # imf1s, imf2s = read_pairs(root)
        co1,co2 ,out = DKM_matching(imf1s,imf2s,path_to_weights,images,step,draw_img=False)
        # co1=co1[:8000,]
        # co2=co2[:8000,]
        data = network_forward(co1,co2 ,out)
        self.data = data

        return self.data,out


    def check_affine(self,data):
        coor1 = data["coor1"]
        coor2 = data["coor2"]
        pred_A = data["pred_A"]
        F_gt = data["F_gt"]


        # for i in range(len(coor1)):
        #     pred_A[i] = get_optimal_affine_transformation(pred_A[i].detach().cpu().numpy(),F_gt,coor1[i].cpu().detach().numpy(),coor2[i].cpu().detach().numpy(),)
        #
        # pred_A.cuda()
        #
        fmatrix_ = F_gt.unsqueeze(0).cuda()

        a1 = pred_A[:, 0]
        a2 = pred_A[:, 1]
        a3 = pred_A[:, 2]
        a4 = pred_A[:, 3]
        u1 = coor1[:, 0]
        v1 = coor1[:, 1]
        u2 = coor2[:, 0]
        v2 = coor2[:, 1]
        f1 = fmatrix_[:, 0, 0]
        f2 = fmatrix_[:, 0, 1]
        f3 = fmatrix_[:, 0, 2]
        f4 = fmatrix_[:, 1, 0]
        f5 = fmatrix_[:, 1, 1]
        f6 = fmatrix_[:, 1, 2]
        f7 = fmatrix_[:, 2, 0]
        f8 = fmatrix_[:, 2, 1]
        f9 = fmatrix_[:, 2, 2]

        loss_a = torch.mul((u2 + torch.mul(a1, u1)), f1) + torch.mul(torch.mul(a1, v1), f2) + torch.mul(a1,
                                                                                                        f3) + torch.mul(
            (v2 + torch.mul(a3, u1)), f4) + torch.mul(torch.mul(a3, v1), f5) + torch.mul(a3, f6) + f7

        loss_b = torch.mul(torch.mul(a2, u1), f1) + torch.mul((u2 + torch.mul(a2, v1)), f2) + torch.mul(a2,
                                                                                                        f3) + torch.mul(
            torch.mul(a4, u1), f4) + torch.mul((v2 + torch.mul(a4, v1)), f5) + torch.mul(a4, f6) + f8

        indexes = torch.nonzero((loss_a < 0.001) & (loss_b < 0.001))

        coor1_ = coor1[indexes]
        coor2_ = coor2[indexes]
        pred_A_ = pred_A[indexes]


        return{
            "coor1_":coor1_,
            "coor2_":coor2_,
            "pred_A_":pred_A_,
            "F_gt":F_gt,
        }

    def forward(self,imf1s,imf2s,images,path_to_weights,step):
        # images = read_img_cam(root)
        # imf1s, imf2s = read_pairs(root)
        co1,co2 ,out , im1_path, im2_path = DKM_matching(imf1s,imf2s,path_to_weights,images,step,draw_img=False)
        # co1=co1[:8000,]
        # co2=co2[:8000,]
        self.data = network_forward(co1,co2 ,out,im1_path, im2_path)


    def backward_net(self ):
        if self.data != 0:
            affine_loss = affine_trans_loss.affine_trans_loss(self.data).to(self.device)
            self.loss = affine_loss(self.data).requires_grad_(True)
            self.loss.backward()
        return  self.loss


    def optimize_parameters(self,imf1s,imf2s,images,path_to_weights,step):
        self.forward(imf1s,imf2s,images,path_to_weights,step)
        loss = self.backward_net()
        self.optimizer_affnet.step()
        self.optimizer_scale.step()
        self.optimizer_angle.step()
        self.scheduler.step()
        return loss

    # def val(self, imf1s, imf2s, images, path_to_weights, step):
    #     self.optimizer.zero_grad()
    #     self.forward(imf1s, imf2s, images, path_to_weights, step)



    def set_input(self, data):
        self.im1 = Variable(data['im1'].to(self.device))
        self.im2 = Variable(data['im2'].to(self.device))

        self.fmatrix = data['F'].cuda()
        self.pose = Variable(data['pose'].to(self.device))
        self.intrinsic1 = data['intrinsic1'].to(self.device)
        self.intrinsic2 = data['intrinsic2'].to(self.device)

        self.im1_ori = data['im1_ori']
        self.im2_ori = data['im2_ori']
        self.batch_size = len(self.im1)
        self.imsize = self.im1.size()[2:]


    def write_summary(self, writer, n_iter,):
        # print("%s | Step: %d, Loss: %2.5f" % (self.args.exp_name, n_iter, self.j_loss.item()))
        print("%s | Step: %d, ,affine_Loss: %2.5f" % (self.args.exp_name, n_iter, self.loss))
        # print("%s | Step: %d, Loss: %2.5f"% (self.args.exp_name, n_iter, self.j_loss.item()))

        # write scalar
        if n_iter % self.args.log_scalar_interval == 0:
            writer.add_scalar('Total_loss', self.loss, n_iter)

        # write image
        """

        if n_iter % self.args.log_img_interval == 0:
            # this visualization shows a number of query points in the first image,
            # and their predicted correspondences in the second image,
            # the groundtruth epipolar lines for the query points are plotted in the second image
            num_kpts_display = 20
            im1_o = self.im1_ori[0].numpy()
            im2_o = self.im2_ori[0].numpy()
            kpt1 = self.coord1.cpu().numpy()[0][:num_kpts_display, :]
            # predicted correspondence
            correspondence = self.out['coord2_ef_add']
            kpt2 = correspondence.detach().cpu().numpy()[0][:num_kpts_display, :]
            lines2 = cv2.computeCorrespondEpilines(kpt1.reshape(-1, 1, 2), 1, self.fmatrix[0].cpu().numpy())
            lines2 = lines2.reshape(-1, 3)
            im2_o, im1_o = utils.drawlines(im2_o, im1_o, lines2, kpt2, kpt1)
            vis = np.concatenate((im1_o, im2_o), 1)
            vis = torch.from_numpy(vis.transpose(2, 0, 1)).float().unsqueeze(0)
            x = vutils.make_grid(vis, normalize=True)
            writer.add_image('Image', x, n_iter)
            """

    def load_model(self, filename):
        to_load = torch.load(filename)
        self.model.load_state_dict(to_load['state_dict'])

        if 'optimizer' in to_load.keys():
            self.optimizer.load_state_dict(to_load['optimizer'])
        if 'scheduler' in to_load.keys():
            self.scheduler.load_state_dict(to_load['scheduler'])


        # return to_load['step']
        return self.model.load_state_dict(to_load['state_dict'])

    def load_from_ckpt(self):
        '''
        load model from existing checkpoints and return the current step
        :param ckpt_dir: the directory that stores ckpts
        :return: the current starting step
        '''

        # load from the specified ckpt path
        if self.args.ckpt_path != "":
            print("Reloading from {}".format(self.args.ckpt_path))
            if os.path.isfile(self.args.ckpt_path):
                step = self.load_model(self.args.ckpt_path)
            else:
                raise Exception('no checkpoint found in the following path:{}'.format(self.args.ckpt_path))

        else:
            ckpt_folder = os.path.join(self.args.outdir, self.args.exp_name)
            os.makedirs(ckpt_folder, exist_ok=True)
            # load from the most recent ckpt from all existing ckpts
            ckpts = [os.path.join(ckpt_folder, f) for f in sorted(os.listdir(ckpt_folder)) if f.endswith('.pth')]
            if len(ckpts) > 0:
                fpath = ckpts[-1]
                step = self.load_model(fpath)
                print('Reloading from {}, starting at step={}'.format(fpath, step))
            else:
                print('No ckpts found, training from scratch...')
                step = 0

        return step

    def save_model(self, step):
        ckpt_folder = os.path.join(self.args.outdir, self.args.exp_name)
        os.makedirs(ckpt_folder, exist_ok=True)

        save_path = os.path.join(ckpt_folder, "{:06d}.pth".format(step))
        print('saving ckpts {}...'.format(save_path))
        # torch.save({'step': step,
        #             'state_dict': self.model.state_dict(),
        #             'optimizer':  self.optimizer.state_dict(),
        #             'scheduler': self.scheduler.state_dict(),
        #             },
        #            save_path)

        torch.save({
            'model_scale': self.model_scale.state_dict(),
            'optimizer_scale': self.optimizer_scale.state_dict(),
            'model_angle': self.model_angle.state_dict(),
            'optimizer_angle': self.optimizer_angle.state_dict(),
            'model_affnet':self.affnet.state_dict(),
            'optimizer_affnet':self.optimizer_affnet.state_dict(),
            'step': step,
        }, save_path)
