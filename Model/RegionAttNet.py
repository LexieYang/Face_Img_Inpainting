import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.networks import get_scheduler, Vgg16, PFDiscriminator, NLayerDiscriminator, gated_ResU_Net
from .loss import GANLoss, StyleLoss, PerceptualLoss
import numpy as np
import functools


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=True)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal(m.weight.data, 1.0, gain)
            init.constant(m.bias.data, 0.0)

def init_net(net, init_type='normal', init_gain=0.02):

    init_weights(net, init_type, gain=init_gain)
    return net

def define_D(input_nc, ndf, n_layers_D=3, norm='batch',  init_type='normal', init_gain=0.02):
    netD = None
    norm_layer = get_norm_layer(norm_type=norm)
    netD = NLayerDiscriminator(input_nc, ndf, n_layers=n_layers_D, norm_layer=norm_layer, use_sigmoid=False)
    
    netF = PFDiscriminator()
    
    return init_net(netD, init_type, init_gain), init_net(netF, init_type, init_gain)

def define_G(input_nc, output_nc, ngf,  norm='batch', init_type='normal', init_gain=0.02):
    
    norm_layer = get_norm_layer(norm_type=norm)
    netG = gated_ResU_Net(input_nc+1, output_nc)
    return init_net(netG, init_type, init_gain)


class RegionAttNet(nn.Module):
    def __init__(self, in_ch, out_ch, args, isTrain=True):
        super(RegionAttNet, self).__init__()
        self.att_save_path = 'att_map/'
        self.isTrain = isTrain
        self.args = args
        self.vgg = Vgg16()                   # for NetF
        self.vgg = torch.nn.DataParallel(self.vgg, list(range(self.args.ngpu)))
        self.gener = define_G(in_ch, out_ch, args.base_dim, args.norm,
                                args.init_type, args.init_gain)
        self.gener = torch.nn.DataParallel(self.gener, list(range(self.args.ngpu)))
        self.save_dir = os.path.join(args.ckpt_dir, args.name)

        # descriminators
        if self.isTrain:
            # TODO 
            self.netD, self.netF = define_D(3, args.base_dim,
                                          args.n_layers_D, args.norm, args.init_type, args.init_gain)
            
            self.netF = torch.nn.DataParallel(self.netF, list(range(self.args.ngpu)))
            self.netD = torch.nn.DataParallel(self.netD, list(range(self.args.ngpu)))


        # Testing
        if not self.isTrain or args.resume:
            print("loading ckpt ... ")
            self.load_network(self.gener, 'PENnet', args.which_epoch)
            # resume
            if self.isTrain:
                self.load_network(self.netD, 'D', args.which_epoch)
                self.load_network(self.netF, 'F', args.which_epoch)

        if self.isTrain:
            self.old_lr = args.lr
            # Losses
            self.PerceptualLoss = PerceptualLoss()
            self.StyleLoss = StyleLoss()
            self.criterionDiv = nn.KLDivLoss()              # attention supervised 
            self.criterionMSE = nn.MSELoss()
            self.criterionGAN = GANLoss(target_real_label=0.9, target_fake_label=0.1)
            self.criterionL1 = nn.L1Loss()
            # optimizers
            self.schedulers = []
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.gener.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
            self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_F)

            for optimizer in self.optimizers:
                self.schedulers.append(get_scheduler(optimizer, args))

    def set_input(self, gt, img, mask, device):
        # generate input:
        batch_size = gt.size(0)
        # set inputs
        self.img = img.float().to(device)   # img is the masked img
        self.gt  = gt.float().to(device)    # ground truth, clean img
        self.mask = mask.float().to(device) # binary mask, 1 for mask 0 for not 
    def __rect(self):
        low, high, full = self.args.sizes
        rect_size = np.random.choice(high-low+1) + low
        assert rect_size >= low and rect_size <= high, "value error"
        
        top_l_x = np.random.choice(full - rect_size)
        top_l_y = np.random.choice(full - rect_size)

        return [top_l_x, top_l_y, rect_size, full]
    


    @staticmethod
    def init_weights(m):
        if type(m) == nn.Conv2d:
            nn.init.xavier_normal_(m.weight)
            try:
                nn.init.constant_(m.bias, 0)
            except AttributeError:
                pass

        if type(m) == nn.Linear:
            nn.init.normal_(m.weight)
            try:
                nn.init.constant_(m.bias, 0)
            except AttributeError:
                pass
        
    def forward(self):

        self.gt_inputs = torch.cat((self.gt, self.mask), dim=1)
        self.inputs = torch.cat((self.img, self.mask), dim=1)
        # siamese training
        self.gt_pred_img = self.gener(self.gt_inputs)  
        self.pred_img = self.gener(self.inputs)

        self.comp_img = (1 - self.mask)*self.img + self.mask*self.pred_img

    def optimize_parameters(self, update_d=False):
        self.train()
        self.forward()
        # Discriminator
        if update_d:
            self.optimizer_D.zero_grad()
            self.optimizer_F.zero_grad()
            self.backward_D()
            self.optimizer_D.step()
            self.optimizer_F.step()
        # Generator
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def backward_D(self):

        fake_AB = self.comp_img
        real_AB = self.gt
        self.gt_latent_fake = self.vgg(self.comp_img.clone().detach())
        self.gt_latent_real = self.vgg(self.gt.clone().detach())

        self.pred_fake = self.netD(fake_AB.detach())
        self.pred_real = self.netD(real_AB)
        self.loss_D_fake = self.criterionGAN(self.pred_fake, self.pred_real, True)

        self.pred_fake_F = self.netF(self.gt_latent_fake['relu3_3'].detach())
        self.pred_real_F = self.netF(self.gt_latent_real['relu3_3'])
        self.loss_F_fake = self.criterionGAN(self.pred_fake_F, self.pred_real_F, True)

        self.loss_D = self.loss_D_fake + self.loss_F_fake 
        self.loss_D.backward()

    def backward_G(self):
        fake_AB = self.comp_img
        pred_img = self.pred_img

        pred_fake = self.netD(fake_AB)
        pred_fake_f = self.netF(self.gt_latent_fake['relu3_3'])

        pred_real = self.netD(self.gt)
        pred_real_f = self.netF(self.gt_latent_real['relu3_3'])
        self.style_loss = self.StyleLoss(fake_AB, self.gt)
        self.perc_loss = self.PerceptualLoss(fake_AB, self.gt)
        self.loss_G_GAN = self.criterionGAN(pred_fake, pred_real, False) + self.criterionGAN(pred_fake_f, pred_real_f, False)
       
        self.loss_l1 = self.criterionL1(self.comp_img, self.gt)

        self.loss_G = self.style_loss * 250 + self.loss_G_GAN * self.args.gan_weight + self.loss_l1 + 0.1 * self.perc_loss  #+ 0.1 * self.loss_G_atten_x2#self.loss_G_L1 + 0.01 * self.loss_G_atten_x5

        self.loss_G.backward()



    def get_current_visual(self):
        img = self.img.data
        real_B = self.gt.data
        comp_img = self.comp_img.data
        mask = self.mask.data
        output = self.pred_img

        return img, mask, comp_img, output, real_B

    def call_tfboard(self, writer, step):
        writer.add_scalar("G_GAN", self.loss_G_GAN.data.item(), step)
        writer.add_scalar("loss_l1", self.loss_l1.data.item(), step)
        writer.add_scalar("D", self.loss_D_fake.data.item(), step)
        writer.add_scalar("F", self.loss_F_fake.data.item(), step)

    def get_GAN_loss(self):
        return self.loss_G_GAN.data.item()

    def save_network(self, network, network_label, epoch_label, gpu_ids=[0]):
        if os.path.exists( self.save_dir ) is False:
            os.makedirs( self.save_dir)
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)

        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(gpu_ids[0])

    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        network.load_state_dict(torch.load(save_path))

    def save(self, epoch):
        self.save_network(self.gener, 'PENnet', epoch)
        self.save_network(self.netD, 'D', epoch)
        self.save_network(self.netF, 'F', epoch)

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        return lr
    


    def test(self):
        with torch.no_grad():
            self.inputs = torch.cat((self.img, self.mask), dim=1)
            self.pred_img = self.gener(self.inputs)
            self.comp_img = (1 - self.mask)*self.img + self.mask*self.pred_img

    