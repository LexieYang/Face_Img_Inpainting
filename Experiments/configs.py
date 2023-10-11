import os 
import json

class Base_Config():
    def __init__(self):
        self.seed = 2333
        self.img_dir  = "../data/CelebA_With_Masks/" 
        self.mask_dir  = "../data/Binary_Masks/"  
        self.irr_mask_dir = "../data/CelebA/irregular_masks_ratio/"
        self.data_dir = "../data/CelebA/img_align_celeba/"
        self.ckpt_dir = "./model_weights/SPD/"
        self.dataset = "CelebA"
        self.visual_dir = "./Visual/"
        self.summary_dir = "./tfboards/"
        self.json_path = './Experiments/logs/'
        self.device = 'cuda'
        # self.weights_
        self.ngpu  = 4
        # place holder:
        self.model = 'MODEL'
        self.name = "NAME"
        
        self.lr_policy='lambda'
        self.lr_decay_iters=50
        self.epoch_count=0

        self.beta1 = 0.5
        self.lambda_A = 100
        self.gan_weight = 0.1

        self.visual_freq = 15000     # default 15000
        self.writer_freq = 500
        self.save_epoch_freq = 1
        self.mask_type =  "face_mask"    #["face_mask", "cnt_mask","irr_mask"]  # three types of masks, including face masks,\
                                                                                # centered masks and irregular masks.

    def save_config(self):
        with open(os.path.join(self.json_path, "{}-{}-{}".format(self.dataset, self.model, self.name)), "w") as f:
            json.dump(self.__dict__, f, indent=4)


class ISSUE_17_EXP21_V3(Base_Config):

    def __init__(self, mode="train"):
        super(ISSUE_17_EXP21_V3, self).__init__()
        self.niter=20        # final num of epochs equal to (niter + niter_decay + 1)
        self.niter_decay=100
        self.batch_size = 16   # around 8
        self.in_ch = 3   # the input channel of coarse stage
        self.out_ch = 3
        self.base_dim = 64  # The base dim of discriminator 

        self.sizes =  (256, 256) # min, max value of rect. and size of img

        self.resume = False          # loading model from ckpt. 
        self.which_epoch = '29'        # loading from which epoch
        self.lr = 2e-4
        self.model = "SPD"  # [Unet or PSA]
        self.name = "ISSUE_17_EXP21_V3"
        self.mode = mode          # which mode
        self.summary = True         # flg for tfboards
        self.summary_register = ['train']
        self.teacher_forcing_ratio = 1.0 # 

        
        # add for resUnet
        self.norm = 'instance'
        # self.use_dropout = 
        self.init_type = 'normal'
        self.init_gain = 0.02
        self.n_layers_D = 0 
        self.gpu_ids = [0]
        # self.wait = 10000          # if after |wait| images,  GANloss still not decreasing, then stop the update the discriminators
