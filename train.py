import os
import datetime
from tqdm import trange
from tqdm import tqdm
import torch
from Dataset.datasets import CelebA
from tensorboardX import SummaryWriter
import numpy as np
import random
from Experiments import ISSUE_17_EXP21_V3
import sys
import torchvision
import torchvision.transforms as transforms
from torch.utils import data
from Model.RegionAttNet import RegionAttNet
import sys
import torch.nn as nn
####### EXPERIMENTS DEFINE #######################################
args = ISSUE_17_EXP21_V3('train')   
##################################################################

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)


if __name__ == "__main__":
    # Prepare TF writer
    if args.summary:
        if args.mode in args.summary_register:
            if not os.path.isdir(args.summary_dir):
                os.makedirs(args.summary_dir)
            summary_dir = os.path.join(args.summary_dir, args.model, args.name+ '_fold' + '/', datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
            writer = SummaryWriter(summary_dir)
            args.save_config()
        else:
            writer = None
    else:
        writer = None

    # device
    if not torch.cuda.is_available():
        args.device = 'cpu'
    device = torch.device(args.device)


    isDebug = True if sys.gettrace() else False
    if isDebug:
        args.summary = False
    else:
        args.summary = True


    with open("./Dataset/CelebA/face_mask/legi_test.txt", 'r') as f:
        lines = f.readlines()
        legi_test = [l.rstrip() for l in lines]

    with open("./Dataset/CelebA/face_mask/legi_train.txt", 'r') as f:
        lines = f.readlines()
        legi_train = [l.rstrip() for l in lines]

    with open("./Dataset/CelebA/face_mask/legi_eval.txt", 'r') as f:
        lines = f.readlines()
        legi_eval = [l.rstrip() for l in lines]
    
    img_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std = [0.5] * 3)
    ])
    mask_transforms = transforms.ToTensor()

    dataset_train = CelebA("train", args.data_dir, args.img_dir, args.mask_dir, legi_train, args.sizes, args.mask_type, transform=img_transforms, mask_transform=mask_transforms)
    dataset_eval = CelebA("test", args.data_dir, args.img_dir, args.mask_dir, legi_test, args.sizes, args.mask_type, transform=img_transforms, mask_transform=mask_transforms)

    train_loader = data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=12, pin_memory=True, drop_last=True)
    eval_loader = data.DataLoader(dataset_eval, batch_size=args.batch_size, shuffle=False,  num_workers=12, pin_memory=True, drop_last=True)

    print("Train: {} batches, {} images".format(len(train_loader), args.batch_size * len(train_loader)))
    print("Eval : {} batches, {} images".format(len(eval_loader), args.batch_size * len(eval_loader)))


    visual_dir = os.path.join(args.visual_dir, args.mode, args.name, args.mask_type)
    if not os.path.isdir(visual_dir):
        os.makedirs(visual_dir)

    eval_save_dir = os.path.join(args.visual_dir, "val", args.name, args.mask_type)
    if not os.path.isdir(eval_save_dir):
        os.makedirs(eval_save_dir)

    # PREPARE MODEL ...

    model = RegionAttNet(args.in_ch, args.out_ch, args)
    model = model.to(device)

    start_epoch = 0
    total_steps = 0
    epoch_ = 0
    update_d = True
    val_total_steps = 0
    criterionL1 = nn.L1Loss()
    min_val_loss = np.inf
    # TRAIN
    t = trange(start_epoch+1, args.niter + args.niter_decay)
    for epoch in t:
        t.set_description("Epoch: {}/{}".format(epoch, args.niter + args.niter_decay + 1))
        for data in train_loader:
            gt_img, img_data, mask_data = data
            total_steps += args.batch_size
            # MODEL
            model.set_input(gt_img, img_data, mask_data, device)
            model.optimize_parameters(update_d=True)
            if total_steps % args.visual_freq == 0:
                img_, mask, fake_B, _, real_B = model.get_current_visual()
                mask = mask.expand(img_.size())
                pic = ( torch.cat([img_, mask, fake_B, real_B], dim=0) + 1 ) / 2.0
                grid_pic = torchvision.utils.make_grid(pic, nrow=args.batch_size)
                torchvision.utils.save_image(grid_pic, os.path.join(visual_dir, "Epoch_{}_({}).png".format(epoch, total_steps)))

            if total_steps % args.writer_freq == 0:
                if writer is not None:
                    model.call_tfboard(writer, total_steps)

        t.write("start validation...")
        val_loss = 0
        avg_val_losses = 0
        for eval_data in eval_loader:
            val_total_steps += args.batch_size
            eval_gt_img, eval_img_data, eval_mask_data = eval_data
            model.set_input(eval_gt_img, eval_img_data, eval_mask_data, device)
            model.test()
            img, mask, fake_B, pred_img, real_B = model.get_current_visual()

            val_loss = criterionL1(fake_B, real_B)
            avg_val_losses += val_loss.item()

            if val_total_steps % 600 == 0:
                
                mask = mask.expand(img.size())
                pic = ( torch.cat([img, fake_B, real_B], dim=0) + 1 ) / 2.0
                grid_pic = torchvision.utils.make_grid(pic, nrow=args.batch_size)
                torchvision.utils.save_image(grid_pic, os.path.join(eval_save_dir, "eval_Epoch_{}_({}).png".format(epoch, val_total_steps)))
            
                if writer is not None:
                    writer.add_scalar("val_loss_iter", val_loss, val_total_steps)

        avg_val_losses = avg_val_losses / len(eval_loader)
        if writer is not None:
            writer.add_scalar("val_loss_epoch", avg_val_losses, epoch)
        if min_val_loss > avg_val_losses:
            min_val_loss = avg_val_losses
            model.save(epoch)
            epoch_ = epoch
        lr_ = model.update_learning_rate()
        t.set_postfix(last_saved='{}'.format(epoch_), lr='{:.3e}'.format(lr_)) 
