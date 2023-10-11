import os
import datetime
from tqdm import tqdm
import torch
from Dataset.datasets import CelebA
from tensorboardX import SummaryWriter
import numpy as np
import random
from Experiments import ISSUE_17_EXP21_V3
import torchvision.transforms as transforms
from torch.utils import data
from Model.RegionAttNet import RegionAttNet
import skimage
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from torchvision.utils import save_image

args = ISSUE_17_EXP21_V3('test')

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

def PSNR(pred, gt, data_range=255):
    return psnr(pred, gt)

def L1(pred, gt):
    return np.mean(np.abs((np.mean(pred,2) - np.mean(gt,2))/255))


def SSIM(pred, gt, data_range, win_size=11, multichannel=True):
    return ssim(pred, gt, win_size=11, multichannel=True)

def img2photo(imgs):
    return ((imgs+1)*127.5).transpose(1,2).transpose(2,3).detach().cpu().numpy()

def save_img(path, name, img):
    # img (H,W,C) or (H,W) np.uint8
    skimage.io.imsave(path+'/'+name+'.png', img)

def evaluate_batch(batch_size, img_batch, output_batch, gt_batch, pred_batch, mask_batch, save=False, path=None, count=None, index=None):
    gt_batch_np = ((gt_batch.detach().permute(0,2,3,1).cpu().numpy()+1)*127.5).astype(np.uint8)
    pred_batch_np = ((pred_batch.detach().permute(0,2,3,1).cpu().numpy()+1)*127.5).astype(np.uint8)
    if save:
        mask_batch_np = ((mask_batch.detach().permute(0,2,3,1).cpu().numpy()[:,:,:,0]+1)*127.5).astype(np.uint8)

    psnr, ssim, l1 = 0., 0., 0.
    for i in range(batch_size):
        gt, pred = gt_batch_np[i], pred_batch_np[i]	# data_range = pred.max() - gt.max()
        data_range = pred.max() - pred.min()
        psnr += PSNR(pred, gt, data_range)
        ssim += SSIM(pred, gt, data_range)
        l1 += L1(pred, gt)
        
        gt_tr = (gt_batch[i]+1)/2.0
        pred_tr = (pred_batch[i]+1)/2.0
        img_tr = (img_batch[i]+1)/2.0
        output_tr = (output_batch[i]+1)/2.0
        mask_tr = (mask_batch[i]+1)/2.0
        if save:
            save_image(pred_tr, path+str(count)+'_'+str(i)+'_output.png')
            save_image(gt_tr, path+str(count)+'_'+str(i)+'_gt.png')
            save_image(img_tr, path+str(count)+'_'+str(i)+'_img.png')
            save_image(output_tr, path+str(count)+'_'+str(i)+'_pred.png')
            save_image(mask_tr, path+str(count)+'_'+str(i)+'_mask.png')

    return psnr/batch_size, ssim/batch_size, l1/batch_size

save_path = 'test_results/'

if __name__ == "__main__":
    # Prepare TF writer
    if args.summary:
        if args.mode in args.summary_register:
            if not os.path.isdir(args.summary_dir):
                os.mkdir(args.summary_dir)
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


    with open("./Dataset/CelebA/face_mask/legi_eval.txt", 'r') as f:
        lines = f.readlines()
        legi_test = [l.rstrip() for l in lines]

    img_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std = [0.5] * 3)
    ])
    mask_transforms = transforms.ToTensor()

    dataset_test = CelebA("test", args.data_dir, args.img_dir, args.mask_dir, legi_test, args.sizes, args.mask_type, transform=img_transforms, mask_transform=mask_transforms)
    test_loader = data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True, drop_last=True)

    save_dir = os.path.join(args.visual_dir, args.mode, args.name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # PREPARE MODEL ...
    model = RegionAttNet(args.in_ch, args.out_ch, args, isTrain=False)
    model = model.to(device)

    start_epoch = 0
    total_steps = 0
    epoch = int(args.which_epoch)
    print("WORKING UNDER {}".format(args.mode))
    avg_psnr, avg_ssim, avg_l1 = 0., 0., 0.

    for i, data in enumerate(tqdm(test_loader)):
        img, gt_img, mask = data

        total_steps += args.batch_size

        model.set_input(img, gt_img, mask, device)
        model.test()

        img, mask, fake_B, output, real_B = model.get_current_visual()

        batch_avg_psnr, batch_avg_ssim, batch_avg_l1 = evaluate_batch(
            batch_size=args.batch_size,
            img_batch = img,
            output_batch = output,
            gt_batch=real_B,
            pred_batch=fake_B,
            mask_batch=mask,
            save=False,
            path=save_path,
            count=(i+1)
            )
        avg_psnr = avg_psnr + ((batch_avg_psnr- avg_psnr) / (i+1))
        avg_ssim = avg_ssim + ((batch_avg_ssim- avg_ssim) / (i+1))
        avg_l1 = avg_l1 + ((batch_avg_l1- avg_l1) / (i+1))

        print(
            "Number: %05d" % ((i+1) * args.batch_size),
            " | Average: PSNR: %.4f" % (avg_psnr),
            " SSIM: %.4f" % (avg_ssim),
            " L1: %.4f" % (avg_l1),
            "| Current batch:", (i+1),
            " PSNR: %.4f" % (batch_avg_psnr),
            " SSIM: %.4f" % (batch_avg_ssim),
            " L1: %.4f" % (batch_avg_l1), flush=True
        )
 