import argparse
import os
from pickle import TRUE
import sys
import random
import time
import torch
import cv2
import math
import numpy as np
import torch.backends.cudnn as cudnn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm

class _AvgMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self._sum = 0.0
        self._n = 0

    def add(self, value):
        self._sum += float(value)
        self._n += 1

    def value(self):
        if self._n == 0:
            return [0.0]
        return [self._sum / self._n]


class meter:
    AverageValueMeter = _AvgMeter

import utils
import json
from data.load_data import loadingData
from load_test_data import loadingTestData, loadingTestChikuseiData
from data.load_tiff_data import loadingTiffData
from tiff_utils import build_pairs, evaluate_tiff_pairs


def _set_cuda_visible_devices_from_args(argv):
    if argv is None:
        return
    if "--gpus" not in argv:
        return
    try:
        i = argv.index("--gpus")
        if i + 1 < len(argv):
            os.environ["CUDA_VISIBLE_DEVICES"] = str(argv[i + 1])
    except Exception:
        return


_set_cuda_visible_devices_from_args(sys.argv)


from OverallModel import General_VolFormer



from basicModule import *
import scipy.io
# loss
from Loss import HybridLoss, CrossEntropy2d
from metrics import quality_assessment
from torch.autograd import Variable


# global settings
resume = False
log_interval = 10
model_name = ''
test_data_dir = ''


def main():
    # parsers UseUnLabeledMixUp
    main_parser = argparse.ArgumentParser(description="parser for SR network")
    subparsers = main_parser.add_subparsers(title="subcommands", dest="subcommand",)
    
    
    train_parser = subparsers.add_parser("train", help="parser for training arguments")
    train_parser.add_argument("--cuda", type=int, required=False, default=1,
                              help="set it to 1 for running on GPU, 0 for CPU")
    train_parser.add_argument("--batch_size", type=int, default=1, help="batch size, default set to 64")
    train_parser.add_argument("--UseLabeledSpectralMixUp", type=int, default=0, help="if we use gan loss, 0  for false, 1 for yes")
    train_parser.add_argument("--theta_LabeledSpectralMixUp", type=int, default=1, help="if we use gan loss, 0  for false, 1 for yes")
    train_parser.add_argument("--UseUnlabelConsistency", type=int, default=0, help="if we use unlabeled consistency, 0  for false, 1 for yes")
    train_parser.add_argument("--UseRGB", type=int, default=0, help="if we use rgb, 0  for false, 1 for yes")
    train_parser.add_argument("--epochs", type=int, default=20, help="epochs, default set to 20")
    train_parser.add_argument("--conversionMat_path", type=str, default="./data/conversion_8channels.mat",
                              help="path to conversion matrix transforming spectral images of 31 channels to images of 8 channels ")
    train_parser.add_argument("--n_feats", type=int, default=192, help="n_feats, default set to 256")
    train_parser.add_argument("--n_blocks", type=int, default=3, help="n_blocks, default set to 6")
    train_parser.add_argument("--n_subs", type=int, default=8, help="n_subs, default set to 8")
    train_parser.add_argument("--n_ovls", type=int, default=2, help="n_ovls, default set to 1")
    train_parser.add_argument("--n_scale", type=int, default=4, help="n_scale, default set to 2")
    train_parser.add_argument("--use_share", type=bool, default=True, help="f_share, default set to True")
    train_parser.add_argument("--dataset_name", type=str, default="Cave", help="dataset_name, default set to dataset_name")
    train_parser.add_argument("--train_dir_mslabel", type=str, default='./dataset/Cave_x8/trains/', help="directory of train spectral dataset")
    train_parser.add_argument("--eval_dir_ms", type=str, default='./dataset/Cave_x8/evals/', help="directory of evaluation spectral dataset")
    train_parser.add_argument("--test_dir", type=str, default='./dataset/Cave_x8/tests/', help="directory of test spectral dataset")
    train_parser.add_argument("--train_dir_rgb", type=str, default='./dataset/RGB/Div2k/Div2k_and_Flickr2k_train_data/', help="directory of train rgb dataset")
    train_parser.add_argument("--theta_rgb", type=int, default=24, help="how many times of rgb images regard to ms images")   ###24    ###34
    train_parser.add_argument("--theta_unlabel", type=int, default=3, help="how many times of unlabel ms images regard to ms label images")
    train_parser.add_argument("--data_train_num", type=int, default=4500, help="how many .mat files used in each epoch")  ##4500
    train_parser.add_argument("--data_eval_num", type=int, default=12,help="how many .mat files used in each epoch")
    train_parser.add_argument("--model_title", type=str, default="VolFormer", help="model_title, default set to model_title")
    train_parser.add_argument("--seed", type=int, default=3000, help="start seed for model")
    train_parser.add_argument("--learning_rate", type=float, default=1e-4,
                              help="learning rate, default set to 1e-4")
    train_parser.add_argument("--weight_decay", type=float, default=0, help="weight decay, default set to 0")
    train_parser.add_argument("--save_dir", type=str, default="./trained_model/",
                              help="directory for saving trained models, default is trained_model folder")
    train_parser.add_argument("--gpus", type=str, default="7", help="gpu ids (default: 7)")

    # TIFF dataset options (for HISR/lzy TIFF pairs)
    train_parser.add_argument("--use_tiff", type=int, default=1, help="1: use *_LR.tif/*_HR.tif pair dataset; 0: use original .mat dataset")
    train_parser.add_argument("--train_dir_tiff", type=str, default="/home/fdw/code/HISR/lzy/tiff/train", help="TIFF train dir containing *_LR.tif and *_HR.tif")
    train_parser.add_argument("--eval_dir_tiff", type=str, default="/home/fdw/code/HISR/lzy/tiff/test", help="TIFF eval dir containing *_LR.tif and *_HR.tif")
    train_parser.add_argument("--test_dir_tiff", type=str, default="/home/fdw/code/HISR/lzy/tiff/test", help="TIFF test dir containing *_LR.tif and *_HR.tif")
    train_parser.add_argument("--lr_patch", type=int, default=50, help="LR patch size for TIFF training")
    train_parser.add_argument("--samples_per_image", type=int, default=200, help="samples per LR/HR image pair for TIFF training")
    train_parser.add_argument("--tile", type=int, default=50, help="TIFF eval/test sliding window tile (LR space)")
    train_parser.add_argument("--overlap", type=int, default=0, help="TIFF eval/test sliding window overlap (LR space)")
    train_parser.add_argument("--cache_images", type=int, default=0, help="1: cache TIFF images in RAM")
    train_parser.add_argument("--num_workers", type=int, default=4, help="dataloader workers")
    train_parser.add_argument("--n_colors", type=int, default=224, help="spectral channels for TIFF (bands)")
    train_parser.add_argument("--scale", type=int, default=4, help="scale factor for TIFF SR (must match LR/HR size ratio)")
    train_parser.add_argument("--eval_interval", type=int, default=1, help="run eval every N epochs")
    train_parser.add_argument("--eval_iters", type=int, default=300, help="run eval every N iterations, 0 to disable")
    train_parser.add_argument("--eval_strict", type=int, default=1, help="1: TIFF eval 遇到异常直接报错停止；0: 打印并跳过异常样本")
    train_parser.add_argument("--save_best", type=int, default=1, help="save best checkpoint by PSNR when using TIFF")

    # VolFormer backbone lightweight config
    train_parser.add_argument("--vf_embed_dim", type=int, default=120, help="VolFormer embed_dim")
    train_parser.add_argument("--vf_depth", type=int, default=4, help="VolFormer depth of each stage")
    train_parser.add_argument("--vf_stages", type=int, default=4, help="VolFormer number of stages")
    train_parser.add_argument("--vf_num_heads", type=int, default=4, help="VolFormer number of heads in each stage")
    train_parser.add_argument("--amp", type=int, default=1, help="Use AMP (1) or not (0)")
    train_parser.add_argument("--resume", type=str, default='/home/fdw/code/HISR/lzy/code/HISR/VolFormer/checkpoints_old/Cave_Cave_Cave_VolFormer_Blocks=3_Subs8_Ovls2_Feats=192_ckpt_epoch_4.pth', help="checkpoint path to resume (optional)")

    test_parser = subparsers.add_parser("test", help="parser for testing arguments")
    test_parser.add_argument("--cuda", type=int, required=False, default=1,
                             help="set it to 1 for running on GPU, 0 for CPU")
    test_parser.add_argument("--gpus", type=str, default="7", help="gpu ids (default: 7)")
    test_parser.add_argument("--test_dir", type=str, default='./dataset/Cave_x8/tests/', help="directory of test spectral dataset")
    test_parser.add_argument("--model_dir", type=str, default='./checkpoints/Cave_Cave_Cave_VolFormer_Blocks=3_Subs8_Ovls2_Feats=256_ckpt_epoch_20.pth', help="directory of trained model")
    test_parser.add_argument("--n_feats", type=int, default=256, help="n_feats, default set to 256")
    test_parser.add_argument("--n_blocks", type=int, default=3, help="n_blocks, default set to 6")
    test_parser.add_argument("--n_subs", type=int, default=8, help="n_subs, default set to 8")
    test_parser.add_argument("--n_ovls", type=int, default=2, help="n_ovls, default set to 1")
    test_parser.add_argument("--n_colors", type=int, default=31, help="n_colors, default set to 31")
    test_parser.add_argument("--n_scale", type=int, default=8, help="n_scale, default set to 2")
    test_parser.add_argument("--model_title", type=str, default="VolFormer",
                             help="model_title, default set to model_title")
    test_parser.add_argument("--result_path", type=str, default="./results/",
                             help="result_path, directory of result")
    args = main_parser.parse_args()
    if hasattr(args, "gpus"):
        print(args.gpus)
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    if args.subcommand is None:
        print("ERROR: specify either train or test")
        sys.exit(1)
    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)
    if args.subcommand == "train":
        train(args)
    else: 
        test(args)
    pass


bce_loss = torch.nn.BCEWithLogitsLoss()


def loss_calc(pred, label, device):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = Variable(label.long()).to(device)
    criterion = CrossEntropy2d().to(device)

    return criterion(pred, label)


def train(args):
    device = torch.device("cuda" if args.cuda else "cpu")
    print("Start seed: ", args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True

    # load conversion_matrix and by multiplying the conversion matrix with images, we can get images with 8 channels
    conversion_matix = scipy.io.loadmat(args.conversionMat_path)  # ("~/Data/Chikusei/conversion_matrix_128_8.mat") #
    conversion_matix = np.array(conversion_matix['conversion_matrx'], dtype=np.float32)
    conversion_matix = torch.from_numpy(conversion_matix.copy())
    conversion_matix = conversion_matix.to(device)

    print('===> Loading datasets')

    if int(getattr(args, "use_tiff", 0)) == 1:
        train_mslabel_set = loadingTiffData(
            image_dir=args.train_dir_tiff,
            scale=args.scale,
            lr_patch=args.lr_patch,
            samples_per_image=args.samples_per_image,
            cache_images=bool(args.cache_images),
        )
        train_rgb_set = None
        eval_pairs = build_pairs(args.eval_dir_tiff)
        eval_ms_loader = None # Not used in TIFF mode for validation
        colors = int(args.n_colors)
        args.n_scale = int(args.scale)
    else:
        train_mslabel_set = loadingData(image_dir=args.train_dir_mslabel, augment=False, total_num=args.data_train_num)
        train_rgb_set = loadingData(image_dir=args.train_dir_rgb, augment=False, total_num=args.theta_rgb * args.data_train_num)
        eval_mslabel_set = loadingData(image_dir=args.eval_dir_ms, augment=False, total_num=args.data_eval_num)

        eval_ms_loader = DataLoader(eval_mslabel_set, batch_size=1, num_workers=8, shuffle=False)
        if args.dataset_name == 'Cave':
            colors = 31
        elif args.dataset_name == 'Chikusei':
            colors = 128
        elif args.dataset_name == 'Pavia':
            colors = 102
        else:
            colors = 121


    print('===> Building model')
    if args.model_title == "VolFormer":
        net = General_VolFormer(
            n_subs=args.n_subs,
            n_ovls=args.n_ovls,
            n_colors=colors,
            n_blocks=args.n_blocks,
            n_feats=args.n_feats,
            n_scale=args.n_scale,
            res_scale=0.1,
            use_share=args.use_share,
            conv=default_conv,
            vf_embed_dim=int(getattr(args, "vf_embed_dim", 60)),
            vf_depth=int(getattr(args, "vf_depth", 2)),
            vf_stages=int(getattr(args, "vf_stages", 4)),
            vf_num_heads=int(getattr(args, "vf_num_heads", 2)),
        )

    model_title = args.dataset_name + "_"+args.dataset_name + "_" + args.model_title + '_Blocks=' + str(args.n_blocks) + '_Subs' + str(
        args.n_subs) + '_Ovls' + str(args.n_ovls) + '_Feats=' + str(args.n_feats)
    model_name = './checkpoints5/' + model_title + "_ckpt_epoch_" + str(16) + ".pth"
    args.model_title = model_title

    if torch.cuda.device_count() > 1:
        print("===> Let's use", torch.cuda.device_count(), "GPUs.")
        net = torch.nn.DataParallel(net)
    start_epoch = 0
    if resume:
        if os.path.isfile(model_name):
            print("=> loading checkpoint '{}'".format(model_name))
            checkpoint = torch.load(model_name)
            start_epoch = checkpoint["epoch"]
            net.load_state_dict(checkpoint["model"].state_dict())


        else:
            print("=> no checkpoint found at '{}'".format(model_name))
    net.to(device).train()

    # Loss
    h_loss = HybridLoss(spatial_tv=True, spectral_tv=True)
    L1_loss = torch.nn.L1Loss()

    best_psnr = -1e9
    if args.resume is not None and os.path.exists(args.resume):
        print("=> loading resume checkpoint '{}'".format(args.resume))
        try:
            ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(args.resume, map_location="cpu")
        if isinstance(ckpt, dict) and "model" in ckpt:
            net.load_state_dict(ckpt["model"].state_dict() if hasattr(ckpt["model"], "state_dict") else ckpt["model"])
            if "optim" in ckpt and ckpt["optim"] is not None:
                optimizer_state = ckpt["optim"]
            else:
                optimizer_state = None
            if "best_psnr" in ckpt and ckpt["best_psnr"] is not None:
                best_psnr = float(ckpt["best_psnr"])
            if "epoch" in ckpt:
                start_epoch = int(ckpt["epoch"])
        else:
            optimizer_state = None
    else:
        optimizer_state = None

    print("===> Setting optimizer and logger")
    # add L2 regularization

    optimizer = Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        print("=> optimizer state loaded")

    epoch_meter_mslabel = meter.AverageValueMeter()
    epoch_meter_mslabelmixup = meter.AverageValueMeter()
    epoch_meter_mslabelrgb = meter.AverageValueMeter()
    epoch_meter_msunlabelmixup = meter.AverageValueMeter()
    epoch_meter_msunlabel = meter.AverageValueMeter()
    epoch_meter_rgb = meter.AverageValueMeter()
    writer = SummaryWriter('runs/' + model_title + '_' + str(time.localtime()))

    print('===> Start training')

    for e in range(start_epoch, args.epochs):
        adjust_learning_rate(args.learning_rate, optimizer, e + 1, args.epochs)

        epoch_meter_mslabel.reset()
        print("Start epoch {}, labeled ms learning rate = {}".format(e + 1, optimizer.param_groups[0]["lr"]))
        epoch_meter_mslabelmixup.reset()
        epoch_meter_msunlabel.reset()
        epoch_meter_mslabelrgb.reset()
        epoch_meter_msunlabelmixup.reset()
        epoch_meter_rgb.reset()

        
        iteration = 0
        train_mslabel_loader = DataLoader(train_mslabel_set, batch_size=args.batch_size, num_workers=args.num_workers if int(getattr(args, "use_tiff", 0)) == 1 else 8, shuffle=True)
        train_mslabel_iter = iter(train_mslabel_loader)

        if int(getattr(args, "use_tiff", 0)) == 1:
            pbar = tqdm(train_mslabel_loader, desc=f"Train E{e+1}", dynamic_ncols=True, mininterval=10.0, leave=True)

            use_amp = bool(int(getattr(args, "amp", 1))) and device.type == "cuda"
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

            for iteration, batch_mslabel in enumerate(pbar, start=1):
                x, lms, gt = batch_mslabel
                x, lms, gt = x.to(device), lms.to(device), gt.to(device)

                optimizer.zero_grad(set_to_none=True)
                img_size = x.shape[2:4]

                if use_amp:
                    with torch.amp.autocast(device_type="cuda"):
                        y_ms_l = net(x, lms, modality="spectral", img_size=img_size)
                        loss = h_loss(y_ms_l, gt)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    y_ms_l = net(x, lms, modality="spectral", img_size=img_size)
                    loss = h_loss(y_ms_l, gt)
                    loss.backward()
                    optimizer.step()

                loss_v = float(loss.item())
                epoch_meter_mslabel.add(loss_v)
                pbar.set_postfix({"loss": f"{loss_v:.6f}"})

                if (iteration % log_interval) == 0:
                    print(
                        "===> {} B{} Sub{} Fea{} GPU{}\tEpoch[{}]({}/{}): ms Loss: {:.6f}".format(
                            time.ctime(),
                            args.n_blocks,
                            args.n_subs,
                            args.n_feats,
                            args.gpus,
                            e + 1,
                            iteration,
                            len(train_mslabel_loader),
                            loss_v,
                        ),
                        flush=True,
                    )
                    writer.add_scalar('scalar/train_loss_ms', loss_v, e * len(train_mslabel_loader) + iteration)
                
                # ---- MODIFICATION: Intermediate Validation ----
                current_total_iter = e * len(train_mslabel_loader) + iteration
                if args.eval_iters > 0 and (current_total_iter % args.eval_iters == 0):
                    print(f"\n[Validation] Triggered at total_iter {current_total_iter}...", flush=True)
                    mean_psnr, mean_sam = evaluate_tiff_pairs(
                        net,
                        eval_pairs,
                        device=device,
                        scale=int(args.scale),
                        tile=int(getattr(args, "tile", 50)),
                        overlap=int(getattr(args, "overlap", 0)),
                        save_dir=None,
                        strict=bool(int(getattr(args, "eval_strict", 0))),
                    )
                    print(f"\n[ITER {current_total_iter}] TIFF eval PSNR={mean_psnr:.4f} dB | SAM={mean_sam:.6f} rad", flush=True)

                    if np.isfinite(mean_psnr) and np.isfinite(mean_sam):
                        writer.add_scalar('scalar/tiff_eval_psnr_iters', mean_psnr, current_total_iter)
                        writer.add_scalar('scalar/tiff_eval_sam_iters', mean_sam, current_total_iter)

                        if int(getattr(args, "save_best", 1)) == 1 and mean_psnr > best_psnr:
                            best_psnr = mean_psnr
                            save_checkpoint(args, net, optimizer, e + 1, best_psnr=best_psnr, is_best=True)
                    else:
                        print(f"[TIFF EVAL][ITER {current_total_iter}] 指标为NaN/Inf，跳过写tensorboard/更新best", flush=True)
                    
                    net.to(device).train()
                # -----------------------------------------------

        else:
            train_rgb_loader = DataLoader(train_rgb_set, batch_size=args.theta_rgb * args.batch_size, num_workers=8, shuffle=True)
            train_rgb_iter = iter(train_rgb_loader)

            for batch_mslabel, batch_rgb in zip(train_mslabel_iter, train_rgb_iter):
                # training for spectral images

                x, lms, gt = batch_mslabel
                x, lms, gt = x.to(device), lms.to(device), gt.to(device)
                optimizer.zero_grad()
                img_size=x.shape[2:4]
                y_ms_l = net(x, lms, modality="spectral",img_size=img_size)
                loss = h_loss(y_ms_l, gt)
                epoch_meter_mslabel.add(loss.item())
                loss.backward()
                optimizer.step()

                # tensorboard visualization
                if (iteration + log_interval) % log_interval == 0:
                    print("===> {} B{} Sub{} Fea{} GPU{}\tEpoch[{}]({}/{}): ms Loss: {:.6f}".format(time.ctime(),
                                                                                                args.n_blocks,
                                                                                                args.n_subs,
                                                                                                args.n_feats,
                                                                                                args.gpus, e + 1,
                                                                                                iteration + 1,
                                                                                                len(train_mslabel_loader),
                                                                                                loss.item()))
                n_iter = e * len(train_mslabel_loader) + iteration + 1
                writer.add_scalar('scalar/train_loss_ms', loss, n_iter)

                if args.UseLabeledSpectralMixUp == 1 and e <= (args.epochs - 1):
                    # spectral mixup on labeled data

                    bsize, fsize, hsize, wsize = gt.shape
                    for mixid in range(args.theta_LabeledSpectralMixUp):
                        conversion_matrix_rand = np.random.rand(fsize, fsize)
                        conversion_matrix_rand = np.array(conversion_matrix_rand / (np.mean(np.sum(conversion_matrix_rand, axis=0))), dtype=np.float32)
                        conversion_matrix_rand = torch.from_numpy(conversion_matrix_rand)
                        conversion_matrix_rand = conversion_matrix_rand.to(device)

                        x_mixup, lms_mixup, gt_mixup = (x + conversion(x, conversion_matrix_rand)) / 2, (lms + conversion(lms, conversion_matrix_rand)) / 2, (
                                        gt + conversion(gt, conversion_matrix_rand)) / 2
                        optimizer.zero_grad()
                        img_size=x_mixup.shape[2:4]
                        y_ms_l = net(x_mixup, lms_mixup, modality="spectral",img_size=img_size)
                        loss = h_loss(y_ms_l, gt_mixup)
                        epoch_meter_mslabelmixup.add(loss.item())
                        loss.backward()
                        optimizer.step()

                        # tensorboard visualization
                        if (iteration + log_interval) % log_interval == 0:
                            print("===> {} B{} Sub{} Fea{} GPU{}\tEpoch[{}]({}/{}): ms spectral mixup Loss: {:.6f}".format(time.ctime(),
                                                                                                                           args.n_blocks,
                                                                                                                           args.n_subs,
                                                                                                                           args.n_feats,
                                                                                                                           args.gpus, e + 1,
                                                                                                                           iteration + 1,
                                                                                                                           len(train_mslabel_loader),
                                                                                                                           loss.item()))
                        n_iter = e * len(train_mslabel_loader) + iteration + 1
                        writer.add_scalar('scalar/train_loss_ms_mixup', loss, n_iter)

                # training for rgb images batch 1
                if args.UseRGB == 1 and e <= (args.epochs - 1):
                    x, lms, gt_rgb = batch_rgb
                    x, lms, gt_rgb = x.to(device), lms.to(device), gt_rgb.to(device)
                    y_rgb = torch.zeros(gt_rgb.size())
                    for i in range(0, args.theta_rgb):
                        x_i, lms_i, gt_i = x[i * args.batch_size:(i + 1) * args.batch_size, :, :, :], lms[i * args.batch_size:(i + 1) * args.batch_size, :, :, :], gt_rgb[
                                                                                                                                                                   i * args.batch_size:(
                                                                                                                                                                           i + 1) * args.batch_size,
                                                                                                                                                                   :, :, :]
                        optimizer.zero_grad()
                        img_size=x_i.shape[2:4]                   
                        y_rgb_i = net(x_i, lms_i, modality="rgb",img_size=img_size)
                        y_rgb[i * args.batch_size:(i + 1) * args.batch_size, :, :, :] = y_rgb_i
                        loss = h_loss(y_rgb_i, gt_i)
                        epoch_meter_rgb.add(loss.item())
                        loss.backward()
                        optimizer.step()

                        if (iteration + log_interval) % log_interval == 0:
                            print("===> {} B{} Sub{} Fea{} GPU{}\tEpoch[{}]({}/{}): rgb Loss: {:.6f}".format(time.ctime(),
                                                                                                         args.n_blocks,
                                                                                                         args.n_subs,
                                                                                                         args.n_feats,
                                                                                                         args.gpus,
                                                                                                         e + 1,
                                                                                                         iteration + 1,
                                                                                                         len(train_rgb_loader),
                                                                                                         loss.item()))
                        n_iter = e * len(train_rgb_loader) + iteration + 1
                        writer.add_scalar('scalar/train_loss_rgb', loss, n_iter)


                iteration += 1
        

        print("===> {}\tEpoch {} Training mslabel Complete: Avg. Loss: {:.6f}".format(time.ctime(), e + 1,
                                                                                    epoch_meter_mslabel.value()[0]))
        print("===> {}\tEpoch {} Training msunlabel Complete: Avg. Loss: {:.6f}".format(time.ctime(), e + 1,
                                                                                        epoch_meter_msunlabel.value()[0]))
        print("===> {}\tEpoch {} Training rgb Complete: Avg. Loss: {:.6f}".format(time.ctime(), e + 1,
                                                                                epoch_meter_rgb.value()[0]))

        # run validation set every epoch
        if int(getattr(args, "use_tiff", 0)) == 1:
            # Check if we should skip end-of-epoch eval if eval_iters is used to avoid redundancy, 
            # or keep it as a safeguard. Here we keep it but check eval_interval.
            if args.eval_iters == 0 and ((e + 1) % int(getattr(args, "eval_interval", 1)) == 0):
                mean_psnr, mean_sam = evaluate_tiff_pairs(
                    net,
                    eval_pairs,
                    device=device,
                    scale=int(args.scale),
                    tile=int(getattr(args, "tile", 50)),
                    overlap=int(getattr(args, "overlap", 0)),
                    save_dir=None,
                )
                print(f"[EPOCH {e+1}] TIFF eval PSNR={mean_psnr:.4f} dB | SAM={mean_sam:.6f} rad", flush=True)
                writer.add_scalar('scalar/tiff_eval_psnr', mean_psnr, e + 1)
                writer.add_scalar('scalar/tiff_eval_sam', mean_sam, e + 1)

                if int(getattr(args, "save_best", 1)) == 1 and mean_psnr > best_psnr:
                    best_psnr = mean_psnr
                    save_checkpoint(args, net, optimizer, e + 1, best_psnr=best_psnr, is_best=True)
        else:
            eval_loss_ms = validate(args, eval_ms_loader, "spectral", net, L1_loss, args.theta_rgb)
            writer.add_scalar('scalar/avg_validation_loss_ms', eval_loss_ms, iteration)


        # tensorboard visualization
        writer.add_scalar('scalar/avg_epoch_loss_mslabel', epoch_meter_mslabel.value()[0], e + 1)
        writer.add_scalar('scalar/avg_epoch_loss_msunlabel', epoch_meter_msunlabel.value()[0], e + 1)
        writer.add_scalar('scalar/avg_epoch_loss_rgb', epoch_meter_rgb.value()[0], e + 1)
        # save model weights at checkpoints every epoch
        # save checkpoint every epoch; best checkpoint will be updated after TIFF eval
        save_checkpoint(args, net, optimizer, e + 1, best_psnr=best_psnr, is_best=False)

    # save model after training
    net.eval().cpu()
    save_model_filename = model_title + "_epoch_" + str(args.epochs) + ".pth"
    save_model_path = os.path.join(args.save_dir, save_model_filename)
    # Use the model directly if it's not a DataParallel instance.
    model_state = net.module.state_dict() if isinstance(net, torch.nn.DataParallel) else net.state_dict()
    torch.save(model_state, save_model_path)
    print("\nDone, trained model saved at", save_model_path)


def conversion(m_input, m_conversion):
    b, c, h, w = m_input.shape
    m_input = m_input.permute(0, 3, 2, 1)
    m_input = torch.reshape(m_input, (b * w * h, c))
    x, y = m_conversion.shape
    if c == x:
        m_new = torch.matmul(m_input, m_conversion)
        m_new = torch.reshape(m_new, (b, w, h, y))
        m_new = m_new.permute(0, 3, 2, 1)

        return m_new
    else:
        raise
    #    print("Wrong dimensions for matrix multiplication")


def sum_dict(a, b):
    temp = dict()
    for key in a.keys() | b.keys():
        temp[key] = sum([d.get(key, 0) for d in (a, b)])
    return temp


def adjust_learning_rate(start_lr, optimizer, epoch, total_epoch_num):
    """Sets the learning rate to the initial LR decayed by 10 every 50 epochs"""
    # lr = start_lr * (0.1 ** (epoch // 30))
    lr = start_lr * (0.3 ** (epoch // 5))
    if epoch == total_epoch_num:
        lr = lr * 0.3

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate_D(start_lr, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 50 epochs"""
    # lr = start_lr * (0.1 ** (epoch // 30))
    lr = start_lr * (0.3 ** (epoch // 5))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def validate(args, loader, modality, model, criterion, theta):
    device = torch.device("cuda" if args.cuda else "cpu")
    # switch to evaluate mode
    model.eval()
    epoch_meter = meter.AverageValueMeter()
    epoch_meter.reset()
    with torch.no_grad():
        if modality == "spectral":
            for i, (ms, lms, gt) in enumerate(loader):
                ms, lms, gt = ms.to(device), lms.to(device), gt.to(device)
                img_size=ms.shape[2:4]
                y = model(ms, lms, modality,img_size=img_size)
                loss = criterion(y, gt)
                epoch_meter.add(loss.item())
        elif modality == "rgb":
            for i, (ms, lms, gt) in enumerate(loader):
                for j in range(0, theta):
                    ms_j, lms_j, gt_j = ms[j * args.batch_size:(j + 1) * args.batch_size, :, :, :], lms[j * args.batch_size:(j + 1) * args.batch_size, :, :, :], gt[
                                                                                                                                                                   j * args.batch_size:(
                                                                                                                                                                           j + 1) * args.batch_size,
                                                                                                                                                                   :, :, :]
                    ms_j, lms_j, gt_j = ms_j.to(device), lms_j.to(device), gt_j.to(device)
                    y = model(ms_j, lms_j, modality)
                    loss = criterion(y, gt_j)
                    epoch_meter.add(loss.item())
                    
        mesg = "===> {}\tEpoch {} evaluation Complete: Avg. Loss: {:.6f}".format(time.ctime(), modality, epoch_meter.value()[0])
        print(mesg)



    # back to training mode
    model.train()
    return epoch_meter.value()[0]


def test(args):
    device = torch.device("cuda" if args.cuda else "cpu")
    print('===> Loading testset')
    test_set = loadingTestData(image_dir=args.test_dir, augment=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    print('===> Start testing')
    with torch.no_grad():
        epoch_meter = meter.AverageValueMeter()
        epoch_meter.reset()
        if(args.model_title == 'VolFormer'):
            model = General_VolFormer(n_subs=args.n_subs, n_ovls=args.n_ovls, n_colors=args.n_colors, n_blocks=args.n_blocks, n_feats=args.n_feats,
                                      n_scale=args.n_scale, res_scale=0.1, use_share=True, conv=default_conv)

        ####test checkpoints
        checkpoint = torch.load(args.model_dir)
        model.load_state_dict(checkpoint["model"].state_dict())

        model.to(device).eval()
        mse_loss = torch.nn.MSELoss()
        output = []
        test_number = 0
        for i, (x, lms, gt) in enumerate(test_loader):
            x, lms, gt = x.to(device), lms.to(device), gt.to(device)
            img_size=x.shape[2:4]
            y = model(x, lms, modality="spectral",img_size=img_size)

            y, gt = y.squeeze().cpu().numpy().transpose(1, 2, 0), gt.squeeze().cpu().numpy().transpose(1, 2, 0)
            y = y[:gt.shape[0], :gt.shape[1], :]
            if i == 0:
                indices = quality_assessment(gt, y, data_range=1., ratio=4)
            else:
                indices = sum_dict(indices, quality_assessment(gt, y, data_range=1., ratio=4))
            output.append(y)
            test_number += 1
        for index in indices:
            indices[index] = indices[index] / test_number

    # save_dir = "/data/test.npy"
    # save_dir = args.result_path + args.model_dir.split('/')[-1].split('.')[0] + '.npy'
    save_dir = args.result_path + 'Ours_Real_Pavia_factor_x4.npy'
    # img = output[1][:, :, 0]
    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    np.save(save_dir, output)
    print("Test finished, test results saved to .npy file at ", save_dir)
    print(indices)

    # QIstr = args.model_title + '_' + args.model_dir.split('/')[-1].split('.')[0] + '_'+ str(time.ctime()) + ".txt"
    QIstr = args.result_path + 'Ours_Real_Pavia_factor_x4.txt'
    json.dump(indices, open(QIstr, 'w'))


def save_checkpoint(args, model, optimizer, epoch, best_psnr=None, is_best: bool = False):
    device = torch.device("cuda" if args.cuda else "cpu")
    model.eval().cpu()

    checkpoint_model_dir = './checkpoints/'
    if not os.path.exists(checkpoint_model_dir):
        os.makedirs(checkpoint_model_dir)

    # save epoch checkpoint
    ckpt_model_filename = args.dataset_name + "_" + args.model_title + "_ckpt_epoch_" + str(epoch) + ".pth"
    ckpt_model_path = os.path.join(checkpoint_model_dir, ckpt_model_filename)

    state = {
        "epoch": epoch,
        "model": model,
        "optim": optimizer.state_dict() if optimizer is not None else None,
        "best_psnr": best_psnr,
        "args": vars(args),
    }
    torch.save(state, ckpt_model_path)

    # also maintain a last.pth
    last_path = os.path.join(checkpoint_model_dir, "last.pth")
    torch.save(state, last_path)

    if is_best:
        best_path = os.path.join(checkpoint_model_dir, "best.pth")
        torch.save(state, best_path)
        print("[BEST] update best_psnr={}, saved to {}".format(best_psnr, best_path))

    model.to(device).train()
    print("Checkpoint saved to {}".format(ckpt_model_path))


if __name__ == "__main__":
    main()