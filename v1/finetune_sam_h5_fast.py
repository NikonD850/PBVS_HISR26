#!/usr/bin/env python3
"""
Fast Finetune script for SAM reduction - H5 version
优化版本：固定backbone，只finetune头部，增大batch size
"""

import argparse
import os
import random
import time
import glob
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy as np
import h5py

from OverallModel import General_VolFormer
from basicModule import default_conv
from Loss import HybridLossWithSAM


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


def parse_args():
    parser = argparse.ArgumentParser(description="Fast Finetune VolFormer for SAM reduction")
    
    # Model config
    parser.add_argument("--n_feats", type=int, default=192)
    parser.add_argument("--n_blocks", type=int, default=3)
    parser.add_argument("--n_subs", type=int, default=8)
    parser.add_argument("--n_ovls", type=int, default=2)
    parser.add_argument("--n_colors", type=int, default=224)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--use_share", type=bool, default=True)
    parser.add_argument("--vf_embed_dim", type=int, default=120)
    parser.add_argument("--vf_depth", type=int, default=4)
    parser.add_argument("--vf_stages", type=int, default=4)
    parser.add_argument("--vf_num_heads", type=int, default=4)
    
    # Data paths
    parser.add_argument("--train_dir", type=str, default="/home/fdw/data/HISR/train")
    parser.add_argument("--val_dir", type=str, default="/home/fdw/data/HISR/test")
    
    # Training config - 优化参数
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4, help="增大batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="初始学习率")
    parser.add_argument("--final_learning_rate", type=float, default=1e-7, help="最终学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--lr_patch", type=int, default=50, help="减小patch size以适应更大batch")
    parser.add_argument("--samples_per_image", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=4)
    
    # Loss config - 激进SAM优化
    parser.add_argument("--sam_weight", type=float, default=5.0)
    parser.add_argument("--l1_weight", type=float, default=0.5)
    
    # Finetune策略
    parser.add_argument("--freeze_backbone", type=int, default=1, help="1: 固定VolFormer backbone")
    parser.add_argument("--freeze_trunk", type=int, default=0, help="1: 固定trunk")
    
    # Checkpoint
    parser.add_argument("--resume", type=str, 
                        default="/home/fdw/code/HISR/lzy/code/HISR/VolFormer/checkpoints/trae/psnr_24.49_sam_0.0520_VolFormer_Blocks=3_Subs8_Ovls2_Feats=192_epoch_42_iter_128100.pth")
    parser.add_argument("--save_dir", type=str, default="./checkpoints_finetune_fast")
    
    # Evaluation
    parser.add_argument("--eval_interval", type=int, default=1)
    parser.add_argument("--eval_iters", type=int, default=200)
    parser.add_argument("--tile", type=int, default=50)
    parser.add_argument("--overlap", type=int, default=0)
    
    # Hardware
    parser.add_argument("--gpus", type=str, default="2")
    parser.add_argument("--amp", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True


def load_h5_scene(h5_path: str) -> Tuple[np.ndarray, np.ndarray]:
    with h5py.File(h5_path, 'r') as f:
        lr = np.asarray(f['LR_uint8'][...])
        hr = np.asarray(f['HR_uint8'][...])
    return lr, hr


def build_h5_pairs(dir_path: str) -> List[Tuple[str, str]]:
    h5_files = sorted(glob.glob(os.path.join(dir_path, "*.h5")))
    pairs = []
    for h5_path in h5_files:
        scene_id = os.path.splitext(os.path.basename(h5_path))[0]
        pairs.append((scene_id, h5_path))
    return pairs


class H5Dataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]], scale: int, lr_patch: int, samples_per_image: int):
        self.pairs = pairs
        self.scale = scale
        self.lr_patch = lr_patch
        self.samples_per_image = samples_per_image
        
    def __len__(self):
        return len(self.pairs) * self.samples_per_image
    
    def __getitem__(self, idx):
        img_idx = idx // self.samples_per_image
        scene_id, h5_path = self.pairs[img_idx]
        
        lr_hwc, hr_hwc = load_h5_scene(h5_path)
        lr_chw = np.transpose(lr_hwc, (2, 0, 1))
        hr_chw = np.transpose(hr_hwc, (2, 0, 1))
        
        c, hlr, wlr = lr_chw.shape
        s = self.scale
        
        lp = self.lr_patch
        lp_eff = min(lp, hlr, wlr)
        hp_eff = int(lp_eff * s)
        
        if lp_eff == hlr and lp_eff == wlr:
            y, x = 0, 0
        else:
            y = np.random.randint(0, hlr - lp_eff + 1)
            x = np.random.randint(0, wlr - lp_eff + 1)
        
        lr_patch = lr_chw[:, y:y+lp_eff, x:x+lp_eff]
        hr_patch = hr_chw[:, y*s:y*s+hp_eff, x*s:x*s+hp_eff]
        
        lr_f = lr_patch.astype(np.float32) / 255.0
        hr_f = hr_patch.astype(np.float32) / 255.0
        
        ms = torch.from_numpy(lr_f)
        gt = torch.from_numpy(hr_f)
        lms = F.interpolate(ms.unsqueeze(0), scale_factor=s, mode="bicubic", align_corners=False).squeeze(0)
        
        ms = torch.where(torch.isnan(ms), torch.full_like(ms, 0), ms)
        lms = torch.where(torch.isnan(lms), torch.full_like(lms, 0), lms)
        gt = torch.where(torch.isnan(gt), torch.full_like(gt, 0), gt)
        
        return ms, lms, gt, scene_id


def freeze_backbone(model: General_VolFormer):
    """固定VolFormer backbone (branch1)"""
    print("=> Freezing VolFormer backbone (branch1)")
    for param in model.branch1.parameters():
        param.requires_grad = False
    
    # 统计可训练参数
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"=> Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    return model


def load_checkpoint(ckpt_path: str, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    print(f"=> Loading checkpoint '{ckpt_path}'")
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            model_state = ckpt["model"]
            if hasattr(model_state, "state_dict"):
                model_state = model_state.state_dict()
        elif "model_state_dict" in ckpt:
            model_state = ckpt["model_state_dict"]
        else:
            model_state = ckpt
        model.load_state_dict(model_state, strict=False)
        # 从checkpoint中读取epoch信息
        start_epoch = ckpt.get("epoch", 0)
        print(f"=> Loaded checkpoint from epoch {start_epoch}")
    else:
        model.load_state_dict(ckpt, strict=False)
        start_epoch = 0
        print(f"=> Loaded checkpoint (no epoch info)")
    
    return start_epoch


def save_checkpoint(args, model, optimizer, epoch, best_psnr=None, best_sam=None, is_best=False, is_best_sam=False, iter_num=None, psnr_val=None, sam_val=None):
    device = next(model.parameters()).device
    model.eval().cpu()
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Always save with unique name including metrics
    if iter_num is not None:
        model_title = f"VolFormer_SAM_E{epoch}_I{iter_num}_PSNR{psnr_val:.2f}_SAM{sam_val:.4f}"
    else:
        model_title = f"VolFormer_SAM_E{epoch}_PSNR{psnr_val:.2f}_SAM{sam_val:.4f}"
    ckpt_filename = f"{model_title}.pth"
    ckpt_path = os.path.join(args.save_dir, ckpt_filename)
    
    state = {
        "epoch": epoch,
        "model": model,
        "optim": optimizer.state_dict() if optimizer is not None else None,
        "best_psnr": best_psnr,
        "best_sam": best_sam,
        "args": vars(args),
    }
    torch.save(state, ckpt_path)
    
    last_path = os.path.join(args.save_dir, "last.pth")
    torch.save(state, last_path)
    
    if is_best:
        best_path = os.path.join(args.save_dir, "best_psnr.pth")
        torch.save(state, best_path)
        print(f"[BEST PSNR] Saved")
    
    if is_best_sam:
        best_sam_path = os.path.join(args.save_dir, "best_sam.pth")
        torch.save(state, best_sam_path)
        print(f"[BEST SAM] Saved")
    
    model.to(device).train()
    print(f"Checkpoint saved to {ckpt_path}")


def psnr_torch(sr, hr, eps=1e-12):
    mse = torch.mean((sr - hr) ** 2, dim=(1, 2, 3))
    return 10.0 * torch.log10(1.0 / (mse + eps))


def sam_torch(sr, hr, eps=1e-8):
    n, c, h, w = sr.shape
    sr_v = sr.permute(0, 2, 3, 1).reshape(n, h * w, c)
    hr_v = hr.permute(0, 2, 3, 1).reshape(n, h * w, c)
    dot = torch.sum(sr_v * hr_v, dim=-1)
    sr_n = torch.linalg.norm(sr_v, dim=-1)
    hr_n = torch.linalg.norm(hr_v, dim=-1)
    cos = dot / (sr_n * hr_n + eps)
    cos = torch.clamp(cos, -1.0, 1.0)
    ang = torch.acos(cos)
    return torch.mean(ang, dim=-1)


def evaluate_per_scene_h5(model, pairs, device, scale, tile, overlap):
    model.eval()
    results = {}
    
    for scene_id, h5_path in pairs:
        try:
            lr_hwc, hr_hwc = load_h5_scene(h5_path)
            lr_chw = np.transpose(lr_hwc, (2, 0, 1))
            hr_chw = np.transpose(hr_hwc, (2, 0, 1))
            
            # Simple inference without sliding window for speed
            lr = lr_chw.astype(np.float32) / 255.0
            h, w = lr.shape[1], lr.shape[2]
            
            # If image is too large, use center crop for eval
            if h > 100 or w > 100:
                # Use sliding window for large images
                c, h, w = lr_chw.shape
                lr_norm = lr_chw.astype(np.float32) / 255.0
                
                # Process full image with sliding window
                out_h, out_w = h * scale, w * scale
                out = np.zeros((c, out_h, out_w), dtype=np.float32)
                wgt = np.zeros((1, out_h, out_w), dtype=np.float32)
                
                tile_size = tile
                stride = tile_size - overlap
                
                ys = list(range(0, h, stride))
                xs = list(range(0, w, stride))
                if ys[-1] + tile_size < h:
                    ys.append(h - tile_size)
                if xs[-1] + tile_size < w:
                    xs.append(w - tile_size)
                
                for y in ys:
                    y = min(max(int(y), 0), h - tile_size)
                    for x in xs:
                        x = min(max(int(x), 0), w - tile_size)
                        patch = lr_norm[:, y:y+tile_size, x:x+tile_size]
                        inp = torch.from_numpy(patch).unsqueeze(0).to(device)
                        lms = F.interpolate(inp, scale_factor=scale, mode="bicubic", align_corners=False)
                        with torch.no_grad():
                            pred = model(inp, lms, modality="spectral", img_size=inp.shape[2:4]).clamp(0.0, 1.0).cpu().numpy()[0]
                        
                        oy0, ox0 = y * scale, x * scale
                        oy1, ox1 = oy0 + tile_size * scale, ox0 + tile_size * scale
                        out[:, oy0:oy1, ox0:ox1] += pred
                        wgt[:, oy0:oy1, ox0:ox1] += 1.0
                
                out /= np.maximum(wgt, 1e-6)
                sr_f = out
            else:
                inp = torch.from_numpy(lr).unsqueeze(0).to(device)
                lms = F.interpolate(inp, scale_factor=scale, mode="bicubic", align_corners=False)
                with torch.no_grad():
                    sr_f = model(inp, lms, modality="spectral", img_size=inp.shape[2:4]).clamp(0.0, 1.0).cpu().numpy()[0]
            
            hr_f = hr_chw.astype(np.float32) / 255.0
            sr_t = torch.from_numpy(sr_f).unsqueeze(0).to(device)
            hr_t = torch.from_numpy(hr_f).unsqueeze(0).to(device)
            
            p = psnr_torch(sr_t, hr_t).mean().item()
            s = sam_torch(sr_t, hr_t).mean().item()
            results[scene_id] = {"psnr": p, "sam": s}
        except Exception as e:
            print(f"[ERROR] {scene_id}: {e}")
            results[scene_id] = {"psnr": 0.0, "sam": 1.0}
    
    return results


def train(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    set_seed(args.seed)
    
    print("===> Building model")
    model = General_VolFormer(
        n_subs=args.n_subs, n_ovls=args.n_ovls, n_colors=args.n_colors,
        n_blocks=args.n_blocks, n_feats=args.n_feats, n_scale=args.scale,
        res_scale=0.1, use_share=args.use_share, conv=default_conv,
        vf_embed_dim=args.vf_embed_dim, vf_depth=args.vf_depth,
        vf_stages=args.vf_stages, vf_num_heads=args.vf_num_heads,
    )
    model.to(device)
    
    # 加载checkpoint
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        start_epoch = load_checkpoint(args.resume, model)
    print(f"Starting training from epoch {start_epoch + 1}")
    
    # 固定backbone
    if args.freeze_backbone:
        model = freeze_backbone(model)
    
    # 只优化可训练参数
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    
    print(f"===> Using HybridLossWithSAM: sam_weight={args.sam_weight}, l1_weight={args.l1_weight}")
    criterion = HybridLossWithSAM(
        sam_weight=args.sam_weight, l1_weight=args.l1_weight,
        spatial_tv=False, spectral_tv=False,
    )
    
    # 数据
    print("===> Loading datasets")
    train_pairs = build_h5_pairs(args.train_dir)
    val_pairs = build_h5_pairs(args.val_dir)
    
    # 合并训练集
    all_pairs = train_pairs + val_pairs
    print(f"Combined dataset: {len(train_pairs)} train + {len(val_pairs)} val = {len(all_pairs)} total")
    
    train_dataset = H5Dataset(all_pairs, args.scale, args.lr_patch, args.samples_per_image)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, 
        num_workers=args.num_workers, shuffle=True, pin_memory=True,
    )
    
    # Tensorboard
    model_title = f"VolFormer_Fast_SAM{args.sam_weight}"
    writer = SummaryWriter(os.path.join('runs', model_title + '_' + time.strftime('%Y%m%d_%H%M%S')))
    
    print("===> Start training")
    use_amp = bool(args.amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    
    best_psnr = -1e9
    best_sam = 1e9
    global_iter = 0
    
    for epoch in range(start_epoch, start_epoch + args.epochs):
        # Cosine annealing LR schedule from initial_lr to final_lr
        total_epochs = start_epoch + args.epochs
        if total_epochs > 1:
            progress = (epoch - start_epoch) / (total_epochs - 1)
            lr = args.final_learning_rate + (args.learning_rate - args.final_learning_rate) * 0.5 * (1 + np.cos(np.pi * progress))
        else:
            lr = args.learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print(f"\nEpoch {epoch+1}/{total_epochs}, LR: {lr:.6e}")
        
        model.train()
        epoch_meter = meter.AverageValueMeter()
        epoch_meter.reset()
        
        pbar = tqdm(train_loader, desc=f"Train E{epoch+1}")
        for iteration, batch in enumerate(pbar, start=1):
            x, lms, gt, _ = batch
            x, lms, gt = x.to(device), lms.to(device), gt.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            if use_amp:
                with torch.amp.autocast("cuda"):
                    y = model(x, lms, modality="spectral", img_size=x.shape[2:4])
                    loss = criterion(y, gt)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                y = model(x, lms, modality="spectral", img_size=x.shape[2:4])
                loss = criterion(y, gt)
                loss.backward()
                optimizer.step()
            
            loss_v = float(loss.item())
            epoch_meter.add(loss_v)
            global_iter += 1
            pbar.set_postfix({"loss": f"{loss_v:.6f}"})
            
            if iteration % 10 == 0:
                writer.add_scalar('train/loss', loss_v, global_iter)
            
            # Intermediate evaluation
            if args.eval_iters > 0 and global_iter % args.eval_iters == 0:
                print(f"\n[Validation] Iter {global_iter}")
                per_scene_results = evaluate_per_scene_h5(
                    model, val_pairs, device, args.scale, args.tile, args.overlap
                )
                mean_psnr = np.mean([m["psnr"] for m in per_scene_results.values()])
                mean_sam = np.mean([m["sam"] for m in per_scene_results.values()])
                
                print(f"[Iter {global_iter}] PSNR={mean_psnr:.4f} | SAM={mean_sam:.6f}")
                writer.add_scalar('eval/psnr', mean_psnr, global_iter)
                writer.add_scalar('eval/sam', mean_sam, global_iter)
                
                is_best = mean_psnr > best_psnr
                is_best_sam = mean_sam < best_sam
                if is_best:
                    best_psnr = mean_psnr
                if is_best_sam:
                    best_sam = mean_sam
                
                save_checkpoint(args, model, optimizer, epoch+1, best_psnr, best_sam, is_best, is_best_sam, 
                               iter_num=global_iter, psnr_val=mean_psnr, sam_val=mean_sam)
                model.train()
        
        avg_loss = epoch_meter.value()[0]
        print(f"Epoch {epoch+1} completed. Avg Loss: {avg_loss:.6f}")
        writer.add_scalar('train/epoch_loss', avg_loss, epoch+1)
        
        # End of epoch eval
        if (epoch + 1) % args.eval_interval == 0:
            print(f"\n[Validation] Epoch {epoch+1}")
            per_scene_results = evaluate_per_scene_h5(
                model, val_pairs, device, args.scale, args.tile, args.overlap
            )
            
            print(f"{'Scene':<15} {'PSNR':>10} {'SAM':>10}")
            print("-" * 40)
            for sid, metrics in sorted(per_scene_results.items(), key=lambda x: x[1]["sam"]):
                print(f"{sid:<15} {metrics['psnr']:>10.4f} {metrics['sam']:>10.6f}")
            
            mean_psnr = np.mean([m["psnr"] for m in per_scene_results.values()])
            mean_sam = np.mean([m["sam"] for m in per_scene_results.values()])
            print(f"{'MEAN':<15} {mean_psnr:>10.4f} {mean_sam:>10.6f}")
            
            writer.add_scalar('eval_epoch/psnr', mean_psnr, epoch+1)
            writer.add_scalar('eval_epoch/sam', mean_sam, epoch+1)
            
            is_best = mean_psnr > best_psnr
            is_best_sam = mean_sam < best_sam
            if is_best:
                best_psnr = mean_psnr
            if is_best_sam:
                best_sam = mean_sam
            
            save_checkpoint(args, model, optimizer, epoch+1, best_psnr, best_sam, is_best, is_best_sam,
                           iter_num=None, psnr_val=mean_psnr, sam_val=mean_sam)
            print(f"Best PSNR: {best_psnr:.4f} | Best SAM: {best_sam:.6f}")
    
    print("\n===> Training completed")
    os.makedirs(args.save_dir, exist_ok=True)
    final_path = os.path.join(args.save_dir, f"{model_title}_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved to {final_path}")
    writer.close()


if __name__ == "__main__":
    args = parse_args()
    train(args)
