#!/usr/bin/env python3
"""
优化版动态随机采样Finetune - Tensor数据增强
- 数据增强在Tensor上操作（更快）
- DataLoader多进程
- HDF5切片读取
"""
import os
import glob
import argparse
import numpy as np
import torch
import h5py
import random
from torch.optim import Adam
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm
import torchvision.transforms.functional as TF
from OverallModel import General_VolFormer
from basicModule import default_conv


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    elif 'model' in ckpt:
        model = ckpt['model']
        state_dict = model.state_dict() if hasattr(model, 'state_dict') else model
    else:
        state_dict = ckpt
    
    net = General_VolFormer(
        n_subs=8, n_ovls=2, n_colors=224, n_blocks=3, n_feats=192,
        n_scale=4, res_scale=0.1, use_share=True, conv=default_conv,
        vf_embed_dim=120, vf_depth=4, vf_stages=4, vf_num_heads=4,
    )
    net.load_state_dict(state_dict, strict=True)
    return net


def freeze_layers(model):
    trainable_keywords = ['upsample', 'tail', 'conv_after_body', 'conv_last', 'conv_hr']
    
    for name, param in model.named_parameters():
        param.requires_grad = False
    
    for name, param in model.named_parameters():
        for kw in trainable_keywords:
            if kw in name:
                param.requires_grad = True
                break
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f'总参数: {total_params:,}')
    print(f'可训练参数: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)')
    
    return model


def calc_psnr(sr, hr):
    mse = np.mean((sr - hr) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(1.0 / mse)


class TensorAugment:
    """Tensor数据增强（GPU上操作）"""
    
    @staticmethod
    def augment(lr_tensor, hr_tensor):
        """
        lr_tensor: (C, H, W) tensor
        hr_tensor: (C, H, W) tensor
        """
        # 1. 水平翻转 (50%)
        if random.random() < 0.5:
            lr_tensor = torch.flip(lr_tensor, dims=[2])
            hr_tensor = torch.flip(hr_tensor, dims=[2])
        
        # 2. 垂直翻转 (50%)
        if random.random() < 0.5:
            lr_tensor = torch.flip(lr_tensor, dims=[1])
            hr_tensor = torch.flip(hr_tensor, dims=[1])
        
        # 3. 90度倍数旋转 (25%)
        if random.random() < 0.25:
            k = random.randint(1, 3)
            lr_tensor = torch.rot90(lr_tensor, k=k, dims=[1, 2])
            hr_tensor = torch.rot90(hr_tensor, k=k, dims=[1, 2])
        
        # 4. Random Resize Crop (30%)
        if random.random() < 0.3:
            scale = random.uniform(0.7, 1.3)
            lr_tensor, hr_tensor = TensorAugment.random_resize_crop(lr_tensor, hr_tensor, scale)
        
        return lr_tensor, hr_tensor
    
    @staticmethod
    def random_resize_crop(lr, hr, scale):
        """Random Resize Crop"""
        c, h, w = lr.shape
        scale_factor = hr.shape[1] // h
        
        new_h = max(int(h * scale), h // 2)
        new_w = max(int(w * scale), w // 2)
        
        # 缩放
        lr_scaled = TF.resize(lr, [new_h, new_w], interpolation=TF.InterpolationMode.BILINEAR)
        hr_scaled = TF.resize(hr, [new_h * scale_factor, new_w * scale_factor], interpolation=TF.InterpolationMode.BILINEAR)
        
        # 裁剪或填充
        if new_h >= h and new_w >= w:
            i = random.randint(0, new_h - h)
            j = random.randint(0, new_w - w)
            lr_cropped = lr_scaled[:, i:i+h, j:j+w]
            hr_cropped = hr_scaled[:, i*scale_factor:i*scale_factor+h*scale_factor, j*scale_factor:j*scale_factor+w*scale_factor]
        else:
            pad_h = h - new_h
            pad_w = w - new_w
            lr_cropped = TF.pad(lr_scaled, [pad_w//2, pad_h//2, pad_w-pad_w//2, pad_h-pad_h//2], padding_mode='reflect')
            hr_cropped = TF.pad(hr_scaled, [(pad_w*scale_factor)//2, (pad_h*scale_factor)//2, 
                                              (pad_w*scale_factor)-(pad_w*scale_factor)//2, 
                                              (pad_h*scale_factor)-(pad_h*scale_factor)//2], padding_mode='reflect')
        
        return lr_cropped, hr_cropped


class FastPatchDataset(IterableDataset):
    """快速Patch数据集"""
    
    def __init__(self, file_list, patch_size=50, scale=4, rotate_crop=False):
        self.file_list = file_list
        self.patch_size = patch_size
        self.scale = scale
        self.rotate_crop = rotate_crop
        
        self.file_info = []
        for fpath in file_list:
            with h5py.File(fpath, 'r') as f:
                lr_ds = f['LR_uint8']
                h, w, c = lr_ds.shape
            self.file_info.append({
                'path': fpath,
                'lr_shape': (h, w, c),
            })
    
    def __iter__(self):
        while True:
            yield self._sample_one()
    
    def _sample_one(self):
        file_idx = random.randint(0, len(self.file_info) - 1)
        info = self.file_info[file_idx]
        patch_size = self.patch_size
        scale = self.scale
        h, w, _ = info['lr_shape']
        
        with h5py.File(info['path'], 'r') as f:
            if self.rotate_crop and random.random() < 0.3:
                # 任意角度裁剪
                angle = random.uniform(-45, 45)
                center_x = random.randint(patch_size, w - patch_size)
                center_y = random.randint(patch_size, h - patch_size)
                
                lr = self._rotate_crop(f['LR_uint8'], center_x, center_y, patch_size, angle)
                hr = self._rotate_crop(f['HR_uint8'], center_x * scale, center_y * scale, 
                                       patch_size * scale, angle)
            else:
                # 标准水平竖直裁剪
                y = random.randint(0, h - patch_size)
                x = random.randint(0, w - patch_size)
                lr = f['LR_uint8'][y:y+patch_size, x:x+patch_size, :]
                hr_y, hr_x = y * scale, x * scale
                hr = f['HR_uint8'][hr_y:hr_y+patch_size*scale, hr_x:hr_x+patch_size*scale, :]
        
        lr_tensor = torch.from_numpy(np.transpose(lr, (2, 0, 1)).astype(np.float32) / 255.0)
        hr_tensor = torch.from_numpy(np.transpose(hr, (2, 0, 1)).astype(np.float32) / 255.0)
        
        return lr_tensor, hr_tensor
    
    def _rotate_crop(self, dataset, cx, cy, size, angle):
        """从数据集中以任意角度裁剪矩形区域"""
        import cv2
        h, w, c = dataset.shape
        
        # 创建旋转矩阵
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        
        # 计算旋转后的边界
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        
        # 调整旋转矩阵以避免裁剪
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2
        
        # 对每个通道进行旋转
        result = np.zeros((size, size, c), dtype=np.uint8)
        for i in range(c):
            rotated = cv2.warpAffine(dataset[:, :, i], M, (new_w, new_h), 
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            # 从旋转后的图像中心裁剪
            start_y = max(0, new_h // 2 - size // 2)
            start_x = max(0, new_w // 2 - size // 2)
            end_y = min(new_h, start_y + size)
            end_x = min(new_w, start_x + size)
            
            crop = rotated[start_y:end_y, start_x:end_x]
            
            # 如果裁剪区域不够大，填充
            if crop.shape[0] < size or crop.shape[1] < size:
                crop = cv2.copyMakeBorder(crop, 
                    max(0, size - crop.shape[0]) // 2, 
                    max(0, size - crop.shape[0]) - max(0, size - crop.shape[0]) // 2,
                    max(0, size - crop.shape[1]) // 2, 
                    max(0, size - crop.shape[1]) - max(0, size - crop.shape[1]) // 2,
                    cv2.BORDER_REFLECT)
            
            result[:, :, i] = crop[:size, :size]
        
        return result


def collate_fn(batch):
    lrs, hrs = zip(*batch)
    return torch.stack(lrs), torch.stack(hrs)


def evaluate_fast(model, dataset, device, n_samples=100, scale=4, batch_size=40):
    """批量评估（更快）"""
    model.eval()
    psnrs = []
    
    lrs, hrs = [], []
    for _ in range(n_samples):
        lr, hr = dataset._sample_one()
        lrs.append(lr)
        hrs.append(hr)
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch_lr = torch.stack(lrs[i:i+batch_size]).to(device)
            batch_hr = torch.stack(hrs[i:i+batch_size])
            
            lms = torch.nn.functional.interpolate(batch_lr, scale_factor=scale, mode='bicubic', align_corners=False)
            pred = model(batch_lr, lms, modality='spectral', img_size=batch_lr.shape[2:4]).clamp(0.0, 1.0)
            
            for j in range(pred.shape[0]):
                sr = pred[j].cpu().numpy()
                hr = batch_hr[j].numpy()
                psnrs.append(calc_psnr(sr, hr))
    
    model.train()
    return np.mean(psnrs)


def main():
    parser = argparse.ArgumentParser(description='Tensor数据增强Finetune')
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--train_dir', type=str, default='/home/fdw/data/HISR/train')
    parser.add_argument('--val_dir', type=str, default='/home/fdw/data/HISR/test')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--iters_per_epoch', type=int, default=1000)
    parser.add_argument('--patch_size', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--save_dir', type=str, default='checkpoints/finetune')
    parser.add_argument('--eval_samples', type=int, default=500)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    log_file = open(os.path.join(args.save_dir, 'train_log.txt'), 'w')
    
    def log_print(msg):
        print(msg)
        log_file.write(msg + '\n')
        log_file.flush()
    
    log_print('=' * 60)
    log_print('Tensor数据增强Finetune')
    log_print('=' * 60)
    log_print(f'预训练权重: {args.ckpt}')
    log_print(f'学习率: {args.lr}')
    log_print(f'Epochs: {args.epochs}')
    log_print(f'Batch size: {args.batch_size}')
    log_print(f'每epoch迭代: {args.iters_per_epoch}')
    log_print(f'Num workers: {args.num_workers}')
    log_print(f'Device: {device}')
    log_print('')

    log_print('加载模型...')
    model = load_model(args.ckpt)
    
    log_print('冻结前端层...')
    model = freeze_layers(model)
    model.to(device).train()
    
    train_files = sorted(glob.glob(os.path.join(args.train_dir, '*.h5')))
    val_files = sorted(glob.glob(os.path.join(args.val_dir, '*.h5')))
    all_files = train_files + val_files
    
    log_print(f'训练集文件: {len(train_files)} 个')
    log_print(f'验证集文件: {len(val_files)} 个')
    log_print(f'总文件: {len(all_files)} 个')
    
    dataset = FastPatchDataset(all_files, args.patch_size, rotate_crop=True)
    eval_dataset = FastPatchDataset(all_files, args.patch_size, rotate_crop=False)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    criterion = torch.nn.MSELoss()
    
    log_print('\n初始评估...')
    init_psnr = evaluate_fast(model, eval_dataset, device, args.eval_samples)
    log_print(f'初始PSNR: {init_psnr:.4f} dB')
    
    best_psnr = init_psnr
    
    log_print('\n开始Finetune...')
    log_print('=' * 60)
    
    for epoch in range(args.epochs):
        model.train()
        
        total_loss = 0
        n_batches = 0
        
        pbar = tqdm(dataloader, total=args.iters_per_epoch, desc=f'Epoch {epoch+1}/{args.epochs}')
        
        for lr_batch, hr_batch in pbar:
            if n_batches >= args.iters_per_epoch:
                break
            
            # Tensor数据增强（在GPU上）
            lr_batch = lr_batch.to(device)
            hr_batch = hr_batch.to(device)
            
            aug_lr = []
            aug_hr = []
            for i in range(lr_batch.shape[0]):
                lr, hr = TensorAugment.augment(lr_batch[i], hr_batch[i])
                
                # Mixup (20%概率)
                if random.random() < 0.2:
                    j = random.randint(0, lr_batch.shape[0] - 1)
                    lr2, hr2 = TensorAugment.augment(lr_batch[j], hr_batch[j])
                    lam = max(0.1, min(0.9, np.random.beta(0.2, 0.2)))
                    lr = lam * lr + (1 - lam) * lr2
                    hr = lam * hr + (1 - lam) * hr2
                
                aug_lr.append(lr)
                aug_hr.append(hr)
            
            inp = torch.stack(aug_lr)
            gt = torch.stack(aug_hr)
            
            lms = torch.nn.functional.interpolate(inp, scale_factor=4, mode='bicubic', align_corners=False)
            pred = model(inp, lms, modality='spectral', img_size=inp.shape[2:4])
            
            loss = criterion(pred, gt)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        avg_loss = total_loss / n_batches
        val_psnr = evaluate_fast(model, eval_dataset, device, args.eval_samples)
        
        log_print(f'Epoch {epoch+1}/{args.epochs}: Loss={avg_loss:.6f}, PSNR={val_psnr:.4f} dB')
        
        save_path = os.path.join(args.save_dir, f'finetune_epoch{epoch+1}.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': epoch + 1,
            'psnr': val_psnr,
            'args': {'n_subs': 8, 'n_ovls': 2, 'n_colors': 224, 'n_blocks': 3, 'n_feats': 192, 'n_scale': 4, 'use_share': True, 'vf_embed_dim': 120, 'vf_depth': 4, 'vf_stages': 4, 'vf_num_heads': 4}
        }, save_path)
        log_print(f'  -> 保存模型: {save_path}')
        
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            save_path = os.path.join(args.save_dir, f'finetune_best.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_psnr': best_psnr,
                'args': {'n_subs': 8, 'n_ovls': 2, 'n_colors': 224, 'n_blocks': 3, 'n_feats': 192, 'n_scale': 4, 'use_share': True, 'vf_embed_dim': 120, 'vf_depth': 4, 'vf_stages': 4, 'vf_num_heads': 4}
            }, save_path)
            log_print(f'  -> 保存最佳模型: {save_path}')
    
    log_print('')
    log_print('=' * 60)
    log_print('Finetune完成!')
    log_print('=' * 60)
    log_print(f'初始PSNR: {init_psnr:.4f} dB')
    log_print(f'最佳PSNR: {best_psnr:.4f} dB')
    log_print(f'提升: {best_psnr - init_psnr:.4f} dB')
    
    log_file.close()


if __name__ == '__main__':
    main()
