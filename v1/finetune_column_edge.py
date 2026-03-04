#!/usr/bin/env python3
"""
深度Finetune - 列方向结构损失 + 边缘感知损失
- 更小学习率 (1e-6)
- 每500步评估
- 列方向一致性约束
- 边缘感知损失
"""
import os
import glob
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import h5py
import random
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm
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


def unfreeze_layers(model, mode='partial'):
    if mode == 'more':
        trainable_keywords = [
            'upsample', 'tail', 'conv_after_body', 'conv_last', 'conv_hr',
            'embed', 'layers', 'norm', 'patch_embed', 'pos_drop'
        ]
    elif mode == 'all':
        for name, param in model.named_parameters():
            param.requires_grad = True
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'总参数: {total_params:,}')
        print(f'可训练参数: {trainable_params:,} (100.00%)')
        return model
    else:
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


def calc_sam(sr, hr, eps=1e-8):
    sr = np.clip(sr, 0, 1)
    hr = np.clip(hr, 0, 1)
    
    dot_product = np.sum(sr * hr, axis=0)
    norm_sr = np.sqrt(np.sum(sr ** 2, axis=0) + eps)
    norm_hr = np.sqrt(np.sum(hr ** 2, axis=0) + eps)
    
    cos_angle = dot_product / (norm_sr * norm_hr + eps)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    angle = np.arccos(cos_angle)
    return np.mean(angle)


class ColumnConsistencyLoss(torch.nn.Module):
    def __init__(self, weight=0.1):
        super(ColumnConsistencyLoss, self).__init__()
        self.weight = weight
    
    def forward(self, pred, target):
        pred_diff = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        target_diff = target[:, :, :, 1:] - target[:, :, :, :-1]
        
        loss = F.l1_loss(pred_diff, target_diff)
        return loss * self.weight


class EdgeAwareLoss(torch.nn.Module):
    def __init__(self, weight=0.1):
        super(EdgeAwareLoss, self).__init__()
        self.weight = weight
        
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
    
    def forward(self, pred, target):
        pred_gray = pred.mean(dim=1, keepdim=True)
        target_gray = target.mean(dim=1, keepdim=True)
        
        pred_edge_x = F.conv2d(pred_gray, self.sobel_x, padding=1)
        pred_edge_y = F.conv2d(pred_gray, self.sobel_y, padding=1)
        pred_edge = torch.sqrt(pred_edge_x ** 2 + pred_edge_y ** 2 + 1e-8)
        
        target_edge_x = F.conv2d(target_gray, self.sobel_x, padding=1)
        target_edge_y = F.conv2d(target_gray, self.sobel_y, padding=1)
        target_edge = torch.sqrt(target_edge_x ** 2 + target_edge_y ** 2 + 1e-8)
        
        loss = F.l1_loss(pred_edge, target_edge)
        return loss * self.weight


class FastPatchDataset(IterableDataset):
    def __init__(self, file_list, patch_size=50, scale=4):
        self.file_list = file_list
        self.patch_size = patch_size
        self.scale = scale
        
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
            y = random.randint(0, h - patch_size)
            x = random.randint(0, w - patch_size)
            lr = f['LR_uint8'][y:y+patch_size, x:x+patch_size, :]
            hr_y, hr_x = y * scale, x * scale
            hr = f['HR_uint8'][hr_y:hr_y+patch_size*scale, hr_x:hr_x+patch_size*scale, :]
        
        lr_tensor = torch.from_numpy(np.transpose(lr, (2, 0, 1)).astype(np.float32) / 255.0)
        hr_tensor = torch.from_numpy(np.transpose(hr, (2, 0, 1)).astype(np.float32) / 255.0)
        
        return lr_tensor, hr_tensor


def collate_fn(batch):
    lrs, hrs = zip(*batch)
    return torch.stack(lrs), torch.stack(hrs)


def evaluate_fast(model, dataset, device, n_samples=100, scale=4, batch_size=20):
    model.eval()
    psnrs = []
    sams = []
    
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
                sams.append(calc_sam(sr, hr))
    
    model.train()
    return np.mean(psnrs), np.mean(sams)


def main():
    parser = argparse.ArgumentParser(description='深度Finetune - 列方向结构损失 + 边缘感知损失')
    parser.add_argument('--ckpt', type=str, 
                        default='/home/fdw/code/HISR/lzy/code/VolFormer/checkpoints/trae/psnr_25.06_finetune_epoch1.pth')
    parser.add_argument('--train_dir', type=str, default='/home/fdw/data/HISR/train')
    parser.add_argument('--val_dir', type=str, default='/home/fdw/data/HISR/test')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--iters_per_epoch', type=int, default=1000)
    parser.add_argument('--eval_steps', type=int, default=500)
    parser.add_argument('--patch_size', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu', type=str, default='2')
    parser.add_argument('--save_dir', type=str, default='/home/fdw/code/HISR/lzy/code/VolFormer/checkpoints/finetune/column_edge')
    parser.add_argument('--eval_samples', type=int, default=500)
    parser.add_argument('--unfreeze', type=str, default='partial', choices=['more', 'all', 'partial'])
    parser.add_argument('--column_weight', type=float, default=0.1)
    parser.add_argument('--edge_weight', type=float, default=0.1)
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
    log_print('深度Finetune - 列方向结构损失 + 边缘感知损失')
    log_print('=' * 60)
    log_print(f'预训练权重: {args.ckpt}')
    log_print(f'学习率: {args.lr}')
    log_print(f'权重衰减: {args.weight_decay}')
    log_print(f'解冻模式: {args.unfreeze}')
    log_print(f'列方向损失权重: {args.column_weight}')
    log_print(f'边缘感知损失权重: {args.edge_weight}')
    log_print(f'每{args.eval_steps}步评估一次')
    log_print(f'Epochs: {args.epochs}')
    log_print(f'Batch size: {args.batch_size}')
    log_print(f'每epoch迭代: {args.iters_per_epoch}')
    log_print(f'Device: {device}')
    log_print('')

    log_print('加载模型...')
    model = load_model(args.ckpt)
    
    log_print(f'解冻层 (模式: {args.unfreeze})...')
    model = unfreeze_layers(model, mode=args.unfreeze)
    model.to(device).train()
    
    criterion_l1 = torch.nn.L1Loss()
    criterion_column = ColumnConsistencyLoss(weight=args.column_weight).to(device)
    criterion_edge = EdgeAwareLoss(weight=args.edge_weight).to(device)
    
    train_files = sorted(glob.glob(os.path.join(args.train_dir, '*.h5')))
    val_files = sorted(glob.glob(os.path.join(args.val_dir, '*.h5')))
    all_files = train_files + val_files
    
    log_print(f'训练集文件: {len(train_files)} 个')
    log_print(f'验证集文件: {len(val_files)} 个')
    log_print(f'总文件: {len(all_files)} 个')
    
    dataset = FastPatchDataset(all_files, args.patch_size)
    eval_dataset = FastPatchDataset(all_files, args.patch_size)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                      lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    
    log_print('\n初始评估...')
    init_psnr, init_sam = evaluate_fast(model, eval_dataset, device, args.eval_samples)
    log_print(f'初始PSNR: {init_psnr:.4f} dB, SAM: {init_sam:.4f} rad ({np.degrees(init_sam):.2f} deg)')
    
    best_psnr = init_psnr
    best_sam = init_sam
    global_step = 0
    
    log_print('\n开始Finetune...')
    log_print('=' * 60)
    
    for epoch in range(args.epochs):
        model.train()
        
        total_loss = 0
        total_l1 = 0
        total_col = 0
        total_edge = 0
        n_batches = 0
        
        pbar = tqdm(dataloader, total=args.iters_per_epoch, desc=f'Epoch {epoch+1}/{args.epochs}')
        
        for lr_batch, hr_batch in pbar:
            if n_batches >= args.iters_per_epoch:
                break
            
            lr_batch = lr_batch.to(device)
            hr_batch = hr_batch.to(device)
            
            lms = torch.nn.functional.interpolate(lr_batch, scale_factor=4, mode='bicubic', align_corners=False)
            pred = model(lr_batch, lms, modality='spectral', img_size=lr_batch.shape[2:4])
            
            loss_l1 = criterion_l1(pred, hr_batch)
            loss_col = criterion_column(pred, hr_batch)
            loss_edge = criterion_edge(pred, hr_batch)
            loss = loss_l1 + loss_col + loss_edge
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_l1 += loss_l1.item()
            total_col += loss_col.item()
            total_edge += loss_edge.item()
            n_batches += 1
            global_step += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'l1': f'{loss_l1.item():.4f}',
                'col': f'{loss_col.item():.4f}',
                'edge': f'{loss_edge.item():.4f}'
            })
            
            if global_step % args.eval_steps == 0:
                val_psnr, val_sam = evaluate_fast(model, eval_dataset, device, args.eval_samples)
                log_print(f'  [Step {global_step}] PSNR={val_psnr:.4f} dB, SAM={val_sam:.4f} rad')
                
                if val_psnr > best_psnr:
                    best_psnr = val_psnr
                    best_sam = val_sam
                    save_path = os.path.join(args.save_dir, f'finetune_best.pth')
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'epoch': epoch + 1,
                        'step': global_step,
                        'best_psnr': best_psnr,
                        'best_sam': best_sam,
                        'args': {'n_subs': 8, 'n_ovls': 2, 'n_colors': 224, 'n_blocks': 3, 'n_feats': 192, 'n_scale': 4, 'use_share': True, 'vf_embed_dim': 120, 'vf_depth': 4, 'vf_stages': 4, 'vf_num_heads': 4}
                    }, save_path)
                    log_print(f'    -> 保存最佳模型 (PSNR提升: {best_psnr - init_psnr:.4f} dB)')
                
                model.train()
        
        scheduler.step()
        avg_loss = total_loss / n_batches
        avg_l1 = total_l1 / n_batches
        avg_col = total_col / n_batches
        avg_edge = total_edge / n_batches
        val_psnr, val_sam = evaluate_fast(model, eval_dataset, device, args.eval_samples)
        current_lr = scheduler.get_last_lr()[0]
        
        log_print(f'Epoch {epoch+1}/{args.epochs}: Loss={avg_loss:.6f}, L1={avg_l1:.6f}, Col={avg_col:.6f}, Edge={avg_edge:.6f}, PSNR={val_psnr:.4f} dB, SAM={val_sam:.4f} rad ({np.degrees(val_sam):.2f} deg), LR={current_lr:.2e}')
        
        save_path = os.path.join(args.save_dir, f'finetune_epoch{epoch+1}.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': epoch + 1,
            'psnr': val_psnr,
            'sam': val_sam,
            'args': {'n_subs': 8, 'n_ovls': 2, 'n_colors': 224, 'n_blocks': 3, 'n_feats': 192, 'n_scale': 4, 'use_share': True, 'vf_embed_dim': 120, 'vf_depth': 4, 'vf_stages': 4, 'vf_num_heads': 4}
        }, save_path)
        log_print(f'  -> 保存模型: {save_path}')
    
    log_print('')
    log_print('=' * 60)
    log_print('Finetune完成!')
    log_print('=' * 60)
    log_print(f'初始PSNR: {init_psnr:.4f} dB, SAM: {init_sam:.4f} rad')
    log_print(f'最佳PSNR: {best_psnr:.4f} dB, SAM: {best_sam:.4f} rad')
    log_print(f'PSNR提升: {best_psnr - init_psnr:.4f} dB')
    log_print(f'SAM变化: {best_sam - init_sam:.4f} rad')
    
    log_file.close()


if __name__ == '__main__':
    main()
