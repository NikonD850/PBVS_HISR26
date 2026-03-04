import numpy as np
import torch
from tqdm import tqdm

from data import crop_to_ref, pad_pair_to_window


def compare_mpsnr(x_true, x_pred, data_range=1.0):
    gt = np.asarray(x_true, dtype=np.float64)
    pred = np.asarray(x_pred, dtype=np.float64)
    if gt.shape != pred.shape:
        raise ValueError(f"shape 不一致: gt={gt.shape}, pred={pred.shape}")
    if gt.ndim != 3:
        raise ValueError(f"输入必须是 (H, W, C)，当前是 {gt.shape}")

    mse = np.mean((gt - pred) ** 2, axis=(0, 1))
    mse = np.maximum(mse, 1e-12)
    psnr = 10.0 * np.log10((float(data_range) ** 2) / mse)
    return float(np.mean(psnr))


def compare_sam(x_true, x_pred):
    gt = np.asarray(x_true, dtype=np.float64)
    pred = np.asarray(x_pred, dtype=np.float64)
    if gt.shape != pred.shape:
        raise ValueError(f"shape 不一致: gt={gt.shape}, pred={pred.shape}")
    if gt.ndim != 3:
        raise ValueError(f"输入必须是 (H, W, C)，当前是 {gt.shape}")

    gt_vec = gt.reshape(-1, gt.shape[-1])
    pred_vec = pred.reshape(-1, pred.shape[-1])

    numerator = np.sum(gt_vec * pred_vec, axis=1)
    gt_norm = np.linalg.norm(gt_vec, axis=1)
    pred_norm = np.linalg.norm(pred_vec, axis=1)

    valid = (gt_norm > 0) & (pred_norm > 0)
    if not np.any(valid):
        return 0.0

    cosine = numerator[valid] / (gt_norm[valid] * pred_norm[valid])
    cosine = np.clip(cosine, -1.0, 1.0)
    sam_deg = np.mean(np.arccos(cosine)) * 180.0 / np.pi
    return float(sam_deg)


def compute_psnr_sam(model, val_loader, device, n_scale, use_amp, show_progress=True, max_batches=None):
    was_training = model.training
    model.eval()
    psnr_sum = 0.0
    sam_sum = 0.0
    sample_count = 0

    iterator = tqdm(val_loader, desc="Val", ncols=100, leave=False) if show_progress else val_loader

    with torch.no_grad():
        for batch_idx, (ms, lms, gt) in enumerate(iterator):
            if max_batches is not None and batch_idx >= int(max_batches):
                break
            ms = ms.to(device, non_blocking=True)
            lms = lms.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)

            ms_pad, lms_pad = pad_pair_to_window(ms, lms, n_scale=n_scale, window_size=8)
            img_size = ms_pad.shape[2:4]

            with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
                pred = model(ms_pad, lms_pad, modality="spectral", img_size=img_size)
            pred = crop_to_ref(pred, gt)

            pred_np = pred.detach().float().cpu().numpy().transpose(0, 2, 3, 1)
            gt_np = gt.detach().float().cpu().numpy().transpose(0, 2, 3, 1)

            for index in range(pred_np.shape[0]):
                psnr_sum += compare_mpsnr(gt_np[index], pred_np[index], data_range=1.0)
                sam_sum += compare_sam(gt_np[index], pred_np[index])
                sample_count += 1

    if was_training:
        model.train()

    if sample_count == 0:
        raise RuntimeError("验证集为空，无法计算 PSNR/SAM。")

    return psnr_sum / sample_count, sam_sum / sample_count, sample_count
