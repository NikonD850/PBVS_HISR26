import glob
import os
import re
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from osgeo import gdal

gdal.UseExceptions()


def read_tiff_chw(path: str, dtype=np.uint8) -> np.ndarray:
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"无法打开TIFF: {path}")
    c = ds.RasterCount
    h = ds.RasterYSize
    w = ds.RasterXSize

    arr = np.empty((c, h, w), dtype=dtype)
    for i in range(c):
        band = ds.GetRasterBand(i + 1)
        a = band.ReadAsArray()
        if a is None:
            raise RuntimeError(f"读取band失败: {path}, band={i+1}")
        if a.dtype != dtype:
            a = a.astype(dtype, copy=False)
        arr[i] = a
    return arr


def _numpy_dtype_to_gdal(dtype: np.dtype) -> int:
    dt = np.dtype(dtype)
    if dt == np.uint8:
        return gdal.GDT_Byte
    if dt == np.uint16:
        return gdal.GDT_UInt16
    if dt == np.int16:
        return gdal.GDT_Int16
    if dt == np.uint32:
        return gdal.GDT_UInt32
    if dt == np.int32:
        return gdal.GDT_Int32
    if dt == np.float32:
        return gdal.GDT_Float32
    if dt == np.float64:
        return gdal.GDT_Float64
    return gdal.GDT_Float32


def write_tiff_chw(path: str, cube_chw: np.ndarray):
    if cube_chw.ndim != 3:
        raise ValueError(f"write_tiff_chw expects (C,H,W), got {cube_chw.shape}")

    c, h, w = cube_chw.shape
    gdal_dtype = _numpy_dtype_to_gdal(cube_chw.dtype)

    bytes_size = int(cube_chw.size) * int(cube_chw.dtype.itemsize)
    options = ["COMPRESS=NONE", "TILED=NO", "INTERLEAVE=BAND"]
    if bytes_size >= (4 * 1024**3 - 1024**2):
        options.append("BIGTIFF=YES")

    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(path, int(w), int(h), int(c), gdal_dtype, options=options)
    if ds is None:
        raise RuntimeError(f"GDAL Create失败: {path}")

    try:
        for i in range(c):
            band = ds.GetRasterBand(i + 1)
            band.WriteArray(cube_chw[i])
        ds.FlushCache()
    finally:
        ds = None


def normalize_u8_to_float(x_chw_u8: np.ndarray) -> np.ndarray:
    return x_chw_u8.astype(np.float32) / 255.0


def float_to_u8(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0 + 0.5).astype(np.uint8)


def _parse_scene_id(path: str) -> Optional[str]:
    base = os.path.basename(path)
    m = re.match(r"^(.*)_LR\.tif$", base)
    if not m:
        return None
    return m.group(1)


def build_pairs(dir_path: str) -> List[Tuple[str, str, str]]:
    lr_files = sorted(glob.glob(os.path.join(dir_path, "*_LR.tif")))
    pairs = []
    for lr in lr_files:
        sid = _parse_scene_id(lr)
        if sid is None:
            continue
        hr = os.path.join(dir_path, f"{sid}_HR.tif")
        if not os.path.exists(hr):
            raise FileNotFoundError(f"找不到HR配对文件: {hr} (for {lr})")
        pairs.append((sid, lr, hr))
    return pairs


def psnr(sr: torch.Tensor, hr: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    mse = torch.mean((sr - hr) ** 2, dim=(1, 2, 3))
    return 10.0 * torch.log10(1.0 / (mse + eps))


def sam(sr: torch.Tensor, hr: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
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


def predict_sliding_window(
    model: nn.Module,
    lr_chw_u8: np.ndarray,
    scale: int,
    tile: int,
    overlap: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()

    c, h, w = lr_chw_u8.shape

    if tile is None or int(tile) <= 0:
        tile = 50
    tile = int(tile)

    if overlap is None:
        overlap = 0
    overlap = int(overlap)

    if overlap < 0:
        raise ValueError("overlap不能为负数")

    if h <= tile and w <= tile:
        lr = normalize_u8_to_float(lr_chw_u8)
        inp = torch.from_numpy(lr).unsqueeze(0).to(device)
        pred = model(inp, F.interpolate(inp, scale_factor=scale, mode="bicubic", align_corners=False), modality="spectral", img_size=inp.shape[2:4]).clamp(0.0, 1.0).cpu().numpy()[0]
        return pred.astype(np.float32, copy=False)

    lr = normalize_u8_to_float(lr_chw_u8)

    out_h = h * scale
    out_w = w * scale
    out = np.zeros((c, out_h, out_w), dtype=np.float32)
    wgt = np.zeros((1, out_h, out_w), dtype=np.float32)

    stride = tile - overlap
    if stride <= 0:
        raise ValueError("tile必须大于overlap")

    ys = list(range(0, h, stride))
    xs = list(range(0, w, stride))
    if len(ys) == 0:
        ys = [0]
    if len(xs) == 0:
        xs = [0]

    if ys[-1] + tile < h:
        ys.append(h - tile)
    if xs[-1] + tile < w:
        xs.append(w - tile)

    for y in ys:
        y = min(max(int(y), 0), h - tile)
        for x in xs:
            x = min(max(int(x), 0), w - tile)
            patch = lr[:, y : y + tile, x : x + tile]
            inp = torch.from_numpy(patch).unsqueeze(0).to(device)
            lms = F.interpolate(inp, scale_factor=scale, mode="bicubic", align_corners=False)
            with torch.no_grad():
                pred = model(inp, lms, modality="spectral", img_size=inp.shape[2:4]).clamp(0.0, 1.0).cpu().numpy()[0]
            del inp, lms
            if device.type == "cuda":
                torch.cuda.empty_cache()

            oy0 = y * scale
            ox0 = x * scale
            oy1 = oy0 + tile * scale
            ox1 = ox0 + tile * scale

            out[:, oy0:oy1, ox0:ox1] += pred
            wgt[:, oy0:oy1, ox0:ox1] += 1.0

    out /= np.maximum(wgt, 1e-6)
    return out


def evaluate_tiff_pairs(
    model: nn.Module,
    pairs,
    device: torch.device,
    scale: int,
    tile: int,
    overlap: int,
    save_dir: Optional[str] = None,
    strict: bool = False,
):
    model.eval()
    psnr_all = []
    sam_all = []

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    if pairs is None or len(pairs) == 0:
        msg = "[TIFF EVAL] pairs为空（没有找到任何 *_LR.tif/*_HR.tif 配对）"
        if strict:
            raise RuntimeError(msg)
        print(msg, flush=True)
        return float("nan"), float("nan")

    skipped = 0
    total = 0

    for sid, lr_path, hr_path in pairs:
        total += 1
        lr_u8 = read_tiff_chw(lr_path, dtype=np.uint8)
        hr_u8 = read_tiff_chw(hr_path, dtype=np.uint8)

        if lr_u8.ndim != 3 or hr_u8.ndim != 3:
            skipped += 1
            msg = f"[TIFF EVAL][{sid}] 维度错误 lr={lr_u8.shape} hr={hr_u8.shape}"
            if strict:
                raise RuntimeError(msg)
            print(msg, flush=True)
            continue

        c1, hlr, wlr = lr_u8.shape
        c2, hhr, whr = hr_u8.shape
        if c1 != c2:
            skipped += 1
            msg = f"[TIFF EVAL][{sid}] 通道数不一致 lrC={c1} hrC={c2} lr={lr_path} hr={hr_path}"
            if strict:
                raise RuntimeError(msg)
            print(msg, flush=True)
            continue

        if hhr != hlr * int(scale) or whr != wlr * int(scale):
            skipped += 1
            msg = (
                f"[TIFF EVAL][{sid}] 尺度不匹配: LR({hlr},{wlr}) HR({hhr},{whr}) scale={scale} "
                f"lr={lr_path} hr={hr_path}"
            )
            if strict:
                raise RuntimeError(msg)
            print(msg, flush=True)
            continue

        sr_f = predict_sliding_window(model, lr_u8, scale=scale, tile=tile, overlap=overlap, device=device)
        hr_f = normalize_u8_to_float(hr_u8)

        if (not np.isfinite(sr_f).all()) or (not np.isfinite(hr_f).all()):
            skipped += 1
            msg = f"[TIFF EVAL][{sid}] sr/hr 出现 NaN/Inf (sr_f finite={np.isfinite(sr_f).all()}, hr_f finite={np.isfinite(hr_f).all()})"
            if strict:
                raise RuntimeError(msg)
            print(msg, flush=True)
            continue

        if sr_f.shape != hr_f.shape:
            skipped += 1
            msg = f"[TIFF EVAL][{sid}] SR/HR shape不一致 sr={sr_f.shape} hr={hr_f.shape}"
            if strict:
                raise RuntimeError(msg)
            print(msg, flush=True)
            continue

        sr_t = torch.from_numpy(sr_f).unsqueeze(0).to(device)
        hr_t = torch.from_numpy(hr_f).unsqueeze(0).to(device)

        p = psnr(sr_t, hr_t).mean().item()
        a = sam(sr_t, hr_t).mean().item()

        if not (np.isfinite(p) and np.isfinite(a)):
            skipped += 1
            msg = f"[TIFF EVAL][{sid}] 指标为 NaN/Inf: psnr={p} sam={a}"
            if strict:
                raise RuntimeError(msg)
            print(msg, flush=True)
            continue

        psnr_all.append(p)
        sam_all.append(a)

        if save_dir is not None:
            out_u8 = float_to_u8(sr_f)
            out_path = os.path.join(save_dir, f"{sid}_SR.tif")
            write_tiff_chw(out_path, out_u8)

    if len(psnr_all) == 0:
        msg = f"[TIFF EVAL] 没有任何有效样本用于计算均值：total={total}, skipped={skipped}"
        if strict:
            raise RuntimeError(msg)
        print(msg, flush=True)
        return float("nan"), float("nan")

    if skipped > 0:
        print(f"[TIFF EVAL] total={total}, used={len(psnr_all)}, skipped={skipped}", flush=True)

    return float(np.mean(psnr_all)), float(np.mean(sam_all))
