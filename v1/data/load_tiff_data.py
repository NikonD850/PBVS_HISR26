import glob
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data

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


def normalize_u8_to_float(x_chw_u8: np.ndarray) -> np.ndarray:
    return x_chw_u8.astype(np.float32) / 255.0


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


@dataclass
class PatchSpec:
    lr_patch: int
    scale: int

    @property
    def hr_patch(self) -> int:
        return int(self.lr_patch * self.scale)


class loadingTiffData(data.Dataset):
    def __init__(
        self,
        image_dir: str,
        scale: int,
        lr_patch: int = 50,
        samples_per_image: int = 200,
        cache_images: bool = False,
    ):
        super().__init__()
        self.pairs = build_pairs(image_dir)
        self.patch = PatchSpec(lr_patch=lr_patch, scale=scale)
        self.samples_per_image = int(samples_per_image)
        self.cache_images = bool(cache_images)
        self._cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def __len__(self):
        return len(self.pairs) * self.samples_per_image

    def _get_pair(self, idx: int) -> Tuple[str, str, str]:
        img_idx = idx // self.samples_per_image
        return self.pairs[img_idx]

    def _load(self, lr_path: str, hr_path: str) -> Tuple[np.ndarray, np.ndarray]:
        if self.cache_images and lr_path in self._cache:
            return self._cache[lr_path]
        lr = read_tiff_chw(lr_path, dtype=np.uint8)
        hr = read_tiff_chw(hr_path, dtype=np.uint8)
        if self.cache_images:
            self._cache[lr_path] = (lr, hr)
        return lr, hr

    def __getitem__(self, idx: int):
        sid, lr_path, hr_path = self._get_pair(idx)
        lr_u8, hr_u8 = self._load(lr_path, hr_path)

        c, hlr, wlr = lr_u8.shape
        _, hhr, whr = hr_u8.shape
        s = self.patch.scale
        if hhr != hlr * s or whr != wlr * s:
            raise ValueError(f"尺度不匹配: {sid} LR({hlr},{wlr}) HR({hhr},{whr}) scale={s}")

        lp = self.patch.lr_patch
        lp_eff = min(lp, hlr, wlr)
        hp_eff = int(lp_eff * s)

        if lp_eff == hlr and lp_eff == wlr:
            y = 0
            x = 0
        else:
            y = np.random.randint(0, hlr - lp_eff + 1)
            x = np.random.randint(0, wlr - lp_eff + 1)

        lr_patch = lr_u8[:, y : y + lp_eff, x : x + lp_eff]
        hr_patch = hr_u8[:, y * s : y * s + hp_eff, x * s : x * s + hp_eff]

        lr_f = normalize_u8_to_float(lr_patch)
        hr_f = normalize_u8_to_float(hr_patch)

        ms = torch.from_numpy(lr_f)
        gt = torch.from_numpy(hr_f)

        # lms: bicubic upsampled LR patch -> HR size
        lms = F.interpolate(ms.unsqueeze(0), scale_factor=s, mode="bicubic", align_corners=False).squeeze(0)


        ms = torch.where(torch.isnan(ms), torch.full_like(ms, 0), ms)
        lms = torch.where(torch.isnan(lms), torch.full_like(lms, 0), lms)
        gt = torch.where(torch.isnan(gt), torch.full_like(gt, 0), gt)

        return ms, lms, gt
