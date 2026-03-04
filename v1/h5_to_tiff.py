#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
load_h5_and_export_dual_tiff.py

用途：
批量处理指定文件夹下的 .h5 文件：
1) 同时提取 HR (High Res) 和 LR (Low Res) 数据
2) 分别导出为：
   - {filename}_HR.tif / {filename}_LR.tif
   - {filename}_HR_preview.png / {filename}_LR_preview.png
"""

import argparse
import os
import glob
import traceback
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import h5py

try:
    from osgeo import gdal
    gdal.UseExceptions()
    _HAS_GDAL = True
except ImportError:
    _HAS_GDAL = False
    gdal = None

try:
    import imageio.v2 as imageio
except Exception:
    imageio = None


# ================= 配置区 =================
# 在这里定义 H5 内部 dataset 的搜索关键字
# 脚本会优先匹配名字里包含这些字符串的 dataset
HR_KEYWORDS = ["hr", "high", "label", "gt", "groundtruth"]
LR_KEYWORDS = ["lr", "low", "data", "input"]
# =========================================


@dataclass
class H5DatasetInfo:
    name: str
    shape: Tuple[int, ...]
    dtype: str


def list_datasets(h5_path: str) -> List[H5DatasetInfo]:
    infos: List[H5DatasetInfo] = []

    def _visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            infos.append(H5DatasetInfo(name=name, shape=tuple(obj.shape), dtype=str(obj.dtype)))

    with h5py.File(h5_path, "r") as f:
        f.visititems(_visit)
    
    # 按大小排序，通常大的是主数据
    infos.sort(key=lambda x: int(np.prod(x.shape)), reverse=True)
    return infos


def _looks_like_cube(shape: Tuple[int, ...]) -> bool:
    # 过滤掉标量或非图像数据
    return len(shape) in (3, 4)


def _infer_band_axis(shape: Tuple[int, ...]) -> int:
    """推断 3D 数据体 (cube) 的波段轴 (band axis)。

    约定：你的数据维度是 HWC（例如 temp.py 打印的 (2188, 640, 224)），
    因此 band 轴默认就是最后一维（C）。

    如果未来遇到不是 HWC 的情况，可再扩展为更复杂的推断策略。
    """
    if len(shape) != 3:
        return int(np.argmin(shape))
    return 2


def normalize_cube(arr: np.ndarray) -> np.ndarray:
    """
    统一将数据转为 (B, H, W) 格式
    """
    # 1. 处理 4D (N, ...) -> 取第一个样本
    if arr.ndim == 4:
        arr = arr[0]

    if arr.ndim != 3:
        return arr # 这种可能是 mask 或者异常数据，原样返回或抛错

    # 2. 识别 Band 轴并转置
    band_axis = _infer_band_axis(arr.shape)
    
    if band_axis == 0:   # (B, H, W) -> OK
        return arr
    elif band_axis == 2: # (H, W, B) -> 转置
        return np.transpose(arr, (2, 0, 1))
    elif band_axis == 1: # (H, B, W) -> 少见，转置
        return np.transpose(arr, (1, 0, 2))
    
    return arr


def find_dataset_by_keywords(infos: List[H5DatasetInfo], keywords: List[str]) -> Optional[str]:
    """在 info 列表中查找包含关键字的 dataset name"""
    # 1. 精确/包含匹配
    for kw in keywords:
        for info in infos:
            if kw.lower() in info.name.lower() and _looks_like_cube(info.shape):
                return info.name
    return None


def _numpy_dtype_to_gdal(dtype: np.dtype) -> int:
    if not _HAS_GDAL:
        raise RuntimeError("未安装 GDAL")
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


def save_tiff_bandwise(out_tif: str, cube_bhw: np.ndarray):
    """使用 GDAL 保存 GeoTIFF（按 band 写入）。

    输入必须为 (B, H, W)。
    """
    if not _HAS_GDAL:
        raise RuntimeError(
            "未安装 GDAL，无法保存 TIFF。\n"
            "安装方法：pip install gdal 或 conda install -c conda-forge gdal"
        )

    if cube_bhw.ndim != 3:
        raise ValueError(f"save_tiff_bandwise 期望输入为 3D (B,H,W)，但得到 ndim={cube_bhw.ndim}, shape={cube_bhw.shape}")

    B, H, W = cube_bhw.shape

    gdal_dtype = _numpy_dtype_to_gdal(cube_bhw.dtype)

    bytes_size = int(cube_bhw.size) * int(cube_bhw.dtype.itemsize)

    options = [
        "COMPRESS=NONE",
        "TILED=NO",
        "INTERLEAVE=BAND",
    ]
    if bytes_size >= (4 * 1024**3 - 1024**2):
        options.append("BIGTIFF=YES")

    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(out_tif, int(W), int(H), int(B), gdal_dtype, options=options)
    if ds is None:
        raise RuntimeError(f"GDAL Create 失败: {out_tif}")

    try:
        for b in range(B):
            band = ds.GetRasterBand(b + 1)
            band.WriteArray(cube_bhw[b])
        ds.FlushCache()
    finally:
        ds = None


def save_preview_png(out_png: str, cube_bhw: np.ndarray, band_indices: List[int]):
    """保存预览图"""
    if imageio is None: return

    B, H, W = cube_bhw.shape
    
    # 简单的归一化到 0-255
    def to_u8(x):
        x = x.astype(np.float32)
        lo, hi = np.percentile(x, (2, 98)) # 稍微激进一点的拉伸
        x = (x - lo) / (hi - lo + 1e-6)
        return (np.clip(x, 0, 1) * 255).astype(np.uint8)

    # 修正索引范围
    safe_bands = [min(b, B-1) for b in band_indices]
    
    if len(safe_bands) == 0:
        img = to_u8(cube_bhw[0])
    elif len(safe_bands) == 1:
        img = to_u8(cube_bhw[safe_bands[0]])
    else:
        # 取前三个作为 RGB
        while len(safe_bands) < 3: safe_bands.append(safe_bands[-1])
        r = to_u8(cube_bhw[safe_bands[0]])
        g = to_u8(cube_bhw[safe_bands[1]])
        b = to_u8(cube_bhw[safe_bands[2]])
        img = np.stack([r, g, b], axis=-1)

    imageio.imwrite(out_png, img)


def process_one_file(h5_path, out_dir, args):
    filename = os.path.basename(h5_path)
    basename = os.path.splitext(filename)[0]
    
    print(f"正在分析: {filename} ...")
    
    # 1. 扫描所有 dataset
    infos = list_datasets(h5_path)
    
    # 2. 自动匹配 key
    hr_key = find_dataset_by_keywords(infos, HR_KEYWORDS)
    lr_key = find_dataset_by_keywords(infos, LR_KEYWORDS)

    # 如果自动匹配失败且还是同一个文件，防止互相覆盖，可以用备选逻辑
    # 如果两个 key 相同（比如文件名里既有 lr 又有 hr），则需要更细致的判断（这里暂且略过，假设名字区分度够）
    
    tasks = []
    if hr_key: tasks.append(("HR", hr_key))
    if lr_key: tasks.append(("LR", lr_key))
    
    if not tasks:
        print(f"  -> [警告] {filename} 中未找到匹配 HR 或 LR 关键字的 dataset，跳过。")
        print(f"  -> 可用 datasets: {[i.name for i in infos]}")
        return

    with h5py.File(h5_path, 'r') as f:
        for tag, key in tasks:
            print(f"  -> 提取 {tag}: {key}")
            data = f[key][...]
            cube = normalize_cube(np.asarray(data))
            
            # 生成文件名
            out_tif = os.path.join(out_dir, f"{basename}_{tag}.tif")
            out_png = os.path.join(out_dir, f"{basename}_{tag}_preview.png")
            
            # 导出
            save_tiff_bandwise(out_tif, cube)
            
            bands = [int(x) for x in args.preview_bands.split(",") if x.strip()]
            save_preview_png(out_png, cube, bands)

    print(f"  -> {filename} 处理完成。")


def main():
    parser = argparse.ArgumentParser(description="批量提取 H5 中的 HR 和 LR 数据并转为 TIFF")
    
    parser.add_argument("--in_dir", type=str, default="/home/fdw/data/HISR/final_test", 
                        help="输入 H5 文件夹路径")
    parser.add_argument("--out_dir", type=str, default="/home/fdw/code/HISR/lzy/tiff/final_test", 
                        help="输出 TIFF/PNG 文件夹路径")
    
    parser.add_argument("--preview-bands", default="10,50,100", 
                        help="预览用的 band 索引，例如 '0' (灰度) 或 '10,50,100' (RGB)")
    
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)

    h5_files = sorted(glob.glob(os.path.join(args.in_dir, "*.h5")))
    
    if not h5_files:
        print(f"未找到 H5 文件: {args.in_dir}")
        return

    print(f"找到 {len(h5_files)} 个文件。目标: 提取 HR 和 LR 双份数据。")
    print("-" * 50)

    success_cnt = 0
    
    for f in h5_files:
        try:
            process_one_file(f, args.out_dir, args)
            success_cnt += 1
        except Exception:
            print(f"!! 处理出错: {f}")
            traceback.print_exc()
        print("-" * 30)
        
    print(f"全部完成。成功处理文件数: {success_cnt}")


if __name__ == "__main__":
    main()