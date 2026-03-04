#!/usr/bin/env python3
import argparse
import glob
import os
import shutil
import time
import zipfile

import numpy as np
import torch
import h5py

try:
    import imageio.v2 as imageio
except Exception:
    imageio = None

try:
    from PIL import Image
except Exception:
    Image = None

from OverallModel import General_VolFormer
from basicModule import default_conv


def float_to_u8(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0 + 0.5).astype(np.uint8)


def _save_rgb_png(out_png: str, cube_chw_u8: np.ndarray):
    if cube_chw_u8.ndim != 3:
        return
    c, h, w = cube_chw_u8.shape
    if c < 3:
        return
    
    def to_u8_band(band_data):
        band_data = band_data.astype(np.float32)
        lo, hi = np.percentile(band_data, (2, 98))
        band_data = (band_data - lo) / (hi - lo + 1e-6)
        return (np.clip(band_data, 0, 1) * 255).astype(np.uint8)
    
    r = to_u8_band(cube_chw_u8[0])
    g = to_u8_band(cube_chw_u8[1])
    b = to_u8_band(cube_chw_u8[2])
    rgb_img = np.stack([r, g, b], axis=-1)
    
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    if Image is not None:
        Image.fromarray(rgb_img, mode='RGB').save(out_png)
    elif imageio is not None:
        imageio.imwrite(out_png, rgb_img)


def _list_h5_datasets(h5_path: str):
    infos = []

    def _visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            infos.append((name, tuple(obj.shape), str(obj.dtype), obj.compression))

    with h5py.File(h5_path, "r") as f:
        f.visititems(_visit)

    infos.sort(key=lambda x: int(np.prod(x[1])), reverse=True)
    return infos


def _looks_like_cube(shape):
    return len(shape) in (3, 4)


def _infer_band_axis(shape):
    if len(shape) != 3:
        return int(np.argmin(shape))
    return 2


def _normalize_cube_to_chw(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 4:
        arr = arr[0]

    if arr.ndim != 3:
        raise ValueError(f"h5 cube expects 3D/4D array, got shape={arr.shape}")

    band_axis = _infer_band_axis(arr.shape)
    if band_axis == 0:
        cube = arr
    elif band_axis == 2:
        cube = np.transpose(arr, (2, 0, 1))
    elif band_axis == 1:
        cube = np.transpose(arr, (1, 0, 2))
    else:
        cube = arr

    return np.asarray(cube)


def _find_lr_dataset(infos):
    for name, shape, _dtype, _comp in infos:
        if name == "LR" and _looks_like_cube(shape):
            return name

    keywords = ["lr", "low", "data", "input"]
    for kw in keywords:
        for name, shape, _dtype, _comp in infos:
            if kw in name.lower() and _looks_like_cube(shape):
                return name

    for name, shape, _dtype, _comp in infos:
        if _looks_like_cube(shape):
            return name

    return None


def _save_sr_h5(out_h5: str, sr_chw_u8: np.ndarray):
    if sr_chw_u8.ndim != 3:
        raise ValueError(f"sr cube expects CHW 3D array, got shape={sr_chw_u8.shape}")

    sr_hwc_u8 = np.transpose(sr_chw_u8, (1, 2, 0))
    sr_hwc_u8 = np.asarray(sr_hwc_u8, dtype=np.uint8)

    os.makedirs(os.path.dirname(out_h5), exist_ok=True)
    with h5py.File(out_h5, "w") as f:
        f.create_dataset("HR", data=sr_hwc_u8, compression="gzip")


def _load_ckpt_like_mains(ckpt_path: str):
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")

    if not isinstance(ckpt, dict):
        raise ValueError(f"checkpoint format error: {ckpt_path}")

    if "model" in ckpt:
        model_obj = ckpt["model"]
    elif "model_state_dict" in ckpt:
        model_obj = ckpt["model_state_dict"]
    else:
        raise ValueError(f"checkpoint missing 'model' or 'model_state_dict' key: {ckpt_path}")

    ckpt_args = ckpt.get("args", {})
    return model_obj, ckpt_args


@torch.no_grad()
def predict_sliding_window(
    model: torch.nn.Module,
    lr_chw_u8: np.ndarray,
    scale: int,
    tile: int,
    overlap: int,
    device: torch.device,
):
    model.eval()

    lr = lr_chw_u8.astype(np.float32) / 255.0
    c, h, w = lr.shape

    tile = int(tile) if tile is not None else 50
    if tile <= 0:
        tile = 50

    overlap = int(overlap) if overlap is not None else 0
    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    out_h = h * int(scale)
    out_w = w * int(scale)
    out = np.zeros((c, out_h, out_w), dtype=np.float32)
    wgt = np.zeros((1, out_h, out_w), dtype=np.float32)

    stride = tile - overlap
    if stride <= 0:
        raise ValueError("tile must be greater than overlap")

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
            lms = torch.nn.functional.interpolate(inp, scale_factor=scale, mode="bicubic", align_corners=False)
            pred = model(inp, lms, modality="spectral", img_size=inp.shape[2:4]).clamp(0.0, 1.0)
            pred_np = pred.detach().cpu().numpy()[0]

            oy0 = y * scale
            ox0 = x * scale
            oy1 = oy0 + tile * scale
            ox1 = ox0 + tile * scale

            out[:, oy0:oy1, ox0:ox1] += pred_np
            wgt[:, oy0:oy1, ox0:ox1] += 1.0

            del inp, lms, pred, pred_np

    out /= np.maximum(wgt, 1e-6)
    return out


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt",
        type=str,
        default="./checkpoints/v1.pth",
        help="checkpoint path",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default="./datasets/final_test",
        help="input directory with LR_*.h5",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="save SR results directory (default: auto from ckpt name)",
    )
    parser.add_argument(
        "--submit_dir",
        type=str,
        default=None,
        help="copy final h5 to x4 subdirectory (default: auto from ckpt name)",
    )
    parser.add_argument(
        "--zip_name",
        type=str,
        default=None,
        help="final submission zip filename (default: auto from ckpt name)",
    )
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--tile", type=int, default=50)
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--gpus", type=str, default="0", help="CUDA_VISIBLE_DEVICES")

    args = parser.parse_args()

    ckpt_basename = os.path.splitext(os.path.basename(args.ckpt))[0]
    if "iter_" in ckpt_basename:
        ckpt_short = "iter_" + ckpt_basename.split("iter_")[-1]
    else:
        ckpt_short = ckpt_basename[-20:] if len(ckpt_basename) > 20 else ckpt_basename
    result_root = "./result"
    default_save_dir = os.path.join(result_root, ckpt_short)
    default_submit_dir = os.path.join(result_root, f"submit_{ckpt_short}")

    if args.save_dir is None:
        args.save_dir = default_save_dir
    if args.submit_dir is None:
        args.submit_dir = default_submit_dir
    if args.zip_name is None:
        args.zip_name = f"{ckpt_short}.zip"

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"ckpt not found: {args.ckpt}")

    h5_files = sorted(glob.glob(os.path.join(args.test_dir, "LR_*.h5")))
    if not h5_files:
        raise FileNotFoundError(f"No LR_*.h5 found in test_dir: {args.test_dir}")

    state_dict, ckpt_args = _load_ckpt_like_mains(args.ckpt)

    net = General_VolFormer(
        n_subs=int(ckpt_args.get("n_subs", 8)),
        n_ovls=int(ckpt_args.get("n_ovls", 2)),
        n_colors=int(ckpt_args.get("n_colors", 224)),
        n_blocks=int(ckpt_args.get("n_blocks", 3)),
        n_feats=int(ckpt_args.get("n_feats", 192)),
        n_scale=int(ckpt_args.get("n_scale", args.scale)),
        res_scale=0.1,
        use_share=bool(ckpt_args.get("use_share", True)),
        conv=default_conv,
        vf_embed_dim=int(ckpt_args.get("vf_embed_dim", 120)),
        vf_depth=int(ckpt_args.get("vf_depth", 4)),
        vf_stages=int(ckpt_args.get("vf_stages", 4)),
        vf_num_heads=int(ckpt_args.get("vf_num_heads", 4)),
    )
    
    if isinstance(state_dict, dict):
        net.load_state_dict(state_dict, strict=False)
    elif isinstance(state_dict, torch.nn.Module):
        net = state_dict

    net.to(device).eval()

    save_dir = args.save_dir.strip() if isinstance(args.save_dir, str) else ""
    if save_dir == "":
        save_dir = None

    submit_dir = args.submit_dir.strip() if isinstance(args.submit_dir, str) else ""
    if submit_dir == "":
        submit_dir = None

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    if submit_dir is not None:
        os.makedirs(submit_dir, exist_ok=True)

    print("===> device:", device)
    print("===> ckpt:", args.ckpt)
    print("===> test_dir:", args.test_dir)
    print("===> h5_files:", len(h5_files))

    t_all = time.time()

    for h5_path in h5_files:
        base = os.path.splitext(os.path.basename(h5_path))[0]
        sid = base

        infos = _list_h5_datasets(h5_path)
        lr_key = _find_lr_dataset(infos)
        if lr_key is None:
            raise KeyError(f"No LR dataset found in h5: {h5_path}, datasets={[n for n,_,_,_ in infos]}")

        with h5py.File(h5_path, "r") as f:
            lr_arr = np.asarray(f[lr_key][...])

        lr_chw_u8 = _normalize_cube_to_chw(lr_arr).astype(np.uint8, copy=False)

        t0 = time.time()
        sr_f = predict_sliding_window(
            net,
            lr_chw_u8,
            scale=int(args.scale),
            tile=int(args.tile),
            overlap=int(args.overlap),
            device=device,
        )
        sr_chw_u8 = float_to_u8(sr_f)
        dt = time.time() - t0

        print(f"[{sid}] done, time={dt:.2f}s", flush=True)

        if save_dir is not None:
            out_h5 = os.path.join(save_dir, f"{sid}.h5")
            _save_sr_h5(out_h5, sr_chw_u8)

            if Image is not None or imageio is not None:
                out_png = os.path.join(save_dir, f"{sid}_rgb.png")
                _save_rgb_png(out_png, sr_chw_u8)

            if submit_dir is not None:
                x4_dir = os.path.join(submit_dir, "x4")
                os.makedirs(x4_dir, exist_ok=True)
                hr_name = base.replace("LR_", "HR_") + ".h5"
                if not hr_name.startswith("HR_"):
                    hr_name = "HR_" + base.replace("LR_", "") + ".h5"
                idx = base.replace("LR_", "")
                hr_name = f"HR_{idx}.h5"
                shutil.copy2(out_h5, os.path.join(x4_dir, hr_name))

    if submit_dir is not None:
        zip_name = args.zip_name.strip() if isinstance(args.zip_name, str) else ""
        if zip_name == "":
            zip_name = None

        if zip_name is not None:
            base_dir = os.path.dirname(submit_dir.rstrip("/"))
            zip_path = os.path.join(base_dir, zip_name)
            src_dir = os.path.join(submit_dir, "x4")

            if not os.path.isdir(src_dir):
                raise FileNotFoundError(f"Source directory not found: {src_dir}")

            if os.path.exists(zip_path):
                os.remove(zip_path)

            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for root, _dirs, files in os.walk(src_dir):
                    for fn in files:
                        fp = os.path.join(root, fn)
                        arcname = os.path.relpath(fp, submit_dir)
                        zf.write(fp, arcname=arcname)

            print(f"[ZIP] saved: {zip_path}")

    dt_all = time.time() - t_all
    print(f"[DONE] h5 inference completed, total_time={dt_all:.2f}s")


if __name__ == "__main__":
    main()
