import argparse
import os
import time

import numpy as np
import torch

# 恢复使用 tiff_utils.py (基于 GDAL)
from tiff_utils import build_pairs, evaluate_tiff_pairs
from tiff_utils import read_tiff_chw, normalize_u8_to_float, predict_sliding_window, psnr, sam
from OverallModel import General_VolFormer


def _load_ckpt_like_mains(ckpt_path: str):
    # PyTorch 2.6+ 默认 weights_only=True，但 best.pth 保存了完整模型对象，需要设为 False
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        # 兼容旧版 PyTorch，没有 weights_only 参数
        ckpt = torch.load(ckpt_path, map_location="cpu")

    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise ValueError(f"checkpoint格式不符合 mains.py 的 save_checkpoint: {ckpt_path}")

    model_obj = ckpt["model"]
    ckpt_args = ckpt.get("args", {})

    return model_obj, ckpt_args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="/home/fdw/code/HISR/lzy/code/VolFormer/checkpoints/best.pth", help="best.pth 路径")
    parser.add_argument("--data_dir", type=str, default="/home/fdw/code/HISR/lzy/tiff/test", help="包含 *_LR.tif/*_HR.tif 的目录")
    parser.add_argument("--gpus", type=str, default="6", help="CUDA_VISIBLE_DEVICES")
    parser.add_argument("--tile", type=int, default=50, help="LR 滑窗 tile")
    parser.add_argument("--overlap", type=int, default=0, help="LR 滑窗 overlap")
    parser.add_argument("--scale", type=int, default=4, help="倍率（必须与数据匹配）")
    parser.add_argument("--save_dir", type=str, default='/home/fdw/code/HISR/lzy/code/VolFormer/result', help="可选：保存 SR tif 到该目录")
    parser.add_argument("--strict", type=int, default=1, help="1: 遇到异常直接报错；0: 跳过异常样本")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("===> device:", device)
    print("===> ckpt:", args.ckpt)
    print("===> data_dir:", args.data_dir)

    pairs = build_pairs(args.data_dir)
    if len(pairs) == 0:
        print("警告：在 data_dir 中没有找到任何 *_LR.tif/*_HR.tif 配对文件。请检查路径和文件名。")
    print("===> pairs:", len(pairs))

    net, ckpt_args = _load_ckpt_like_mains(args.ckpt)

    # 如果 checkpoint 里保存了 args，就用它来覆盖命令行参数（保证一致）
    if isinstance(ckpt_args, dict) and len(ckpt_args) > 0:
        for k, v in ckpt_args.items():
            if hasattr(args, k):
                setattr(args, k, v)

    net.to(device).eval()

    # 打印每个样本的 PSNR/SAM（便于看到三张图分别多少）
    for sid, lr_path, hr_path in pairs:
        try:
            lr_u8 = read_tiff_chw(lr_path, dtype=np.uint8)
            hr_u8 = read_tiff_chw(hr_path, dtype=np.uint8)
            sr_f = predict_sliding_window(net, lr_u8, scale=int(args.scale), tile=int(args.tile), overlap=int(args.overlap), device=device)
            hr_f = normalize_u8_to_float(hr_u8)
            sr_t = torch.from_numpy(sr_f).unsqueeze(0).to(device)
            hr_t = torch.from_numpy(hr_f).unsqueeze(0).to(device)
            p = psnr(sr_t, hr_t).mean().item()
            a = sam(sr_t, hr_t).mean().item()
            print(f"[PER-IMG] {sid}: PSNR={p:.4f} dB | SAM={a:.6f} rad", flush=True)
        except Exception as e:
            print(f"[PER-IMG] {sid}: 失败: {e}", flush=True)

    t0 = time.time()
    mean_psnr, mean_sam = evaluate_tiff_pairs(
        net,
        pairs,
        device=device,
        scale=int(args.scale),
        tile=int(args.tile),
        overlap=int(args.overlap),
        save_dir=args.save_dir,
        strict=bool(int(args.strict)),
    )
    dt = time.time() - t0

    print(f"===> DONE: mean PSNR={mean_psnr:.4f} dB | mean SAM={mean_sam:.6f} rad | time={dt:.2f}s")


if __name__ == "__main__":
    main()
