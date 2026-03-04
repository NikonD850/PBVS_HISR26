import argparse
import os
from typing import Dict, List, Tuple

import numpy as np

from tiff_utils import read_tiff_chw, normalize_u8_to_float


def _parse_sid_from_sr(name: str) -> str:
    # e.g. Scene_25_SR.tif -> Scene_25
    base = os.path.basename(name)
    if base.endswith("_SR.tif"):
        return base[: -len("_SR.tif")]
    if base.endswith(".tif"):
        return base[: -len(".tif")]
    return base


def _psnr_per_channel(sr: np.ndarray, hr: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    # sr/hr: (C,H,W) float in [0,1]
    diff = sr.astype(np.float32) - hr.astype(np.float32)
    mse = np.mean(diff * diff, axis=(1, 2))
    return 10.0 * np.log10(1.0 / (mse + eps))


def _psnr_mean(sr: np.ndarray, hr: np.ndarray) -> float:
    return float(np.mean(_psnr_per_channel(sr, hr)))


def _iter_lr_grid(hlr: int, wlr: int, tile: int, overlap: int):
    stride = int(tile) - int(overlap)
    if stride <= 0:
        raise ValueError("tile必须大于overlap")

    ys = list(range(0, int(hlr), int(stride)))
    xs = list(range(0, int(wlr), int(stride)))
    if len(ys) == 0:
        ys = [0]
    if len(xs) == 0:
        xs = [0]
    if ys[-1] + int(tile) < int(hlr):
        ys.append(int(hlr) - int(tile))
    if xs[-1] + int(tile) < int(wlr):
        xs.append(int(wlr) - int(tile))

    for y in ys:
        y = min(max(int(y), 0), int(hlr) - int(tile))
        for x in xs:
            x = min(max(int(x), 0), int(wlr) - int(tile))
            yield int(y), int(x)


def _block_psnr_from_sr_hr(
    sr: np.ndarray,
    hr: np.ndarray,
    scale: int,
    tile: int,
    overlap: int,
):
    # sr/hr: (C, Hhr, Whr)
    c, hhr, whr = hr.shape
    hlr = hhr // int(scale)
    wlr = whr // int(scale)

    tile_hr = int(tile) * int(scale)
    blocks = []

    for y, x in _iter_lr_grid(hlr, wlr, tile=int(tile), overlap=int(overlap)):
        oy0 = y * int(scale)
        ox0 = x * int(scale)
        oy1 = oy0 + tile_hr
        ox1 = ox0 + tile_hr

        sr_b = sr[:, oy0:oy1, ox0:ox1]
        hr_b = hr[:, oy0:oy1, ox0:ox1]
        if sr_b.shape != hr_b.shape or sr_b.shape[1] != tile_hr or sr_b.shape[2] != tile_hr:
            continue

        psnr_c = _psnr_per_channel(sr_b, hr_b)
        psnr_m = float(np.mean(psnr_c))
        blocks.append({
            "y_lr": y,
            "x_lr": x,
            "psnr_mean": psnr_m,
            "psnr_c": psnr_c,
        })

    return blocks


def _build_sr_hr_pairs(sr_dir: str, hr_dir: str) -> List[Tuple[str, str, str]]:
    out = []
    for fn in sorted(os.listdir(sr_dir)):
        if not fn.endswith("_SR.tif"):
            continue
        sid = _parse_sid_from_sr(fn)
        sr_path = os.path.join(sr_dir, fn)
        hr_path = os.path.join(hr_dir, f"{sid}_HR.tif")
        if not os.path.exists(hr_path):
            continue
        out.append((sid, sr_path, hr_path))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr_dir", type=str, default="/home/fdw/trained_model", help="包含 *_SR.tif 的目录")
    parser.add_argument("--hr_dir", type=str, default="/home/fdw/code/HISR/lzy/tiff/test", help="包含 *_HR.tif 的目录")

    # 块级统计：tile/overlap 是 LR 尺度（与你推理时的滑窗一致），scale 用于换算到 HR/SR 尺度
    parser.add_argument("--scale", type=int, default=4, help="倍率（HR = LR * scale）")
    parser.add_argument("--tile", type=int, default=200, help="LR滑窗tile大小，用于块级PSNR统计")
    parser.add_argument("--overlap", type=int, default=0, help="LR滑窗overlap，用于块级PSNR统计")

    parser.add_argument("--topk", type=int, default=15, help="输出每张图最差的前K个通道")
    parser.add_argument("--block_topk", type=int, default=20, help="输出每张图最差的前K个块")
    parser.add_argument("--save_csv", type=int, default=1, help="1: 保存块级PSNR到CSV（默认保存到sr_dir）；0: 不保存")
    args = parser.parse_args()

    pairs = _build_sr_hr_pairs(args.sr_dir, args.hr_dir)
    if len(pairs) == 0:
        raise RuntimeError(f"在 {args.sr_dir} 中没找到可配对的 *_SR.tif（或在 {args.hr_dir} 中缺少对应 *_HR.tif）")

    # 汇总：跨所有图的通道平均/最小
    ch_sum = None
    ch_cnt = None
    ch_min = None

    per_img: Dict[str, Dict] = {}

    for sid, sr_path, hr_path in pairs:
        sr_u8 = read_tiff_chw(sr_path, dtype=np.uint8)
        hr_u8 = read_tiff_chw(hr_path, dtype=np.uint8)

        if sr_u8.shape != hr_u8.shape:
            raise RuntimeError(f"[{sid}] SR/HR shape不一致 sr={sr_u8.shape} hr={hr_u8.shape}")

        sr = normalize_u8_to_float(sr_u8)
        hr = normalize_u8_to_float(hr_u8)

        # 全图（逐通道 & 均值）
        psnr_c = _psnr_per_channel(sr, hr)
        psnr_mean = float(np.mean(psnr_c))

        # 按 LR tile 切块后统计（块对应 HR/SR 尺度 tile*scale）
        blocks = _block_psnr_from_sr_hr(sr, hr, scale=int(args.scale), tile=int(args.tile), overlap=int(args.overlap))
        if len(blocks) == 0:
            raise RuntimeError(f"[{sid}] 没有产生任何有效块，请检查 scale/tile/overlap 是否匹配")

        block_psnr = np.array([b["psnr_mean"] for b in blocks], dtype=np.float64)
        block_mean = float(np.mean(block_psnr))
        block_min = float(np.min(block_psnr))

        per_img[sid] = {
            "psnr_mean": psnr_mean,
            "psnr_c": psnr_c,
            "block_mean": block_mean,
            "block_min": block_min,
            "n_blocks": int(len(blocks)),
            "blocks": blocks,
            "sr_path": sr_path,
            "hr_path": hr_path,
        }

        c = int(psnr_c.size)
        if ch_sum is None:
            ch_sum = np.zeros((c,), dtype=np.float64)
            ch_cnt = np.zeros((c,), dtype=np.int64)
            ch_min = np.full((c,), np.inf, dtype=np.float64)

        ch_sum += psnr_c
        ch_cnt += 1
        ch_min = np.minimum(ch_min, psnr_c)

        # 打印每张图整体PSNR + 块级PSNR
        print(
            f"[PER-IMG] {sid}: PSNR(full)={psnr_mean:.4f} dB | PSNR(block_mean)={block_mean:.4f} dB | PSNR(block_min)={block_min:.4f} dB | n_blocks={len(blocks)} (sr={sr_path})",
            flush=True,
        )

        # 打印该图最差通道（全图逐通道）
        k = min(int(args.topk), c)
        worst = np.argsort(psnr_c)[:k]
        msg = " ".join([f"ch{int(i)}={float(psnr_c[i]):.2f}" for i in worst])
        print(f"[PER-IMG] {sid}: worst_ch_full{ k } {msg}", flush=True)

        # 打印该图最差块（按块均值PSNR排序）
        bk = min(int(getattr(args, "block_topk", 20)), int(len(blocks)))
        worst_blocks = np.argsort(block_psnr)[:bk]
        msgb = " ".join([
            f"(y={int(blocks[i]['y_lr'])},x={int(blocks[i]['x_lr'])},psnr={float(block_psnr[i]):.2f})" for i in worst_blocks
        ])
        print(f"[PER-IMG] {sid}: worst_blocks{ bk } {msgb}", flush=True)

        # 保存块级CSV
        if bool(int(getattr(args, "save_csv", 1))):
            out_csv = os.path.join(args.sr_dir, f"{sid}_block_psnr_tile{int(args.tile)}_over{int(args.overlap)}_scale{int(args.scale)}.csv")
            with open(out_csv, "w", encoding="utf-8") as f:
                f.write("y_lr,x_lr,psnr_mean_db\n")
                for b in blocks:
                    f.write(f"{int(b['y_lr'])},{int(b['x_lr'])},{float(b['psnr_mean']):.6f}\n")
            print(f"[PER-IMG] {sid}: 已保存块级PSNR: {out_csv}", flush=True)

    # 跨图汇总：哪些通道整体差
    ch_mean = ch_sum / np.maximum(ch_cnt, 1)
    overall_mean = float(np.mean(ch_mean))
    drop = overall_mean - ch_mean
    order = np.argsort(-drop)

    print(f"[ALL] n_imgs={len(pairs)} overall_channel_mean={overall_mean:.4f} dB", flush=True)
    topk = min(20, int(ch_mean.size))
    for r in range(topk):
        ci = int(order[r])
        print(
            f"[ALL] rank={r+1:02d} ch={ci:03d} mean={float(ch_mean[ci]):.4f} dB min={float(ch_min[ci]):.4f} dB drop={float(drop[ci]):.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
