import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import h5py
import numpy as np
import scipy.io
import torch
import torch.nn.functional as F


SUPPORTED_SUFFIX = (".mat", ".h5", ".hdf5")
MS_KEYS = ["LR_uint8", "ms", "lr", "LR"]
GT_KEYS = ["HR_uint8", "gt", "HR", "hr"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="构建 patch shard 数据集：先切 50x50 patch，再按 inference 一致方式做 4x bicubic。"
    )
    parser.add_argument("--input_dir", type=str, required=True, help="输入目录（支持 .mat/.h5/.hdf5）。")
    parser.add_argument("--output_dir", type=str, required=True, help="输出 shard 目录。")
    parser.add_argument("--patch_size", type=int, default=50, help="LR patch 大小。")
    parser.add_argument("--overlap", type=float, default=0.5, help="patch overlap 比例，必须在 [0.4, 0.5]。")
    parser.add_argument("--stride", type=int, default=0, help="可选。若 >0 则优先使用该 stride（自动校验 overlap 范围）。")
    parser.add_argument("--n_scale", type=int, default=4, help="上采样倍率。")
    parser.add_argument("--shard_size", type=int, default=4096, help="每个 shard 最多样本数。")
    parser.add_argument("--interp_batch", type=int, default=128, help="插值批大小。")
    parser.add_argument("--num_workers", type=int, default=0, help="CPU worker 数，0 表示自动使用 os.cpu_count()。")
    parser.add_argument("--threads_per_worker", type=int, default=1, help="每个 worker 内 torch 线程数。")
    parser.add_argument("--gpus", type=str, default="0", help="可选，逗号分隔 GPU id，如 0,1,2,3,4,5,6,7。")
    parser.add_argument("--save_dtype", type=str, choices=["uint8", "float32"], default="uint8", help="保存 dtype。")
    parser.add_argument("--compression", type=str, choices=["none", "lzf", "gzip"], default="none", help="HDF5 压缩方式。")
    parser.add_argument("--gzip_level", type=int, default=1, help="gzip 压缩级别。")
    parser.add_argument("--chunk_size", type=int, default=64, help="每个 dataset 的第一维 chunk 大小。")
    parser.add_argument("--limit_files", type=int, default=0, help="可选。仅处理前 N 个输入文件。")
    return parser.parse_args()


def _is_supported_file(path: Path):
    return path.suffix.lower() in SUPPORTED_SUFFIX


def _normalize_to_float(array):
    np_array = np.asarray(array)
    original_dtype = np_array.dtype
    np_array = np_array.astype(np.float32)

    should_divide = np.issubdtype(original_dtype, np.integer)
    if not should_divide:
        finite_mask = np.isfinite(np_array)
        if np.any(finite_mask):
            max_value = float(np.max(np_array[finite_mask]))
            if max_value > 1.5:
                should_divide = True

    if should_divide:
        np_array = np_array / 255.0
    return np_array


def _read_first_key(container, keys):
    for key in keys:
        if key in container:
            return np.asarray(container[key])
    raise KeyError(f"missing keys {keys}, available keys: {list(container.keys())}")


def _load_ms_gt(path: Path):
    lower_path = str(path).lower()
    if lower_path.endswith(".mat"):
        raw = scipy.io.loadmat(str(path))
        ms = np.asarray(raw["ms"])
        gt = np.asarray(raw["gt"])
    else:
        with h5py.File(str(path), "r") as raw:
            ms = _read_first_key(raw, MS_KEYS)
            gt = _read_first_key(raw, GT_KEYS)

    ms = _normalize_to_float(ms)
    gt = _normalize_to_float(gt)
    if ms.ndim != 3 or gt.ndim != 3:
        raise ValueError(f"{path} 输入维度错误，期望 3D(H,W,C)，当前 ms={ms.shape}, gt={gt.shape}")
    return ms, gt


def _build_patch_positions(length, patch_size, stride):
    if patch_size <= 0:
        raise ValueError(f"patch_size 必须 > 0，当前是 {patch_size}")
    if stride <= 0:
        raise ValueError(f"stride 必须 > 0，当前是 {stride}")
    if length < patch_size:
        raise ValueError(f"输入尺寸 {length} 小于 patch_size {patch_size}")

    positions = list(range(0, length - patch_size + 1, stride))
    last = length - patch_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def _resolve_stride_and_overlap(patch_size, overlap, stride):
    if stride > 0:
        stride = int(stride)
        if stride >= patch_size:
            raise ValueError(f"stride 必须小于 patch_size，当前 stride={stride}, patch_size={patch_size}")
        actual_overlap = 1.0 - (float(stride) / float(patch_size))
    else:
        actual_overlap = float(overlap)
        if not (0.4 <= actual_overlap <= 0.5):
            raise ValueError(f"overlap 必须在 [0.4, 0.5]，当前是 {actual_overlap}")
        stride = int(round(float(patch_size) * (1.0 - actual_overlap)))
        stride = max(1, min(stride, patch_size - 1))
        actual_overlap = 1.0 - (float(stride) / float(patch_size))

    if not (0.4 <= actual_overlap <= 0.5):
        raise ValueError(
            f"当前 stride={stride} 对应 overlap={actual_overlap:.6f}，不在 [0.4, 0.5]。"
            "请调整 patch_size/overlap/stride。"
        )
    return int(stride), float(actual_overlap)


def _float_to_storage(array, save_dtype):
    if save_dtype == "float32":
        return array.astype(np.float32)
    return np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)


def _resolve_compression(compression, gzip_level):
    if compression == "none":
        return None, None
    if compression == "lzf":
        return "lzf", None
    return "gzip", int(gzip_level)


class ShardWriter:
    def __init__(
        self,
        output_dir,
        worker_id,
        shard_size,
        chunk_size,
        save_dtype,
        compression,
        gzip_level,
        patch_size,
        stride,
        overlap,
        n_scale,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.worker_id = int(worker_id)
        self.shard_size = int(shard_size)
        self.chunk_size = int(chunk_size)
        self.save_dtype = str(save_dtype)
        self.compression = str(compression)
        self.gzip_level = int(gzip_level)
        self.patch_size = int(patch_size)
        self.stride = int(stride)
        self.overlap = float(overlap)
        self.n_scale = int(n_scale)

        self._buffer_ms = []
        self._buffer_lms = []
        self._buffer_gt = []
        self._buffer_count = 0
        self._shard_index = 0

        self.total_patches = 0
        self.total_shards = 0

    def add_batch(self, ms_batch, lms_batch, gt_batch):
        if ms_batch.shape[0] == 0:
            return

        start = 0
        batch_size = int(ms_batch.shape[0])
        while start < batch_size:
            room = self.shard_size - self._buffer_count
            take = min(room, batch_size - start)

            end = start + take
            self._buffer_ms.append(ms_batch[start:end])
            self._buffer_lms.append(lms_batch[start:end])
            self._buffer_gt.append(gt_batch[start:end])
            self._buffer_count += take
            start = end

            if self._buffer_count >= self.shard_size:
                self.flush()

    def flush(self):
        if self._buffer_count == 0:
            return

        ms = np.concatenate(self._buffer_ms, axis=0)
        lms = np.concatenate(self._buffer_lms, axis=0)
        gt = np.concatenate(self._buffer_gt, axis=0)
        self._write_one_shard(ms=ms, lms=lms, gt=gt)

        self._buffer_ms = []
        self._buffer_lms = []
        self._buffer_gt = []
        self._buffer_count = 0

    def _write_one_shard(self, ms, lms, gt):
        shard_name = f"shard_w{self.worker_id:03d}_{self._shard_index:06d}.h5"
        shard_path = self.output_dir / shard_name

        compression, compression_opts = _resolve_compression(self.compression, self.gzip_level)
        chunk_n = max(1, min(self.chunk_size, int(ms.shape[0])))

        with h5py.File(str(shard_path), "w") as h5f:
            ms_kwargs = {"chunks": (chunk_n, ms.shape[1], ms.shape[2], ms.shape[3])}
            lms_kwargs = {"chunks": (chunk_n, lms.shape[1], lms.shape[2], lms.shape[3])}
            gt_kwargs = {"chunks": (chunk_n, gt.shape[1], gt.shape[2], gt.shape[3])}
            if compression is not None:
                ms_kwargs["compression"] = compression
                lms_kwargs["compression"] = compression
                gt_kwargs["compression"] = compression
                if compression_opts is not None:
                    ms_kwargs["compression_opts"] = compression_opts
                    lms_kwargs["compression_opts"] = compression_opts
                    gt_kwargs["compression_opts"] = compression_opts

            h5f.create_dataset("ms", data=ms, **ms_kwargs)
            h5f.create_dataset("lms", data=lms, **lms_kwargs)
            h5f.create_dataset("gt", data=gt, **gt_kwargs)

            h5f.attrs["count"] = int(ms.shape[0])
            h5f.attrs["patch_size"] = int(self.patch_size)
            h5f.attrs["stride"] = int(self.stride)
            h5f.attrs["overlap"] = float(self.overlap)
            h5f.attrs["n_scale"] = int(self.n_scale)
            h5f.attrs["save_dtype"] = str(self.save_dtype)

        self.total_patches += int(ms.shape[0])
        self.total_shards += 1
        self._shard_index += 1

    def close(self):
        self.flush()


def _interpolate_lms(ms_batch_hwc, n_scale, device):
    ms_bchw = np.ascontiguousarray(ms_batch_hwc.transpose(0, 3, 1, 2))
    ms_t = torch.from_numpy(ms_bchw).to(device=device, dtype=torch.float32, non_blocking=(device.type == "cuda"))
    with torch.no_grad():
        lms_t = F.interpolate(ms_t, scale_factor=int(n_scale), mode="bicubic", align_corners=False)
    lms_hwc = lms_t.permute(0, 2, 3, 1).contiguous().cpu().numpy().astype(np.float32)
    return lms_hwc


def _process_one_file(path, writer, patch_size, stride, n_scale, interp_batch, save_dtype, device):
    ms, gt = _load_ms_gt(path)
    ms_h, ms_w, ms_c = ms.shape
    gt_h, gt_w, gt_c = gt.shape
    if ms_c != gt_c:
        raise ValueError(f"{path} ms/gt 通道数不一致: ms_c={ms_c}, gt_c={gt_c}")
    if gt_h % ms_h != 0 or gt_w % ms_w != 0:
        raise ValueError(f"{path} gt 与 ms 尺寸不整除: ms={ms.shape}, gt={gt.shape}")

    scale_h_gt = gt_h // ms_h
    scale_w_gt = gt_w // ms_w
    if scale_h_gt != int(n_scale) or scale_w_gt != int(n_scale):
        raise ValueError(
            f"{path} gt/ms 比例与 n_scale 不一致: "
            f"scale_h_gt={scale_h_gt}, scale_w_gt={scale_w_gt}, n_scale={n_scale}"
        )

    lr_h_positions = _build_patch_positions(ms_h, patch_size, stride)
    lr_w_positions = _build_patch_positions(ms_w, patch_size, stride)
    coords = [(top, left) for top in lr_h_positions for left in lr_w_positions]

    gt_patch_h = int(patch_size * scale_h_gt)
    gt_patch_w = int(patch_size * scale_w_gt)
    total = len(coords)
    processed = 0

    while processed < total:
        end = min(total, processed + int(interp_batch))
        sub_coords = coords[processed:end]

        ms_batch = np.empty((len(sub_coords), patch_size, patch_size, ms_c), dtype=np.float32)
        gt_batch = np.empty((len(sub_coords), gt_patch_h, gt_patch_w, gt_c), dtype=np.float32)

        for idx, (top, left) in enumerate(sub_coords):
            ms_batch[idx] = ms[top : top + patch_size, left : left + patch_size, :]
            gt_top = int(top * scale_h_gt)
            gt_left = int(left * scale_w_gt)
            gt_batch[idx] = gt[gt_top : gt_top + gt_patch_h, gt_left : gt_left + gt_patch_w, :]

        lms_batch = _interpolate_lms(ms_batch_hwc=ms_batch, n_scale=n_scale, device=device)
        if lms_batch.shape[1] != gt_patch_h or lms_batch.shape[2] != gt_patch_w:
            raise RuntimeError(
                f"{path} 插值后 lms patch 尺寸异常: "
                f"lms={lms_batch.shape}, 期望 HxW={gt_patch_h}x{gt_patch_w}"
            )

        writer.add_batch(
            ms_batch=_float_to_storage(ms_batch, save_dtype),
            lms_batch=_float_to_storage(lms_batch, save_dtype),
            gt_batch=_float_to_storage(gt_batch, save_dtype),
        )
        processed = end

    return total


def _worker_main(worker_id, file_list, cfg):
    worker_id = int(worker_id)
    file_list = [Path(path) for path in file_list]
    torch.set_num_threads(int(cfg["threads_per_worker"]))
    torch.set_num_interop_threads(1)

    gpu_id = cfg.get("gpu_id", None)
    if gpu_id is not None:
        device = torch.device(f"cuda:{int(gpu_id)}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    writer = ShardWriter(
        output_dir=cfg["output_dir"],
        worker_id=worker_id,
        shard_size=cfg["shard_size"],
        chunk_size=cfg["chunk_size"],
        save_dtype=cfg["save_dtype"],
        compression=cfg["compression"],
        gzip_level=cfg["gzip_level"],
        patch_size=cfg["patch_size"],
        stride=cfg["stride"],
        overlap=cfg["overlap"],
        n_scale=cfg["n_scale"],
    )

    processed_files = 0
    processed_patches = 0
    for idx, path in enumerate(file_list):
        count = _process_one_file(
            path=path,
            writer=writer,
            patch_size=cfg["patch_size"],
            stride=cfg["stride"],
            n_scale=cfg["n_scale"],
            interp_batch=cfg["interp_batch"],
            save_dtype=cfg["save_dtype"],
            device=device,
        )
        processed_files += 1
        processed_patches += int(count)

        log_every = int(cfg["log_every"])
        if log_every > 0 and ((idx + 1) % log_every == 0):
            print(
                f"[worker {worker_id}] files={processed_files}/{len(file_list)} "
                f"patches={processed_patches} device={device}"
            )

    writer.close()
    return {
        "worker_id": worker_id,
        "device": str(device),
        "input_files": processed_files,
        "patches": int(processed_patches),
        "shards": int(writer.total_shards),
    }


def _split_round_robin(items, num_parts):
    groups = [[] for _ in range(num_parts)]
    for idx, item in enumerate(items):
        groups[idx % num_parts].append(str(item))
    return groups


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir 不存在: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    stride, overlap = _resolve_stride_and_overlap(
        patch_size=int(args.patch_size),
        overlap=float(args.overlap),
        stride=int(args.stride),
    )

    files = sorted([path for path in input_dir.iterdir() if path.is_file() and _is_supported_file(path)])
    if args.limit_files > 0:
        files = files[: int(args.limit_files)]
    if len(files) == 0:
        raise RuntimeError(f"{input_dir} 下未找到支持文件（.mat/.h5/.hdf5）。")

    gpus = []
    if args.gpus.strip() != "":
        gpus = [int(token.strip()) for token in args.gpus.split(",") if token.strip() != ""]
        if len(gpus) == 0:
            raise ValueError("--gpus 传入为空。")
        if not torch.cuda.is_available():
            raise RuntimeError("传入了 --gpus，但当前环境未检测到 CUDA。")

    if len(gpus) > 0:
        worker_count = len(gpus)
    else:
        worker_count = int(args.num_workers) if int(args.num_workers) > 0 else int(os.cpu_count() or 1)
    worker_count = max(1, min(worker_count, len(files)))

    file_groups = _split_round_robin(files, worker_count)
    task_payloads = []
    for worker_id, group in enumerate(file_groups):
        if len(group) == 0:
            continue

        cfg = {
            "output_dir": str(output_dir),
            "patch_size": int(args.patch_size),
            "stride": int(stride),
            "overlap": float(overlap),
            "n_scale": int(args.n_scale),
            "shard_size": int(args.shard_size),
            "chunk_size": int(args.chunk_size),
            "interp_batch": int(args.interp_batch),
            "save_dtype": str(args.save_dtype),
            "compression": str(args.compression),
            "gzip_level": int(args.gzip_level),
            "threads_per_worker": int(args.threads_per_worker),
            "log_every": 10,
            "gpu_id": (int(gpus[worker_id]) if worker_id < len(gpus) else None),
        }
        task_payloads.append((worker_id, group, cfg))

    print(f"Input dir: {input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Input files: {len(files)}")
    print(f"Workers: {len(task_payloads)}")
    print(f"Patch size: {args.patch_size}, stride: {stride}, overlap: {overlap:.4f}")
    print(f"n_scale: {args.n_scale}, shard_size: {args.shard_size}, interp_batch: {args.interp_batch}")
    print(f"save_dtype: {args.save_dtype}, compression: {args.compression}")
    if len(gpus) > 0:
        print(f"GPU workers: {gpus}")

    start_time = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(task_payloads)) as pool:
        results = pool.starmap(_worker_main, task_payloads)

    total_patches = int(sum(item["patches"] for item in results))
    total_shards = int(sum(item["shards"] for item in results))
    elapsed_sec = float(time.time() - start_time)

    manifest = {
        "created_time": time.ctime(),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "num_input_files": int(len(files)),
        "num_workers": int(len(task_payloads)),
        "patch_size": int(args.patch_size),
        "stride": int(stride),
        "overlap": float(overlap),
        "n_scale": int(args.n_scale),
        "shard_size": int(args.shard_size),
        "interp_batch": int(args.interp_batch),
        "save_dtype": str(args.save_dtype),
        "compression": str(args.compression),
        "total_patches": total_patches,
        "total_shards": total_shards,
        "elapsed_sec": elapsed_sec,
        "workers": results,
    }
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)

    print("Done.")
    print(f"total_patches={total_patches}")
    print(f"total_shards={total_shards}")
    print(f"manifest={manifest_path}")
    print(f"elapsed_sec={elapsed_sec:.2f}")


if __name__ == "__main__":
    main()
