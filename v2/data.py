import math
import os
from pathlib import Path

import h5py
import numpy as np
import scipy.io
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler


SUPPORTED_SUFFIX = (".mat", ".h5", ".hdf5")


def _is_supported_file(filename: str):
    return filename.lower().endswith(SUPPORTED_SUFFIX)


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
            return np.array(container[key])
    raise KeyError(f"missing keys {keys}, available keys: {list(container.keys())}")


def _read_first_key_ref(container, keys):
    for key in keys:
        if key in container:
            return container[key]
    raise KeyError(f"missing keys {keys}, available keys: {list(container.keys())}")


def _load_triplet_full(path: Path):
    lower_path = str(path).lower()
    if lower_path.endswith(".mat"):
        raw = scipy.io.loadmat(str(path))
        ms = np.array(raw["ms"])
        lms = np.array(raw["ms_bicubic"])
        gt = np.array(raw["gt"])
    else:
        with h5py.File(str(path), "r") as raw:
            gt = _read_first_key(raw, ["HR_uint8", "gt", "HR", "hr"])
            lms = _read_first_key(raw, ["LR_4x_uint8", "ms_bicubic", "lms", "LR_4x"])
            ms = _read_first_key(raw, ["LR_uint8", "ms", "lr", "LR"])

    ms = _normalize_to_float(ms)
    lms = _normalize_to_float(lms)
    gt = _normalize_to_float(gt)
    return ms, lms, gt


def _load_triplet_shapes(path: Path):
    lower_path = str(path).lower()
    if lower_path.endswith(".mat"):
        raw = scipy.io.loadmat(str(path))
        ms_shape = np.array(raw["ms"]).shape
        lms_shape = np.array(raw["ms_bicubic"]).shape
        gt_shape = np.array(raw["gt"]).shape
        return ms_shape, lms_shape, gt_shape

    with h5py.File(str(path), "r") as raw:
        gt_shape = tuple(_read_first_key_ref(raw, ["HR_uint8", "gt", "HR", "hr"]).shape)
        lms_shape = tuple(_read_first_key_ref(raw, ["LR_4x_uint8", "ms_bicubic", "lms", "LR_4x"]).shape)
        ms_shape = tuple(_read_first_key_ref(raw, ["LR_uint8", "ms", "lr", "LR"]).shape)
    return ms_shape, lms_shape, gt_shape


def _build_patch_positions(length: int, patch_size: int, stride: int):
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


def _to_tensor_chw(array):
    tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1).float()
    tensor = torch.where(torch.isnan(tensor), torch.full_like(tensor, 0), tensor)
    return tensor


def _apply_random_flip_triplet(ms, lms, gt, enable_hflip, enable_vflip):
    if bool(enable_hflip) and torch.rand(1).item() < 0.5:
        ms = torch.flip(ms, dims=[2])
        lms = torch.flip(lms, dims=[2])
        gt = torch.flip(gt, dims=[2])
    if bool(enable_vflip) and torch.rand(1).item() < 0.5:
        ms = torch.flip(ms, dims=[1])
        lms = torch.flip(lms, dims=[1])
        gt = torch.flip(gt, dims=[1])
    return ms, lms, gt


class loadingData(Dataset):
    def __init__(
        self,
        image_dir,
        augment=None,
        total_num=None,
        lr_patch_size=50,
        lr_patch_stride=25,
        scale=4,
        augment_modes=None,
        preload_in_memory=True,
        patch_mode=False,
        hflip=True,
        vflip=True,
    ):
        super(loadingData, self).__init__()
        self.image_paths = []
        for name in sorted(os.listdir(image_dir)):
            if _is_supported_file(name):
                self.image_paths.append(Path(image_dir) / name)

        if isinstance(total_num, int):
            self.image_paths = self.image_paths[:total_num]

        self.preload_in_memory = bool(preload_in_memory)
        self.patch_mode = bool(patch_mode)
        self.enable_hflip = bool(augment) and bool(hflip)
        self.enable_vflip = bool(augment) and bool(vflip)
        self.lr_patch_size = int(lr_patch_size)
        self.lr_patch_stride = int(lr_patch_stride)
        self.samples = []
        self.patch_records = []

        self._cache_file_idx = -1
        self._cache_ms = None
        self._cache_lms = None
        self._cache_gt = None

        if self.preload_in_memory:
            for path in self.image_paths:
                ms, lms, gt = _load_triplet_full(path)
                self.samples.append((ms, lms, gt))

        if self.patch_mode:
            self._build_patch_records()

    def _build_patch_records(self):
        if len(self.image_paths) == 0:
            return
        for file_idx, path in enumerate(self.image_paths):
            ms_shape, lms_shape, gt_shape = _load_triplet_shapes(path)
            ms_h, ms_w = int(ms_shape[0]), int(ms_shape[1])
            lms_h, lms_w = int(lms_shape[0]), int(lms_shape[1])
            gt_h, gt_w = int(gt_shape[0]), int(gt_shape[1])

            if lms_h % ms_h != 0 or lms_w % ms_w != 0:
                raise ValueError(f"{path} 的 LR_4x 与 LR 尺寸不整除: lms={lms_shape}, ms={ms_shape}")
            if gt_h % ms_h != 0 or gt_w % ms_w != 0:
                raise ValueError(f"{path} 的 HR 与 LR 尺寸不整除: gt={gt_shape}, ms={ms_shape}")

            scale_h_lms = lms_h // ms_h
            scale_w_lms = lms_w // ms_w
            scale_h_gt = gt_h // ms_h
            scale_w_gt = gt_w // ms_w

            lr_h_positions = _build_patch_positions(ms_h, self.lr_patch_size, self.lr_patch_stride)
            lr_w_positions = _build_patch_positions(ms_w, self.lr_patch_size, self.lr_patch_stride)

            lms_patch_h = self.lr_patch_size * scale_h_lms
            lms_patch_w = self.lr_patch_size * scale_w_lms
            gt_patch_h = self.lr_patch_size * scale_h_gt
            gt_patch_w = self.lr_patch_size * scale_w_gt

            for top in lr_h_positions:
                for left in lr_w_positions:
                    self.patch_records.append(
                        {
                            "file_idx": file_idx,
                            "ms_top": int(top),
                            "ms_left": int(left),
                            "ms_h": int(self.lr_patch_size),
                            "ms_w": int(self.lr_patch_size),
                            "lms_top": int(top * scale_h_lms),
                            "lms_left": int(left * scale_w_lms),
                            "lms_h": int(lms_patch_h),
                            "lms_w": int(lms_patch_w),
                            "gt_top": int(top * scale_h_gt),
                            "gt_left": int(left * scale_w_gt),
                            "gt_h": int(gt_patch_h),
                            "gt_w": int(gt_patch_w),
                        }
                    )

    def _load_cached_file(self, file_idx: int):
        if self._cache_file_idx == int(file_idx):
            return self._cache_ms, self._cache_lms, self._cache_gt
        ms, lms, gt = _load_triplet_full(self.image_paths[file_idx])
        self._cache_file_idx = int(file_idx)
        self._cache_ms = ms
        self._cache_lms = lms
        self._cache_gt = gt
        return ms, lms, gt

    def __getitem__(self, index):
        if self.patch_mode:
            record = self.patch_records[index]
            ms_full, lms_full, gt_full = self._load_cached_file(record["file_idx"])
            ms = ms_full[
                record["ms_top"] : record["ms_top"] + record["ms_h"],
                record["ms_left"] : record["ms_left"] + record["ms_w"],
                :,
            ]
            lms = lms_full[
                record["lms_top"] : record["lms_top"] + record["lms_h"],
                record["lms_left"] : record["lms_left"] + record["lms_w"],
                :,
            ]
            gt = gt_full[
                record["gt_top"] : record["gt_top"] + record["gt_h"],
                record["gt_left"] : record["gt_left"] + record["gt_w"],
                :,
            ]
            ms_t = _to_tensor_chw(ms)
            lms_t = _to_tensor_chw(lms)
            gt_t = _to_tensor_chw(gt)
            ms_t, lms_t, gt_t = _apply_random_flip_triplet(
                ms_t,
                lms_t,
                gt_t,
                enable_hflip=self.enable_hflip,
                enable_vflip=self.enable_vflip,
            )
            return ms_t, lms_t, gt_t

        if self.preload_in_memory:
            ms, lms, gt = self.samples[index]
        else:
            ms, lms, gt = _load_triplet_full(self.image_paths[index])

        ms_t = _to_tensor_chw(ms)
        lms_t = _to_tensor_chw(lms)
        gt_t = _to_tensor_chw(gt)
        ms_t, lms_t, gt_t = _apply_random_flip_triplet(
            ms_t,
            lms_t,
            gt_t,
            enable_hflip=self.enable_hflip,
            enable_vflip=self.enable_vflip,
        )
        return ms_t, lms_t, gt_t

    def __len__(self):
        if self.patch_mode:
            return len(self.patch_records)
        return len(self.image_paths)


class DistributedContiguousSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, drop_last=False):
        if num_replicas is None:
            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError("DistributedContiguousSampler 需要已初始化的分布式环境。")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError("DistributedContiguousSampler 需要已初始化的分布式环境。")
            rank = dist.get_rank()
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"rank 非法: {rank}, num_replicas={num_replicas}")

        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.drop_last = bool(drop_last)
        self.dataset_len = len(dataset)

        if self.drop_last:
            self.num_samples = self.dataset_len // self.num_replicas
        else:
            self.num_samples = int(math.ceil(self.dataset_len / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(self.dataset_len))
        if self.drop_last:
            indices = indices[: self.total_size]
        else:
            if len(indices) == 0:
                return iter([])
            padding_size = self.total_size - len(indices)
            if padding_size > 0:
                indices += [indices[-1]] * padding_size

        start = self.rank * self.num_samples
        end = start + self.num_samples
        return iter(indices[start:end])

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        _ = epoch


def pad_pair_to_window(ms, lms, n_scale, window_size=8):
    _, _, h_lr, w_lr = ms.shape
    target_h_lr = int(math.ceil(h_lr / window_size) * window_size)
    target_w_lr = int(math.ceil(w_lr / window_size) * window_size)
    pad_h_lr = target_h_lr - h_lr
    pad_w_lr = target_w_lr - w_lr

    if pad_h_lr == 0 and pad_w_lr == 0:
        return ms, lms

    h_hr, w_hr = lms.shape[2], lms.shape[3]
    scale_h = h_hr // h_lr if (h_lr > 0 and h_hr % h_lr == 0) else n_scale
    scale_w = w_hr // w_lr if (w_lr > 0 and w_hr % w_lr == 0) else n_scale

    pad_h_hr = pad_h_lr * scale_h
    pad_w_hr = pad_w_lr * scale_w
    ms_pad = F.pad(ms, (0, pad_w_lr, 0, pad_h_lr), mode="reflect")
    lms_pad = F.pad(lms, (0, pad_w_hr, 0, pad_h_hr), mode="reflect")
    return ms_pad, lms_pad


def crop_to_ref(output, ref):
    return output[:, :, :ref.shape[2], :ref.shape[3]]


def build_loader(dataset, batch_size, shuffle, args, distributed=None):
    if distributed is None:
        distributed = bool(getattr(getattr(args, "system", None), "distributed_runtime", 0))
    data_cfg = getattr(args, "data", args)
    sampler = None
    if distributed:
        if bool(getattr(dataset, "patch_mode", False)) and not bool(shuffle):
            sampler = DistributedContiguousSampler(dataset, drop_last=False)
        else:
            sampler = DistributedSampler(dataset, shuffle=shuffle)

    num_workers = int(data_cfg.num_workers)
    if bool(getattr(dataset, "patch_mode", False)) and not bool(shuffle):
        num_workers = 0

    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": bool(data_cfg.pin_memory),
        "persistent_workers": bool(data_cfg.persistent_workers) if num_workers > 0 else False,
        "prefetch_factor": int(data_cfg.prefetch_factor) if num_workers > 0 else None,
        "timeout": int(data_cfg.dataloader_timeout) if num_workers > 0 else 0,
        "multiprocessing_context": data_cfg.mp_context if num_workers > 0 else None,
        "drop_last": bool(data_cfg.drop_last) if shuffle else False,
    }
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle if sampler is None else False),
        sampler=sampler,
        **loader_kwargs,
    )
