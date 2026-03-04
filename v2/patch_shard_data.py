import bisect
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


SUPPORTED_SUFFIX = (".h5", ".hdf5")


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


class PatchShardDataset(Dataset):
    def __init__(self, shard_dir, total_num=None, augment=False, hflip=True, vflip=True):
        super().__init__()
        self.patch_mode = False
        self.enable_hflip = bool(augment) and bool(hflip)
        self.enable_vflip = bool(augment) and bool(vflip)

        self.shard_dir = Path(shard_dir)
        if not self.shard_dir.exists():
            raise FileNotFoundError(f"shard_dir 不存在: {self.shard_dir}")

        all_paths = sorted(
            [path for path in self.shard_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIX]
        )
        if len(all_paths) == 0:
            raise RuntimeError(f"{self.shard_dir} 下未找到 h5/hdf5 shard 文件。")

        self.shard_paths = []
        self.shard_lengths = []
        self.cumulative_lengths = []
        running = 0

        for path in all_paths:
            with h5py.File(str(path), "r") as raw:
                for key in ("ms", "lms", "gt"):
                    if key not in raw:
                        raise KeyError(f"{path} 缺少 key={key}")

                ms_shape = tuple(raw["ms"].shape)
                lms_shape = tuple(raw["lms"].shape)
                gt_shape = tuple(raw["gt"].shape)
                if len(ms_shape) != 4 or len(lms_shape) != 4 or len(gt_shape) != 4:
                    raise ValueError(f"{path} 数据维度必须是 [N,H,W,C]，当前 ms={ms_shape}, lms={lms_shape}, gt={gt_shape}")
                if ms_shape[0] != lms_shape[0] or ms_shape[0] != gt_shape[0]:
                    raise ValueError(f"{path} 样本数不一致: ms={ms_shape[0]}, lms={lms_shape[0]}, gt={gt_shape[0]}")
                sample_count = int(ms_shape[0])

            if sample_count <= 0:
                continue

            self.shard_paths.append(path)
            self.shard_lengths.append(sample_count)
            running += sample_count
            self.cumulative_lengths.append(running)

        if len(self.shard_paths) == 0:
            raise RuntimeError(f"{self.shard_dir} 下的 shard 文件都为空。")

        self.total_len = running
        if isinstance(total_num, int):
            self.total_len = min(self.total_len, max(0, int(total_num)))
        if self.total_len <= 0:
            raise RuntimeError("total_num 设置后数据集为空。")

        self._handles = {}

    def __len__(self):
        return self.total_len

    def _locate_index(self, index):
        idx = int(index)
        if idx < 0:
            idx += self.total_len
        if idx < 0 or idx >= self.total_len:
            raise IndexError(f"index 越界: {index}")

        shard_idx = bisect.bisect_right(self.cumulative_lengths, idx)
        start = 0 if shard_idx == 0 else self.cumulative_lengths[shard_idx - 1]
        local_idx = idx - start
        return shard_idx, local_idx

    def _get_handle(self, shard_idx):
        handle = self._handles.get(shard_idx)
        if handle is None:
            handle = h5py.File(str(self.shard_paths[shard_idx]), "r")
            self._handles[shard_idx] = handle
        return handle

    def __getitem__(self, index):
        shard_idx, local_idx = self._locate_index(index)
        handle = self._get_handle(shard_idx)

        ms = _normalize_to_float(np.asarray(handle["ms"][local_idx]))
        lms = _normalize_to_float(np.asarray(handle["lms"][local_idx]))
        gt = _normalize_to_float(np.asarray(handle["gt"][local_idx]))
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

    def close(self):
        for handle in self._handles.values():
            try:
                handle.close()
            except Exception:
                pass
        self._handles.clear()

    def __del__(self):
        self.close()
