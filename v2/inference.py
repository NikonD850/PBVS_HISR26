import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Keep local project modules ahead of same-name modules from sibling repos.
PROJECT_ROOT = str(Path(__file__).resolve().parent)
if PROJECT_ROOT in sys.path:
    sys.path.remove(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from data import pad_pair_to_window
from model import build_model
from options import get_options


SUPPORTED_SUFFIX = (".h5", ".hdf5")


def parse_args():
    parser = argparse.ArgumentParser(description="VolFormer v2 H5 inference (single-key input/output).")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/mnt/afs/users/fandawei/data/PBSR_HISR/final_test",
        help="输入 h5 目录（每个文件仅一个 key，默认 50x50x224）。",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./result/v2",
        help="输出目录（默认: ./result/v2）。",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="./checkpoints_v2/volformer_v2_epoch_0030.pth",
        help="checkpoint 路径（默认 epoch 30）。",
    )
    parser.add_argument(
        "--model_file",
        type=str,
        default="origin",
        help="模型架构文件名，对应 model/archs/<name>.py。",
    )
    parser.add_argument(
        "--save_key",
        type=str,
        default="",
        help="输出 key。默认空字符串表示与输入 key 同名。",
    )
    parser.add_argument(
        "--save_dtype",
        type=str,
        choices=["uint8", "float32"],
        default="uint8",
        help="输出数据类型。uint8 会将网络输出按 [0,1] 映射到 [0,255]。",
    )
    parser.add_argument(
        "--strict",
        type=int,
        default=1,
        help="load_state_dict 是否严格匹配（1/0）。",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="",
        help="CUDA_VISIBLE_DEVICES，默认空字符串表示不改环境变量。",
    )
    return parser.parse_args()


def _is_h5_file(path: Path):
    return path.suffix.lower() in SUPPORTED_SUFFIX


def _normalize_to_float01(array: np.ndarray):
    np_array = np.asarray(array)
    orig_dtype = np_array.dtype
    np_array = np_array.astype(np.float32)

    should_divide = np.issubdtype(orig_dtype, np.integer)
    if not should_divide:
        finite_mask = np.isfinite(np_array)
        if np.any(finite_mask):
            if float(np.max(np_array[finite_mask])) > 1.5:
                should_divide = True

    if should_divide:
        np_array = np_array / 255.0
    return np_array


def _read_single_key_h5(path: Path):
    with h5py.File(str(path), "r") as h5f:
        keys = list(h5f.keys())
        if len(keys) != 1:
            raise RuntimeError(f"{path} 期望只有一个 key，实际 keys={keys}")
        key = keys[0]
        data = np.asarray(h5f[key])
    return key, data


def _to_tensor_bchw(array_hwc: np.ndarray):
    chw = np.ascontiguousarray(array_hwc.transpose(2, 0, 1))
    tensor = torch.from_numpy(chw).unsqueeze(0).float()
    return tensor


def _to_hwc(array_bchw: torch.Tensor):
    # array_bchw: [1, C, H, W]
    hwc = array_bchw.squeeze(0).permute(1, 2, 0).contiguous().cpu().numpy()
    return hwc


def _resolve_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        if "model_state_dict" in ckpt_obj and isinstance(ckpt_obj["model_state_dict"], dict):
            return ckpt_obj["model_state_dict"]
        if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
            return ckpt_obj["state_dict"]
    if isinstance(ckpt_obj, dict):
        return ckpt_obj
    raise RuntimeError("无法从 checkpoint 解析 state_dict。")


def _strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    has_module_prefix = any(k.startswith("module.") for k in state_dict.keys())
    if not has_module_prefix:
        return state_dict
    return {k[len("module.") :]: v for k, v in state_dict.items()}


def build_network(device: torch.device, args):
    options = get_options()
    options.model.model_file = args.model_file

    gpu_count = torch.cuda.device_count() if device.type == "cuda" else 1
    gpu_count = max(1, int(gpu_count))
    model, _ = build_model(options=options, device=device, gpu_count=gpu_count)

    ckpt_obj = torch.load(args.ckpt_path, map_location=device)
    state_dict = _resolve_state_dict(ckpt_obj)
    state_dict = _strip_module_prefix(state_dict)
    load_info = model.load_state_dict(state_dict, strict=bool(args.strict))
    if hasattr(load_info, "missing_keys") and hasattr(load_info, "unexpected_keys"):
        print(f"load_state_dict: missing_keys={len(load_info.missing_keys)}, unexpected_keys={len(load_info.unexpected_keys)}")

    model.eval()
    return model, options


@torch.no_grad()
def run_single(model, options, input_array_hwc: np.ndarray, device: torch.device):
    ms = _normalize_to_float01(input_array_hwc)
    ms_t = _to_tensor_bchw(ms).to(device, non_blocking=True)

    # 训练时第二输入为 4x bicubic，上采样后尺寸应为 200x200（对于 50x50 输入）。
    lms_t = F.interpolate(
        ms_t,
        scale_factor=options.model.n_scale,
        mode="bicubic",
        align_corners=False,
    )

    ms_pad, lms_pad = pad_pair_to_window(ms_t, lms_t, n_scale=options.model.n_scale, window_size=8)
    img_size = ms_pad.shape[2:4]

    pred = model(ms_pad, lms_pad, modality="spectral", img_size=img_size)
    pred = pred[:, :, : lms_t.shape[2], : lms_t.shape[3]]
    pred_hwc = _to_hwc(pred)
    return pred_hwc


def convert_output_dtype(pred_hwc: np.ndarray, save_dtype: str):
    if save_dtype == "float32":
        return pred_hwc.astype(np.float32)
    pred_uint8 = np.clip(np.rint(pred_hwc * 255.0), 0, 255).astype(np.uint8)
    return pred_uint8


def main():
    args = parse_args()

    if args.gpus.strip() != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus.strip()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    ckpt_path = Path(args.ckpt_path)

    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir 不存在: {input_dir}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"ckpt_path 不存在: {ckpt_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in input_dir.iterdir() if p.is_file() and _is_h5_file(p)])
    if len(files) == 0:
        raise RuntimeError(f"{input_dir} 下未找到 h5/hdf5 文件。")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Input dir: {input_dir}")
    print(f"Output dir: {out_dir}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Num files: {len(files)}")

    model, options = build_network(device=device, args=args)

    for path in tqdm(files, desc="Inference", ncols=120):
        in_key, input_data = _read_single_key_h5(path)
        if input_data.ndim != 3:
            raise RuntimeError(f"{path} 输入必须是 3 维，当前 shape={input_data.shape}")

        pred_hwc = run_single(model=model, options=options, input_array_hwc=input_data, device=device)
        output_data = convert_output_dtype(pred_hwc, save_dtype=args.save_dtype)
        out_key = args.save_key if args.save_key != "" else in_key

        out_path = out_dir / path.name
        with h5py.File(str(out_path), "w") as h5f:
            h5f.create_dataset(out_key, data=output_data)

    print("Done.")


if __name__ == "__main__":
    main()
