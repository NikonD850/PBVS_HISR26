import os
import h5py
import argparse
import numpy as np
from tqdm import tqdm

def uniform_indices(L, P, N):
    """均匀 + 整数 + 全覆盖"""
    idx = np.linspace(0, L - P, N)
    idx = np.round(idx).astype(int)
    idx[-1] = L - P
    idx[0] = 0
    return np.unique(idx)   # 防止 round 重复

def process_h5(in_path, out_dir):
    """每个 patch 单独保存"""
    base_name = os.path.splitext(os.path.basename(in_path))[0]

    with h5py.File(in_path, 'r') as f:
        lr = f['LR_uint8'][:]   # (546 or 547, 160, 224)
        hr = f['HR_uint8'][:]   # (2184 or 2188, 640, 224)
        lr4x = f['LR_4x_uint8'][:]

    H_lr, W_lr, C = lr.shape
    assert W_lr == 160
    assert H_lr in (546, 547)

    ys = uniform_indices(H_lr, 50, 21)
    xs = uniform_indices(W_lr, 50, 5)

    os.makedirs(out_dir, exist_ok=True)

    # 遍历所有 patch
    for y_idx, y in enumerate(ys):
        for x_idx, x in enumerate(xs):
            lr_patch = lr[y:y+50, x:x+50, :]
            hr_patch = hr[y*4:(y*4)+200, x*4:(x*4)+200, :]
            lr4x_patch = lr4x[y*4:(y*4)+200, x*4:(x*4)+200, :]

            out_name = f"{base_name}_{x_idx}_{y_idx}.h5"
            out_path = os.path.join(out_dir, out_name)

            with h5py.File(out_path, 'w') as f:
                f.create_dataset('LR_uint8', data=lr_patch, dtype=np.uint8)
                f.create_dataset('HR_uint8', data=hr_patch, dtype=np.uint8)
                f.create_dataset('LR_4x_uint8', data=lr4x_patch, dtype=np.uint8)

def main(args):
    files = sorted([f for f in os.listdir(args.root_in) if f.endswith('.h5')])

    for name in tqdm(files, desc="Extracting patches"):
        in_path = os.path.join(args.root_in, name)
        process_h5(in_path, args.root_out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_in', type=str, required=True)
    parser.add_argument('--root_out', type=str, required=True)
    args = parser.parse_args()
    main(args)
