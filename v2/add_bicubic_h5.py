import os
import h5py
import cv2
import numpy as np
from tqdm import tqdm
import argparse

def bicubic_upsample(LR, scale=4):
    """
    LR: numpy array, H x W x C
    返回: H*scale x W*scale x C, uint8
    """
    H, W, C = LR.shape
    HR_up = np.zeros((H*scale, W*scale, C), dtype=np.uint8)
    for c in range(C):
        HR_up[:, :, c] = cv2.resize(LR[:, :, c], (W*scale, H*scale), interpolation=cv2.INTER_CUBIC)
    return HR_up

def process_h5_file(in_path, out_path):
    with h5py.File(in_path, 'r') as f:
        lr = f['LR_uint8'][...]   # H x W x 224
        hr = f['HR_uint8'][...]   # 4H x 4W x 224

    # 生成 LR_4x
    lr_4x = bicubic_upsample(lr, scale=4)

    # 保存到新 h5
    with h5py.File(out_path, 'w') as f:
        f.create_dataset('LR_uint8', data=lr, dtype='uint8')
        f.create_dataset('HR_uint8', data=hr, dtype='uint8')
        f.create_dataset('LR_4x_uint8', data=lr_4x, dtype='uint8')

def batch_process_h5(in_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    files = [f for f in os.listdir(in_dir) if f.endswith('.h5')]
    for file in tqdm(files):
        in_path = os.path.join(in_dir, file)
        out_path = os.path.join(out_dir, file)
        process_h5_file(in_path, out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LR_4x_uint8 for H5 datasets")
    parser.add_argument('--in_dir', type=str, required=True, help="Path to original h5 directory")
    parser.add_argument('--out_dir', type=str, required=True, help="Path to save new h5 files")
    args = parser.parse_args()

    batch_process_h5(args.in_dir, args.out_dir)

