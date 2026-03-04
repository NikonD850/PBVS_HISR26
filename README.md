## Introduction

This project is modified from [VolFormer](https://github.com/yudadabing/VolFormer).

Based on the original project, we developed two architectures, named v1 and v2, and trained them separately to evaluate their performance, resulting in two pre-trained models.

Considering that a single model may have limited stability, we further improved it by fusing the inference results.

## Installation

```bash
conda create -n volformer2 python=3.10 -y
conda activate volformer2
conda install -c conda-forge gdal=3.12.1
pip install -r requirements.txt
```

## Inference

### H5 Generation

You can download our pre-trained models (v1 & v2) on [Google Drive](https://drive.google.com/drive/folders/10Zys2Wk3QavvLRmKm4gqgJC4FuUgyxW8).

Put `v1.pth` into `./v1/checkpoints`, `v2.pth` into `./v2/checkpoints`, `final_test` dataset into `./datasets`.

Run `./v1/test_h5_no_gdal.py` for v1 inference:

```bash
cd ./v1
python test_h5_no_gdal.py \
  --ckpt ./checkpoints/v1.pth \
  --test_dir ../datasets/final_test \
  --save_dir ./result/v1
```

Run `./v2/inference.py` for v2 inference:

```bash
cd ./v2
python inference.py \
  --ckpt ./checkpoints/v2.pth \
  --input_dir ../datasets/final_test \
  --out_dir ./result/v2
```

### H5 Fusion

Run `./merge_h5_weighted.py` to fuse h5 files:

```bash
python merge_h5_weighted.py \
  --input_a ./v1/result/v1 \
  --input_b ./v2/result/v2 \
  --weight_a 0.4 --weight_b 0.6 \
  --out_path ./result/v1_v2 \
  --strict_files 1 --strict_keys 1
```

The v1 model performs moderately in terms of PSNR but achieves better SAM metrics, so it is used as an auxiliary model. Meanwhile, the v2 model demonstrates strong performance in both PSNR and SAM. Thus, the weights are set to 0.4 (v1) and 0.6 (v2).

ZIP file will be automatically generated under the `./result` directory for submission.

## Data Preparation

### Model v1

Model v1 uses `TIFF (.tif)` files for pre-training and `HDF5 (.h5)` files for fine-tuning.

**TIFF**: `{name}_LR.tif` & `{name}_HR.tif` are paired, with LR × 4 = HR.

**HDF5**: Contains `LR_uint8: (H, W, C)` & `HR_uint8: (H*4, W*4, C)`.

Use `h5_to_tiff.py` to convert from `HDF5` to `TIFF`:

```bash
cd ./v1
python h5_to_tiff.py
```

Place the datasets under `v1/datasets` as the following structure:

```plaintext
v1/
├── datasets/
│   ├── tiff/              # TIFF (Pre-train)
│   │   ├── train/
│   │   │   ├── Scene_1_LR.tif
│   │   │   ├── Scene_1_HR.tif
│   │   │   └── ...
│   │   └── test/
│   └── h5/                # HDF5 (Finetune)
│       ├── train/
│       │   └── Scene_1.h5
│       └── test/
......
```

### Model v2

Build patch shards using `build_patch_shards.py`:

```bash
cd ./v2
python build_patch_shards.py
```

Place the original h5 files and patch shards under `v2/datasets` as the following structure:

```plaintext
v2/
├── datasets/
│   ├── train/             # Original h5 files for training
│   │   └── *.h5
│   ├── val/               # Original h5 files for validating
│   │   └── *.h5
│   └── patch_shards/      # Output of build_patch_shards.py
│       ├── train/
│       │   ├── shard_*.h5
│       │   └── manifest.json
│       └── val/
......
```

## Training

### Model v1

```bash
cd ./v1

# Visualization (optional)
tensorboard --logdir ./runs/

# Base model training
python mains.py train

# Fine-tuning
python finetune_tensor.py

# Column-wise structure + Edge-aware fine-tuning
python finetune_column_edge.py

# Fast fine-tuning with SAM optimization
python finetune_sam_h5_fast.py
```

### Model v2

```bash
cd ./v2

# Training & fine-tuning
torchrun train.py --nproc_per_node=4
```

