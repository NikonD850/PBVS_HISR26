from dataclasses import asdict, dataclass, field
import json


@dataclass
class SystemOptions:
    cuda: int = 1
    gpus: str = "0,1,2,3"
    seed: int = 3000

    distributed: int = 1
    dist_backend: str = "nccl"
    dist_url: str = "env://"
    local_rank: int = -1
    distributed_runtime: int = 0

    amp: int = 1
    amp_dtype: str = "bf16"

    tf32: int = 1
    cudnn_benchmark: int = 1
    cudnn_deterministic: int = 0

    compile: int = 0
    compile_mode: str = "max-autotune"


@dataclass
class DataOptions:
    dataset_backend: str = "patch_shard"

    train_dir_mslabel: str = "datasets_origin/train"
    val_dir_ms: str = "datasets_origin/test_crop50"
    train_shard_dir: str = "datasets/patch_shards/train"
    val_shard_dir: str = "datasets/patch_shards/test"

    data_train_num: int = 100000
    data_val_num: int = 100000

    preload_in_memory: int = 0
    train_patch_mode: int = 1
    train_lr_patch_size: int = 50
    train_lr_patch_stride: int = 25
    train_hflip: int = 1
    train_vflip: int = 1

    num_workers: int = 16
    pin_memory: int = 1
    persistent_workers: int = 1
    prefetch_factor: int = 1
    dataloader_timeout: int = 120
    mp_context: str = "fork"
    drop_last: int = 0


@dataclass
class ModelOptions:
    model_file: str = "origin"

    dataset_name: str = "Cave"
    n_feats: int = 192
    n_blocks: int = 2
    n_subs: int = 24
    n_ovls: int = 8
    n_scale: int = 4
    n_colors: int = 224

    vf_embed_dim: int = 144
    vf_depth: int = 8
    vf_layers: int = 8
    vf_num_heads: int = 8
    vf_use_checkpoint: int = 0


@dataclass
class TrainOptions:
    epochs: int = 1000
    val_start_epoch: int = 1
    val_interval: int = 0
    batch_size_per_gpu: int = 2
    debug_mode: int = 0
    debug_train_batches: int = 8
    debug_val_batches: int = 8
    iter_ckpt_interval_steps: int = 1000

    learning_rate: float = 3e-5
    min_learning_rate: float = 1e-6
    weight_decay: float = 0.0


@dataclass
class PathOptions:
    ckpt_dir: str = "./checkpoints_v2.5"
    log_dir: str = "./logs_v2.5"
    run_name: str = "volformer_v2"
    resume_path: str = "./checkpoints_v2/volformer_v2_epoch_0030_0031_w07_03.pth"
    resume_strict: int = 1


@dataclass
class Options:
    system: SystemOptions = field(default_factory=SystemOptions)
    data: DataOptions = field(default_factory=DataOptions)
    model: ModelOptions = field(default_factory=ModelOptions)
    train: TrainOptions = field(default_factory=TrainOptions)
    paths: PathOptions = field(default_factory=PathOptions)


def get_options() -> Options:
    return Options()


def dump_options(path: str, options: Options):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(asdict(options), file, ensure_ascii=False, indent=2)
