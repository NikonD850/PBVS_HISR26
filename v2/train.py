import json
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

# Reduce CUDA allocator fragmentation unless user explicitly overrides it.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from tqdm import tqdm

# Keep local project modules ahead of same-name modules from sibling repos.
PROJECT_ROOT = str(Path(__file__).resolve().parent)
if PROJECT_ROOT in sys.path:
    sys.path.remove(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from data import build_loader, crop_to_ref, loadingData, pad_pair_to_window
from loss import build_loss
from metrics import compute_psnr_sam
from model import build_model
from options import dump_options, get_options
from patch_shard_data import PatchShardDataset


def is_main_process(options):
    return not bool(options.system.distributed_runtime) or int(options.system.local_rank) == 0


def set_seed(seed: int, use_cuda: bool):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)


def setup_distributed(options):
    system = options.system
    use_cuda = bool(system.cuda) and torch.cuda.is_available()
    if bool(system.cuda) and not use_cuda:
        raise RuntimeError("你设置了 cuda=1，但当前环境未检测到 CUDA。")

    local_rank = int(os.environ.get("LOCAL_RANK", system.local_rank))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = bool(system.distributed) and use_cuda and world_size > 1

    if distributed:
        if local_rank < 0:
            raise RuntimeError("分布式模式下缺少 LOCAL_RANK。请使用 torchrun 启动。")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=system.dist_backend, init_method=system.dist_url)
        system.local_rank = local_rank
        system.distributed_runtime = 1
        device = torch.device("cuda", local_rank)
        gpu_count = world_size
    else:
        system.local_rank = 0
        system.distributed_runtime = 0
        device = torch.device("cuda" if use_cuda else "cpu")
        gpu_count = torch.cuda.device_count() if use_cuda else 1
        gpu_count = max(1, gpu_count)

    return device, use_cuda, gpu_count


def setup_speed_flags(options, use_cuda):
    system = options.system
    cudnn.benchmark = bool(system.cudnn_benchmark)
    cudnn.deterministic = bool(system.cudnn_deterministic)

    if use_cuda and bool(system.tf32):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def maybe_compile_model(model, options):
    if not bool(options.system.compile):
        return model
    if not hasattr(torch, "compile"):
        return model
    return torch.compile(model, mode=options.system.compile_mode)


def move_batch_to_device(ms, lms, gt, device):
    return (
        ms.to(device, non_blocking=True),
        lms.to(device, non_blocking=True),
        gt.to(device, non_blocking=True),
    )


def get_autocast_dtype(options):
    amp_dtype = str(options.system.amp_dtype).lower()
    if amp_dtype == "bf16":
        return torch.bfloat16
    return torch.float16


def reduce_scalar(value, device, distributed):
    tensor = torch.tensor([float(value)], device=device, dtype=torch.float64)
    if distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= dist.get_world_size()
    return float(tensor.item())


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    ckpt_path,
    val_psnr,
    val_sam,
    step_in_epoch=0,
    global_step=0,
):
    core_model = model.module if hasattr(model, "module") else model
    payload = {
        "epoch": int(epoch),
        "step_in_epoch": int(step_in_epoch),
        "global_step": int(global_step),
        "model_state_dict": core_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "val_mpsnr": float(val_psnr),
        "val_sam": float(val_sam),
        "saved_time": time.ctime(),
    }
    torch.save(payload, ckpt_path)

def maybe_resume(model, optimizer, scheduler, options, device):
    resume_path = options.paths.resume_path
    strict = bool(options.paths.resume_strict)
    path = None
    if isinstance(resume_path, str) and resume_path.strip() != "":
        path = Path(resume_path.strip())
        if not path.exists():
            raise FileNotFoundError(f"resume checkpoint 不存在: {path}")

    if path is None:
        return 1, 0

    if is_main_process(options):
        print(f"Resume checkpoint: {path}")

    ckpt = torch.load(str(path), map_location=device)
    state_dict = None
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "model" in ckpt and hasattr(ckpt["model"], "state_dict"):
            state_dict = ckpt["model"].state_dict()
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
    if state_dict is None:
        raise RuntimeError("无法从 resume checkpoint 解析 model state_dict。")

    core_model = model.module if hasattr(model, "module") else model
    load_info = core_model.load_state_dict(state_dict, strict=strict)
    if hasattr(load_info, "missing_keys") and hasattr(load_info, "unexpected_keys") and is_main_process(options):
        print(
            f"resume load missing_keys={len(load_info.missing_keys)}, "
            f"unexpected_keys={len(load_info.unexpected_keys)}"
        )

    if isinstance(ckpt, dict) and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if isinstance(ckpt, dict) and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    last_epoch = int(ckpt.get("epoch", 0)) if isinstance(ckpt, dict) else 0
    step_in_epoch = int(ckpt.get("step_in_epoch", 0)) if isinstance(ckpt, dict) else 0
    global_step = int(ckpt.get("global_step", 0)) if isinstance(ckpt, dict) else 0
    if step_in_epoch > 0:
        return max(1, last_epoch), max(0, global_step)
    return max(1, last_epoch + 1), max(0, global_step)


def create_output_dirs(options):
    ckpt_dir = Path(options.paths.ckpt_dir)
    log_dir = Path(options.paths.log_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir, log_dir


def train():
    options = get_options()
    os.environ["CUDA_VISIBLE_DEVICES"] = options.system.gpus

    device, use_cuda, gpu_count = setup_distributed(options)
    setup_speed_flags(options, use_cuda)
    set_seed(options.system.seed, use_cuda)

    per_gpu_batch_size = int(options.train.batch_size_per_gpu)
    global_batch_size = per_gpu_batch_size * gpu_count
    debug_mode = bool(options.train.debug_mode)
    debug_train_batches = max(1, int(options.train.debug_train_batches))
    debug_val_batches = max(1, int(options.train.debug_val_batches))
    iter_ckpt_interval_steps = max(0, int(getattr(options.train, "iter_ckpt_interval_steps", 1000)))
    validation_enabled = int(options.train.val_interval) > 0
    val_start_epoch_effective = 1 if debug_mode else int(options.train.val_start_epoch)
    is_main = is_main_process(options)

    if is_main:
        data_backend = str(getattr(options.data, "dataset_backend", "legacy")).strip().lower()
        print(f"Device: {device}")
        print(f"Visible GPUs: {options.system.gpus}")
        print(
            f"Batch config: batch_size_per_gpu={per_gpu_batch_size}, "
            f"gpus={gpu_count}, global_batch_size={global_batch_size}"
        )
        if debug_mode:
            print(
                f"Debug mode: on (train_batches={debug_train_batches}, "
                f"val_batches={debug_val_batches})"
            )
        print(f"Data backend: {data_backend}")
        if data_backend == "patch_shard":
            print(f"Train shard dir: {options.data.train_shard_dir}")
            print(f"Val shard dir: {options.data.val_shard_dir}")
        else:
            load_mode = "preload_in_memory" if bool(options.data.preload_in_memory) else "from_file"
            print(f"Data load mode: {load_mode}")
            if bool(options.data.train_patch_mode):
                print(
                    "Train patch mode: on "
                    f"(lr_patch={options.data.train_lr_patch_size}, "
                    f"stride={options.data.train_lr_patch_stride})"
                )
        print(
            f"Train augment: hflip={bool(options.data.train_hflip)}, "
            f"vflip={bool(options.data.train_vflip)}"
        )
        if validation_enabled:
            print(
                f"Val schedule: start@{val_start_epoch_effective}, "
                f"every {options.train.val_interval} epochs"
            )
        else:
            print("Validation: disabled (val_interval<=0)")
        print(f"Model file: {options.model.model_file}")

    data_backend = str(getattr(options.data, "dataset_backend", "legacy")).strip().lower()
    if data_backend == "patch_shard":
        train_shard_dir = str(getattr(options.data, "train_shard_dir", "")).strip()
        val_shard_dir = str(getattr(options.data, "val_shard_dir", "")).strip()
        if train_shard_dir == "":
            raise RuntimeError("使用 patch_shard 后端时，必须设置 data.train_shard_dir。")
        if validation_enabled and val_shard_dir == "":
            raise RuntimeError("启用验证时，patch_shard 后端必须设置 data.val_shard_dir。")
        train_set = PatchShardDataset(
            shard_dir=train_shard_dir,
            total_num=options.data.data_train_num,
            augment=True,
            hflip=bool(options.data.train_hflip),
            vflip=bool(options.data.train_vflip),
        )
        val_set = None
        if validation_enabled:
            val_set = PatchShardDataset(
                shard_dir=val_shard_dir,
                total_num=options.data.data_val_num,
                augment=False,
                hflip=False,
                vflip=False,
            )
        train_shuffle = True
    else:
        train_set = loadingData(
            image_dir=options.data.train_dir_mslabel,
            augment=True,
            total_num=options.data.data_train_num,
            lr_patch_size=options.data.train_lr_patch_size,
            lr_patch_stride=options.data.train_lr_patch_stride,
            preload_in_memory=bool(options.data.preload_in_memory),
            patch_mode=bool(options.data.train_patch_mode),
            hflip=bool(options.data.train_hflip),
            vflip=bool(options.data.train_vflip),
        )
        val_set = None
        if validation_enabled:
            val_set = loadingData(
                image_dir=options.data.val_dir_ms,
                augment=False,
                total_num=options.data.data_val_num,
                lr_patch_size=options.data.train_lr_patch_size,
                lr_patch_stride=options.data.train_lr_patch_stride,
                preload_in_memory=bool(options.data.preload_in_memory),
                patch_mode=False,
                hflip=False,
                vflip=False,
            )
        train_shuffle = not bool(options.data.train_patch_mode)

    if is_main:
        print(f"Train samples: {len(train_set)}")
        if val_set is not None:
            print(f"Val samples: {len(val_set)}")
        else:
            print("Val samples: skipped (validation disabled)")

    train_loader = build_loader(train_set, batch_size=per_gpu_batch_size, shuffle=train_shuffle, args=options, distributed=bool(options.system.distributed_runtime))
    val_loader = None
    if val_set is not None:
        val_loader = build_loader(val_set, batch_size=per_gpu_batch_size, shuffle=False, args=options, distributed=False)

    if is_main:
        print(f"Train loader workers: {int(getattr(train_loader, 'num_workers', 0))}")
        print(f"Train shuffle: {bool(train_shuffle)}")

    model, _ = build_model(options=options, device=device, gpu_count=gpu_count)
    model = maybe_compile_model(model, options)

    if bool(options.system.distributed_runtime):
        model = DDP(
            model,
            device_ids=[options.system.local_rank],
            output_device=options.system.local_rank,
            find_unused_parameters=True,
        )

    criterion = build_loss()
    optimizer = Adam(model.parameters(), lr=options.train.learning_rate, weight_decay=options.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, options.train.epochs),
        eta_min=options.train.min_learning_rate,
    )

    amp_enabled = bool(options.system.amp) and use_cuda
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled and get_autocast_dtype(options) == torch.float16)

    ckpt_dir, log_dir = create_output_dirs(options)
    start_epoch, global_step = maybe_resume(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        options=options,
        device=device,
    )
    history_path = log_dir / f"{options.paths.run_name}_history.jsonl"
    summary_path = log_dir / f"{options.paths.run_name}_summary.json"
    config_path = log_dir / f"{options.paths.run_name}_options.json"

    if is_main:
        print(f"Iter checkpoint interval (steps): {iter_ckpt_interval_steps}")
        dump_options(str(config_path), options)

    best_psnr = None
    best_sam = None
    best_epoch = None

    for epoch in range(start_epoch, options.train.epochs + 1):
        if hasattr(train_loader, "sampler") and train_loader.sampler is not None and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        model.train()
        epoch_loss_sum = 0.0
        step_count = 0

        iterator = tqdm(train_loader, desc=f"Train {epoch}/{options.train.epochs}", ncols=120, disable=not is_main)
        for batch_idx, (ms, lms, gt) in enumerate(iterator):
            if debug_mode and batch_idx >= debug_train_batches:
                break
            ms, lms, gt = move_batch_to_device(ms, lms, gt, device)

            optimizer.zero_grad(set_to_none=True)
            ms_pad, lms_pad = pad_pair_to_window(ms, lms, n_scale=options.model.n_scale, window_size=8)
            img_size = ms_pad.shape[2:4]

            autocast_dtype = get_autocast_dtype(options)
            with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=autocast_dtype):
                pred = model(ms_pad, lms_pad, modality="spectral", img_size=img_size)
                pred = crop_to_ref(pred, gt)
                loss = criterion(pred, gt)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            epoch_loss_sum += float(loss.item())
            step_count += 1
            global_step += 1
            if is_main:
                iterator.set_postfix(loss=f"{loss.item():.6f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
                if iter_ckpt_interval_steps > 0 and global_step % iter_ckpt_interval_steps == 0:
                    iter_ckpt_path = ckpt_dir / f"{options.paths.run_name}_iter_{global_step:08d}.pth"
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        ckpt_path=str(iter_ckpt_path),
                        val_psnr=float("nan"),
                        val_sam=float("nan"),
                        step_in_epoch=step_count,
                        global_step=global_step,
                    )
                    print(
                        f"[CKPT] iter={global_step} epoch={epoch} "
                        f"step_in_epoch={step_count} ckpt={iter_ckpt_path}"
                    )

        if is_main:
            iterator.close()

        scheduler.step()

        avg_train_loss_local = epoch_loss_sum / max(1, step_count)
        avg_train_loss = reduce_scalar(avg_train_loss_local, device=device, distributed=bool(options.system.distributed_runtime))

        if is_main:
            print(f"Epoch {epoch}: avg_train_loss={avg_train_loss:.6f}")

        should_validate = (
            validation_enabled
            and val_loader is not None
            and epoch >= val_start_epoch_effective
            and (epoch - val_start_epoch_effective) % options.train.val_interval == 0
        )
        val_psnr = float("nan")
        val_sam = float("nan")
        val_count = 0

        if should_validate:
            val_psnr, val_sam, val_count = compute_psnr_sam(
                model=model,
                val_loader=val_loader,
                device=device,
                n_scale=options.model.n_scale,
                use_amp=amp_enabled,
                show_progress=is_main,
                max_batches=(debug_val_batches if debug_mode else None),
            )

            val_psnr = reduce_scalar(val_psnr, device=device, distributed=bool(options.system.distributed_runtime))
            val_sam = reduce_scalar(val_sam, device=device, distributed=bool(options.system.distributed_runtime))

            if is_main:
                record = {
                    "epoch": int(epoch),
                    "avg_train_loss": float(avg_train_loss),
                    "val_mpsnr": float(val_psnr),
                    "val_sam": float(val_sam),
                    "val_files": int(val_count),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "ckpt": str(ckpt_dir / f"{options.paths.run_name}_epoch_{epoch:04d}.pth"),
                    "time": time.ctime(),
                }
                with history_path.open("a", encoding="utf-8") as file:
                    file.write(json.dumps(record, ensure_ascii=False) + "\n")

                if best_psnr is None or val_psnr > best_psnr:
                    best_psnr = val_psnr
                    best_sam = val_sam
                    best_epoch = epoch

                print(
                    f"[VAL] epoch={epoch} MPSNR={val_psnr:.6f} SAM={val_sam:.6f} "
                    f"files={val_count} ckpt={ckpt_dir / f'{options.paths.run_name}_epoch_{epoch:04d}.pth'}"
                )

        if is_main:
            ckpt_path = ckpt_dir / f"{options.paths.run_name}_epoch_{epoch:04d}.pth"
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                ckpt_path=str(ckpt_path),
                val_psnr=val_psnr,
                val_sam=val_sam,
                step_in_epoch=0,
                global_step=global_step,
            )
            if not should_validate:
                print(f"[CKPT] epoch={epoch} ckpt={ckpt_path} (validation skipped)")

    if is_main:
        summary = {
            "run_name": options.paths.run_name,
            "epochs": int(options.train.epochs),
            "val_start_epoch": int(options.train.val_start_epoch),
            "val_interval": int(options.train.val_interval),
            "validation_enabled": bool(validation_enabled),
            "best_epoch_by_psnr": (int(best_epoch) if best_epoch is not None else None),
            "best_psnr": (float(best_psnr) if best_psnr is not None else None),
            "best_sam_at_best_psnr": (float(best_sam) if best_sam is not None else None),
            "history_path": str(history_path),
            "ckpt_dir": str(ckpt_dir),
            "options_path": str(config_path),
            "finished_time": time.ctime(),
        }
        with summary_path.open("w", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2)

        print("Training finished.")
        print(f"Summary: {summary_path}")

    if bool(options.system.distributed_runtime):
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    train()
