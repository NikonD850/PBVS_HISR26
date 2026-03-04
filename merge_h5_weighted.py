import argparse
import shutil
import zipfile
from contextlib import ExitStack
from pathlib import Path
from typing import Dict, Optional, Tuple

import h5py
import numpy as np


SUPPORTED_SUFFIX = (".h5", ".hdf5")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Weighted merge for two h5/hdf5 files or directories."
    )
    parser.add_argument(
        "--input_a",
        type=str,
        required=True,
        help="First input h5 file or directory.",
    )
    parser.add_argument(
        "--input_b",
        type=str,
        required=True,
        help="Second input h5 file or directory.",
    )
    parser.add_argument(
        "--weight_a",
        type=float,
        default=0.5,
        help="Weight for input_a. Will be normalized with weight_b.",
    )
    parser.add_argument(
        "--weight_b",
        type=float,
        default=0.5,
        help="Weight for input_b. Will be normalized with weight_a.",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        required=True,
        help="Output h5 file path (file mode) or output directory (dir mode).",
    )
    parser.add_argument(
        "--strict_files",
        type=int,
        default=1,
        help="Dir mode only. 1: require same h5 file set in both dirs; 0: merge by union.",
    )
    parser.add_argument(
        "--strict_keys",
        type=int,
        default=1,
        help="1: require same dataset keys in paired h5; 0: merge by union keys.",
    )
    parser.add_argument(
        "--zip_name",
        type=str,
        default=None,
        help="Name of the output zip file (without .zip extension). If not provided, uses the basename of --out_path.",
    )
    return parser.parse_args()


def _is_h5_file(path: Path):
    return path.suffix.lower() in SUPPORTED_SUFFIX


def _normalize_weights(weight_a: float, weight_b: float) -> Tuple[float, float]:
    total = float(weight_a) + float(weight_b)
    if total == 0.0:
        raise ValueError("weight_a + weight_b must not be 0")
    return float(weight_a) / total, float(weight_b) / total


def _collect_h5_files(root_dir: Path) -> Dict[Path, Path]:
    files: Dict[Path, Path] = {}
    for path in sorted(root_dir.rglob("*")):
        if path.is_file() and _is_h5_file(path):
            rel = path.relative_to(root_dir)
            files[rel] = path
    return files


def _merge_numeric_arrays(
    arr_a: np.ndarray,
    arr_b: np.ndarray,
    weight_a: float,
    weight_b: float,
):
    if arr_a.shape != arr_b.shape:
        raise RuntimeError(f"shape mismatch: {arr_a.shape} vs {arr_b.shape}")

    merged_float = arr_a.astype(np.float32) * weight_a + arr_b.astype(np.float32) * weight_b
    out_dtype = arr_a.dtype

    if np.issubdtype(out_dtype, np.integer):
        info = np.iinfo(out_dtype)
        return np.clip(np.rint(merged_float), info.min, info.max).astype(out_dtype)

    if np.issubdtype(out_dtype, np.floating):
        return merged_float.astype(out_dtype)

    return merged_float


def _select_fallback(val_a, val_b, weight_a: float, weight_b: float):
    return val_a if weight_a >= weight_b else val_b


def _merge_single_h5(
    file_a: Optional[Path],
    file_b: Optional[Path],
    out_file: Path,
    weight_a: float,
    weight_b: float,
    strict_keys: bool,
):
    if file_a is None and file_b is None:
        raise RuntimeError("both file_a and file_b are None")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        h5_a = stack.enter_context(h5py.File(str(file_a), "r")) if file_a is not None else None
        h5_b = stack.enter_context(h5py.File(str(file_b), "r")) if file_b is not None else None
        h5_out = stack.enter_context(h5py.File(str(out_file), "w"))
        keys_a = set(h5_a.keys()) if h5_a is not None else set()
        keys_b = set(h5_b.keys()) if h5_b is not None else set()

        # Special-case: if both files exist and each contains exactly one dataset key,
        # treat them as the single data array to be merged regardless of the key names.
        # This supports datasets where the single key name differs across files.
        if (h5_a is not None) and (h5_b is not None) and (len(keys_a) == 1) and (len(keys_b) == 1):
            key_a = next(iter(keys_a))
            key_b = next(iter(keys_b))

            arr_a = np.asarray(h5_a[key_a])
            arr_b = np.asarray(h5_b[key_b])

            if arr_a.shape != arr_b.shape:
                raise RuntimeError(f"shape mismatch: {arr_a.shape} vs {arr_b.shape}")

            fallback_keys = 0
            copied_from_single_side = 0

            if np.issubdtype(arr_a.dtype, np.number) and np.issubdtype(arr_b.dtype, np.number):
                merged = _merge_numeric_arrays(arr_a, arr_b, weight_a, weight_b)
            else:
                merged = _select_fallback(arr_a, arr_b, weight_a, weight_b)
                fallback_keys = 1

            # write merged dataset using the key from A to keep deterministic naming
            h5_out.create_dataset(key_a, data=merged)
            return fallback_keys, copied_from_single_side

        only_a = keys_a - keys_b
        only_b = keys_b - keys_a
        if strict_keys and (h5_a is not None) and (h5_b is not None) and (only_a or only_b):
            raise RuntimeError(
                f"dataset keys mismatch at {out_file.name}: "
                f"only_in_a={len(only_a)}, only_in_b={len(only_b)}"
            )

        all_keys = sorted(keys_a | keys_b)
        fallback_keys = 0
        copied_from_single_side = 0

        for key in all_keys:
            in_a = h5_a is not None and key in h5_a
            in_b = h5_b is not None and key in h5_b

            if in_a and in_b:
                arr_a = np.asarray(h5_a[key])
                arr_b = np.asarray(h5_b[key])

                if np.issubdtype(arr_a.dtype, np.number) and np.issubdtype(arr_b.dtype, np.number):
                    merged = _merge_numeric_arrays(arr_a, arr_b, weight_a, weight_b)
                else:
                    merged = _select_fallback(arr_a, arr_b, weight_a, weight_b)
                    fallback_keys += 1
            elif in_a:
                merged = np.asarray(h5_a[key])
                copied_from_single_side += 1
            else:
                merged = np.asarray(h5_b[key])
                copied_from_single_side += 1

            h5_out.create_dataset(key, data=merged)

    return fallback_keys, copied_from_single_side


def _create_submission(out_root: Path, out_x4: Path, zip_name: Optional[str]):
    # generate zip archive for submission
    print(f"Generating zip archive......")
    if not out_x4.exists() or not any(out_x4.iterdir()):
        print("No files in x4 directory, skipping zip creation.")
        return

    if zip_name:
        # ensure zip_name does not have .zip extension
        zip_filename = Path(zip_name).name + ".zip"
    else:
        zip_filename = f"{out_root.name}.zip"

    zip_path = out_root.parent / zip_filename

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in out_x4.iterdir():
            if file.is_file():
                arcname = f"x4/{file.name}"
                zipf.write(file, arcname)
    print(f"Generated zip archive: {zip_path}")


def _run_file_mode(
    input_a: Path,
    input_b: Path,
    out_path: Path,
    weight_a: float,
    weight_b: float,
    strict_keys: bool,
):
    fallback_keys, copied_from_single_side = _merge_single_h5(
        file_a=input_a,
        file_b=input_b,
        out_file=out_path,
        weight_a=weight_a,
        weight_b=weight_b,
        strict_keys=strict_keys,
    )
    print(f"saved_out_path={out_path}")
    print("mode=file")
    print(f"fallback_non_numeric_keys={fallback_keys}")
    print(f"copied_single_side_keys={copied_from_single_side}")


def _run_dir_mode(
    input_a: Path,
    input_b: Path,
    out_path: Path,
    weight_a: float,
    weight_b: float,
    strict_files: bool,
    strict_keys: bool,
    zip_name: Optional[str],
):
    files_a = _collect_h5_files(input_a)
    files_b = _collect_h5_files(input_b)
    rel_a = set(files_a.keys())
    rel_b = set(files_b.keys())
    only_a = rel_a - rel_b
    only_b = rel_b - rel_a

    if strict_files and (only_a or only_b):
        raise RuntimeError(
            f"h5 file set mismatch: only_in_a={len(only_a)}, only_in_b={len(only_b)}"
        )

    all_rel = sorted(rel_a | rel_b)

    # create output directory and x4 subdirectory
    out_root = out_path
    out_x4 = out_root / "x4"
    out_x4.mkdir(parents=True, exist_ok=True)

    total_fallback_keys = 0
    total_copied_keys = 0
    merged_file_count = 0

    for rel in all_rel:
        file_a = files_a.get(rel)
        file_b = files_b.get(rel)

        # output file path: replace "LR_" with "HR_" in the relative path, and put under out_x4
        out_rel = rel.with_name(rel.name.replace("LR_", "HR_", 1))
        out_file = out_x4 / out_rel

        fallback_keys, copied_keys = _merge_single_h5(
            file_a=file_a,
            file_b=file_b,
            out_file=out_file,
            weight_a=weight_a,
            weight_b=weight_b,
            strict_keys=strict_keys,
        )
        total_fallback_keys += int(fallback_keys)
        total_copied_keys += int(copied_keys)
        merged_file_count += 1

    print(f"saved_out_path={out_root}")
    print("mode=dir")
    print(f"merged_files={merged_file_count}")
    print(f"files_only_in_a={len(only_a)}")
    print(f"files_only_in_b={len(only_b)}")
    print(f"fallback_non_numeric_keys={total_fallback_keys}")
    print(f"copied_single_side_keys={total_copied_keys}")

    # create zip archive for submission
    _create_submission(out_root, out_x4, zip_name)


def main():
    args = parse_args()
    input_a = Path(args.input_a).resolve()
    input_b = Path(args.input_b).resolve()
    out_path = Path(args.out_path).resolve()
    strict_files = bool(int(args.strict_files))
    strict_keys = bool(int(args.strict_keys))
    weight_a, weight_b = _normalize_weights(args.weight_a, args.weight_b)
    zip_name = args.zip_name

    if not input_a.exists():
        raise FileNotFoundError(f"input_a not found: {input_a}")
    if not input_b.exists():
        raise FileNotFoundError(f"input_b not found: {input_b}")

    is_file_mode = input_a.is_file() and input_b.is_file()
    is_dir_mode = input_a.is_dir() and input_b.is_dir()
    if not (is_file_mode or is_dir_mode):
        raise RuntimeError("input_a and input_b must both be files or both be directories")

    print(f"input_a={input_a}")
    print(f"input_b={input_b}")
    print(f"normalized_weight_a={weight_a:.6f}")
    print(f"normalized_weight_b={weight_b:.6f}")
    print(f"strict_files={int(strict_files)}")
    print(f"strict_keys={int(strict_keys)}")
    if zip_name:
        print(f"zip_name={zip_name}")

    if is_file_mode:
        _run_file_mode(
            input_a=input_a,
            input_b=input_b,
            out_path=out_path,
            weight_a=weight_a,
            weight_b=weight_b,
            strict_keys=strict_keys,
        )
    else:
        _run_dir_mode(
            input_a=input_a,
            input_b=input_b,
            out_path=out_path,
            weight_a=weight_a,
            weight_b=weight_b,
            strict_files=strict_files,
            strict_keys=strict_keys,
            zip_name=zip_name,
        )


if __name__ == "__main__":
    main()
