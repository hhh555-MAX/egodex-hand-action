"""Extract RGB frames from EgoDex MP4 files into model-ready image folders."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--frame-root", required=True, type=Path)
    parser.add_argument("--limit-pairs", type=int, default=None)
    parser.add_argument("--image-extension", default=".jpg")
    parser.add_argument("--quality", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = list(iter_hdf5_mp4_pairs(args.dataset_root))
    if args.limit_pairs is not None:
        pairs = pairs[: args.limit_pairs]
    if not pairs:
        raise SystemExit(f"No paired .hdf5/.mp4 files found under {args.dataset_root}")

    args.frame_root.mkdir(parents=True, exist_ok=True)
    extension = normalize_extension(args.image_extension)

    for index, (_hdf5_path, mp4_path) in enumerate(pairs, start=1):
        video_id = video_id_for(args.dataset_root, mp4_path)
        output_dir = args.frame_root / video_id
        output_dir.mkdir(parents=True, exist_ok=True)
        output_pattern = output_dir / f"%06d{extension}"

        if should_report(index, len(pairs), args.progress_every):
            print(f"[EgoDex] Extracting {index}/{len(pairs)}: {mp4_path} -> {output_dir}", flush=True)

        if not args.overwrite and any(output_dir.glob(f"*{extension}")):
            continue

        command = (
            "ffmpeg",
            "-y" if args.overwrite else "-n",
            "-i",
            str(mp4_path),
            "-start_number",
            "0",
            "-q:v",
            str(args.quality),
            str(output_pattern),
        )
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise SystemExit(
                "ffmpeg failed for "
                f"{mp4_path}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )


def iter_hdf5_mp4_pairs(dataset_root: Path):
    for hdf5_path in sorted(dataset_root.rglob("*.hdf5")):
        mp4_path = hdf5_path.with_suffix(".mp4")
        if mp4_path.exists():
            yield hdf5_path, mp4_path


def video_id_for(dataset_root: Path, mp4_path: Path) -> str:
    relative = mp4_path.relative_to(dataset_root).with_suffix("")
    return "_".join(relative.parts)


def normalize_extension(extension: str) -> str:
    return extension if extension.startswith(".") else f".{extension}"


def should_report(index: int, count: int, progress_every: int) -> bool:
    if progress_every <= 0:
        return False
    return index == 1 or index == count or index % progress_every == 0


if __name__ == "__main__":
    main()
