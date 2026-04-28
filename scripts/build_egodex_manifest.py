"""Build EgoDex manifest/split JSON files from official HDF5/MP4 folders."""

from __future__ import annotations

import argparse
from pathlib import Path

from egodex_hand_action.contracts.data import Handedness
from egodex_hand_action.datasets import EgoDexHdf5ManifestBuilder, EgoDexHdf5ManifestConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--hand", choices=("left", "right"), default="left")
    parser.add_argument("--frame-root", type=Path, default=None)
    parser.add_argument("--frame-extension", default=".jpg")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    builder = EgoDexHdf5ManifestBuilder(
        EgoDexHdf5ManifestConfig(
            hand=Handedness(args.hand),
            frame_root=args.frame_root,
            frame_extension=args.frame_extension,
        )
    )
    paths = builder.build(args.dataset_root, args.output_dir)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
