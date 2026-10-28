# SPDX-License-Identifier: Apache-2.0
"""Prepare a GR00T LeRobot dataset for ``--data_loader knonik``.

Ingests the LeRobot v2.x dataset at DATASET with knonik and packs it, writing
``hlp_manifest.json`` and its packs to ``DATASET/knonik`` (or ``--output``),
which is where ``launch_finetune.py --data_loader knonik`` looks by default.

The output is self-contained: the LeRobot ``meta/`` (modality config, tasks,
episodes) is copied beside the packs, and GR00T's ``meta/stats.json`` is
written from knonik's statistics of the packs (``norm_stats.json``: every
episode, exact, unless ``--sample-episodes``). With ``--embodiment-tag``,
``meta/relative_stats.json`` is written too; otherwise training writes it on
first use. The LeRobot dataset itself is left untouched and is not needed to
train.

Needs the knonik package and a login:

    pip install knonik
    knonik login

    python scripts/knonik/prepare_dataset.py /data/my_lerobot_dataset --embodiment-tag <tag>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys


def _cameras(info: dict) -> list[str]:
    return [
        key.split(".", 2)[-1]
        for key, feat in info.get("features", {}).items()
        if key.startswith("observation.images.") and feat.get("dtype") == "video"
    ]


def ingest_config(dataset: Path, cameras: list[str], fps: float, columns: list[str]) -> dict:
    from gr00t.data.knonik.stats import numeric_stream

    # state/action, plus any other parquet column modality.json slices from
    # (its `original_key`), each under the stream name the loader reads.
    numeric = {"state": "state", "action": "action"}
    numeric.update({numeric_stream(c): c for c in columns if numeric_stream(c) == c})
    return {
        "dir_type": "lerobot_v2",
        "data_dir": str(dataset),
        "fps": fps,
        "layout": {
            "images": {c: {"key": c, "fps": fps} for c in cameras},
            "numeric": numeric,
        },
    }


def copy_meta(dataset: Path, output: Path) -> None:
    """The LeRobot meta/ GR00T reads, without the statistics: those are
    written from the packs."""
    from gr00t.data.stats import LE_ROBOT_REL_STATS_FILENAME, LE_ROBOT_STATS_FILENAME

    skip = {Path(LE_ROBOT_STATS_FILENAME).name, Path(LE_ROBOT_REL_STATS_FILENAME).name}
    (output / "meta").mkdir(parents=True, exist_ok=True)
    for src in sorted((dataset / "meta").iterdir()):
        if src.is_file() and src.name not in skip:
            shutil.copy2(src, output / "meta" / src.name)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("dataset", type=Path, help="LeRobot v2.x dataset root (meta/ data/ videos/)")
    ap.add_argument("--output", type=Path, default=None, help="default: DATASET/knonik")
    ap.add_argument(
        "--cameras", default="", help="comma list of camera keys (default: every video feature)"
    )
    ap.add_argument("--workers", type=int, default=1, help="packs built in parallel")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--embodiment-tag",
        default=None,
        help="also write meta/relative_stats.json for this embodiment",
    )
    ap.add_argument(
        "--modality-config-path",
        default=None,
        help=".py modality config registering a custom --embodiment-tag",
    )
    ap.add_argument(
        "--sample-episodes",
        type=int,
        default=None,
        metavar="N",
        help="compute statistics over N randomly chosen episodes (default: all, exact)",
    )
    ap.add_argument("--stats-seed", type=int, default=0, help="seed for --sample-episodes")
    ap.add_argument(
        "--keep-episodes",
        action="store_true",
        help="keep the intermediate ingested episodes (OUTPUT/episodes)",
    )
    args = ap.parse_args()

    try:
        from knonik import stats as knonik_stats
        from knonik.ingest import pack, run
    except ImportError:
        print("knonik is not installed: pip install knonik", file=sys.stderr)
        return 2

    dataset = args.dataset.resolve()
    info_path = dataset / "meta" / "info.json"
    if not info_path.is_file():
        print(f"{info_path} not found: DATASET must be a LeRobot v2.x dataset", file=sys.stderr)
        return 2
    info = json.loads(info_path.read_text())
    fps = float(info["fps"])
    cameras = [c.strip() for c in args.cameras.split(",") if c.strip()] or _cameras(info)
    if not cameras:
        print(f"no video features in {info_path}", file=sys.stderr)
        return 2
    if not (dataset / "meta" / "modality.json").is_file():
        print(f"{dataset}/meta/modality.json not found: GR00T needs it", file=sys.stderr)
        return 2

    from gr00t.data.knonik import stats as gr00t_stats

    columns = gr00t_stats.numeric_columns(dataset)
    tag = None
    if args.embodiment_tag:
        from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
        from gr00t.data.embodiment_tags import EmbodimentTag

        if args.modality_config_path:
            import importlib

            sys.path.append(str(Path(args.modality_config_path).resolve().parent))
            importlib.import_module(Path(args.modality_config_path).stem)
        tag = EmbodimentTag.resolve(args.embodiment_tag)
        # Checked before the ingest, which is the slow part.
        if tag.value not in MODALITY_CONFIGS:
            print(
                f"no modality config for embodiment {tag.value!r}: pass --modality-config-path",
                file=sys.stderr,
            )
            return 2

    output = (args.output or dataset / "knonik").resolve()
    episodes = output / "episodes"
    output.mkdir(parents=True, exist_ok=True)
    print(f"ingesting {dataset} ({len(cameras)} cameras: {', '.join(cameras)}) -> {episodes}")
    run(config=ingest_config(dataset, cameras, fps, columns), output=episodes)

    print(f"packing -> {output}")
    pack(
        input_dir=episodes,
        output_dir=output,
        dataset_name=dataset.name,
        window_size=256,
        fps=int(round(fps)),
        seed=args.seed,
        pack_mode="shard_batch",
        workers=args.workers,
        progress=print,
        stats=args.sample_episodes is None,
    )
    if not args.keep_episodes:
        shutil.rmtree(episodes, ignore_errors=True)
    manifest = output / "hlp_manifest.json"
    if not manifest.is_file():
        print(f"packing finished but {manifest} is missing", file=sys.stderr)
        return 1
    if args.sample_episodes is not None:
        print(f"computing norm_stats.json over {args.sample_episodes} sampled episodes")
        knonik_stats.compute(output, sample_episodes=args.sample_episodes, seed=args.stats_seed)

    copy_meta(dataset, output)
    gr00t_stats.generate_stats(output)
    if tag is not None:
        gr00t_stats.generate_rel_stats(output, tag)
    print(f"done: {manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
