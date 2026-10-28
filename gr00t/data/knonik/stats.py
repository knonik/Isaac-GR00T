# SPDX-License-Identifier: Apache-2.0
"""GR00T's ``meta/stats.json`` and ``meta/relative_stats.json`` from a knonik dataset.

The same two files ``gr00t/data/stats.py`` writes from a LeRobot dataset's
parquet, computed from the knonik packs instead, so a prepared knonik directory
trains with no LeRobot copy beside it:

- ``stats.json`` is knonik's ``norm_stats.json`` (written at pack time, or
  computed here when missing or out of date) under the LeRobot column names
  GR00T looks statistics up by.
- ``relative_stats.json`` runs GR00T's own relative-action chunking over the
  state and action trajectories knonik reads back from the packs.

Both are cached with fingerprints, as upstream does, and regenerated when the
dataset's episodes or the embodiment's action config change.

Usage:
    python -m gr00t.data.knonik.stats --knonik-dir <dataset>/knonik --embodiment-tag <tag>
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.dataset.lerobot_episode_loader import DEFAULT_COLUMN_NAMES
from gr00t.data.stats import (
    LE_ROBOT_INFO_FILENAME,
    LE_ROBOT_REL_STATS_FILENAME,
    LE_ROBOT_STATS_FILENAME,
    STATS_FINGERPRINTS_KEY,
    RelativeActionLoader,
    _compute_relative_action_fingerprint,
    _compute_stats_fingerprint,
    _dump_stats_cache_atomic,
    _load_stats_cache,
)
from gr00t.data.types import ActionRepresentation, EmbodimentTag
from gr00t.data.utils import to_json_serializable


MODALITY_FILENAME = "meta/modality.json"
_STAT_KEYS = ("mean", "std", "min", "max", "q01", "q99")
# Ingest names LeRobot's two standard vectors `state` and `action`, and every
# other parquet column after itself.
_STREAM_FOR_COLUMN = {
    DEFAULT_COLUMN_NAMES["state"]: "state",
    DEFAULT_COLUMN_NAMES["action"]: "action",
}


def numeric_stream(column: str) -> str:
    """The knonik stream holding LeRobot parquet column ``column``."""
    return _STREAM_FOR_COLUMN.get(column, column)


def numeric_columns(meta_root: str | Path) -> list[str]:
    """The parquet columns ``modality.json`` slices state and action from."""
    modality = json.loads((Path(meta_root) / MODALITY_FILENAME).read_text())
    return sorted(
        {
            info.get("original_key", DEFAULT_COLUMN_NAMES[m])
            for m in ("state", "action")
            for info in modality.get(m, {}).values()
        }
    )


def meta_root(manifest: str | Path, dataset_path: str | Path) -> Path:
    """Where GR00T's ``meta/`` is read from: the prepared knonik directory when
    it carries one (``prepare_dataset.py`` copies it there), else the LeRobot
    dataset it was prepared from."""
    knonik_dir = Path(manifest).parent
    return knonik_dir if (knonik_dir / MODALITY_FILENAME).is_file() else Path(dataset_path)


def _source_fingerprint(norm: dict[str, Any]) -> dict[str, Any]:
    """What the knonik statistics were computed over."""
    return {
        "catalog": norm.get("catalog_fingerprint", ""),
        "episodes": norm.get("episodes"),
        "sampled": bool(norm.get("sampled")),
        "seed": norm.get("seed", 0),
    }


def _hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def norm_stats(knonik_dir: str | Path, sample_episodes: int | None = None, seed: int = 0) -> dict:
    """The dataset's ``norm_stats.json``, computed first when it is missing or
    no longer describes the dataset's episodes. ``sample_episodes`` applies
    only to a fresh computation; a stale file is recomputed with the sampling
    it recorded."""
    from knonik import stats as ks

    norm = ks.load(knonik_dir)
    if norm is not None and ks.is_current(knonik_dir, norm):
        return norm
    if norm is not None and norm.get("sampled") and sample_episodes is None:
        sample_episodes, seed = int(norm["episodes"]), int(norm.get("seed", 0))
    print(f"Computing knonik norm_stats.json for {knonik_dir}")
    return ks.compute(knonik_dir, sample_episodes=sample_episodes, seed=seed)


def generate_stats(
    knonik_dir: str | Path, sample_episodes: int | None = None, seed: int = 0
) -> None:
    """Write ``meta/stats.json`` from the dataset's knonik statistics."""
    root = Path(knonik_dir)
    norm = norm_stats(root, sample_episodes, seed)
    streams = norm["streams"]
    missing = [c for c in numeric_columns(root) if numeric_stream(c) not in streams]
    if missing:
        raise ValueError(
            f"{root} has no knonik stream for column(s) {missing} that modality.json reads: "
            "re-run scripts/knonik/prepare_dataset.py on the LeRobot dataset"
        )

    features = json.loads((root / LE_ROBOT_INFO_FILENAME).read_text())["features"]
    source = _source_fingerprint(norm)
    stats: dict[str, Any] = {}
    fingerprints: dict[str, str] = {}
    for column, feature in features.items():
        s = streams.get(numeric_stream(column))
        if "float" not in feature["dtype"] or s is None:
            continue
        width = int(np.prod(feature.get("shape") or [1]))
        if int(s["dim"]) != width:
            raise ValueError(
                f"knonik stream {numeric_stream(column)!r} is {s['dim']} wide but {column} is "
                f"{width}: {root} was packed from the wrong parquet column. Re-run "
                "scripts/knonik/prepare_dataset.py with a current knonik"
            )
        stats[column] = {k: s[k] for k in _STAT_KEYS}
        fingerprints[column] = _hash(
            {"feature": _compute_stats_fingerprint(column, feature), "source": source}
        )

    path = root / LE_ROBOT_STATS_FILENAME
    if _load_stats_cache(path).get(STATS_FINGERPRINTS_KEY) == fingerprints:
        return
    print(f"Generating stats for {root} from knonik norm_stats.json")
    stats[STATS_FINGERPRINTS_KEY] = fingerprints
    _dump_stats_cache_atomic(path, stats)


def _relative_stats_for_key(
    root: Path, embodiment_tag: EmbodimentTag, action_key: str, norm: dict[str, Any]
) -> dict[str, np.ndarray]:
    from knonik import stats as ks

    # Reads root/meta for the modality layout and never touches a parquet: each
    # episode's frame is handed to it below.
    loader = RelativeActionLoader(root, embodiment_tag, action_key)
    modality = loader.loader.modality_meta
    groups = []
    for m in ("state", "action"):
        for key in loader.modality_configs[m].modality_keys:
            info = modality[m][key]
            column = info.get("original_key", DEFAULT_COLUMN_NAMES[m])
            groups.append((f"{m}.{key}", numeric_stream(column), info["start"], info["end"]))

    trajectories = []
    for _episode_id, arrays in ks.iter_numeric(
        root,
        streams=sorted({g[1] for g in groups}),
        sample_episodes=int(norm["episodes"]) if norm.get("sampled") else None,
        seed=int(norm.get("seed", 0)),
    ):
        # The frame LeRobotEpisodeLoader would build from the parquet: one
        # float32 array per row per joint group.
        frame = pd.DataFrame(
            {
                col: list(np.asarray(arrays[stream][:, a:b], dtype=np.float32))
                for col, stream, a, b in groups
            }
        )
        loader.loader = [frame]
        trajectories.extend(loader.load_relative_actions(0))
    return {
        "max": np.max(trajectories, axis=0),
        "min": np.min(trajectories, axis=0),
        "q01": np.quantile(trajectories, 0.01, axis=0),
        "q99": np.quantile(trajectories, 0.99, axis=0),
        "mean": np.mean(trajectories, axis=0),
        "std": np.std(trajectories, axis=0),
    }


def generate_rel_stats(knonik_dir: str | Path, embodiment_tag: EmbodimentTag) -> None:
    """Write ``meta/relative_stats.json``, over the same episodes as
    ``norm_stats.json`` (all of them, or its seeded sample)."""
    from knonik import stats as ks

    root = Path(knonik_dir)
    action_config = MODALITY_CONFIGS[embodiment_tag.value]["action"]
    if action_config.action_configs is None:
        return
    action_keys = [
        key
        for key, config in zip(action_config.modality_keys, action_config.action_configs)
        if config.rep == ActionRepresentation.RELATIVE
    ]
    norm = ks.load(root)
    if norm is None:
        raise FileNotFoundError(f"{root} has no norm_stats.json: run generate_stats first")
    source = _source_fingerprint(norm)
    path = root / LE_ROBOT_REL_STATS_FILENAME
    stats = _load_stats_cache(path)
    fingerprints = stats.setdefault(STATS_FINGERPRINTS_KEY, {})
    for action_key in sorted(action_keys):
        expected = _hash(
            {
                "action": _compute_relative_action_fingerprint(embodiment_tag, action_key),
                "source": source,
            }
        )
        if action_key in stats and fingerprints.get(action_key) == expected:
            continue
        print(f"Generating relative stats for {root} {embodiment_tag} {action_key} from knonik")
        stats[action_key] = _relative_stats_for_key(root, embodiment_tag, action_key, norm)
        fingerprints[action_key] = expected
    _dump_stats_cache_atomic(path, to_json_serializable(dict(stats)))


def main(
    knonik_dir: str,
    embodiment_tag: EmbodimentTag | None = None,
    modality_config_path: str | None = None,
    sample_episodes: int | None = None,
    seed: int = 0,
):
    """Generate GR00T statistics for a prepared knonik dataset.

    Args:
        knonik_dir: Directory written by scripts/knonik/prepare_dataset.py.
        embodiment_tag: Embodiment whose relative actions to summarise; without
            it only meta/stats.json is written.
        modality_config_path: .py modality config registering a custom embodiment tag.
        sample_episodes: Compute knonik's statistics over this many randomly chosen
            episodes when they have to be computed (default: every episode).
        seed: Seed for sample_episodes.
    """
    if modality_config_path is not None:
        import importlib
        import sys

        config_path = Path(modality_config_path)
        if not (config_path.exists() and config_path.suffix == ".py"):
            raise FileNotFoundError(
                f"Modality config path does not exist or is not a .py file: {modality_config_path}"
            )
        sys.path.append(str(config_path.parent))
        importlib.import_module(config_path.stem)
    if embodiment_tag is not None and embodiment_tag.value not in MODALITY_CONFIGS:
        raise ValueError(
            f"No modality config for embodiment tag '{embodiment_tag.value}': pass "
            "--modality-config-path for a custom embodiment"
        )
    generate_stats(knonik_dir, sample_episodes=sample_episodes, seed=seed)
    if embodiment_tag is not None:
        generate_rel_stats(knonik_dir, embodiment_tag)


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
