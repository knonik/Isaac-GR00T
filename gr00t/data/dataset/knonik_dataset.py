# SPDX-License-Identifier: Apache-2.0
"""GR00T training batches from a knonik dataset (``data.loader=knonik``).

Everything is read from the prepared knonik directory
(``scripts/knonik/prepare_dataset.py``): the frames and the state/action
windows from its packs, and the modality metadata, task strings and
normalisation statistics from the ``meta/`` beside them, which
``gr00t/data/knonik/stats.py`` fills from the packs. A directory prepared
before it carried a ``meta/`` reads those from the LeRobot dataset at
``dataset_path`` instead. Each item this dataset yields is one collated
per-device batch.
"""

from __future__ import annotations

import functools
import json
import logging
from pathlib import Path
import threading
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import IterableDataset

from gr00t.data.dataset.sharded_mixture_dataset import ShardedMixtureDataset
from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset
from gr00t.data.knonik import loader as kl
from gr00t.data.knonik.source import resolve_manifest
from gr00t.data.knonik.stats import meta_root, numeric_stream
from gr00t.data.types import EmbodimentTag, MessageType, ModalityConfig, VLAStepData


logger = logging.getLogger(__name__)

_DEFAULT_COLUMN = {"state": "observation.state", "action": "action"}


def _video_stream(column: str) -> str:
    # Ingest names a camera after the last part of observation.images.<camera>.
    return column.split(".")[-1]


def raw_batch_steps(raw: dict, embodiment_tag: EmbodimentTag) -> Iterator[VLAStepData]:
    n = len(raw["text"])
    keep = raw.get("keep")
    for i in range(n):
        if keep is not None and not keep[i]:
            continue
        yield VLAStepData(
            images={k: list(v[i].numpy()) for k, v in raw["images"].items()},
            masks=None,
            states={k: v[i] for k, v in raw["states"].items()},
            actions={k: v[i] for k, v in raw["actions"].items()},
            text=raw["text"][i],
            embodiment=embodiment_tag,
        )


def _pin_batch(obj):
    if torch.is_tensor(obj):
        return obj.pin_memory()
    if hasattr(obj, "data") and hasattr(obj, "keys") and isinstance(getattr(obj, "data"), dict):
        obj.data = {k: _pin_batch(v) for k, v in obj.data.items()}
        return obj
    if isinstance(obj, dict):
        return {k: _pin_batch(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_pin_batch(v) for v in obj)
    return obj


def _pin(obj):
    kl.set_local_cuda_device()
    return _pin_batch(obj)


def _child_dataset(ctor: dict) -> "KnonikGr00tDataset":
    return KnonikGr00tDataset(**ctor, child=True)


class Gr00tBatchTransform:
    """Processor + collator over one raw knonik batch; runs in the transform pool."""

    def __init__(self, processor, embodiment_tag: EmbodimentTag):
        self.processor = processor
        self.embodiment_tag = embodiment_tag

    def __call__(self, item: dict):
        if "features" in item:
            return self.processor.collator(item["features"])
        features = [
            self.processor([{"type": MessageType.EPISODE_STEP.value, "content": step}])
            for step in raw_batch_steps(item, self.embodiment_tag)
        ]
        if item.get("collate", True):
            return self.processor.collator(features)
        return {"features": features, "dropped": len(item["text"]) - len(features)}


class KnonikGr00tDataset(IterableDataset):
    is_knonik = True
    yields_batches = True

    def __init__(
        self,
        dataset_path: str | Path,
        embodiment_tag: EmbodimentTag,
        modality_configs: dict[str, ModalityConfig],
        processor,
        knonik: dict | None,
        *,
        per_device_batch_size: int,
        steps_needed: int,
        baseline_num_workers: int,
        seed: int = 42,
        shard_size: int = 2**10,
        episode_sampling_rate: float = 0.1,
        num_shards_per_epoch: int = int(1e5),
        override_pretraining_statistics: bool = False,
        allow_padding: bool = False,
        child: bool = False,
    ):
        super().__init__()
        self._ctor = dict(
            dataset_path=str(dataset_path),
            embodiment_tag=embodiment_tag,
            modality_configs=modality_configs,
            processor=processor,
            knonik=dict(knonik or {}),
            per_device_batch_size=per_device_batch_size,
            steps_needed=steps_needed,
            baseline_num_workers=baseline_num_workers,
            seed=seed,
            shard_size=shard_size,
            episode_sampling_rate=episode_sampling_rate,
            num_shards_per_epoch=num_shards_per_epoch,
            override_pretraining_statistics=override_pretraining_statistics,
            allow_padding=allow_padding,
        )
        self._child = bool(child)
        self.allow_padding = bool(allow_padding)
        self.k = kl.merged_settings(knonik)
        self.batch_size = int(per_device_batch_size)
        self.dataset_path = Path(dataset_path)
        self.manifest = resolve_manifest(
            self.k["source"], self.dataset_path, self.k["hub_cache_dir"] or None
        )
        self.meta_root = meta_root(self.manifest, self.dataset_path)
        self.embodiment_tag = embodiment_tag
        self.modality_configs = modality_configs
        self.seed = int(seed)

        stock = ShardedSingleStepDataset(
            dataset_path=self.meta_root,
            embodiment_tag=embodiment_tag,
            modality_configs=modality_configs,
            shard_size=shard_size,
            episode_sampling_rate=episode_sampling_rate,
            seed=seed,
            allow_padding=allow_padding,
        )
        self._mixture = (
            None
            if self._child
            else ShardedMixtureDataset(
                datasets=[stock],
                weights=[1.0],
                processor=processor,
                seed=seed,
                training=True,
                num_shards_per_epoch=num_shards_per_epoch,
                override_pretraining_statistics=override_pretraining_statistics,
            )
        )
        self.processor = processor
        episodes = stock.episode_loader
        mconf = episodes.modality_configs
        meta = episodes.modality_meta

        self._video: list[tuple[str, str]] = []
        for key in mconf["video"].modality_keys:
            meta_key = episodes._video_key_mapping.get(key, key)
            col = meta["video"][meta_key].get("original_key", f"observation.images.{meta_key}")
            self._video.append((key, _video_stream(col)))
        self._numeric: dict[str, list[tuple[str, str, int, int]]] = {}
        for modality in ("state", "action"):
            rows = []
            for key in mconf[modality].modality_keys:
                info = meta[modality][key]
                col = info.get("original_key", _DEFAULT_COLUMN[modality])
                rows.append((key, numeric_stream(col), int(info["start"]), int(info["end"])))
            self._numeric[modality] = rows

        offsets: dict[str, list[int]] = {}
        for _key, stream in self._video:
            offsets[stream] = list(mconf["video"].delta_indices)
        for modality, rows in self._numeric.items():
            for _key, stream, _a, _b in rows:
                want = list(mconf[modality].delta_indices)
                if offsets.setdefault(stream, want) != want:
                    raise ValueError(
                        f"stream {stream!r} feeds two modalities with different offsets"
                    )
        self._offsets = offsets
        if min(min(v) for v in offsets.values()) < 0 and not allow_padding:
            raise ValueError(
                "negative delta_indices need data.allow_padding=True: without it the stock "
                "loader reads frames from the END of the episode"
            )
        acts = mconf["action"].delta_indices
        if max(max(v) for v in offsets.values()) > max(acts):
            raise ValueError("an observation offset reaches past the action horizon")

        info = json.loads((self.meta_root / "meta" / "info.json").read_text())
        self._fps = float(info["fps"])
        self._frame_shapes = {}
        for _key, stream in self._video:
            feat = info["features"][f"observation.images.{stream}"]
            shape = feat.get("shape") or [
                feat["info"]["video.height"],
                feat["info"]["video.width"],
                3,
            ]
            self._frame_shapes[stream] = [int(shape[0]), int(shape[1]), 3]
        self._lang_by_ep = kl.episode_languages(self.meta_root)

        lengths = np.asarray(episodes.episode_lengths, dtype=np.int64)
        self.admit_fraction = (
            float(
                np.clip(lengths - stock.action_horizon + 1, 0, None).sum() / max(1, lengths.sum())
            )
            if allow_padding
            else 1.0
        )
        self._plan_batches = kl.plan_batches(
            int(np.ceil(int(steps_needed) * self.batch_size / max(0.05, self.admit_fraction))),
            self.batch_size,
        )
        self._buf: list = []
        self._buf_lock = threading.Lock()
        self._in_process = self._child or not bool(self.k["loader_process"])
        self._pin = (not self._child) and bool(torch.cuda.is_available())
        self._proc: kl.ProcessStream | None = None
        workers = int(self.k["transform_workers"]) or max(1, int(baseline_num_workers))
        self._pool = kl.TransformPool(
            Gr00tBatchTransform(processor, embodiment_tag), workers=workers
        )
        self._stream = kl.BatchStream(
            name=f"gr00t_{embodiment_tag.value}",
            cfg_fn=self._loader_cfg,
            raw_batches=self._raw_batches,
            pool=self._pool,
            post=self._finish,
        )

    def _loader_cfg(self, batch_transform) -> tuple[dict, dict]:
        return kl.loader_cfg(
            self.manifest,
            batch_size=self.batch_size,
            num_decoders=int(self.k["num_decoders"]),
            fps=self._fps,
            frame_offsets=self._offsets,
            frame_shapes=self._frame_shapes,
            pad_missing=self.allow_padding,
            seed=self.seed,
            max_steps=self._plan_batches,
            batch_transform=batch_transform,
            transform_threads=max(4, self._pool.workers),
        )

    def _text(self, episode_id, fallback) -> str:
        ep = kl.source_episode(episode_id)
        if ep is not None and ep in self._lang_by_ep:
            return self._lang_by_ep[ep]
        return str(fallback or "")

    def raw_batch(self, batch: dict) -> dict:
        n = int(np.asarray(batch[self._numeric["action"][0][1]]).shape[0])
        langs = batch.get("_languages") or [""] * n
        ep_ids = batch.get("_episode_ids") or [None] * n
        numeric = {
            stream: np.asarray(batch[stream])
            for rows in self._numeric.values()
            for _k, stream, _a, _b in rows
        }
        out = {
            "images": {
                key: torch.from_numpy(np.array(batch[stream], copy=True))
                for key, stream in self._video
            },
            "states": {
                key: np.ascontiguousarray(numeric[stream][:, :, a:b], dtype=np.float32)
                for key, stream, a, b in self._numeric["state"]
            },
            "actions": {
                key: np.ascontiguousarray(numeric[stream][:, :, a:b], dtype=np.float32)
                for key, stream, a, b in self._numeric["action"]
            },
            "text": [self._text(ep_ids[i], langs[i]) for i in range(n)],
            "collate": not self.allow_padding,
        }
        if self.allow_padding:
            act = self._numeric["action"][0][1]
            out["keep"] = np.asarray(batch[f"{act}_valid_mask"], dtype=bool).all(axis=1)
        return out

    def _raw_batches(self, batch: dict):
        yield self.raw_batch(batch)

    def _finish(self, out: list) -> list:
        batches = self._refill(out) if self.allow_padding else list(out)
        if self._pin and self._in_process:
            batches = [_pin(b) for b in batches]
        return batches

    def _refill(self, out: list) -> list:
        # With padding, windows that run past the episode end are dropped (the
        # stock loader never samples them); top batches back up to full size.
        B = self.batch_size
        chunks = []
        with self._buf_lock:
            for item in out:
                self._buf.extend(item["features"])
            while len(self._buf) >= B:
                chunks.append(self._buf[:B])
                del self._buf[:B]
        batches = []
        for chunk in chunks:
            batches.extend(self._pool([{"features": chunk}]))
        return batches

    def iter_local(self):
        return iter(self._stream)

    def shutdown_local(self) -> None:
        self._stream.shutdown()

    def __iter__(self):
        if self._in_process:
            yield from self._stream
            return
        if self._proc is None:
            self._proc = kl.ProcessStream(
                functools.partial(_child_dataset, self._ctor),
                post=_pin if self._pin else None,
                name=f"knonik-loader-r{kl.rank_world()[0]}",
            )
        yield from self._proc

    def __len__(self) -> int:
        return int(self._plan_batches)

    def get_dataset_statistics(self):
        return self._mixture.get_dataset_statistics()

    def get_initial_actions(self):
        return self._mixture.get_initial_actions()

    def print_dataset_statistics(self):
        self._mixture.print_dataset_statistics()

    def reset_seed(self, seed: int) -> None:
        self.seed = int(seed)

    def close(self) -> None:
        if self._proc is not None:
            self._proc.close()
        else:
            self._stream.shutdown()

    shutdown = close
