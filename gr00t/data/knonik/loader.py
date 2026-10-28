# SPDX-License-Identifier: Apache-2.0
"""The knonik loader, its batch transform pool, and the process it runs in."""

from __future__ import annotations

import atexit
import concurrent.futures
import copy
import json
import logging
import multiprocessing
import os
from pathlib import Path
import pickle
import queue
import re
import threading
import time
import traceback
from typing import Any, Callable, Iterable


logger = logging.getLogger(__name__)

#: User-facing settings (``config.data.knonik``).
DEFAULTS: dict[str, Any] = {
    # Prepared dataset directory, its hlp_manifest.json, or hub://<org>/<name>.
    # Empty: <dataset_path>/knonik.
    "source": "",
    # Video decode processes per rank.
    "num_decoders": 12,
    # Processes running the GR00T processor + collator per rank.
    # 0: training.dataloader_num_workers.
    "transform_workers": 0,
    # Run the loader in its own process, so it never competes with the training
    # loop for the GIL.
    "loader_process": True,
    # Where Hub datasets are cached (once the Hub is live).
    "hub_cache_dir": "",
}

_PROCESS_QUEUE = 4


def merged_settings(overrides: dict | None) -> dict:
    k = dict(DEFAULTS)
    for key, value in dict(overrides or {}).items():
        if key not in DEFAULTS:
            raise KeyError(f"unknown knonik setting {key!r}; known: {sorted(DEFAULTS)}")
        if value is not None:
            k[key] = value
    return k


def rank_world() -> tuple[int, int]:
    return int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))


def plan_batches(samples_needed: int, batch_size: int, margin: float = 1.25) -> int:
    return max(8, -(-int(samples_needed * margin) // int(batch_size)))


def loader_cfg(
    manifest: str | os.PathLike,
    *,
    batch_size: int,
    num_decoders: int,
    fps: float,
    frame_offsets: dict[str, list[int]],
    frame_shapes: dict[str, list[int]],
    pad_missing: bool,
    seed: int,
    max_steps: int,
    batch_transform: Callable | None,
    transform_threads: int,
) -> tuple[dict, dict]:
    manifest = Path(manifest)
    cams = list(frame_shapes)
    numeric = [s for s in frame_offsets if s not in frame_shapes]
    cfg: dict[str, Any] = {
        "decode_mode": "random_access",
        "hlp_manifest": str(manifest),
        "fs_type": "local",
        "fs_params": {"root": str(manifest.parent)},
        "framework": "numpy",
        "chunk_len": 1,
        "stride": 1,
        "batch_size": int(batch_size),
        "fps": float(fps),
        "shuffle_mode": "global_triplet_fast",
        "delta_timestamps": {
            s: [o / float(fps) for o in offs] for s, offs in frame_offsets.items()
        },
        "pad_missing": bool(pad_missing),
        "num_decoders": int(num_decoders),
        "prefetch_shards": 16,
        "precollate_batches": 12,
        "hlps_in_flight": 4,
        "partial_decode_coalesce_batches": 32,
        "base_seed": int(seed),
        "max_steps": int(max_steps),
        "zerocopy": True,
        "zerocopy_frame_shapes": {c: list(frame_shapes[c]) for c in cams},
    }
    if batch_transform is not None:
        cfg.update(
            {
                "batch_augment_fn": batch_transform,
                "batch_augment_execution": "threaded",
                "batch_augment_workers": int(transform_threads),
                "batch_augment_max_in_flight": int(transform_threads),
            }
        )
    stream_cfg = {"rgb": {c: {} for c in cams}, "numeric": {s: {} for s in numeric}}
    return cfg, stream_cfg


def build_loader(cfg: dict, stream_cfg: dict, *, name: str):
    try:
        from knonik_multidataloader.api.sharded_dataloader import ShardedDataLoader
    except ImportError as exc:
        raise ImportError(
            "data.loader=knonik needs the knonik package: `pip install knonik` and "
            "`knonik login` (see README.md)"
        ) from exc

    if multiprocessing.current_process().daemon:
        raise RuntimeError(
            "knonik loader built in a daemonic process: the DataLoader around the knonik "
            "dataset must use num_workers=0"
        )
    os.environ.setdefault("KNONIK_DECODE_THREADS", "1")
    loader = ShardedDataLoader(cfg, stream_cfg, dataset_name=name)
    if (
        cfg.get("batch_augment_fn") is not None
        and getattr(loader, "batch_augment_fn", None) is None
    ):
        raise RuntimeError("the installed knonik is too old for this integration; upgrade it")
    return loader


def shutdown_loader(loader, timeout_s: float = 60.0) -> None:
    if loader is None:
        return
    t = threading.Thread(target=loader.shutdown, daemon=True)
    t.start()
    t.join(timeout_s)


def episode_languages(meta_root: str | os.PathLike) -> dict[int, str]:
    # Keyed by the LeRobot episode_index: knonik ingest names each episode after
    # it, so it is the number in the loader's episode ids.
    out: dict[int, str] = {}
    with open(Path(meta_root) / "meta" / "episodes.jsonl") as fh:
        for line in fh:
            if line.strip():
                e = json.loads(line)
                names = e.get("tasks") or [""]
                out[int(e["episode_index"])] = str(names[0])
    return out


def source_episode(episode_id: Any) -> int | None:
    m = re.search(r"(\d+)$", str(episode_id))
    return int(m.group(1)) if m else None


def set_local_cuda_device() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    except Exception:
        pass


# --------------------------------------------------------------------------- transforms

_WORKER_FN: Callable | None = None


def _init_worker(payload: bytes, torch_threads: int) -> None:
    global _WORKER_FN
    try:
        import torch

        torch.set_num_threads(max(1, int(torch_threads)))
    except Exception:
        pass
    _WORKER_FN = pickle.loads(payload)


def _run_in_worker(raws: list) -> list:
    assert _WORKER_FN is not None
    return [_WORKER_FN(r) for r in raws]


class TransformPool:
    """Runs the GR00T processor + collator on whole batches in spawned processes."""

    def __init__(self, fn: Callable, *, workers: int, torch_threads: int = 1):
        self.workers = max(1, int(workers))
        self._fn = fn
        self._ex: concurrent.futures.Executor | None = None
        self._torch_threads = torch_threads
        self._lock = threading.Lock()

    def _executor(self) -> concurrent.futures.Executor:
        with self._lock:
            if self._ex is None:
                payload = pickle.dumps(self._fn, protocol=pickle.HIGHEST_PROTOCOL)
                self._ex = concurrent.futures.ProcessPoolExecutor(
                    max_workers=self.workers,
                    mp_context=multiprocessing.get_context("spawn"),
                    initializer=_init_worker,
                    initargs=(payload, self._torch_threads),
                )
            return self._ex

    def start(self) -> None:
        ex = self._executor()
        list(ex.map(_run_in_worker, [[] for _ in range(self.workers)]))

    def __call__(self, raws: list) -> list:
        if not raws:
            return []
        return self._executor().submit(_run_in_worker, raws).result()

    def shutdown(self) -> None:
        ex, self._ex = self._ex, None
        if ex is None:
            return
        procs = list((getattr(ex, "_processes", None) or {}).values())
        ex.shutdown(wait=False, cancel_futures=True)
        for p in procs:
            try:
                p.terminate()
            except Exception:
                pass
        for p in procs:
            try:
                p.join(5)
                if p.is_alive():
                    p.kill()
            except Exception:
                pass


class BatchStream:
    """Iterates the knonik loader, handing each raw batch through ``pool``."""

    def __init__(
        self,
        *,
        name: str,
        cfg_fn: Callable[[Callable | None], tuple[dict, dict]],
        raw_batches: Callable[[dict], Iterable],
        pool: TransformPool,
        post: Callable[[list], list] | None = None,
    ):
        self._name = name
        self._cfg_fn = cfg_fn
        self._raw = raw_batches
        self._post = post
        self.pool = pool
        self.loader = None
        self.rank, self.world = rank_world()

    def _transform(self, batch: dict) -> list:
        raws = list(self._raw(batch))
        del batch
        out = self.pool(raws)
        return self._post(out) if self._post is not None else out

    def build(self) -> None:
        if self.loader is not None:
            return
        self.pool.start()
        cfg, stream_cfg = self._cfg_fn(self._transform)
        self.loader = build_loader(cfg, stream_cfg, name=self._name)

    def __iter__(self):
        self.build()
        while True:
            for out in self.loader:
                for item in out:
                    if item is not None:
                        yield item
            logger.warning("knonik plan exhausted on rank %d; re-planning", self.rank)

    def shutdown(self, timeout_s: float = 60.0) -> None:
        shutdown_loader(self.loader, timeout_s)
        self.loader = None
        self.pool.shutdown()


# --------------------------------------------------------------------------- loader process


def _put(q, stop, msg, timeout: float | None = None) -> bool:
    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        try:
            q.put(msg, timeout=0.5)
            return True
        except queue.Full:
            if msg[0] == "item" and stop.is_set():
                return False
            if deadline is not None and time.monotonic() > deadline:
                return False


def _loader_main(factory: Callable, out_q, stop, drained) -> None:
    ds = None
    try:
        ds = factory()
        for item in ds.iter_local():
            if stop.is_set() or not _put(out_q, stop, ("item", item)):
                break
    except BaseException:
        _put(out_q, stop, ("error", traceback.format_exc()), timeout=30)
    finally:
        _put(out_q, stop, ("end", None), timeout=30)
        if ds is not None:
            t = threading.Thread(target=ds.shutdown_local, daemon=True)
            t.start()
            t.join(60)
        out_q.close()
        t = threading.Thread(target=out_q.join_thread, daemon=True)
        t.start()
        t.join(30)
        drained.wait(120)
        os._exit(0)


class ProcessStream:
    """Runs a dataset in a spawned child process and yields its batches."""

    def __init__(
        self,
        factory: Callable,
        *,
        depth: int = _PROCESS_QUEUE,
        post: Callable[[Any], Any] | None = None,
        name: str = "knonik-loader",
    ):
        self._factory = factory
        self._depth = max(1, int(depth))
        self._post = post
        self._name = name
        self._p = None
        self._q = None
        self._stop = None
        self._drained = None
        self._local: queue.Queue | None = None
        self._ended = threading.Event()
        self._closed = False

    def start(self) -> None:
        if self._p is not None:
            return
        ctx = multiprocessing.get_context("spawn")
        self._q = ctx.Queue(maxsize=self._depth)
        self._stop = ctx.Event()
        self._drained = ctx.Event()
        self._p = ctx.Process(
            target=_loader_main,
            args=(self._factory, self._q, self._stop, self._drained),
            name=self._name,
            daemon=False,
        )
        self._p.start()
        self._local = queue.Queue(maxsize=2)
        threading.Thread(target=self._pump, name=f"{self._name}-pump", daemon=True).start()
        atexit.register(self.close)

    def _deliver(self, msg) -> None:
        while not self._closed:
            try:
                self._local.put(msg, timeout=0.5)
                return
            except queue.Full:
                continue

    def _pump(self) -> None:
        if self._post is not None:
            set_local_cuda_device()
        while True:
            try:
                kind, payload = self._q.get(timeout=1.0)
            except queue.Empty:
                if not self._p.is_alive():
                    self._deliver(("error", f"loader process exited ({self._p.exitcode})"))
                    self._ended.set()
                    self._drained.set()
                    return
                continue
            except Exception:
                if self._closed:
                    continue
                self._deliver(("error", traceback.format_exc()))
                continue
            if kind == "item":
                if self._closed:
                    continue
                if self._post is not None:
                    payload = self._post(payload)
            self._deliver((kind, payload))
            if kind == "end":
                self._ended.set()
                self._drained.set()
                return

    def __iter__(self):
        self.start()
        while True:
            kind, payload = self._local.get()
            if kind == "item":
                yield payload
            elif kind == "error":
                raise RuntimeError(f"knonik loader process failed:\n{payload}")
            else:
                return

    def close(self, timeout: float = 120.0) -> None:
        if self._p is None or self._closed:
            return
        self._closed = True
        self._stop.set()
        deadline = time.monotonic() + timeout
        while not self._ended.is_set() and time.monotonic() < deadline:
            try:
                self._local.get(timeout=0.2)
            except queue.Empty:
                pass
        self._drained.set()
        self._p.join(timeout=15)
        if self._p.is_alive():
            self._p.kill()
            self._p.join(5)
