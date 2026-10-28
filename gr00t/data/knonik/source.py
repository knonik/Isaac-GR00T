# SPDX-License-Identifier: Apache-2.0
"""Where a knonik dataset is read from: local disk, or Knonik Hub."""

from __future__ import annotations

import os
from pathlib import Path

from gr00t.data.knonik import hub


MANIFEST = "hlp_manifest.json"
DEFAULT_SUBDIR = "knonik"


def resolve_manifest(
    source: str | os.PathLike | None,
    dataset_path: str | os.PathLike,
    cache_dir: str | os.PathLike | None = None,
) -> Path:
    """Path of the local ``hlp_manifest.json`` to train from.

    ``source`` is a directory produced by ``scripts/knonik/prepare_dataset.py``,
    the manifest inside it, or ``hub://<org>/<name>[@rev]``. Left empty, it is
    ``<dataset_path>/knonik``, where the prepare script writes by default, or
    ``dataset_path`` itself when that is the prepared directory.
    """
    if not source:
        path = Path(dataset_path)
        if not (path / MANIFEST).is_file():
            path = path / DEFAULT_SUBDIR
    elif hub.is_hub_uri(source):
        path = hub.fetch(str(source), cache_dir)
    else:
        text = str(source)
        if text.startswith("file://"):
            text = text[len("file://") :]
        elif "://" in text:
            raise ValueError(
                f"knonik source {text!r}: only local paths and hub:// datasets are supported. "
                "Copy the prepared dataset to local disk and pass its directory."
            )
        path = Path(text).expanduser()
    if path.is_dir():
        path = path / MANIFEST
    if not path.is_file():
        raise FileNotFoundError(
            f"no {MANIFEST} at {path}: run scripts/knonik/prepare_dataset.py on the "
            "LeRobot dataset first, or pass --knonik-source"
        )
    return path.resolve()
