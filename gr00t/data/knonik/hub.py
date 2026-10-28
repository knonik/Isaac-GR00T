# SPDX-License-Identifier: Apache-2.0
"""Knonik Hub datasets (coming soon).

A Hub dataset is addressed as ``hub://<org>/<name>`` with an optional
``@<revision>``. Once the Hub is live, :func:`fetch` will download (or reuse a
cached copy of) the dataset's KHLP packs and its LeRobot ``meta/`` directory,
and return the local directory that holds ``hlp_manifest.json``. Everything
downstream of :func:`fetch` already works on that local directory, so turning
the Hub on only means implementing :func:`fetch`.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re


HUB_SCHEMES = ("hub://", "knonik://")
HUB_URL = "https://www.knonik.com"

_REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class HubDataset:
    org: str
    name: str
    revision: str = "main"

    @property
    def repo_id(self) -> str:
        return f"{self.org}/{self.name}"


def is_hub_uri(value: str | os.PathLike | None) -> bool:
    return str(value or "").lower().startswith(HUB_SCHEMES)


def parse_hub_uri(uri: str) -> HubDataset:
    text = str(uri)
    for scheme in HUB_SCHEMES:
        if text.lower().startswith(scheme):
            text = text[len(scheme) :]
            break
    else:
        raise ValueError(f"{uri!r} is not a Knonik Hub URI (expected hub://<org>/<name>[@rev])")
    repo, _, revision = text.strip("/").partition("@")
    if not _REPO_RE.match(repo):
        raise ValueError(f"{uri!r}: expected hub://<org>/<name>[@rev]")
    return HubDataset(*repo.split("/"), revision=revision or "main")


def default_cache_dir() -> Path:
    root = os.environ.get("KNONIK_HUB_CACHE") or os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "knonik", "hub"
    )
    return Path(root)


def fetch(uri: str, cache_dir: str | os.PathLike | None = None) -> Path:
    """Local directory holding ``hlp_manifest.json`` for a Hub dataset."""
    dataset = parse_hub_uri(uri)
    raise NotImplementedError(
        f"Loading {dataset.repo_id}@{dataset.revision} from Knonik Hub is coming soon "
        f"({HUB_URL}). For now, prepare the dataset locally "
        "(scripts/knonik/prepare_dataset.py) and pass its directory as the knonik source."
    )
