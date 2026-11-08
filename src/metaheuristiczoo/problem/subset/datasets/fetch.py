"""Download the instance networks into a data root and verify them against their pinned SHA-256.

    fetch(root)          # the eight networks of the maximal-covering archive (problem/subset/transfer/mclp.py)

The files are downloaded one by one from the archive's base URL to `<root>/mclp/<path>`. A directory that
already exists is never modified: if a needed file is missing from it the call fails and says so.
"""

from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path
from typing import List, Union

from metaheuristiczoo.problem.subset.datasets.common import DatasetSpec, resolve_files


def _download(url: str, target: Path, timeout: float = 60.0) -> None:
    """Download `url` to `target` through a temporary file, so an interrupted transfer never leaves a partial file."""
    if not url.startswith(("https://", "http://")):
        raise ValueError(f"refusing to download from {url!r}: only http(s) URLs")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    with urllib.request.urlopen(url, timeout=timeout) as response, open(partial, "wb") as handle:
        shutil.copyfileobj(response, handle)
    partial.replace(target)


def verify_dataset(spec: DatasetSpec, root: Union[str, Path]) -> List[str]:
    """Verify every file of `spec` under `root`; raises DatasetUnavailable or DatasetChanged (see `resolve_files`)."""
    resolve_files(spec, root, verify=True)
    return []


def fetch(root: Union[str, Path], spec: DatasetSpec | None = None) -> Path:
    """Download the files of `spec` (default: the transfer networks) into `root` and verify them; returns the root."""
    if spec is None:
        from metaheuristiczoo.problem.subset.transfer.mclp import MCLP_SPEC as spec
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for pin in spec.repos:
        paths = [source.path for source in spec.files if source.repo == pin.directory]
        dest = root / pin.directory
        if dest.exists():
            missing = [p for p in paths if not (dest / p).is_file()]
            if missing:
                raise FileExistsError(f"{dest} exists but lacks {missing}; it is not modified - remove it or use another root")
            continue
        try:
            for path in paths:
                _download(f"{pin.url.rstrip('/')}/{path}", dest / path)
        except Exception:
            shutil.rmtree(dest, ignore_errors=True)            # no half-filled directory that the next call would trust
            raise
    verify_dataset(spec, root)
    return root
