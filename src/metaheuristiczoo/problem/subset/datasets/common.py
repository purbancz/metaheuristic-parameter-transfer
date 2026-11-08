"""Shared machinery of the dataset loaders: pinned sources, verification, the candidate x feature container.

A loader turns published tables into a `CandidateMatrix`: one row per candidate, one column per
feature, one stored value per (candidate, feature) the source reports, plus a per-candidate table of
whatever else the source measured (groups, cohorts, concentrations) and a per-feature table. It does
not know what a candidate or a feature is; the identifiers are opaque strings and every quantity is a
number. What a stored value means (a reported value, or a value the source dropped below a floor) is
recorded in `absent_means`, because the missing-data structure decides which objectives are honest.

Provenance
----------
Every loader is bound to a `DatasetSpec`: the upstream source (a repository pinned to a commit, or a
base URL), and the SHA-256 of every file it reads. `verify` recomputes the hashes before a loader parses anything, so a
truncated clone, a moved upstream branch or a hand-edited file stops the run with the name of the file
instead of shifting a result. Text files are hashed with their line endings normalised to LF: a
Windows checkout with `core.autocrlf` rewrites them, and the content is the same.

Where the data lives
--------------------
Nothing is stored in the package. A loader takes `root`, a directory or a list of directories searched
in order, or reads `MZOO_DATA_ROOT` (directories separated by `os.pathsep`). Under each root the upstream
files sit in folders named as in `RepoPin.directory`; `fetch.fetch` creates that layout from the pins.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp

DATA_ROOT_ENV = "MZOO_DATA_ROOT"
TEXT_SUFFIXES = (".csv", ".tsv", ".txt")
FETCH_HINT = "python scripts/fetch_subset_data.py --root <directory>"

Root = Union[None, str, os.PathLike, Sequence[Union[str, os.PathLike]]]


class DatasetUnavailable(FileNotFoundError):
    """A file a loader needs is not under any data root."""


class DatasetChanged(ValueError):
    """A file's content differs from the one the loader was written and tested against."""


# -- provenance ----------------------------------------------------------------------------------

@dataclass(frozen=True)
class RepoPin:
    """Where a set of files comes from. kind="git": `url` is a repository and `commit` the full SHA the files were taken
    from. kind="http": `url` is a base URL under which each file sits at its `SourceFile.path` (a static archive such as
    OR-Library); there is no commit, the SHA-256 of every file is the pin."""
    directory: str          # folder name under a data root
    url: str
    commit: str = ""        # full SHA the files were taken from (kind="git")
    kind: str = "git"


@dataclass(frozen=True)
class SourceFile:
    repo: str               # RepoPin.directory
    path: str               # relative to the repository, forward slashes
    sha256: str             # of the content, LF-normalised for text files


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    title: str
    publication: str
    licence: str            # what was found, and where; not legal advice
    tier: str               # "primary" | "secondary" | "unit-test"
    tier_note: str          # the measured reason for the tier
    repos: Tuple[RepoPin, ...]
    files: Tuple[SourceFile, ...]

    def pin(self, directory: str) -> RepoPin:
        return next(p for p in self.repos if p.directory == directory)


def normalised_sha256(path: Union[str, os.PathLike]) -> str:
    """SHA-256 of a file; for text files every CRLF is read as LF (also across read chunks)."""
    path = Path(path)
    text = path.suffix.lower() in TEXT_SUFFIXES
    digest, carry = hashlib.sha256(), b""
    with open(path, "rb") as handle:
        while chunk := handle.read(1 << 22):
            if text:
                chunk = carry + chunk
                carry = b"\r" if chunk.endswith(b"\r") else b""
                if carry:
                    chunk = chunk[:-1]
                chunk = chunk.replace(b"\r\n", b"\n")
            digest.update(chunk)
    digest.update(carry)
    return digest.hexdigest()


def resolve_roots(root: Root = None) -> List[Path]:
    if root is None:
        env = os.environ.get(DATA_ROOT_ENV)
        if not env:
            raise DatasetUnavailable(f"no data root: pass root=... or set {DATA_ROOT_ENV}; "
                                     f"create one with `{FETCH_HINT}`")
        return [Path(p) for p in env.split(os.pathsep) if p]
    if isinstance(root, (str, os.PathLike)):
        return [Path(root)]
    return [Path(p) for p in root]


def locate(spec: DatasetSpec, source: SourceFile, roots: Sequence[Path]) -> Path:
    for base in roots:
        candidate = base / source.repo / source.path
        if candidate.is_file():
            return candidate
    raise DatasetUnavailable(f"{spec.key}: {source.repo}/{source.path} not found under {[str(r) for r in roots]}; "
                             f"fetch it with `{FETCH_HINT} --dataset {spec.key}` "
                             f"(pinned {spec.pin(source.repo).url} @ "
                             f"{spec.pin(source.repo).commit[:10] or spec.pin(source.repo).kind})")


_HASHED: Dict[Tuple[str, int, int], str] = {}


def _hash_cached(path: Path) -> str:
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    if key not in _HASHED:
        _HASHED[key] = normalised_sha256(path)
    return _HASHED[key]


def resolve_files(spec: DatasetSpec, root: Root = None, *, verify: bool = True,
                  only: Optional[Sequence[str]] = None) -> Dict[str, Path]:
    """Locate (and by default verify) the spec's files; keys are 'repo/path'. `only` limits to some paths."""
    roots = resolve_roots(root)
    found, changed = {}, []
    for source in spec.files:
        if only is not None and source.path not in only:
            continue
        path = locate(spec, source, roots)
        if verify and _hash_cached(path) != source.sha256:
            changed.append(f"{source.repo}/{source.path}: expected {source.sha256[:12]}, found {_hash_cached(path)[:12]}")
        found[f"{source.repo}/{source.path}"] = path
    if changed:
        raise DatasetChanged(f"{spec.key}: content differs from the pinned files ({'; '.join(changed)}). "
                             f"Re-fetch at the pinned commit, or pass verify=False to accept the change knowingly.")
    return found


# -- the container -------------------------------------------------------------------------------

@dataclass(frozen=True)
class CandidateMatrix:
    """Candidates x features with stored values, and what else the source measured per candidate."""

    spec: DatasetSpec
    candidates: Tuple[str, ...]
    features: Tuple[str, ...]
    values: sp.csr_matrix
    candidate_table: pd.DataFrame           # index = candidates; NaN where the source did not measure
    feature_table: pd.DataFrame             # index = features
    absent_means: str
    value_description: str
    notes: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n, m = len(self.candidates), len(self.features)
        if self.values.shape != (n, m):
            raise ValueError(f"values are {self.values.shape}, expected {(n, m)}")
        if len(set(self.candidates)) != n or len(set(self.features)) != m:
            raise ValueError("candidate and feature identifiers must be unique")
        if list(self.candidate_table.index) != list(self.candidates):
            raise ValueError("candidate_table must be indexed by the candidates, in order")
        if list(self.feature_table.index) != list(self.features):
            raise ValueError("feature_table must be indexed by the features, in order")
        if self.values.nnz and not np.all(np.isfinite(self.values.data)):
            raise ValueError("stored values must be finite")

    @property
    def key(self) -> str:
        return self.spec.key

    @property
    def shape(self) -> Tuple[int, int]:
        return self.values.shape

    @property
    def density(self) -> float:
        return self.values.nnz / (self.shape[0] * self.shape[1])


    def threshold_for(self, threshold: Optional[float] = None, quantile: Optional[float] = None) -> float:
        """A cut on the stored values: given directly, or the `quantile` of the stored values."""
        if (threshold is None) == (quantile is None):
            raise ValueError("give exactly one of threshold and quantile")
        if threshold is not None:
            return float(threshold)
        if not 0.0 < quantile < 1.0:
            raise ValueError("quantile must be in (0, 1)")
        return float(np.quantile(self.values.data, quantile))

    def binary(self, threshold: Optional[float] = None, quantile: Optional[float] = None) -> Tuple[sp.csr_matrix, float]:
        """The 0/1 matrix of stored values >= the cut (cut > 0), and the cut used."""
        cut = self.threshold_for(threshold, quantile)
        if cut <= 0.0:
            raise ValueError("the cut must be positive: absent entries are not distinguished from zeros")
        keep = self.values.copy()
        keep.data = (keep.data >= cut).astype(np.int8)
        keep.eliminate_zeros()
        return keep.astype(np.int8).tocsr(), cut

    def fingerprint(self) -> str:
        """Content hash of identifiers and stored values (candidate order and feature order are canonical)."""
        digest = hashlib.sha256()
        digest.update(repr((self.key, self.candidates, self.features)).encode("utf-8"))
        for array in (self.values.indptr, self.values.indices, self.values.data):
            digest.update(np.ascontiguousarray(array).tobytes())
        return digest.hexdigest()

    def describe(self) -> dict:
        return {"key": self.key, "tier": self.spec.tier, "n_candidates": self.shape[0], "n_features": self.shape[1],
                "stored_values": int(self.values.nnz), "density": round(self.density, 4),
                "absent_means": self.absent_means, "value": self.value_description,
                "candidate_columns": list(self.candidate_table.columns),
                "notes": {k: v for k, v in self.notes.items() if not str(k).startswith("_")},
                "fingerprint": self.fingerprint()}

    def scalar_table(self, threshold: Optional[float] = None, quantile: Optional[float] = None) -> pd.DataFrame:
        """One row per candidate: the candidate table plus separable summaries of its stored values."""
        table = self.candidate_table.copy()
        table["total_value"] = np.asarray(self.values.sum(axis=1)).ravel()
        table["max_value"] = self.values.max(axis=1).toarray().ravel() if self.values.nnz else 0.0
        if threshold is None and quantile is None:
            table["n_features"] = np.diff(self.values.indptr)
        else:
            binary, _ = self.binary(threshold, quantile)
            table["n_features"] = np.diff(binary.indptr)
        table["n_features"] = table["n_features"].astype(float)
        return table

    def to_coverage_problem(self, **kwargs):
        from metaheuristiczoo.problem.subset.datasets.coverage import CoverageSubsetProblem
        return CoverageSubsetProblem(self, **kwargs)


# -- parsing helpers -----------------------------------------------------------------------------


