"""Pinned data files and the coverage objective.

    common.py           pinned sources (base URL + SHA-256 per file), verification, `CandidateMatrix`
    coverage.py         `CoverageSubsetProblem`: the number (weight) of features a subset of candidates reaches
    reference_values.py how an instance's certified optimum is found and checked
    fetch.py            downloads the pinned files into a data root and verifies them

The data are not part of the package. `python scripts/fetch_subset_data.py --root <dir>` downloads them and
verifies every file.
"""

from metaheuristiczoo.problem.subset.datasets.common import (CandidateMatrix, DatasetChanged, DatasetSpec,
                                                              DatasetUnavailable, RepoPin, SourceFile,
                                                              normalised_sha256, resolve_files)
from metaheuristiczoo.problem.subset.datasets.coverage import CoverageSubsetProblem

__all__ = ["CandidateMatrix", "CoverageSubsetProblem", "DatasetChanged", "DatasetSpec", "DatasetUnavailable",
           "RepoPin", "SourceFile", "normalised_sha256", "resolve_files"]
