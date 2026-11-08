"""The Stage 2 source networks: established maximal-covering benchmark data, pinned, read, and covered.

    network = load_network("SJC324", root)              # distances in the source's own units
    reach   = coverage_matrix(network, radius=800)      # 0/1 site x node; a site covers itself
    problem = build_instance("SJC324", p=2, radius=800, root=root)

The networks come from the archive distributed with the 2024 INFORMS Journal on Computing
cross-entropy paper (github.com/HWangUPV/MCLP, CC BY-NC-SA 4.0), which collects the families used in
the maximal-covering literature. Their own provenance:

    GR100, GR150          Galvao and ReVelle 1996       random networks, edge lists
    SJC324 ... SJC818     Lorena and Pereira 2002       Sao Jose dos Campos, coordinates and demand
    PCB3038               Reinelt 1994 (TSPLIB)         circuit-board geometry

The Beasley family of the same archive (B700, B900) is DELIBERATELY ABSENT: those files are the
OR-Library p-median networks at n = 700 and n = 900, three of the seven graphs Stage 1 tunes on, so
using them would contaminate the transfer test at the level of the data rather than the objective.
The invariant is a test, not a comment (tests/problem/subset/test_transfer_independence.py).

Distance semantics follow the archive's own reader: Euclidean for the coordinate families (SJC, PCB),
shortest path over the edge list for GR. Two deliberate differences from it. That reader fills absent
edges with the sentinel 999 and compares `distance <= r`, which counts an unreachable pair as covered
once the radius passes the sentinel; GR100 IS disconnected (198 unreachable ordered pairs), so this
module uses infinity and an unreachable pair is never covered. And the per-node demands are not read
at all, because the Stage 2 formulation is unweighted - which also means GR's fragile convention of
packing one demand into the fourth column of each of the first n edge rows is never relied on.

The formulation is unweighted fixed-cardinality maximum coverage: choose p of the n sites to maximise
the NUMBER of nodes within distance S of a selected site. The classical MCLP objective is demand-
weighted, so these are not MCLP instances in the strict sense but max-k-cover instances derived from
MCLP benchmark networks, and published MCLP objective values do not apply to them. Every reference
value this suite uses is certified for this formulation (transfer/reference_values.json).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
import scipy.sparse as sp

from metaheuristiczoo.problem.subset.datasets.common import (CandidateMatrix, DatasetSpec, RepoPin, Root,
                                                              SourceFile, resolve_files)

ARCHIVE = RepoPin("mclp", "https://raw.githubusercontent.com/HWangUPV/MCLP/main/MCLP_data", kind="http")

# The archive keeps each family in its own folder, and the path is both the URL suffix and the layout
# under the data root. "G&R" is served literally by raw.githubusercontent (checked), so no encoding.
PINNED_SHA256 = {
    "G&R/GR100.txt": "e4876bbacd6c3023b943aaaba2c932fa3a1b543edf91b683c7365be1928b19e2",
    "G&R/GR150.txt": "d19a47c417102cc8428848b1fbccd0c9091e68662a55e194209da4eb3366e012",
    "SJC/SJC324.txt": "7885c1fa33aea434ab5ad41f0cc0a30e140b27693bf6e36948f242233b17885e",
    "SJC/SJC402.txt": "66d53c2da1cdda907d2968da41afcaa1070909e9fbb511f8b1429d2869e0a6a2",
    "SJC/SJC500.txt": "7f8459596bf134998d8e5c9b98aeaca6fddcf9be4fcaec1736220e76acb3e50c",
    "SJC/SJC708.txt": "f147049232414d8b8af62f75248e6a1013fa6ca90991bafa800f5155321d0e1a",
    "SJC/SJC818.txt": "50d1bf49f2ba2d4cd9d7e19f6565807fea255820cf3770e03f82e08589c56ea7",
    "PCB/PCB3038.txt": "67ec9ff722d0fa141c6a5764dbfc03414129ad130f2208c4b0883d76c32938b3",
}
GRAPH_NETWORKS = ("GR100", "GR150")                   # edge lists; the others are coordinate files
SOURCE_FILE = {path.rsplit("/", 1)[1][:-4]: path for path in PINNED_SHA256}
NETWORKS = tuple(sorted(SOURCE_FILE))

MCLP_SPEC = DatasetSpec(
    "mclp_transfer",
    "Maximal-covering benchmark networks used as Stage 2 transfer instances (GR, SJC, PCB)",
    "Galvao and ReVelle, Eur. J. Oper. Res. 88 (1996) 114-123 (GR); Lorena and Pereira (2002), Sao Jose dos "
    "Campos (SJC); Reinelt, TSPLIB (1991) (PCB); collected in the archive of the 2026 INFORMS Journal on "
    "Computing cross-entropy paper at github.com/HWangUPV/MCLP",
    "the archive is distributed under CC BY-NC-SA 4.0; the original data sets carry their own terms",
    "transfer",
    "Three provenances - random network, real city, circuit-board geometry - and none shares a network with a "
    "Stage 1 tuning graph. The archive's Beasley family is excluded because it IS the OR-Library p-median data.",
    (ARCHIVE,),
    tuple(SourceFile("mclp", name, digest) for name, digest in PINNED_SHA256.items()))


@dataclass(frozen=True)
class Network:
    """A source network: pairwise distances in its own units, and what produced them."""

    name: str
    distances: np.ndarray                             # (n, n); infinite where unreachable
    semantics: str                                    # "euclidean" | "shortest-path"

    @property
    def n(self) -> int:
        return int(self.distances.shape[0])


def read_coordinates(path) -> np.ndarray:
    """`x y demand` per row after one header line, Euclidean distances. The demand is not read."""
    rows = [line.split() for line in open(path).read().splitlines()[1:] if line.strip()]
    xy = np.array([[float(r[0]), float(r[1])] for r in rows], dtype=float)
    return np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))


def read_edge_list(path) -> np.ndarray:
    """`i j weight [demand]` per row after the header; shortest paths, infinity where unreachable."""
    from scipy.sparse.csgraph import floyd_warshall

    lines = [line.split() for line in open(path).read().splitlines() if line.strip()]
    n = int(lines[0][0])
    graph = np.full((n, n), np.inf)
    np.fill_diagonal(graph, 0.0)
    for edge in lines[1:]:
        i, j, weight = int(edge[0]) - 1, int(edge[1]) - 1, float(edge[2])
        graph[i, j] = graph[j, i] = min(graph[i, j], weight)
    return floyd_warshall(graph, directed=False)


@lru_cache(maxsize=16)
def _network(path: str, modified: int, name: str) -> Network:
    if name in GRAPH_NETWORKS:
        return Network(name, read_edge_list(path), "shortest-path")
    return Network(name, read_coordinates(path), "euclidean")


def load_network(name: str, root: Root = None, *, verify: bool = True) -> Network:
    """The pinned file of `name`, hash-checked, read into distances."""
    if name not in NETWORKS:
        raise ValueError(f"unknown network {name!r}; pinned: {list(NETWORKS)}")
    file = SOURCE_FILE[name]
    path = resolve_files(MCLP_SPEC, root, verify=verify, only=(file,))[f"mclp/{file}"]
    return _network(str(path), path.stat().st_mtime_ns, name)


def coverage_matrix(network: Network, radius: float) -> sp.csr_matrix:
    """0/1 site x node: site i covers node j when their distance is at most `radius`. A site covers
    itself, and a pair the network cannot connect is never covered."""
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}")
    return sp.csr_matrix((network.distances <= radius).astype(np.int8))


def candidate_matrix(network: Network, radius: float, reference_values: dict = None) -> CandidateMatrix:
    """The reach matrix in the container `CoverageSubsetProblem` takes, already binary."""
    reach = coverage_matrix(network, radius)
    sites = tuple(f"{network.name}:site{i}" for i in range(network.n))
    nodes = tuple(f"{network.name}:node{j}" for j in range(network.n))
    return CandidateMatrix(
        spec=MCLP_SPEC, candidates=sites, features=nodes, values=reach,
        candidate_table=pd.DataFrame(index=list(sites)), feature_table=pd.DataFrame(index=list(nodes)),
        absent_means="the node lies further from the site than the coverage radius",
        value_description=f"1 when the site covers the node within {radius:g} ({network.semantics})",
        notes={"_reference_values": dict(reference_values or {})})


def build_instance(name: str, p: int, radius: float, root: Root = None, *, verify: bool = True,
                   reference_values: dict = None):
    """One Stage 2 instance: unweighted maximum coverage by `p` sites at `radius` on network `name`."""
    from metaheuristiczoo.problem.subset.datasets.coverage import CoverageSubsetProblem

    network = load_network(name, root, verify=verify)
    if not 0 < p < network.n:
        raise ValueError(f"need 0 < p < n, got p={p}, n={network.n}")
    return CoverageSubsetProblem(candidate_matrix(network, radius, reference_values), threshold=0.5,
                                 direction="maximize",
                                 k=int(p), label=f"{name}/S{radius:g}/p{p}")
