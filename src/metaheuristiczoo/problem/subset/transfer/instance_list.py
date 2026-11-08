"""The ten unweighted maximum-coverage instances of the study, and how they were chosen.

Membership is explicit: an instance is here because it is written here,
and a change to the list, to a radius, to p or to the formulation is a new `TRANSFER_LIST_VERSION`,
not an edit.

The instances test configurations tuned elsewhere on an unseen coverage objective; nothing is tuned on
them (one tuned configuration per algorithm and parameter set):

    tuning (Stage 1)              7 p-median + 7 p-center, pooled   ->  tune  ->  freeze
    main comparison (Stage 2)     these ten instances               ->  run, no tuning

How the ten were chosen (fixed before any metaheuristic ran on them):

* the legitimate parameter space is exactly the (p, S) combinations published in the maximal-covering
  literature for these networks - 159 of them, every one certified here for the unweighted objective;
* a combination whose certified optimum is full coverage is dropped: it cannot separate two solutions;
* a combination the integer program does not certify is dropped, and is reported with its status;
* from the survivors: GR100, GR150, SJC324 (smallest SJC network) and SJC818 (largest) at their two
  most RESTRICTIVE published radii, and PCB3038 at its single published radius; in each cell p is the
  surviving value whose certified optimum is nearest 75 per cent of n, with PCB3038 additionally
  taking its largest surviving p, being alone in its provenance.

The 75 per cent target and the restrictive tiers are an amendment to the rule first registered: the
original "largest surviving p at the smallest and largest published radius" selected the most nearly
saturated cell in every case (six of ten above 96.8 per cent coverage, one leaving two uncovered nodes
in 818). The amendment was decided from certified optima and n alone, before any algorithm had been run
on any of these instances.

The reference values live in `reference_values.json` beside this module - value, status, witness - and
the provenance of the data lives in `mclp.MCLP_SPEC`. They are kept apart on purpose: one records what
somebody else published, the other what we computed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

TRANSFER_LIST_VERSION = 1
ROLES = ("transfer",)


# The legitimate parameter space: the (p, S) combinations published for these networks in the
# maximal-covering literature. The study uses nothing outside it, and every one of the 159 was
# certified for the unweighted objective before the selection rule ran.
PUBLISHED_SETTINGS = {
    "GR100": ((8, 10, 12), (50, 65, 80)),
    "GR150": ((8, 10, 12), (75, 80, 85, 90)),
    "SJC324": (tuple(range(1, 6)), (800, 1200, 1600)),
    "SJC402": (tuple(range(1, 7)), (800, 1200, 1600)),
    "SJC500": (tuple(range(1, 9)), (800, 1200, 1600)),
    "SJC708": (tuple(range(1, 12)), (800, 1200, 1600)),
    "SJC818": (tuple(range(1, 15)), (800, 1200, 1600)),
    "PCB3038": (tuple(range(17, 23)), (400,)),
}


@dataclass(frozen=True)
class TransferInstance:
    """One frozen Stage 2 instance. `name` is `<network>/S<radius>/p<p>`."""

    name: str
    network: str
    p: int
    radius: float
    role: str = "transfer"
    direction: str = "maximize"

    def __post_init__(self) -> None:
        if self.role not in ROLES or self.direction != "maximize":
            raise ValueError(f"{self.name}: role must be in {ROLES} and the objective is maximisation")
        if self.p < 1 or self.radius <= 0:
            raise ValueError(f"{self.name}: need p >= 1 and a positive radius, got p={self.p}, S={self.radius}")
        published_p, published_radii = PUBLISHED_SETTINGS[self.network]
        if self.p not in published_p or self.radius not in published_radii:
            raise ValueError(f"{self.name}: (p={self.p}, S={self.radius:g}) is outside the published space "
                             f"of {self.network}; Stage 2 may use nothing the literature did not")

    def build(self, root=None, *, verify: bool = True):
        """The problem, carrying its certified optimum (found by `known_optimum()`, which the result
        writer stores as the error scale) and the evaluation budget the campaign spends on it."""
        from metaheuristiczoo.problem.subset.transfer.mclp import build_instance
        problem = build_instance(self.network, self.p, self.radius, root=root, verify=verify,
                                 reference_values=reference_entries(self.name))
        problem.evaluation_budget = evaluation_budget(problem.number_of_variables(), self.p)
        return problem


BUDGET_SWEEPS, BUDGET_FLOOR, BUDGET_CAP = 100, 5_000, 1_000_000


def evaluation_budget(n: int, k: int) -> int:
    """Evaluations per run for an instance with n candidates and cardinality k: clamp(100 k (n - k), 5e3, 1e6)."""
    if not 0 < k < n:
        raise ValueError(f"need 0 < k < n, got k={k}, n={n}")
    return int(min(BUDGET_CAP, max(BUDGET_FLOOR, BUDGET_SWEEPS * k * (n - k))))


def _i(network: str, p: int, radius: float) -> TransferInstance:
    return TransferInstance(f"{network}/S{radius:g}/p{p}", network, p, float(radius))


INSTANCES: Tuple[TransferInstance, ...] = (
    # Galvao and ReVelle random networks, shortest-path distances; two most restrictive published radii
    _i("GR100", 10, 50),            # certified 76 of 100   (76.0 %)
    _i("GR100", 8, 65),             # certified 90 of 100   (90.0 %)
    _i("GR150", 10, 75),            # certified 109 of 150  (72.7 %)
    _i("GR150", 10, 80),            # certified 110 of 150  (73.3 %)
    # Sao Jose dos Campos, Euclidean; the smallest and the largest network of the family
    _i("SJC324", 2, 800),           # enumerated 247 of 324 (76.2 %)
    _i("SJC324", 1, 1200),          # enumerated 271 of 324 (83.6 %)
    _i("SJC818", 5, 800),           # certified 582 of 818  (71.2 %)
    _i("SJC818", 3, 1200),          # certified 603 of 818  (73.7 %)
    # TSPLIB circuit board, Euclidean; one published radius, so both published-p extremes that certify
    _i("PCB3038", 17, 400),         # certified 2468 of 3038 (81.2 %)
    _i("PCB3038", 20, 400),         # certified 2736 of 3038 (90.1 %)
)

NETWORKS: Tuple[str, ...] = tuple(dict.fromkeys(instance.network for instance in INSTANCES))


def instances(network: Optional[str] = None) -> Tuple[TransferInstance, ...]:
    """The frozen list, optionally restricted to one source network."""
    if network is None:
        return INSTANCES
    return tuple(i for i in INSTANCES if i.network == network)


def reference_entries(name: str) -> dict:
    """The certified entry of one instance, in the form `CoverageSubsetProblem` looks references up:
    keyed by name, matched on (instance_id, direction, k). Only certified and enumerated values are
    stored here, so `known_optimum()` always finds an optimum and never a best-known value."""
    import json
    from pathlib import Path

    entries = json.loads(Path(__file__).with_name("reference_values.json").read_text(encoding="utf-8"))["entries"]
    entry = dict(entries[name])
    entry.setdefault("k", entry["p"])
    return {name: entry}
