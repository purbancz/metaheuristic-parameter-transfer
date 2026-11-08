"""Reference values of an instance: its certified optimum and how an entry is found and checked.

The transfer instances carry their entries (problem/subset/transfer/reference_values.json) in the candidate
matrix, and `CoverageSubsetProblem.reference()` finds the entry by (instance_id, direction, k). `instance_id` is
the content hash of the problem's data, so a changed network or radius finds nothing rather than a stale value.

Only "enumerated" (every subset scored) and "certified" (an integer program proved it) entries are optima;
`known_optimum()` returns nothing else, because the result writer stores it as the error scale.
`verify(problem, entry)` checks the identity, direction and k, and re-scores the witness subset when present.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

STATUSES = ("enumerated", "certified", "best_known")
EXACT = ("enumerated", "certified")


def find_in(entries: Dict[str, dict], instance_ids: Sequence[str], direction: str, k: int) -> Optional[dict]:
    ids = set(instance_ids)
    for entry in entries.values():
        if entry["instance_id"] in ids and entry["direction"] == direction and entry["k"] == k:
            return entry
    return None


def verify(problem, entry: dict) -> None:
    """Raise ValueError unless the entry describes `problem` and its witness scores the stored value on it."""
    ids = set(getattr(problem, "reference_instance_ids", (problem.instance_id,)))
    if entry["instance_id"] not in ids:
        raise ValueError(f"{entry['name']}: instance_id differs (the data, the cut or the loader changed)")
    if entry["direction"] != problem.direction or entry["k"] != problem.decoder.k:
        raise ValueError(f"{entry['name']}: direction or k differs from the problem")
    if "witness" not in entry:
        return
    index = {candidate: i for i, candidate in enumerate(problem.candidates)}
    subset = tuple(sorted(index[c] for c in entry["witness"]))
    if len(subset) != entry["k"]:
        raise ValueError(f"{entry['name']}: witness has {len(subset)} candidates, expected {entry['k']}")
    scored = problem.aggregate(subset)
    if scored != entry["value"]:
        raise ValueError(f"{entry['name']}: witness scores {scored}, stored value is {entry['value']}")
    bound, value = entry.get("bound"), entry["value"]
    if bound is not None and (bound < value - 1e-6 if problem.direction == "maximize" else bound > value + 1e-6):
        raise ValueError(f"{entry['name']}: bound {bound} is on the wrong side of value {value}")
