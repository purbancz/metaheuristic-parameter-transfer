"""Reference methods that search subset space directly, on the same evaluation budget.

    random_search           independent random subsets
    greedy_forward          add the best candidate until the target cardinality
    local_search            1-swap (and, for a cardinality range, add / remove) descent
    restarted_local_search  local search from random starts until the budget is spent
    iterated_local_search   local search, then repeated random kicks of the local optimum and re-descent

Every method scores through `SubsetProblem.score_subset`, which counts in `evaluation_count`
exactly as `evaluate` does, so a baseline's budget is the same currency as an optimiser's and
the two can be put on one fitness-versus-evaluations plot (`best_so_far_curve`).

Each method returns a `BaselineResult`: the best subset and value, the evaluations it spent
(read off the problem's counter, before and after), the improvement events as
(evaluations, value) pairs, and why it stopped.

Cardinality
-----------
The admissible cardinalities come from the problem's decoder (`cardinality_bounds()`): a top-K
problem is searched at exactly K, a threshold problem over its range. `greedy_forward` is the
exception - it must score subsets SMALLER than the target while it builds one - and the objective
has to be defined there; an application that penalises small subsets will mislead it, and the
docstring says so rather than the code guessing a fix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from metaheuristiczoo.problem.subset.decode import Subset
from metaheuristiczoo.problem.subset.operations import add_neighbours, as_subset, random_subset_in_range, remove_neighbours, swap_neighbours
from metaheuristiczoo.problem.subset.subset_problem import SubsetProblem

Event = Tuple[int, float]           # (problem.evaluation_count after the evaluation, new best value)


@dataclass
class BaselineResult:
    method: str
    best_subset: Optional[Subset]
    best_value: float
    evaluations: int                       # spent by this call (problem counter after - before)
    improvements: List[Event] = field(default_factory=list)
    stopped_because: str = ""
    details: dict = field(default_factory=dict)

    def best_so_far_curve(self, budget: int, interval: int) -> np.ndarray:
        """Best value after every `interval` evaluations up to `budget`, in the runner's shape
        (FitnessObserver): NaN before the first evaluation, the final best after the last."""
        curve = np.full(int(budget) // int(interval), np.nan)
        best = np.nan
        events = list(self.improvements)
        for j in range(curve.size):
            at = (j + 1) * interval
            while events and events[0][0] <= at:
                best = events.pop(0)[1]
            curve[j] = best
        return curve


class _Tracker:
    """Scores through the problem and records improvements against its evaluation counter."""

    def __init__(self, problem: SubsetProblem, budget: Optional[int], method: str):
        self.problem, self.method = problem, method
        self.start = problem.evaluation_count
        self.budget = None if budget is None else int(budget)
        self.best_subset: Optional[Subset] = None
        self.best_value = float("inf")
        self.improvements: List[Event] = []

    @property
    def spent(self) -> int:
        return self.problem.evaluation_count - self.start

    def can_afford(self, n: int = 1) -> bool:
        return self.budget is None or self.spent + n <= self.budget

    def score(self, subset: Subset, admissible: bool = True) -> float:
        """Score a subset (always counted). Only an `admissible` one - a cardinality the decoder can
        produce - may become the reported best: a shorter prefix that no decoded state can equal is
        not an answer to the problem, however good its value."""
        value = self.problem.score_subset(subset)
        if admissible and value < self.best_value:
            self.best_value, self.best_subset = value, subset
            self.improvements.append((self.problem.evaluation_count, value))
        return value

    def result(self, stopped_because: str, **details) -> BaselineResult:
        return BaselineResult(self.method, self.best_subset, self.best_value, self.spent,
                              list(self.improvements), stopped_because, dict(details))


def _bounds(problem: SubsetProblem, cardinality: Optional[int]) -> Tuple[int, int]:
    lo, hi = problem.decoder.cardinality_bounds()
    if cardinality is not None:
        if not lo <= int(cardinality) <= hi:
            raise ValueError(f"cardinality {cardinality} outside the decoder's range [{lo}, {hi}]")
        lo = hi = int(cardinality)
    return lo, hi


# -- exact -------------------------------------------------------------------------------------------


# -- random ------------------------------------------------------------------------------------------

def random_search(problem: SubsetProblem, budget: int, rng=0, cardinality: Optional[int] = None) -> BaselineResult:
    """`budget` independent subsets: cardinality uniform on the decoder's range, then uniform of
    that size (operations.random_subset_in_range, whose docstring states what this is not)."""
    rng = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
    n = problem.number_of_variables()
    lo, hi = _bounds(problem, cardinality)
    tracker = _Tracker(problem, budget, "random")
    while tracker.can_afford():
        tracker.score(random_subset_in_range(rng, n, lo, hi))
    return tracker.result("budget")


# -- greedy ------------------------------------------------------------------------------------------

def greedy_forward(problem: SubsetProblem, target_cardinality: Optional[int] = None, budget: Optional[int] = None,
                   start: Sequence[int] = ()) -> BaselineResult:
    """From `start` (default empty), repeatedly add the single candidate that gives the best
    objective, until `target_cardinality` (default: the decoder's maximum). Scores subsets of
    every intermediate size - the objective must be defined there - and counts those evaluations.
    Deterministic; ties go to the lowest candidate index.

    Reports the best subset seen at an ADMISSIBLE size, i.e. one inside the decoder's cardinality
    range, and the final one in details. Under an exact-cardinality decoder that is the final subset
    only: an earlier, shorter prefix is not a solution, and reporting it would let an objective that
    improves with fewer members (a union to be minimised, a cost) make greedy look better than the
    optimum. Under a range decoder every size in the range counts. If the budget ends before an
    admissible size is reached, `best_subset` is None and `best_value` is inf."""
    n = problem.number_of_variables()
    lo, hi = problem.decoder.cardinality_bounds()
    target = hi if target_cardinality is None else int(target_cardinality)
    current = as_subset(start)
    if not len(current) <= target <= n:
        raise ValueError(f"target cardinality must lie in [{len(current)}, {n}], got {target}")
    tracker = _Tracker(problem, budget, "greedy")
    steps = 0
    while len(current) < target:
        best_next, best_next_value = None, float("inf")
        for candidate in add_neighbours(current, n):
            if not tracker.can_afford():
                return tracker.result("budget", final_subset=current, steps=steps)
            value = tracker.score(candidate, admissible=lo <= len(candidate) <= hi)
            if value < best_next_value:
                best_next, best_next_value = candidate, value
        current = best_next
        steps += 1
    return tracker.result("target cardinality reached", final_subset=current, final_value=best_next_value if steps else None,
                          steps=steps)


# -- local search ------------------------------------------------------------------------------------

def _neighbours(subset: Subset, n: int, lo: int, hi: int, moves: str):
    k = len(subset)
    if moves in ("swap", "all"):
        yield from swap_neighbours(subset, n)
    if moves in ("add", "all") and k < hi:
        yield from add_neighbours(subset, n)
    if moves in ("remove", "all") and k > lo:
        yield from remove_neighbours(subset)


def local_search(problem: SubsetProblem, start: Sequence[int], budget: Optional[int] = None, rng=0,
                 strategy: str = "first", moves: str = "swap") -> BaselineResult:
    """Descent from `start` over the 1-move neighbourhood.

    strategy="first"    scan the neighbourhood in a random order (rng) and take the first improvement;
    strategy="steepest" score the whole neighbourhood and take the best improvement.
    moves="swap" keeps the cardinality; "all" adds add / remove moves inside the decoder's range.
    Stops at a local optimum (details["local_optimum"] = True) or when the budget is exhausted."""
    if strategy not in ("first", "steepest"):
        raise ValueError("strategy must be 'first' or 'steepest'")
    rng = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
    n = problem.number_of_variables()
    lo, hi = problem.decoder.cardinality_bounds()
    current = as_subset(start)
    if not lo <= len(current) <= hi:
        raise ValueError(f"start cardinality {len(current)} outside the decoder's range [{lo}, {hi}]")
    tracker = _Tracker(problem, budget, f"local-{strategy}-{moves}")
    if not tracker.can_afford():
        return tracker.result("budget", local_optimum=False, moves_made=0)
    current_value = tracker.score(current)
    moves_made = 0
    while True:
        neighbours = list(_neighbours(current, n, lo, hi, moves))
        if strategy == "first":
            rng.shuffle(neighbours)
        best_move, best_move_value = None, current_value
        for neighbour in neighbours:
            if not tracker.can_afford():
                return tracker.result("budget", local_optimum=False, moves_made=moves_made, final_subset=current)
            value = tracker.score(neighbour)
            if value < best_move_value:
                best_move, best_move_value = neighbour, value
                if strategy == "first":
                    break
        if best_move is None:
            return tracker.result("local optimum", local_optimum=True, moves_made=moves_made, final_subset=current)
        current, current_value = best_move, best_move_value
        moves_made += 1


def restarted_local_search(problem: SubsetProblem, budget: int, rng=0, strategy: str = "first", moves: str = "swap",
                           cardinality: Optional[int] = None) -> BaselineResult:
    """Local search from independent random starts until `budget` is spent; the domain-independent
    reference. details: restarts, how many ended in a local optimum."""
    rng = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
    n = problem.number_of_variables()
    lo, hi = _bounds(problem, cardinality)
    tracker = _Tracker(problem, budget, f"restarted-local-{strategy}-{moves}")
    restarts, converged = 0, 0
    while tracker.can_afford():
        remaining = tracker.budget - tracker.spent
        run = local_search(problem, random_subset_in_range(rng, n, lo, hi), budget=remaining, rng=rng,
                           strategy=strategy, moves=moves)
        restarts += 1
        converged += bool(run.details.get("local_optimum"))
        if run.best_value < tracker.best_value:
            tracker.best_value, tracker.best_subset = run.best_value, run.best_subset
        tracker.improvements.extend(e for e in run.improvements if e[1] <= tracker.best_value)
    tracker.improvements = _monotone(tracker.improvements)
    return tracker.result("budget", restarts=restarts, local_optima=converged)


def _kick(subset: Subset, n: int, lo: int, hi: int, strength: int, moves: str, rng: np.random.Generator) -> Subset:
    """`strength` random moves from `subset`: swaps (the cardinality is kept) and, with moves="all", also adds and
    removes while the size stays inside [lo, hi]. A move that cannot be made (no candidate outside to swap in, say)
    is skipped, so a kick of a subset with no admissible move returns it unchanged."""
    current = set(subset)
    for _ in range(strength):
        options = ["swap"]
        if moves == "all":
            options += (["add"] if len(current) < hi else []) + (["remove"] if len(current) > lo else [])
        move = options[int(rng.integers(len(options)))]
        outside = [i for i in range(n) if i not in current]
        if move == "swap" and current and outside:
            current.remove(int(rng.choice(sorted(current))))
            current.add(int(rng.choice(outside)))
        elif move == "add" and outside:
            current.add(int(rng.choice(outside)))
        elif move == "remove" and current:
            current.remove(int(rng.choice(sorted(current))))
    return tuple(sorted(current))


ACCEPTANCE = ("better", "not_worse", "always")


def iterated_local_search(problem: SubsetProblem, budget: int, rng=0, strength: int = 2, acceptance: str = "not_worse",
                          restart_after: Optional[int] = None, strategy: str = "first", moves: str = "swap",
                          start: Optional[Sequence[int]] = None, cardinality: Optional[int] = None) -> BaselineResult:
    """Iterated local search (Lourenco, Martin & Stuetzle): descend to a local optimum, then repeat - kick it with
    `strength` random moves, descend again, and keep the new local optimum or the old one by `acceptance`.

    acceptance="better"     accept only a strictly better local optimum;
    acceptance="not_worse"  also accept an equal one - the default, because equal values are common on plateaus and
                            the walk may drift across them (the coverage instances have wide ones);
    acceptance="always"     accept every local optimum (a random walk over local optima).
    restart_after=m         after m consecutive iterations without a new best-so-far, restart from a random subset
                            (None: never), the usual guard against an ILS that has settled in one basin.
    strength                the number of random moves in a kick. It must be large enough that the descent cannot
                            simply undo it: with 1-swap descent a kick of one swap is undone almost every time.
    start                   the first subset (default: random in the decoder's range); strategy and moves as in
                            `local_search`. The kick uses the same moves, so with moves="swap" every subset visited
                            has the cardinality of the start.

    Every evaluation is counted on the problem and the whole budget is spent; the best subset seen is returned.
    details: iterations (kicks), accepted, restarts, local_optima (descents that ended in a local optimum)."""
    if int(strength) < 1:
        raise ValueError("strength must be at least 1")
    if acceptance not in ACCEPTANCE:
        raise ValueError(f"acceptance must be one of {ACCEPTANCE}, got {acceptance!r}")
    if restart_after is not None and int(restart_after) < 1:
        raise ValueError("restart_after must be at least 1 (or None)")
    rng = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
    n = problem.number_of_variables()
    lo, hi = _bounds(problem, cardinality)
    first = random_subset_in_range(rng, n, lo, hi) if start is None else as_subset(start)
    dlo, dhi = problem.decoder.cardinality_bounds()
    if not dlo <= len(first) <= dhi:
        raise ValueError(f"start cardinality {len(first)} outside the decoder's range [{dlo}, {dhi}]")
    tracker = _Tracker(problem, budget, f"iterated-local-{strategy}-{moves}")

    def descend(initial: Subset) -> BaselineResult:
        run = local_search(problem, initial, budget=tracker.budget - tracker.spent, rng=rng, strategy=strategy,
                           moves=moves)
        if run.best_subset is not None and run.best_value < tracker.best_value:
            tracker.best_value, tracker.best_subset = run.best_value, run.best_subset
        tracker.improvements.extend(e for e in run.improvements if e[1] <= tracker.best_value)
        return run

    iterations = accepted = restarts = stagnation = local_optima = 0
    run = descend(first)
    if run.best_subset is not None:
        current, current_value = run.best_subset, run.best_value
        local_optima += bool(run.details.get("local_optimum"))
        while tracker.can_afford():
            iterations += 1
            best_before = tracker.best_value
            run = descend(_kick(current, n, lo, hi, int(strength), moves, rng))
            if run.best_subset is None:
                break
            local_optima += bool(run.details.get("local_optimum"))
            if (run.best_value < current_value or acceptance == "always"
                    or (acceptance == "not_worse" and run.best_value == current_value)):
                current, current_value = run.best_subset, run.best_value
                accepted += 1
            stagnation = 0 if tracker.best_value < best_before else stagnation + 1
            if restart_after is not None and stagnation >= restart_after and tracker.can_afford():
                run = descend(random_subset_in_range(rng, n, lo, hi))
                restarts += 1
                stagnation = 0
                if run.best_subset is None:
                    break
                current, current_value = run.best_subset, run.best_value
                local_optima += bool(run.details.get("local_optimum"))
    tracker.improvements = _monotone(tracker.improvements)
    return tracker.result("budget", iterations=iterations, accepted=accepted, restarts=restarts,
                          local_optima=local_optima, strength=int(strength), acceptance=acceptance)


def _monotone(events: List[Event]) -> List[Event]:
    """Keep only the events that improved on everything before them, in evaluation order."""
    out, best = [], float("inf")
    for at, value in sorted(events):
        if value < best:
            out.append((at, value))
            best = value
    return out


