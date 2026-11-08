"""The evaluation budget every factory gives its algorithm."""

from metaheuristiczoo.experiment.protocol import EVALUATIONS_PER_DIMENSION


def problem_budget(p) -> int:
    """10^4 x D (protocol.EVALUATIONS_PER_DIMENSION), unless the problem carries its own budget.

    A problem may set `evaluation_budget` (an int): the subset-selection instances have a budget rule
    of their own (problem/subset/transfer/instance_list.py), and a check may run at a scaled budget. It is
    read HERE, at construction, because several algorithms register their termination criterion with the
    observable when they are built (LSHADE and the other SOTA reproductions) - a criterion swapped in
    afterwards is never updated and never fires (found 5 Oct 2026: LSHADE ran towards 10^4 x D)."""
    own = getattr(p, "evaluation_budget", None)
    if own is not None:
        budget = int(own)
        if budget <= 0:
            raise ValueError(f"{p.name()}: evaluation_budget must be positive, got {own!r}")
        return budget
    return EVALUATIONS_PER_DIMENSION * p.number_of_variables()
