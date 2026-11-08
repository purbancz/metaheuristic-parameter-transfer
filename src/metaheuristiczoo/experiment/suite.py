"""The problems of the study: the ten maximum-coverage instances of `problem/subset/transfer/`.

Unweighted fixed-cardinality maximum coverage on maximal-covering benchmark networks, with the (p, S)
settings published for them and a certified optimum for every one. Each problem carries its optimum, so
the result writer stores the error against a proven value, and its evaluation budget,
`clamp(100 p (n - p), 5e3, 1e6)`, five times the per-run budget of subset tuning unless the cap applies.
"""

from metaheuristiczoo.experiment.results.layout import TRANSFER


def transfer_problems(root=None):
    """The ten instances, built from the networks under the data root (`MZOO_DATA_ROOT` or `root`)."""
    from metaheuristiczoo.problem.subset.transfer.instance_list import instances
    return [spec.build(root) for spec in instances()]


def build_suite(sections=None, subset_root=None):
    """Section -> problems. The study has one section, `transfer`."""
    if sections not in (None, [TRANSFER], (TRANSFER,)):
        raise ValueError(f"this package has one suite section, {TRANSFER!r}; got {sections!r}")
    return {TRANSFER: transfer_problems(subset_root)}
