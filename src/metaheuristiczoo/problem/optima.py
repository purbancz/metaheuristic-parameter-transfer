"""The known optimum and best known value of a problem, as the result writer stores them.

The value is what turns a stored objective into an ERROR (fitness minus optimum); the result writer stores
it beside each problem (`known_optimum` attribute) so analyses never depend on the problem class being
importable later. The study's instances carry certified optima (problem/subset/transfer/reference_values.json).
"""


def known_optimum(problem):
    """The problem's `known_optimum()` when it defines one, else None.

    A problem may both define the method and have no optimum to report (a subset problem whose stored
    reference is only a best known value returns None), so the result is not assumed to be a number."""
    method = getattr(problem, "known_optimum", None)
    if not callable(method):
        return None
    value = method()
    return None if value is None else float(value)


def best_known(problem):
    """The problem's `best_known()` when it defines one, else None: the best stored value whatever its
    status, so for an open instance it is a value a run may beat, and never an error scale."""
    method = getattr(problem, "best_known", None)
    if not callable(method):
        return None
    value = method()
    return None if value is None else float(value)
