"""Which algorithm each roster name builds, and with which parameter values.

A factory takes a problem and returns a constructed, unrun algorithm. The package is split by
how a factory comes to exist, which is the one thing about it that does not change:

* `declared`         - this repository's own algorithms, one written-out function each.
* `external`         - reproductions of algorithms published by other people.
* `subset_baselines` - the four baselines that search subset space directly (added by the campaign).

Where a factory's numbers come from is not in the factory: since 3 October 2026 every factory
builds through `building.build`, which reads the values of the factory's tuning block from the
active parameter set (`experiment.parameters` - an irace campaign, or the published values), with
the factory's own fixed settings on top. Changing set changes the values, never the algorithms.

`lookup_factory` is how the roster resolves a name and `FACTORIES` is the registry it reads, so
a roster name with no factory fails here rather than deep inside a campaign.
"""

from metaheuristiczoo.experiment.factories.budget import problem_budget  # noqa: F401
from metaheuristiczoo.experiment.factories.declared import *  # noqa: F401,F403
from metaheuristiczoo.experiment.factories.external import *  # noqa: F401,F403

FACTORIES = {_name[len("factory_"):]: _obj for _name, _obj in sorted(globals().items())
             if _name.startswith("factory_") and callable(_obj)}


def lookup_factory(name):
    """The factory registered for an algorithm name.

    The campaign roster refers to algorithms by name and resolves them here; every name it lists
    must appear in `FACTORIES`, which is built from the modules of this package.
    """
    try:
        return FACTORIES[name]
    except KeyError:
        raise KeyError(f"no factory for algorithm {name!r}: "
                       f"experiment.factories defines no factory_{name}") from None
