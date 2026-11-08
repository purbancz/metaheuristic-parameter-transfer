"""Parameter sets: the values the algorithms are built with, one JSON file per set.

A parameter set is the outcome of one way of choosing parameter values - an irace tuning run, or a
recombination of two tuned sets - stored as data rather than written into the factories, so that a
campaign can be run under any set. The factories (`experiment.factories`) read the ACTIVE set at
construction time. The sets of the study:

    tuned_2026_10_subset_combined        S, tuned on subset problems (p-median, p-center)
    tuned_2026_10_diverse_v2             the continuous tuning run
    control_2026_10_continuous_bounce    C, the continuous values with reflection at the bounds
    elite2_* / swap_* / popswap_*        the follow-ups (second-best configurations, group swaps)

File layout (`<name>.json` in this package):

    {
      "name": "<name>",
      "preamble": {                       how the values were obtained
        "kind": "irace" | "control",
        "summary": "...",
        "procedure": {...},               irace sets: tuner, instances, budget rule, runs, options,
                                          seeds; other sets: what they were derived from
        "role": "control"                 a set that may only be run as a control arm
      },
      "algorithms": {
        "<tuning block>": {
          "params": {...},                the values the algorithm is built with
          "notes": "..."                  optional: how the values were derived
        }
      }
    }

Entries are keyed by TUNING BLOCK, the name the tuner raced (here the algorithm's name); a factory
names its block. Values are stored as the tuner evaluated them: scalars,
already repaired and normalised, wrapped into operators by `factories.building`.

The active set is `DEFAULT_PARAMETER_SET`, or the set named by the environment variable
`EXPERIMENT_PARAMETER_SET` (a set name, or a path to a JSON file of this layout). A campaign uses
one set for every algorithm; the few factories bound to another set say so (`bind`), and the
result files record the set each algorithm was built with (`parameter_set_of`).
"""

import copy
import functools
import json
import os

PARAMETER_SETS_DIR = os.path.dirname(os.path.abspath(__file__))

TUNED_2026_10_DIVERSE_V2 = "tuned_2026_10_diverse_v2"
TUNED_2026_10_SUBSET_COMBINED = "tuned_2026_10_subset_combined"

DEFAULT_PARAMETER_SET = TUNED_2026_10_DIVERSE_V2
# The study's default: what a job runs when neither EXPERIMENT_CONTROL_SET nor EXPERIMENT_PARAMETER_SET
# is set (experiment.campaign).
DEFAULT_SUBSET_PARAMETER_SET = TUNED_2026_10_SUBSET_COMBINED
PARAMETER_SET_ENV = "EXPERIMENT_PARAMETER_SET"
# A control arm: a combinatorial job run deliberately with another set, its results kept apart
# (experiment.campaign). A set whose preamble says role = "control" may only be run this way.
CONTROL_SET_ENV = "EXPERIMENT_CONTROL_SET"
# Set alongside PARAMETER_SET_ENV when experiment.campaign put a value there itself (a section's default
# or a control set), so a later resolution in the same process does not mistake it for the user's.
DEFAULTED_SET_ENV = "_EXPERIMENT_PARAMETER_SET_DEFAULTED"
CONTROL_RESULTS_SUBDIR = "control"

# A factory whose values are written in the factory itself (untuned configurations without a
# block): recorded as such in the result files.
IN_CODE = "in code"


class ParameterSetError(KeyError):
    """A parameter set that does not exist, is malformed, or has no entry for a block."""


def active_parameter_set() -> str:
    """The set factories read by default: EXPERIMENT_PARAMETER_SET, else DEFAULT_PARAMETER_SET."""
    return os.environ.get(PARAMETER_SET_ENV) or DEFAULT_PARAMETER_SET


def is_control_set(name: str) -> bool:
    """Whether a set is meant only as a control arm (preamble role = "control")."""
    return load_parameter_set(name).get("preamble", {}).get("role") == "control"


def parameter_set_path(name: str) -> str:
    """The file of a set: a path to a .json file as given, else <name>.json in this package."""
    if name.endswith(".json"):
        return name
    return os.path.join(PARAMETER_SETS_DIR, f"{name}.json")


@functools.lru_cache(maxsize=None)
def load_parameter_set(name: str) -> dict:
    """The parsed set (cached), checked for the layout of the module docstring."""
    path = parameter_set_path(name)
    if not os.path.exists(path):
        known = sorted(f[:-5] for f in os.listdir(PARAMETER_SETS_DIR) if f.endswith(".json"))
        raise ParameterSetError(f"no parameter set {name!r} ({path}); known sets: {known}")
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    missing = [key for key in ("name", "preamble", "algorithms") if key not in data]
    if missing:
        raise ParameterSetError(f"parameter set {path} lacks {missing}")
    for block, entry in data["algorithms"].items():
        if not isinstance(entry, dict) or not isinstance(entry.get("params"), dict):
            raise ParameterSetError(f"parameter set {path}: entry {block!r} has no 'params' object")
    return data


def entry(block: str, parameter_set: str | None = None) -> dict:
    """The whole entry of `block` (params, and elites / source / notes where present), a copy."""
    name = parameter_set or active_parameter_set()
    algorithms = load_parameter_set(name)["algorithms"]
    if block not in algorithms:
        raise ParameterSetError(f"parameter set {name!r} has no values for {block!r}")
    return copy.deepcopy(algorithms[block])


def parameters(block: str, parameter_set: str | None = None) -> dict:
    """The values `block` is built with in the set (default: the active set), a copy."""
    return entry(block, parameter_set)["params"]


def blocks(parameter_set: str | None = None) -> list:
    """The blocks a set has values for."""
    return sorted(load_parameter_set(parameter_set or active_parameter_set())["algorithms"])


def bind(parameter_set: str):
    """Decorator for a factory that is built from `parameter_set` whatever the active set is
    (IN_CODE for values written in the factory, as for the baselines). Recorded on the factory for
    `parameter_set_of`."""
    def mark(factory):
        factory.parameter_set = parameter_set
        return factory
    return mark


def in_code_unless_tuned(block: str):
    """Decorator for a factory that builds from the active set when the set has `block`, and from
    values written in the factory otherwise (an algorithm whose block was added after a set was
    tuned). Recorded on the factory for `parameter_set_of`."""
    def mark(factory):
        factory.in_code_unless_tuned = block
        return factory
    return mark


def parameter_set_of(factory) -> str:
    """The set a factory builds its algorithm from right now."""
    bound = getattr(factory, "parameter_set", None)
    if bound:
        return bound
    block = getattr(factory, "in_code_unless_tuned", None)
    if block is not None and block not in blocks():
        return IN_CODE
    return active_parameter_set()
