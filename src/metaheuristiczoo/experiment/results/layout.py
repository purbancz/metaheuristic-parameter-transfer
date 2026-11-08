"""Where experiment output lives on disk.

Results are filed under the campaign's results directory by suite section. The study has one section:

    transfer   -> transfer/   the ten maximum-coverage instances (problem/subset/transfer/)

(A problem without a section, which the study does not have, is filed under `dim<D>/`.)

Each folder holds one HDF5 file per (problem, algorithm), `<problem>_dim<D>_<algorithm>.h5`,
that accumulates the runs of that pair (see results.writer): the run count is a property of
the file's contents, not of its name, so a campaign can be resumed or extended into the same
files. The section is written into the file's manifest (`section` attribute).
Creating directories is runtime infrastructure: it happens here, when something is about to
be written, never while the campaign is defined.
"""

import os

SCALABLE = "scalable"
TRANSFER = "transfer"
SECTIONS = (SCALABLE, TRANSFER)
_SECTION_DIRS = {TRANSFER: "transfer"}


def results_subdir(results_dir, section, number_of_variables):
    """Folder of the results of one problem: `results_dir/transfer` for the study's instances, and
    `results_dir/dim<D>` for a problem without a section."""
    if section in (None, SCALABLE):
        return f"{results_dir}/dim{number_of_variables}"
    if section not in _SECTION_DIRS:
        raise ValueError(f"unknown suite section {section!r}; expected one of {SECTIONS}")
    return f"{results_dir}/{_SECTION_DIRS[section]}"


def make_dir(path):
    """Create `path` (and its parents) unless it exists. Safe when several jobs create it at once."""
    os.makedirs(path, exist_ok=True)
