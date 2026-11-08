"""Download the networks of the ten maximum-coverage instances and check them, or only check them.

    python scripts/fetch_subset_data.py --root D:/data/mclp                 # download (about 1 MB) and verify
    python scripts/fetch_subset_data.py --root D:/data/mclp --verify-only   # no network; exits 1 if not ready

The networks come from the maximal-covering benchmark archive of Wang and Zhou (github.com/HWangUPV/MCLP,
CC BY-NC-SA 4.0), which collects the GR networks (Galvao and ReVelle 1996), the SJC networks (Lorena and Pereira
2002) and PCB3038 (Reinelt, TSPLIB 1991). Every file is pinned by its SHA-256 in
`metaheuristiczoo/problem/subset/transfer/mclp.py`; a file that differs is refused.

Then `export MZOO_DATA_ROOT=D:/data/mclp` (or pass `root=` to the loaders).
"""
import argparse
import sys

from metaheuristiczoo.problem.subset.datasets.common import DatasetChanged, DatasetUnavailable
from metaheuristiczoo.problem.subset.datasets.fetch import fetch, verify_dataset
from metaheuristiczoo.problem.subset.transfer.mclp import MCLP_SPEC


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, help="directory that holds (or will hold) the networks")
    parser.add_argument("--verify-only", action="store_true", help="check what is under --root; no network")
    args = parser.parse_args()
    if not args.verify_only:
        fetch(args.root, MCLP_SPEC)
    try:
        verify_dataset(MCLP_SPEC, args.root)
    except (DatasetUnavailable, DatasetChanged) as exc:
        print(f"not ready: {exc}")
        sys.exit(1)
    print(f"{MCLP_SPEC.key}: {len(MCLP_SPEC.files)} file(s) verified under {args.root}")


if __name__ == "__main__":
    main()
