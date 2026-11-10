# Code for "Transferring tuned configurations of continuous metaheuristics to subset selection"

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24090959.svg)](https://doi.org/10.5281/zenodo.24090959)
[![Data](https://img.shields.io/badge/data-RODBUK-1f6feb)]([https://doi.org/XXXXX](https://doi.org/10.71580/SANO/QPU2E7))
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![Built on jMetalPy](https://img.shields.io/badge/built%20on-jMetalPy%201.7-orange.svg)](https://github.com/jMetal/jMetalPy)
[![Paper: OLA 2027](https://img.shields.io/badge/paper-OLA%202027%20(in%20preparation)-lightgrey.svg)](https://ola2027.sciencesconf.org/)

This is the code of the paper

> P. Urbańczyk, A. Urbańczyk. *Transferring tuned configurations of continuous metaheuristics to subset
> selection: a parameter-swap study on maximum coverage.* In preparation for OLA 2027.

It contains the 25 metaheuristics of the study and the four baselines, the decoder that turns a continuous
state into a subset, the ten maximum-coverage instances, the tuned configurations, the experiment runner,
and the scripts that produce every table and figure of the paper. It is the part of the authors' research
package `metaheuristiczoo` that the study uses, reduced to what the study runs; the code of the algorithms
is copied from it without changes to what it computes.

## Contents

| Path | What it is |
|---|---|
| `src/metaheuristiczoo/algorithm/` | the 25 algorithms: the role-based PSO line (`basic`, `role_based`) and the reproductions of published algorithms (`sota`) |
| `src/metaheuristiczoo/problem/subset/` | the decoder, the coverage objective, the baselines, and the ten instances (`transfer/`: source pins, certified optima) |
| `src/metaheuristiczoo/experiment/` | parameter sets, factories, campaign, runner, result files, statistics |
| `src/metaheuristiczoo/experiment/parameters/*.json` | the configurations, one file per parameter set (table below) |
| `scripts/fetch_subset_data.py` | downloads the instance networks and checks their SHA-256 |
| `scripts/run_experiment.py` | runs a campaign (main comparison or a follow-up); can be stopped and restarted |
| `scripts/analyze_transfer.py` | paired statistics of the main comparison |
| `papers/parameter_transfer/*.py` | tables and numbers of the paper and of the three follow-ups |
| `papers/parameter_transfer/parameter_spaces.json` | the irace parameter ranges of the 25 algorithms (used for the parameter-difference table) |
| `papers/parameter_transfer/ola2027/` | the paper's figures (`figures_ola.py`) and LaTeX tables (`build_ola.py`) |

### Parameter sets

| File | Role in the paper |
|---|---|
| `tuned_2026_10_subset_combined` | **S**, tuned with irace on p-median and p-center instances (OR-Library graphs) |
| `tuned_2026_10_diverse_v2` | the continuous tuning run (36 instances of nine continuous landscapes, D = 20 and 100) |
| `control_2026_10_continuous_bounce` | **C**: the values of `tuned_2026_10_diverse_v2` with reflection at the bounds |
| `elite2_2026_11_*` | within-run check: the second-best final configuration of each tuning run |
| `swap_2026_11_*` | parameter-group swaps (`base`, `coefficients`, `shares`), both directions |
| `popswap_2026_11_*` | population-size swap, both directions |

Each file has a `preamble` that records how it was produced and holds the values of the study's
algorithms only. The tuning itself (irace and the tuning instances) is not part of this code: the
configurations it produced are the files above, and its traces are in the dataset (`tuning/`).

## Installation

Python 3.12 or later:

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

`requirements.txt` pins jMetalPy 1.7.0 and NumPy 2.2.6, the versions used for the runs (the algorithms'
random streams depend on them), and the versions of SciPy, pandas, h5py and Matplotlib the analysis was
checked with.

## Instance data

The instance networks are not redistributed here. They come from the maximal-covering benchmark archive of
Wang and Zhou (github.com/HWangUPV/MCLP, CC BY-NC-SA 4.0). The fetch script downloads the pinned files
(about 1 MB) and refuses any file whose SHA-256 differs:

```bash
python scripts/fetch_subset_data.py --root <data dir>
export MZOO_DATA_ROOT=<data dir>
```

## Reproducing the paper from the published results

Place (or link) the dataset's `experiment_results/` folder in this directory, then run the commands below.
The dataset stores the result files under shortened names (`<instance>__<algorithm>.h5`); the scripts find
the files by pattern, so the names make no difference to them. Then run

```bash
python scripts/analyze_transfer.py
python papers/parameter_transfer/analysis.py
python papers/parameter_transfer/m2a_analysis.py
python papers/parameter_transfer/m1a_analysis.py
python papers/parameter_transfer/m3_analysis.py
python papers/parameter_transfer/ola2027/figures_ola.py
python papers/parameter_transfer/ola2027/build_ola.py
```

The first command writes `experiment_results/analysis/transfer/`; the others write
`papers/parameter_transfer/tables/`, the `*numbers.json` files, and the paper's figures and LaTeX tables
under `papers/parameter_transfer/ola2027/`. The same files are in the dataset (`derived/`), and a run of
these scripts on the published results reproduces them byte for byte. The figures use the Source Sans Pro
font when it is installed (`FIG_FONT` selects another one).

## Re-running the experiments

Every configuration is run 30 times on every instance with seeds 42 to 71, so runs of different
configurations can be paired by seed. A campaign is scoped with environment variables (see
`scripts/run_experiment.py`) and can be stopped and restarted: finished runs are kept, and several workers
can share one results folder through claim files. `EXPERIMENT_WORKERS` sets the number of processes.

Re-running a stored cell reproduces it exactly: we re-ran cells of PSO, a role-based PSO, a hybrid, L-SHADE,
IKUN-GA and the four baselines under S, C and a follow-up set with this code, and the final values and
convergence curves were identical, bit for bit, to the stored ones. The whole main comparison is large: the
stored runs took about 2,000 core-hours for S and 2,500 for C, most of it on the PCB3038 instances.

Main comparison, configuration S and the four baselines (`experiment_results/transfer/`):

```bash
python scripts/run_experiment.py
```

Configuration C (`experiment_results/control/control_2026_10_continuous_bounce/transfer/`):

```bash
EXPERIMENT_CONTROL_SET=control_2026_10_continuous_bounce python scripts/run_experiment.py
```

One cell, for a quick check (PSO under S on the smallest instance, a few minutes):

```bash
EXPERIMENT_ALGORITHMS=PSO EXPERIMENT_PROBLEMS="*GR100/S50/p10*" python scripts/run_experiment.py
```

Follow-ups. A run with `EXPERIMENT_CONTROL_SET=<set>` writes to
`experiment_results/control/<set>/transfer/`; the analysis scripts read the follow-ups from
`experiment_results/<m2a|m1a|m3>/control/<set>/transfer/`, so move each folder there after the runs
(this is the layout of the dataset).

| Follow-up | Folder | Parameter sets | Algorithms | Instances |
|---|---|---|---|---|
| within-run check | `m2a` | `elite2_2026_11_subset_combined`, `elite2_2026_11_continuous_bounce` | the seven algorithms below (five for the continuous set, which kept one final configuration for the other two) | the four GR instances |
| group swaps | `m1a` | the six `swap_2026_11_*` sets | the seven algorithms below | the four GR instances |
| population swap | `m3` | `popswap_2026_11_subset_takes_continuous`, `popswap_2026_11_continuous_takes_subset` | SHADE, LSHADE, LSRTDE, JSO | the two SJC818 instances |

The seven algorithms are HybridDisjointPSO_WithWanderer, HybridAdditivePSO_WithWanderer,
HybridPartialDisjointPSO_WithRandom, HybridAdditivePSO_WithRandom, HybridPartialDisjointPSO_WithRandom_Var,
HybridAdditivePSO_WithRandom_Var and IKUNGA. For example:

```bash
EXPERIMENT_CONTROL_SET=popswap_2026_11_subset_takes_continuous \
EXPERIMENT_ALGORITHMS=SHADE,LSHADE,LSRTDE,JSO EXPERIMENT_PROBLEMS="*SJC818/*" python scripts/run_experiment.py
```

Each result file stores, with every run, its seed, the configuration it used (`configuration_json`) and
the code version (`git_commit`, a commit of the research repository for the published runs).

## Licence

The code is released under the MIT License (see `LICENSE`). The result data are released under CC BY 4.0
with the dataset. The instance networks keep the licence of their source (CC BY-NC-SA 4.0) and are
therefore fetched rather than included.
