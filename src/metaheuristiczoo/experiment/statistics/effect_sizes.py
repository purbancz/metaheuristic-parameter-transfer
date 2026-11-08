"""Effect sizes for pairwise comparisons: how LARGE a difference is, next to whether it is significant.

With 30 runs per pair and hundreds of algorithms almost every difference is statistically
significant; what a reader needs beside the p-value is the probability that a run of one
algorithm beats a run of the other. That is the Vargha-Delaney A12 (Vargha & Delaney, J. Educ.
Behav. Stat. 25(2), 2000): the probability that a random observation of A is better than a random
observation of B, ties counted half - 0.5 means no difference, 1.0 means A always wins.
Cliff's delta is 2 A12 - 1. Arcuri & Briand (STVR 24(3), 2014) recommend A12 for randomised
algorithms; the magnitude labels below are their thresholds (|A12 - 0.5| < 0.06 negligible,
< 0.14 small, < 0.21 medium, else large).

Everything here works on minimised values per run, so "better" means smaller.
"""

import numpy as np


MAGNITUDES = ((0.06, "negligible"), (0.14, "small"), (0.21, "medium"), (float("inf"), "large"))


def a12(a, b):
    """P(a random value of `a` is SMALLER than a random value of `b`), ties counted half.
    Both are minimised objectives, so a12 > 0.5 means `a` is the better algorithm."""
    a, b = np.asarray(a, dtype=float).ravel(), np.asarray(b, dtype=float).ravel()
    if a.size == 0 or b.size == 0:
        return float("nan")
    less = (a[:, None] < b[None, :]).sum()
    ties = (a[:, None] == b[None, :]).sum()
    return float((less + 0.5 * ties) / (a.size * b.size))


def magnitude(a12_value):
    """Vargha-Delaney / Arcuri-Briand label of |A12 - 0.5|."""
    distance = abs(a12_value - 0.5)
    return next(label for threshold, label in MAGNITUDES if distance < threshold)


