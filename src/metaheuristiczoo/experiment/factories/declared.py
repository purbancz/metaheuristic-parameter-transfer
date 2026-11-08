"""Factories of the repository's own algorithms in the study, one function per algorithm.

Each factory names the class it builds and the tuning block whose values it takes; `building.build` reads the
values from the active parameter set (`experiment.parameters`). The three `_WithRandom_Var` hybrids keep the
values they ran with before they had tuning blocks of their own, for parameter sets without those blocks
(`parameters.in_code_unless_tuned`); every set of the study has them, so the values below are not used by it.
Reproductions of other people's algorithms live in `external`.
"""

from jmetal.util.termination_criterion import StoppingByEvaluations

from metaheuristiczoo.algorithm.role_based.role_hybrids import (HybridAdditivePSO, HybridAdditivePSO_WithRandom,
                                                                HybridAdditivePSO_WithWanderer,
                                                                HybridDisjointPSO_WithWanderer, HybridFullDisjointPSO,
                                                                HybridFullDisjointPSO_WithRandom,
                                                                HybridPartialDisjointPSO,
                                                                HybridPartialDisjointPSO_WithRandom)
from metaheuristiczoo.algorithm.role_based.roles import RebelRejectorPSO, ContrarianDefeatistPSO, EschewerEscapistPSO
from metaheuristiczoo.algorithm.basic.single_objective_pso import SingleObjectivePSO
from metaheuristiczoo.experiment import parameters
from metaheuristiczoo.experiment.factories.budget import problem_budget
from metaheuristiczoo.experiment.factories.building import build
from metaheuristiczoo.experiment.protocol import G_SOLUTIONS_SIZE


def factory_PSO(p):
    return build(SingleObjectivePSO, "PSO", p)


def factory_RebelRejectorPSO(p):
    return build(RebelRejectorPSO, "RebelRejectorPSO", p)


def factory_ContrarianDefeatistPSO(p):
    return build(ContrarianDefeatistPSO, "ContrarianDefeatistPSO", p)


def factory_EschewerEscapistPSO(p):
    return build(EschewerEscapistPSO, "EschewerEscapistPSO", p)


def factory_HybridFullDisjointPSO(p):
    return build(HybridFullDisjointPSO, "HybridFullDisjointPSO", p)


def factory_HybridPartialDisjointPSO(p):
    return build(HybridPartialDisjointPSO, "HybridPartialDisjointPSO", p)


def factory_HybridAdditivePSO(p):
    return build(HybridAdditivePSO, "HybridAdditivePSO", p)


def factory_HybridFullDisjointPSO_WithRandom(p):
    return build(HybridFullDisjointPSO_WithRandom, "HybridFullDisjointPSO_WithRandom", p)


def factory_HybridPartialDisjointPSO_WithRandom(p):
    return build(HybridPartialDisjointPSO_WithRandom, "HybridPartialDisjointPSO_WithRandom", p)


def factory_HybridAdditivePSO_WithRandom(p):
    return build(HybridAdditivePSO_WithRandom, "HybridAdditivePSO_WithRandom", p)


# Values the three '_WithRandom_Var' hybrids ran with before their blocks existed (a March 2026
# tuning under an earlier protocol): used when the active set has no '<name>' block - set 1 has
# none - and recorded as `parameters.IN_CODE` (blocks added 3 Oct 2026).
_UNINFORMED_ONLY_IN_CODE = {
    "HybridFullDisjointPSO_WithRandom_Var": dict(
        w=0.070613324161097,
        c1=1.2648841133643252,
        c2=5.139031539268885,
        rejector_c=0,
        defeatist_c=0,
        escapist_c=0,
        amnesiac_c=3.727243242431227,
        rebel_c=0,
        contrarian_c=0,
        eschewer_c=0,
        anarchic_c=3.4265350576587106,
        rejector_fraction=0,
        defeatist_fraction=0,
        escapist_fraction=0,
        amnesiac_fraction=0.534157673588983,
        rebel_fraction=0,
        contrarian_fraction=0,
        eschewer_fraction=0,
        anarchic_fraction=0.116381850790778,
        assign_roles_every_iteration=False,
    ),
    "HybridPartialDisjointPSO_WithRandom_Var": dict(
        w=0.054655302723429,
        c1=0.429282943785279,
        c2=5.76062395431257,
        rejector_c=0,
        defeatist_c=0,
        escapist_c=0,
        rebel_c=0,
        contrarian_c=0,
        eschewer_c=0,
        amnesiac_c=4.2311484550184915,
        anarchic_c=0.51542902758087,
        rejector_fraction=0,
        defeatist_fraction=0,
        escapist_fraction=0,
        rebel_fraction=0,
        contrarian_fraction=0,
        eschewer_fraction=0,
        amnesiac_fraction=0.172142057161717,
        anarchic_fraction=0.262647267385497,
        assign_roles_every_iteration=False,
    ),
    "HybridAdditivePSO_WithRandom_Var": dict(
        w=0.105078908098416,
        c1=5.922638446294212,
        c2=1.853544278625566,
        rejector_c=0,
        defeatist_c=0,
        escapist_c=0,
        rebel_c=0,
        contrarian_c=0,
        eschewer_c=0,
        anarchic_c=2.803031943603203,
        amnesiac_c=5.298190812498183,
        std_cognitive_prob=0.468507292758563,
        rejector_prob=0,
        defeatist_prob=0,
        escapist_prob=0,
        amnesiac_prob=0.026503169610302,
        std_social_prob=0.322113944701517,
        rebel_prob=0,
        contrarian_prob=0,
        eschewer_prob=0,
        anarchic_prob=0.012611293683424,
        assign_flags_every_iteration=False,
    ),
}


def _uninformed_only(cls, block, p):
    if block in parameters.blocks():
        return build(cls, block, p)
    return cls(problem=p, swarm_size=G_SOLUTIONS_SIZE, **_UNINFORMED_ONLY_IN_CODE[block],
               termination_criterion=StoppingByEvaluations(max_evaluations=problem_budget(p)))


@parameters.in_code_unless_tuned("HybridFullDisjointPSO_WithRandom_Var")
def factory_HybridFullDisjointPSO_WithRandom_Var(p):
    return _uninformed_only(HybridFullDisjointPSO_WithRandom, "HybridFullDisjointPSO_WithRandom_Var", p)


@parameters.in_code_unless_tuned("HybridPartialDisjointPSO_WithRandom_Var")
def factory_HybridPartialDisjointPSO_WithRandom_Var(p):
    return _uninformed_only(HybridPartialDisjointPSO_WithRandom, "HybridPartialDisjointPSO_WithRandom_Var", p)


@parameters.in_code_unless_tuned("HybridAdditivePSO_WithRandom_Var")
def factory_HybridAdditivePSO_WithRandom_Var(p):
    return _uninformed_only(HybridAdditivePSO_WithRandom, "HybridAdditivePSO_WithRandom_Var", p)


def factory_HybridDisjointPSO_WithWanderer(p):
    return build(HybridDisjointPSO_WithWanderer, "HybridDisjointPSO_WithWanderer", p)


def factory_HybridAdditivePSO_WithWanderer(p):
    return build(HybridAdditivePSO_WithWanderer, "HybridAdditivePSO_WithWanderer", p)
