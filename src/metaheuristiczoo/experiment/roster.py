"""The 25 algorithms of the parameter-transfer study, in the order a campaign runs them.

Each name resolves to its factory in `experiment.factories`, which builds the algorithm with the values
of the active parameter set. The four baselines of the main comparison are added by the campaign
(`experiment.factories.subset_baselines`); they do not depend on the parameter set.
"""

from metaheuristiczoo.experiment import factories

FAMILIES = {
    # the canonical PSO, the three role duals, the three hybrid architectures and their variants with
    # uninformed roles (Urbańczyk et al., the composable role-based PSO line)
    'PSO line': ('PSO', 'RebelRejectorPSO', 'ContrarianDefeatistPSO', 'EschewerEscapistPSO',
                 'HybridFullDisjointPSO', 'HybridPartialDisjointPSO', 'HybridAdditivePSO',
                 'HybridDisjointPSO_WithWanderer', 'HybridAdditivePSO_WithWanderer',
                 'HybridFullDisjointPSO_WithRandom', 'HybridPartialDisjointPSO_WithRandom',
                 'HybridAdditivePSO_WithRandom', 'HybridFullDisjointPSO_WithRandom_Var',
                 'HybridPartialDisjointPSO_WithRandom_Var', 'HybridAdditivePSO_WithRandom_Var'),
    # reproductions of published algorithms
    'Reference algorithms': ('LSHADE', 'SPSO2011', 'SHADE', 'JSO', 'LSRTDE', 'AGOA', 'JADE',
                             'IKUNPSO', 'IKUNDE', 'IKUNGA'),
}

ACTIVE_ALGORITHMS = tuple(name for members in FAMILIES.values() for name in members)


def active_factories():
    """Ordered name -> factory mapping of the 25 algorithms."""
    return {name: factories.lookup_factory(name) for name in ACTIVE_ALGORITHMS}
