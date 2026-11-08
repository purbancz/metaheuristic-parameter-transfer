

# Instance seeds, fingerprints and result manifests name a problem class by its module path.
# That path is a persistent protocol, not a packaging detail: the benchmark instances of every
# stored result were derived under the historical top-level layout (`problem.subset...`), so the
# distribution-package prefix introduced by the `src/` layout is stripped here and the module names
# under `problem/` are frozen.
_PACKAGE_PREFIX = "metaheuristiczoo."


def identity_name(problem_class) -> str:
    """Stable `module.QualName` of a problem class, independent of the distribution package."""
    module = problem_class.__module__
    if module.startswith(_PACKAGE_PREFIX):
        module = module[len(_PACKAGE_PREFIX):]
    return f"{module}.{problem_class.__qualname__}"


