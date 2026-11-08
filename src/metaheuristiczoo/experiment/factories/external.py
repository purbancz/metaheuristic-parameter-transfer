"""Factories reproducing algorithms published by other people.

Each factory names its class and its tuning block and is built by `building.build` from the active parameter
set (`experiment.parameters`). Tuning does not make an algorithm ours, which is why these stay apart from the
repository's own algorithms (`declared`) whatever set they run under.
"""

from metaheuristiczoo.algorithm.sota.agoa import AGOA
from metaheuristiczoo.algorithm.sota.ikun import IKUNDE, IKUNGA, IKUNPSO
from metaheuristiczoo.algorithm.sota.jade import JADE
from metaheuristiczoo.algorithm.sota.jso import JSO
from metaheuristiczoo.algorithm.sota.lshade import LSHADE
from metaheuristiczoo.algorithm.sota.lsrtde import LSRTDE
from metaheuristiczoo.algorithm.sota.shade import SHADE
from metaheuristiczoo.algorithm.sota.spso2011 import SPSO2011
from metaheuristiczoo.experiment.factories.building import build


def factory_LSHADE(p):
    return build(LSHADE, "LSHADE", p)


def factory_SPSO2011(p):
    return build(SPSO2011, "SPSO2011", p)


def factory_SHADE(p):
    return build(SHADE, "SHADE", p)


def factory_JSO(p):
    return build(JSO, "JSO", p)


def factory_LSRTDE(p):
    return build(LSRTDE, "LSRTDE", p)


def factory_AGOA(p):
    return build(AGOA, "AGOA", p)


def factory_JADE(p):
    return build(JADE, "JADE", p)


def factory_IKUNPSO(p):
    return build(IKUNPSO, "IKUNPSO", p)


def factory_IKUNDE(p):
    return build(IKUNDE, "IKUNDE", p)


def factory_IKUNGA(p):
    return build(IKUNGA, "IKUNGA", p)
