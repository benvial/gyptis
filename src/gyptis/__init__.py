#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.0.2
# License: MIT
# See the documentation at gyptis.gitlab.io

import importlib
import os
from math import e, pi

from scipy.constants import c, epsilon_0, mu_0

from .__about__ import __author__, __description__, __version__


def use_adjoint(use_adj):
    """Short summary.

    Parameters
    ----------
    use_adj : bool
        Adds automatic differentiation with dolfin-adjoint if True.

    """
    if use_adj:
        os.environ["GYPTIS_ADJOINT"] = "1"
    else:
        try:
            del os.environ["GYPTIS_ADJOINT"]
        except Exception:
            pass

    import sys

    import gyptis

    importlib.reload(gyptis)
    its = [s for s in sys.modules.items() if s[0].startswith("gyptis")]
    for k, v in its:
        importlib.reload(v)


if os.environ.get("GYPTIS_ADJOINT") is not None:
    import dolfin

    importlib.reload(dolfin)
    import dolfin_adjoint

    ADJOINT = True
    dolfin.__dict__.update(dolfin_adjoint.__dict__)
    dolfin.__spec__.name = "dolfin"
    del dolfin_adjoint
    # prevents AttributeError: module 'fenics_adjoint.types.function'
    # has no attribute 'function' when writing solution to files with <<
    dolfin.function.function = dolfin.function
else:
    try:
        del dolfin
    except Exception:
        pass
    import dolfin

    importlib.reload(dolfin)
    ADJOINT = False


from dolfin import MPI

dolfin.parameters["form_compiler"]["cpp_optimize"] = True
dolfin.parameters["form_compiler"]["cpp_optimize_flags"] = "-O2"
# dolfin.parameters["form_compiler"]["quadrature_degree"] = 5
dolfin.parameters["reorder_dofs_serial"] = False
dolfin.parameters["ghost_mode"] = "shared_facet"


from .api import *
from .complex import *
from .plot import *
from .utils import logger, set_log_level
