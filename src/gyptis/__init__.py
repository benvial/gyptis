#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# License: MIT
# See the documentation at gyptis.gitlab.io


import os
from math import e, pi

import dolfin
from scipy.constants import c, epsilon_0, mu_0

from .__about__ import __author__, __description__, __version__

# __doc__ = __description__


if os.environ.get("GYPTIS_ADJOINT") is not None:
    import dolfin_adjoint

    ADJOINT = True
    dolfin.__dict__.update(dolfin_adjoint.__dict__)
    dolfin.__spec__.name = "dolfin"
    del dolfin_adjoint
    ## prevents AttributeError: module 'fenics_adjoint.types.function'
    ## has no attribute 'function' when writing solution to files with <<
    dolfin.function.function = dolfin.function
else:
    ADJOINT = False


from dolfin import MPI

dolfin.parameters["reorder_dofs_serial"] = False
dolfin.parameters["ghost_mode"] = "shared_facet"
dolfin.PETScOptions.set("petsc_prealloc", "200")
dolfin.PETScOptions.set("ksp_type", "preonly")
dolfin.PETScOptions.set("pc_type", "lu")
# dolfin.PETScOptions.set('pc_factor_mat_solver_type', 'mumps')
# dolfin.PETScOptions.set('sub_pc_type', 'lu')
# dolfin.PETScOptions.set('pc_asm_overlap', '10')
# dolfin.PETScOptions.set('ksp_type', 'gmres')
# dolfin.PETScOptions.set('petsc_prealloc', '100')
# dolfin.PETScOptions.set('ksp_rtol', '1.e-12')
# dolfin.PETScOptions.set('ksp_type', 'gmres')
# dolfin.PETScOptions.set('ksp_gmres_restart', '100')
# dolfin.PETScOptions.set('pc_type', 'jacobi')
# dolfin.PETScOptions.set('pc_factor_levels', '3')
# dolfin.PETScOptions.set('mat_mumps_icntl_14', 40)
# dolfin.PETScOptions.set('mat_mumps_icntl_35', 2)
# dolfin.PETScOptions.set('mat_mumps_icntl_36', 1)
# dolfin.PETScOptions.set('mat_mumps_icntl_24', 1)
# dolfin.set_log_level(10000000)

from .api import *
from .complex import *
from .plot import *
from .utils import logger, set_log_level

# logger.warning("Welcome to gyptis!")
# logger.info("Welcome to gyptis!")
# logger.debug("Welcome to gyptis!")
