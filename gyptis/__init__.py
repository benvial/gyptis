import os

import dolfin

from .__about__ import (
    __author__,
    __author_email__,
    __copyright__,
    __description__,
    __license__,
    __status__,
    __version__,
    __website__,
)

__doc__ = __description__


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

dolfin.parameters["reorder_dofs_serial"] = False


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

# dolfin.set_log_level(1)

from .api import *
