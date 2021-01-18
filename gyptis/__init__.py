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

import importlib
import sys

import dolfin

#
# try:
#     __adjoint__
# except:
#     __adjoint__ = False
#
# def adjoint(bool):
#     global __adjoint__
#     global dolfin
#     if bool:
#         __adjoint__ = True
#
#     import dolfin
#     importlib.reload(dolfin)
#     importlib.reload(sys.modules[__name__])
#


__adjoint__ = False

if __adjoint__:
    import dolfin_adjoint as dfa

    #
    # # #
    # dolfin.Function = dfa.Function
    # dolfin.Expression = dfa.Expression
    # dolfin.DirichletBC = dfa.DirichletBC
    # dolfin.Constant = dfa.Constant
    #
    #
    # dolfin.KrylovSolver = dfa.KrylovSolver
    # dolfin.LUSolver = dfa.LUSolver
    # dolfin.NewtonSolver = dfa.NewtonSolver
    # dolfin.NonlinearVariationalSolver = dfa.NonlinearVariationalSolver
    # dolfin.NonlinearVariationalProblem = dfa.NonlinearVariationalProblem
    # dolfin.LinearVariationalSolver = dfa.LinearVariationalSolver
    # dolfin.LinearVariationalProblem = dfa.LinearVariationalProblem
    #
    #
    # dolfin.assemble = dfa.assemble
    # dolfin.assemble_system = dfa.assemble_system
    # dolfin.solve = dfa.solve
    # dolfin.project = dfa.project
    # dolfin.interpolate = dfa.interpolate
    #
    # dolfin.UnitSquareMesh = dfa.UnitSquareMesh
    # dolfin.UnitCubeMesh = dfa.UnitCubeMesh
    # dolfin.Mesh = dfa.Mesh
    #
    # dolfin.mesh = dfa.mesh
    #
    # #
    dolfin.__dict__.update(dfa.__dict__)
    del dfa


# from .complex import *

# __all__ = complex.__all__
