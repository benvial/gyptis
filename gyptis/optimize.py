#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT


from . import dolfin as df
import numpy as np

from .materials import tensor_const


def simp(a, s_min=1, s_max=2, p=1):
    """Solid isotropic material with penalisation (SIMP)"""
    return s_min + (s_max - s_min) * a ** p


def tanh(x):
    return (df.exp(2 * x) - 1) / (df.exp(2 * x) + 1)


def projection(a, beta=1, nu=0.5):
    """Projection operator."""
    return (tanh(beta * nu) + tanh(beta * (a - nu))) / (
        tanh(beta * nu) + tanh(beta * (1 - nu))
    )


def filtering(a, rfilt=0, solver="iterative", function_space=None, order=1, dim=2):
    assert solver in ["direct", "iterative"]
    if np.all(rfilt == 0):
        return a
    else:
        mesh = a.function_space().mesh()
        dim = mesh.ufl_domain().geometric_dimension()
        F = function_space or df.FunctionSpace(mesh, "CG", order)
        bcs = []
        af = df.TrialFunction(F)
        vf = df.TestFunction(F)
        if hasattr(rfilt, "shape"):
            if np.shape(rfilt) in [(2, 2), (3, 3)]:
                rfilt_ = tensor_const(rfilt, dim=dim, real=True)
            else:
                raise ValueError("Wrong shape for rfilt")
        else:
            rfilt_ = df.Constant(rfilt)
        a_ = (
            df.inner(rfilt_ * df.grad(af), rfilt_ * df.grad(vf)) * df.dx
            + df.inner(af, vf) * df.dx
        )
        L_ = df.inner(a, vf) * df.dx
        af = df.Function(F, name="Filtered density")
        if solver == "direct":
            df.solve(a_ == L_, af, bcs)
            # solve(
            #     Ff == 0, af, bcs, solver_parameters=solver_parameters,
            # )
        else:
            solver = df.KrylovSolver("cg", "jacobi")
            # solver.parameters['relative_tolerance'] = 1e-3
            A = df.assemble(a_)
            b = df.assemble(L_)
            solver.solve(A, af.vector(), b)

        return af
