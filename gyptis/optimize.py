#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT


import numpy as np
from scipy.optimize import OptimizeResult, minimize

from . import dolfin as df
from .complex import *
from .helpers import *
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


def derivative(f, x, ctrl_space=None, array=True):
    dfdx = df.compute_gradient(f, df.Control(x))
    if ctrl_space is not None:
        dfdx = project(dfdx, ctrl_space)
    if array:
        return function2array(dfdx)
    else:
        return dfdx


class OptimFunction:
    def __init__(self, fun, stop_val=None, normalization=1):
        self.fun_in = fun
        self.fun_value = None
        self.stop_val = stop_val
        self.normalization = normalization

    def fun(self, x, *args, **kwargs):
        if self.fun_value is not None:
            if self.stop_val is not None:
                if self.fun_value < self.stop_val:
                    raise ValueError(
                        f"Minimum value {self.stop_val} reached, current objective function = {self.fun_value}"
                    )
        obj, grad = self.fun_in(x, *args, **kwargs)
        obj /= self.normalization
        grad /= self.normalization
        self.fun_value = obj
        self.x = x
        return obj, grad


def scipy_minimize(
    f,
    x,
    bounds,
    maxiter=50,
    maxfun=100,
    tol=1e-6,
    args=(),
    stop_val=None,
    normalization=1,
):

    optfun = OptimFunction(fun=f, stop_val=stop_val, normalization=normalization)
    opt = OptimizeResult()
    opt.x = x
    opt.fun = None
    try:
        opt = minimize(
            optfun.fun,
            x,
            tol=tol,
            args=args,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": maxiter, "maxfun": maxfun},
        )
    except Exception as e:
        print(e)
        opt.x = optfun.x
        opt.fun = optfun.fun_value

    return opt


def transfer_sub_mesh(x, geometry, source_space, target_space, subdomain):
    markers = geometry.markers["triangle"]
    domains = geometry.subdomains["surfaces"]
    a = df.Function(source_space)
    mdes = markers.where_equal(domains[subdomain])
    a = function2array(a)
    a[mdes] = function2array(project(x, target_space))
    ctrl = array2function(a, source_space)
    return ctrl
