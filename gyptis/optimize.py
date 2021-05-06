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


def simp(a, s_min=1, s_max=2, p=1, complex=True):
    """Solid isotropic material with penalisation (SIMP)"""
    if complex:
        return Complex(
            simp(a, s_min=s_min.real, s_max=s_max.real, p=p, complex=False),
            simp(a, s_min=s_min.imag, s_max=s_max.imag, p=p, complex=False),
        )
    else:
        return s_min + (s_max - s_min) * a ** p


def tanh(x):
    return (df.exp(2 * x) - 1) / (df.exp(2 * x) + 1)


def projection(a, beta=1, nu=0.5):
    """Projection operator."""
    return (tanh(beta * nu) + tanh(beta * (a - nu))) / (
        tanh(beta * nu) + tanh(beta * (1 - nu))
    )


class Filter:
    def __init__(self, rfilt=0, function_space=None, order=1, solver=None):
        self.rfilt = rfilt
        self.solver = solver
        self.order = order
        self._function_space = function_space

    def weak(self, a):
        self.mesh = a.function_space().mesh()
        self.dim = self.mesh.ufl_domain().geometric_dimension()
        self.function_space = self._function_space or df.FunctionSpace(
            self.mesh, "CG", self.order
        )
        af = df.TrialFunction(self.function_space)
        vf = df.TestFunction(self.function_space)
        if hasattr(self.rfilt, "shape"):
            if np.shape(self.rfilt) in [(2, 2), (3, 3)]:
                self._rfilt = tensor_const(self.rfilt, dim=self.dim, real=True)
            else:
                raise ValueError("Wrong shape for rfilt")
        else:
            self._rfilt = df.Constant(self.rfilt)

        lhs = (
            df.inner(self._rfilt * df.grad(af), self._rfilt * df.grad(vf)) * df.dx
            + df.inner(af, vf) * df.dx
        )
        rhs = df.inner(a, vf) * df.dx
        return lhs, rhs

    def apply(self, a):
        if np.all(self.rfilt == 0):
            return a
        else:
            lhs, rhs = self.weak(a)
            af = df.Function(self.function_space, name="Filtered density")
            vector = df.assemble(rhs)
            if self.solver == None:
                matrix = df.assemble(lhs)
                self.solver = df.KrylovSolver(matrix, "cg", "jacobi")
            self.solver.solve(af.vector(), vector)
            return af


def filtering(a, rfilt=0, function_space=None, order=1):
    filter = Filter(rfilt, function_space, order)
    return filter.apply(a)


def derivative(f, x, ctrl_space=None, array=True):
    dfdx = df.compute_gradient(f, df.Control(x))
    if ctrl_space is not None:
        dfdx = project(
            dfdx,
            ctrl_space,
            solver_type="cg",
            preconditioner_type="jacobi",
        )
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
    markers = geometry.markers
    domains = geometry.domains
    a = df.Function(source_space)
    mdes = markers.where_equal(domains[subdomain])
    a = function2array(a)
    a[mdes] = function2array(
        project(
            x,
            target_space,
            solver_type="cg",
            preconditioner_type="jacobi",
        )
    )
    ctrl = array2function(a, source_space)
    return ctrl
