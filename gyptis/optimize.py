#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

import nlopt
import numpy as np
from scipy.optimize import OptimizeResult, minimize

from . import dolfin as df
from .complex import *
from .materials import tensor_const
from .utils.helpers import *


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


def projection_gradient(a, beta=1, nu=0.5):
    """Projection operator gradient dproj/da."""
    return (
        beta
        * (1 - (tanh(beta * (a - nu))) ** 2)
        / (tanh(beta * nu) + tanh(beta * (1 - nu)))
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
    # a[mdes]
    b = function2array(
        project(
            x,
            target_space,
            solver_type="cg",
            preconditioner_type="jacobi",
        )
    )
    comm = df.MPI.comm_world
    b = comm.gather(b, root=0)
    mdes = comm.gather(mdes, root=0)
    if df.MPI.rank(comm) == 0:
        mdes = np.hstack(mdes)
        b = np.hstack(b)
    else:
        mdes = None
        b = None
    mdes = comm.bcast(mdes, root=0)
    b = comm.bcast(b, root=0)
    a[mdes] = b
    # comm = df.MPI.comm_world
    # a = comm.gather(a, root=0)
    # if df.MPI.rank(comm) == 0:
    #     a1 = np.hstack(a)
    # else:
    #     a1=None
    # a1 = comm.bcast(a1, root=0)
    ctrl = array2function(a, source_space)
    return ctrl


class TopologyOptimizer:
    def __init__(
        self,
        fun,
        x0,
        rfilt=0,
        threshold=(0, 8),
        maxiter=20,
        stopval=None,
        callback=None,
        args=None,
        verbose=True,
    ):
        self.fun = fun
        self.x0 = x0
        self.nvar = len(x0)
        self.threshold = threshold
        self.rfilt = rfilt
        self.maxiter = maxiter
        self.stopval = stopval
        self.callback = callback
        self.args = args or []
        self.verbose = verbose
        self.callback_output = []

    def min_function(self):
        f = self.fun(x)

    def minimize(self):
        if self.verbose:
            print("#################################################")
            print(f"Topology optimization with {self.nvar} variables")
            print("#################################################")
            print("")
        x0 = self.x0

        for iopt in range(*self.threshold):
            self._cbout = []
            if self.verbose:
                print(f"global iteration {iopt}")
                print("---------------------------------------------")
            proj_level = iopt
            args = list(self.args)
            # args[1] = proj_level
            # args = tuple(args)

            def fun_nlopt(x, gradn):
                y, dy = self.fun(
                    x,
                    proj_level=proj_level,
                    rfilt=self.rfilt,
                    reset=True,
                    gradient=True,
                )
                gradn[:] = dy
                cbout = []
                if self.callback is not None:
                    out = self.callback(x, y, dy, *args)
                    self._cbout.append(out)
                return y

            lb = np.zeros(self.nvar, dtype=float)
            ub = np.ones(self.nvar, dtype=float)

            opt = nlopt.opt(nlopt.LD_MMA, self.nvar)
            opt.set_lower_bounds(lb)
            opt.set_upper_bounds(ub)

            # opt.set_ftol_rel(1e-16)
            # opt.set_xtol_rel(1e-16)
            if self.stopval is not None:
                opt.set_stopval(self.stopval)
            if self.maxiter is not None:
                opt.set_maxeval(self.maxiter)

            # opt.set_min_objective(fun_nlopt)
            opt.set_max_objective(fun_nlopt)
            xopt = opt.optimize(x0)
            fopt = opt.last_optimum_value()
            self.callback_output.append(self._cbout)
            x0 = xopt

        self.opt = opt
        self.xopt = xopt
        self.fopt = fopt
        return xopt, fopt
