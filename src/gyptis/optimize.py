#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# License: MIT
# See the documentation at gyptis.gitlab.io

import nlopt
import numpy as np

from . import dolfin as df
from .complex import *
from .materials import tensor_const
from .utils import array2function, function2array, mpi_print, project_iterative, tanh

df.parameters["allow_extrapolation"] = True


def simp(a, s_min=1, s_max=2, p=1, complex=True):
    """Solid isotropic material with penalisation (SIMP)"""
    if complex:
        return Complex(
            simp(a, s_min=s_min.real, s_max=s_max.real, p=p, complex=False),
            simp(a, s_min=s_min.imag, s_max=s_max.imag, p=p, complex=False),
        )
    else:
        return s_min + (s_max - s_min) * a ** p


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
    def __init__(
        self,
        rfilt=0,
        function_space=None,
        degree=1,
        solver=None,
        mesh=None,
        output_function_space=None,
    ):
        self.rfilt = rfilt
        self._rfilt_scaled = self.rfilt / (2 * 3 ** 0.5)
        self.solver = solver
        self.degree = degree
        self._mesh = mesh
        self._function_space = function_space
        self.output_function_space = output_function_space

    def weak(self, a):
        self.mesh = a.function_space().mesh() if self._mesh is None else self._mesh
        self.dim = self.mesh.ufl_domain().geometric_dimension()
        self.function_space = self._function_space or df.FunctionSpace(
            self.mesh, "CG", self.degree
        )
        af = df.TrialFunction(self.function_space)
        vf = df.TestFunction(self.function_space)
        if hasattr(self.rfilt, "shape"):
            if np.shape(self.rfilt) in [(2, 2), (3, 3)]:
                self._rfilt_scaled = tensor_const(
                    self._rfilt_scaled, dim=self.dim, real=True
                )
            else:
                raise ValueError("Wrong shape for rfilt")
        else:
            self._rfilt_scaled = df.Constant(self._rfilt_scaled)

        lhs = (
            df.inner(self._rfilt_scaled * df.grad(af), self._rfilt_scaled * df.grad(vf))
            * df.dx
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
            self.vector = df.assemble(rhs)
            if self.solver == None:
                self.matrix = df.assemble(lhs)
                self.solver = df.KrylovSolver(self.matrix, "cg", "jacobi")
            self.solver.solve(af.vector(), self.vector)
            self.solution = af
            if self.output_function_space is not None:
                return project_iterative(af, self.output_function_space)
            else:
                return af


def filtering(a, rfilt=0, function_space=None, degree=1, solver=None, mesh=None):
    filter = Filter(rfilt, function_space, degree, solver, mesh)
    return filter.apply(a)


def transfer_function(fromFunc, Vto):
    toFunc = df.Function(Vto)
    fromFunc.set_allow_extrapolation(True)
    A = df.PETScDMCollection.create_transfer_matrix(
        fromFunc.ufl_function_space(), toFunc.ufl_function_space()
    )
    toFunc.vector()[:] = A * fromFunc.vector()

    return toFunc


def derivative(f, x, ctrl_space=None, array=False):
    dfdx = df.compute_gradient(f, df.Control(x))
    if ctrl_space is not None:
        dfdx = transfer_function(dfdx, ctrl_space)
    if array:
        return function2array(dfdx)
    else:
        return dfdx


def transfer_sub_mesh(x, geometry, source_space, target_space, subdomain):
    markers = geometry.markers
    domains = geometry.domains
    a0 = df.Function(source_space)
    mdes = markers.where_equal(domains[subdomain])
    b = function2array(project_iterative(x, target_space))
    a = function2array(a0)
    comm = df.MPI.comm_world
    b = comm.gather(b, root=0)
    a = comm.gather(a, root=0)
    mdes = comm.gather(mdes, root=0)
    if df.MPI.rank(comm) == 0:
        mdes = np.hstack(mdes)
        b = np.hstack(b)
        a = np.hstack(a)
        mdes1 = [int(i) for i in mdes]
        a[mdes1] = b
        # sys.stdout.flush()
    else:
        a = None
    a = comm.bcast(a, root=0)
    return array2function(a, source_space)


class TopologyOptimizer:
    def __init__(
        self,
        fun,
        geometry,
        design="design",
        eps_bounds=(1, 3),
        p=1,
        rfilt=0,
        filtering_type="density",
        threshold=(0, 8),
        maxiter=20,
        stopval=None,
        ftol_rel=None,
        xtol_rel=None,
        callback=None,
        args=None,
        verbose=True,
    ):
        self.fun = fun
        self.design = design
        self.threshold = threshold
        self.rfilt = rfilt
        self.filtering_type = filtering_type
        self.maxiter = maxiter
        self.stopval = stopval
        self.ftol_rel = ftol_rel
        self.xtol_rel = xtol_rel
        self.callback = callback
        self.args = args or []
        self.verbose = verbose
        self.callback_output = []
        self.eps_min, self.eps_max = eps_bounds
        self.p = p
        self.geometry = geometry
        self.mesh = self.geometry.mesh
        self.submesh_plt = self.geometry.extract_sub_mesh(self.design)
        self.submesh = self.submesh_plt
        # self.submesh = self.mesh
        # self.submesh = df.SubMesh(self.mesh, self.geometry.markers, self.geometry.domains[self.design])

        self.fs_ctrl = df.FunctionSpace(self.mesh, "DG", 0)
        self.fs_sub = df.FunctionSpace(self.submesh, "DG", 0)
        self.nvar = self.fs_sub.dim()
        self.filter = Filter(self.rfilt, degree=1, output_function_space=self.fs_sub)

    def design_fun(self, density_fp):
        return simp(
            density_fp,
            Constant(self.eps_min),
            Constant(self.eps_max),
            df.Constant(self.p),
        )

    def _topopt_wrapper(
        self,
        objfun,
        Asub,
        Actrl,
        eps_design_min,
        eps_design_max,
        p=1,
        filter=None,
        filtering_type="density",
        proj_level=None,
        reset=True,
        grad=True,
        *args,
    ):
        def wrapper(x):
            proj = proj_level is not None
            filt = filter is not None
            if reset:
                df.set_working_tape(df.Tape())
            density = array2function(x, Asub)
            self.density = density
            filter.solver = None
            if filtering_type == "sensitivity":
                self.density_f = density
            else:
                self.density_f = filter.apply(density) if filt else density

            # density_f = transfer_sub_mesh(density_f, self.geometry, Actrl, Asub, self.design)

            ctrl = transfer_function(self.density_f, Actrl)
            # ctrl = df.interpolate(self.density_f, Actrl)
            # ctrl = project_iterative(self.density_f, Actrl)
            self.density_fp = (
                projection(ctrl, beta=df.Constant(2 ** proj_level))
                if proj
                else self.density_f
            )
            self.epsilon_design = self.design_fun(self.density_fp)
            objective = objfun(self.epsilon_design, *args)

            self.objective = objective

            if grad:
                dobjective_dx = derivative(objective, ctrl)
                if filt:
                    if filtering_type == "sensitivity":
                        f = project_iterative(density * dobjective_dx, Asub)
                        dfdd = filter.apply(f)
                        dobjective_dx = dfdd / (density + 1e-3)
                dobjective_dx = project_iterative(dobjective_dx, Asub)
                dobjective_dx = function2array(dobjective_dx)
                self.dobjective_dx = dobjective_dx
                return objective, dobjective_dx
            else:
                return objective, None

        return wrapper

    def minimize(self, x0):
        self.x0 = x0
        if self.verbose:
            mpi_print("#################################################")
            mpi_print(f"Topology optimization with {self.nvar} variables")
            mpi_print("#################################################")
            mpi_print("")

        for iopt in range(*self.threshold):
            self.proj_level = iopt

            self.wrapper = self._topopt_wrapper(
                self.fun,
                self.fs_sub,
                self.fs_ctrl,
                self.eps_min,
                self.eps_max,
                self.p,
                self.filter,
                self.filtering_type,
                self.proj_level,
                reset=True,
                grad=True,
                *self.args,
            )

            self._cbout = []
            if self.verbose:
                mpi_print(f"global iteration {iopt}")
                mpi_print("---------------------------------------------")

            def _get_gradient(dy):
                comm = df.MPI.comm_world
                rank = df.MPI.rank(comm)

                print(f"rank {rank} dy.shape", dy.shape)
                dyall = comm.gather(dy, root=0)

                if rank == 0:
                    dy = np.hstack(dyall)
                    # print(f"rank {rank} dy.shape",dy.shape)
                else:
                    dy = np.empty(self.nvar)
                    # print(f"rank {rank} dy.shape",dy.shape)
                comm.Bcast(dy, root=0)

                return dy

            def fun_nlopt(x, gradn):
                print("gradn.shape", gradn.shape)
                y, dy = self.wrapper(x)
                # print("dy.shape",dy.shape)
                if self.verbose:
                    mpi_print(f"objective = {y}")
                # dy = (function2array(array2function(dy,self.fs_sub)))
                # print("dy.shape",dy.shape)
                gradn[:] = _get_gradient(dy)
                # _get_gradient(dy)
                cbout = []
                if self.callback is not None:
                    out = self.callback(self)
                    self._cbout.append(out)
                return y

            lb = np.zeros(self.nvar, dtype=float)
            ub = np.ones(self.nvar, dtype=float)

            opt = nlopt.opt(nlopt.LD_MMA, self.nvar)
            opt.set_lower_bounds(lb)
            opt.set_upper_bounds(ub)
            # opt.set_ftol_rel(1e-16)
            # opt.set_xtol_rel(1e-16)
            if self.ftol_rel is not None:
                opt.set_ftol_rel(self.ftol_rel)
            if self.xtol_rel is not None:
                opt.set_xtol_rel(self.xtol_rel)
            if self.stopval is not None:
                opt.set_stopval(self.stopval)
            if self.maxiter is not None:
                opt.set_maxeval(self.maxiter)

            opt.set_min_objective(fun_nlopt)
            xopt = opt.optimize(x0)
            fopt = opt.last_optimum_value()
            if self.callback is not None:
                self.callback_output.append(self._cbout)
            x0 = xopt

        self.opt = opt
        self.xopt = xopt
        self.fopt = fopt
        return xopt, fopt
