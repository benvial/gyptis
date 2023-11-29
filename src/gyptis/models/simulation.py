#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.0.2
# License: MIT
# See the documentation at gyptis.gitlab.io

import glob
import os

import numpy as np
from scipy.constants import c, epsilon_0, mu_0

from .. import ADJOINT, dolfin
from ..bc import *
from ..complex import *
from ..formulations import *
from ..geometry import *
from ..materials import *
from ..materials import _check_len
from ..sources import *
from ..utils import project_iterative
from ..utils.helpers import array2function


def _complexify_items(dictio):
    out = {}
    for k, e in dictio.items():
        lene = _check_len(e)
        out[k] = [[a + 1e-16j for a in b] for b in e] if lene > 0 else e + 1e-16j
    return out


def init_em_materials(geometry, epsilon=None, mu=None):
    if epsilon is None:
        epsilon = {k: 1 for k in geometry.domains.keys()}
    if mu is None:
        mu = {k: 1 for k in geometry.domains.keys()}

    epsilon = _complexify_items(epsilon)
    mu = _complexify_items(mu)
    return epsilon, mu


class Simulation:
    def __init__(self, geometry, formulation=None, direct=True):
        self.geometry = geometry
        self.formulation = formulation
        self.coefficients = formulation.coefficients
        self.function_space = formulation.function_space
        self.real_function_space = formulation.real_function_space
        self._source = formulation.source
        self.boundary_conditions = formulation.boundary_conditions
        self.mesh = self.geometry.mesh
        self._boundary_conditions = []
        self.dx = formulation.dx
        self.ds = formulation.ds
        self.dS = formulation.dS
        self.direct = direct
        self.ndof = self.function_space.dim()

    @property
    def source(self):
        return self._source

    @source.setter
    def source(self, value):
        self.formulation.source = value
        self._source = value

    def assemble_lhs(self):
        """Assemble the left hand side of the weak formulation.

        Returns
        -------
        PETSc matrix
            Assembled matrix.

        """
        self.formulation.build_lhs()
        self.matrix = assemble(self.formulation.lhs)
        return self.matrix

    def assemble_rhs(self, custom_rhs=None):
        """Assemble the right hand side of the weak formulation.

        Returns
        -------
        PETSc vector
            Assembled vector.

        """
        if custom_rhs is not None:
            self.formulation._set_rhs(custom_rhs)
        else:
            self.formulation.build_rhs()
        self.vector = assemble(self.formulation.rhs)
        return self.vector

    def assemble(self):
        """Assemble the weak formulation.

        Returns
        -------
        tuple (PETSc matrix, PETSc vector)
            The assembled matrix and vector.

        """
        self.matrix = self.assemble_lhs()
        self.vector = self.assemble_rhs()
        return self.matrix, self.vector

    def apply_boundary_conditions(self):
        """Apply boundary conditions."""
        bcs = self.formulation.build_boundary_conditions()
        for bc in bcs:
            bc.apply(self.matrix, self.vector)

    def solve_system(self, again=False, vector_function=True):
        """Solve the discretized system.

        Parameters
        ----------
        again : bool
            Reuse solver (the default is False).
        vector_function : bool
            Use a vector function (the default is True).

        Returns
        -------
        dolfin Function
            The solution.

        """
        if vector_function:
            element = self.function_space.split()[0].ufl_element()
            V_vect = dolfin.VectorFunctionSpace(
                self.mesh, element.family(), element.degree()
            )
            u = dolfin.Function(V_vect)
        else:
            u = dolfin.Function(self.function_space)

        if not again:
            if self.direct:
                # self.solver = dolfin.PETScLUSolver(
                #     dolfin.as_backend_type(self.matrix), "mumps"
                # )
                # ksp = self.solver.ksp()
                # ksp.setType(ksp.Type.PREONLY)
                # ksp.pc.setType(ksp.pc.Type.LU)
                # # ksp.pc.setFactorSolverType("MUMPS")
                dolfin.PETScOptions.set("petsc_prealloc", "200")
                dolfin.PETScOptions.set("ksp_type", "preonly")
                dolfin.PETScOptions.set("pc_type", "lu")
                dolfin.PETScOptions.set("pc_factor_mat_solver_type", "mumps")
                self.solver = dolfin.LUSolver(self.matrix, "mumps")
                # ksp.setFromOptions()
            else:
                # self.solver = dolfin.KrylovSolver(self.matrix,"cg", "jacobi")
                self.solver = dolfin.KrylovSolver(self.matrix)
        self.solver.solve(u.vector(), self.vector)
        dolfin.PETScOptions.clear()
        return Complex(*u.split())

    def solve(self):
        """Assemble, apply boundary conditions and computes the solution.

        Returns
        -------
        dolfin Function
            The solution.

        """

        self.assemble()
        self.apply_boundary_conditions()
        return self.solve_system()

    def eigensolve(
        self,
        n_eig=6,
        wavevector_target=0.0,
        tol=1e-6,
        half=True,
        system=True,
        sqrt=True,
        **kwargs
    ):
        wf = self.formulation.weak
        if self.formulation.dim == 1:
            dummy_vector = (
                dolfin.Constant(0) * self.formulation.test * self.formulation.dx
            )
        else:
            dummy_vector = (
                dot(dolfin.Constant((0, 0, 0)), self.formulation.test)
                * self.formulation.dx
            )

        dv = dummy_vector.real + dummy_vector.imag

        # Assemble matrices
        A = dolfin.PETScMatrix()
        B = dolfin.PETScMatrix()
        b = dolfin.PETScVector()

        bcs = self.formulation.build_boundary_conditions()

        if system:
            dolfin.assemble_system(wf[0], dv, bcs, A_tensor=A, b_tensor=b)
            dolfin.assemble_system(wf[1], dv, A_tensor=B, b_tensor=b)

        else:
            dolfin.assemble(wf[0], tensor=A)
            for bc in bcs:
                bc.apply(A)
            dolfin.assemble(wf[1], tensor=B)

        # [bc.zero(A) for bc in bcs]
        # [bc.zero(B) for bc in bcs]

        eigensolver = dolfin.SLEPcEigenSolver(
            dolfin.as_backend_type(A), dolfin.as_backend_type(B)
        )
        eigensolver.parameters["spectrum"] = "target magnitude"
        eigensolver.parameters["solver"] = "krylov-schur"
        eigensolver.parameters["spectral_shift"] = float(wavevector_target**2)
        eigensolver.parameters["spectral_transform"] = "shift-and-invert"
        eigensolver.parameters["tolerance"] = tol
        eigensolver.parameters.update(kwargs)
        eigensolver.set_from_options()
        NEIG = 2 * n_eig if half else n_eig
        eigensolver.solve(NEIG)
        nconv = eigensolver.get_number_converged()

        self.solution = {"converged": nconv}
        KNs = []
        UNs = []

        nconv = min(NEIG, nconv)

        for j in range(nconv):
            ev_re, ev_im, rx, cx = eigensolver.get_eigenpair(j)
            eig_vec_right = array2function(rx, self.formulation.function_space)
            eig_vec_left = array2function(cx, self.formulation.function_space)

            if self.formulation.dim == 1:
                eig_vec = Complex(*eig_vec_right)  # + 1j * Complex(*eig_vec_im)
            else:
                eig_vec_re = [eig_vec_right[i] for i in range(3)]
                eig_vec_im = [eig_vec_right[i] for i in range(3, 6)]
                # eig_vec_im_re = [eig_vec_im[i] for i in range(3)]
                # eig_vec_im_im = [eig_vec_im[i] for i in range(3, 6)]

                eig_vec = vector(Complex(eig_vec_re, eig_vec_im))

            # eig_vec = Complex(eig_vec_re[0],eig_vec_im[1])
            ev = ev_re + 1j * ev_im
            kn = (ev) ** 0.5 if sqrt else ev
            KNs.append(kn)
            UNs.append(eig_vec)
        KNs = np.array(KNs)

        # HACK: We get the complex conjugates as well so compute twice
        # as much eigenvalues and return only half
        if half:
            KNs = KNs[::2]
            UNs = UNs[::2]

        self.solution["eigenvalues"] = KNs
        self.solution["eigenvectors"] = UNs
        self.eigensolver = eigensolver
        return self.solution
