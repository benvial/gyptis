#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
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
from ..sources import *
from ..utils import project_iterative
from ..utils.helpers import array2function


class Simulation:
    def __init__(self, geometry, formulation=None, direct=True):
        self.geometry = geometry
        self.formulation = formulation
        self.coefficients = formulation.coefficients
        self.function_space = formulation.function_space
        self._source = formulation.source
        self.boundary_conditions = formulation.boundary_conditions
        self.mesh = self.geometry.mesh
        self._boundary_conditions = []
        self.dx = formulation.dx
        self.ds = formulation.ds
        self.dS = formulation.dS
        self.direct = direct

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
            The assembled matix and vector.

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
                self.solver = dolfin.LUSolver(self.matrix, "mumps")
            else:
                # self.solver = dolfin.KrylovSolver(self.matrix,"cg", "jacobi")
                self.solver = dolfin.KrylovSolver(self.matrix)
        self.solver.solve(u.vector(), self.vector)
        solution = Complex(*u.split())
        return solution

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

    def eigensolve(self, n_eig=6, wavevector_target=0.0, tol=1e-6, **kwargs):

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

        # [bc.zero(B) for bc in bcs]

        dolfin.assemble_system(wf[0], dv, bcs, A_tensor=A, b_tensor=b)
        dolfin.assemble_system(wf[1], dv, bcs, A_tensor=B, b_tensor=b)

        eigensolver = dolfin.SLEPcEigenSolver(
            dolfin.as_backend_type(A), dolfin.as_backend_type(B)
        )
        # eigensolver.parameters["problem_type"] = "gen_hermitian"
        # eigensolver.parameters["spectrum"] = "target real"
        eigensolver.parameters["spectrum"] = "target magnitude"
        eigensolver.parameters["solver"] = "krylov-schur"
        # eigensolver.parameters["solver"] = "power"
        eigensolver.parameters["spectral_shift"] = float(wavevector_target ** 2)
        eigensolver.parameters["spectral_transform"] = "shift-and-invert"
        eigensolver.parameters["tolerance"] = tol
        # eigensolver.parameters["solver"] = "mumps"
        dolfin.PETScOptions.set("st_ksp_type", "preonly")
        dolfin.PETScOptions.set("st_pc_type", "lu")
        # dolfin.PETScOptions.set("st_pc_factor_mat_solver_type", "mumps")
        # dolfin.PETScOptions.set("eps_max_it", "300")
        # dolfin.PETScOptions.set("eps_target", "0.00001")
        # dolfin.PETScOptions.set("eps_mpd", "600")
        # dolfin.PETScOptions.set("eps_nev", "400")
        eigensolver.parameters.update(kwargs)
        eigensolver.solve(2 * n_eig)

        nconv = eigensolver.get_number_converged()

        self.solution = {}
        self.solution["converged"] = nconv

        KNs = []
        UNs = []

        nconv = min(2 * n_eig, nconv)

        for j in range(nconv):
            ev_re, ev_im, rx, cx = eigensolver.get_eigenpair(j)
            eig_vec_re = array2function(rx, self.formulation.function_space)
            eig_vec_im = array2function(cx, self.formulation.function_space)

            if self.formulation.dim == 1:
                eig_vec = Complex(*eig_vec_re) + 1j * Complex(*eig_vec_im)
            else:

                eig_vec_re_re = [eig_vec_re[i] for i in range(3)]
                eig_vec_re_im = [eig_vec_re[i] for i in range(3, 6)]
                eig_vec_im_re = [eig_vec_im[i] for i in range(3)]
                eig_vec_im_im = [eig_vec_im[i] for i in range(3, 6)]

                re = Complex(eig_vec_re_re, eig_vec_re_im)
                im = Complex(eig_vec_im_re, eig_vec_im_im)
                eig_vec = vector(re) + 1j * vector(im)

            # eig_vec = Complex(eig_vec_re[0],eig_vec_im[1])
            ev = ev_re + 1j * ev_im
            kn = (ev) ** 0.5
            KNs.append(kn)
            UNs.append(eig_vec)

        # HACK: We get the complex conjugates as well so compute twice
        # as much eigenvalues and return only half
        KNs = np.array(KNs)[::2]
        UNs = UNs[::2]

        self.solution["eigenvalues"] = KNs
        self.solution["eigenvectors"] = UNs
        self.eigensolver = eigensolver
        return self.solution
