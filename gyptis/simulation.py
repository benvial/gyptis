#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

import numpy as np

from . import dolfin
from .complex import Complex, Function, assemble
from .helpers import array2function


class Simulation:
    def __init__(self, geometry, formulation=None):
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

    @property
    def source(self):
        return self._source

    @source.setter
    def source(self, value):
        self._source = value

    def assemble_lhs(self, **kwargs):
        self.formulation.build_lhs(**kwargs)
        self.matrix = assemble(self.formulation.lhs)
        return self.matrix

    def assemble_rhs(self, **kwargs):
        self.formulation.build_rhs(**kwargs)
        self.vector = assemble(self.formulation.rhs)
        return self.vector

    def assemble(self, **kwargs):
        self.matrix = self.assemble_lhs(**kwargs)
        self.vector = self.assemble_rhs(**kwargs)
        return self.matrix, self.vector

    def apply_boundary_conditions(self):
        bcs = self.formulation.build_boundary_conditions()
        for bc in bcs:
            bc.apply(self.matrix, self.vector)

    def solve_system(self, again=False, vector_function=True):
        if vector_function:
            element = self.function_space.split()[0].ufl_element()
            V_vect = dolfin.VectorFunctionSpace(
                self.mesh, element.family(), element.degree()
            )
            u = dolfin.Function(V_vect)
        else:
            u = dolfin.Function(self.function_space)

        if not again:
            self.solver = dolfin.LUSolver(self.matrix, "mumps")

        self.solver.solve(u.vector(), self.vector)
        solution = Complex(*u.split())
        return solution

    def solve(self, again=False, **kwargs):
        self.assemble(**kwargs)
        self.apply_boundary_conditions()
        return self.solve_system(again=again, **kwargs)

    def eigensolve(self, n_eig=6, wavevector_target=0.0, tol=1e-6, parameters={}):

        wf = self.formulation.weak
        # A = assemble(wf[0])
        # B = assemble(wf[1])

        dummy_vector = dolfin.Constant(0) * self.formulation.test * self.formulation.dx
        dv = dummy_vector.real + dummy_vector.imag

        # Assemble matrices
        A = dolfin.PETScMatrix()
        B = dolfin.PETScMatrix()
        b = dolfin.PETScVector()

        bcs = self.formulation.build_boundary_conditions()

        dolfin.assemble_system(wf[0], dv, bcs, A_tensor=A, b_tensor=b)
        dolfin.assemble_system(wf[1], dv, bcs, A_tensor=B, b_tensor=b)

        eigensolver = dolfin.SLEPcEigenSolver(
            dolfin.as_backend_type(A), dolfin.as_backend_type(B)
        )
        # eigensolver.parameters["problem_type"] = "gen_hermitian"
        eigensolver.parameters["spectrum"] = "target real"
        # eigensolver.parameters["spectrum"] = "target magnitude"
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

        # eigensolver.parameters["verbose"] = True  # for debugging
        eigensolver.parameters.update(parameters)
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
            eig_vec = Complex(*eig_vec_re) + 1j * Complex(*eig_vec_im)
            # eig_vec = Complex(eig_vec_re[0],eig_vec_im[1])
            ev = ev_re + 1j * ev_im
            kn = (ev) ** 0.5
            KNs.append(kn)
            UNs.append(eig_vec)

        # HACK: We get the complex conjugates as well so compoute twice
        # as much eigenvalues and return only half
        KNs = np.array(KNs)[::2]
        UNs = UNs[::2]

        self.solution["eigenvalues"] = KNs
        self.solution["eigenvectors"] = UNs
        self.eigensolver = eigensolver
        return self.solution
