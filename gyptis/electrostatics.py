#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

from . import dolfin as df
from .base import *
from .helpers import DirichletBC as DirichletBCReal
from .helpers import PeriodicBoundary2DX


class Grating2DElstat(Simulation2D):
    def __init__(
        self,
        geometry,
        epsilon,
        degree=1,
        boundary_conditions={},
    ):
        super().__init__(
            geometry, degree=degree, boundary_conditions=boundary_conditions
        )
        self.epsilon = epsilon

        self.periodic_bcs = PeriodicBoundary2DX(self.geometry.period)
        self.function_space = df.FunctionSpace(
            self.mesh, "CG", self.degree, constrained_domain=self.periodic_bcs
        )

    def _make_subdomains(self):
        return Subdomain(self.markers, self.domains, self.epsilon, degree=self.degree)

    def _prepare_materials(self):
        epsilon = dict(superstrate=1, substrate=1, pml_top=1, pml_bottom=1)
        epsilon.update(self.epsilon)
        self.epsilon["pml_top"] = self.epsilon["superstrate"]
        self.epsilon["pml_bottom"] = self.epsilon["substrate"]
        self.epsilon_coeff = self._make_subdomains()

    def _prepare_bcs(self):
        self._boundary_conditions = []
        self.voltage_bnds = []
        curves = self.geometry.subdomains["curves"]
        markers_curves = self.geometry.mesh_object["markers"]["line"]
        for bnd, cond in self.boundary_conditions.items():
            if cond[0] != "voltage":
                raise ValueError(f"unknown boundary condition {cond}")
            else:
                self.voltage_bnds.append(bnd)
            bc = DirichletBCReal(
                self.function_space, cond[1], markers_curves, bnd, curves
            )
            self._boundary_conditions.append(bc)

    def weak_form(self):
        self._prepare_materials()
        self._prepare_bcs()
        W = self.function_space
        dx = self.dx
        ds = self.ds
        self.potential = df.Function(W)
        Vtrial = df.TrialFunction(W)
        Vtest = df.TestFunction(W)
        n = self.unit_normal_vector

        L = [inner(self.epsilon_coeff * grad(Vtrial), grad(Vtest)) * dx]
        b = [df.Constant(0.0) * Vtest * dx]

        self.lhs = L
        self.rhs = b

    def assemble_lhs(self):
        self.Ah = [assemble(A) for A in self.lhs]

    def assemble_rhs(self):
        self.bh = [assemble(b) for b in self.rhs]

    def assemble(self):
        self.assemble_lhs()
        self.assemble_rhs()

    def solve_system(self, direct=True):
        Ah = self.Ah[0]
        bh = self.bh[0]
        for bc in self._boundary_conditions:
            bc.apply(Ah, bh)

        if direct:
            solver = df.PETScLUSolver("mumps")
            # solver.parameters.update(lu_params)
            solver.solve(Ah, self.potential.vector(), bh)
        else:
            solver = df.PETScKrylovSolver()  ## iterative
            # solver.parameters.update(krylov_params)
            solver.solve(Ah, self.potential.vector(), bh)

    def solve(self, direct=True):
        self.weak_form()
        self.assemble()
        self.solve_system(direct=direct)
