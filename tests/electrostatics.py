#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

from . import dolfin as df
from .base import *
from .bc import PeriodicBoundary2DX
from .bc import _DirichletBC as DirichletBCReal
from .grating3d import *


class Grating2DElstat(Simulation2D):
    def __init__(
        self, geometry, epsilon, degree=1, boundary_conditions={},
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

    def _prepare_bcs(self, doms="curves", marks="line"):
        self._boundary_conditions = []
        self.voltage_bnds = []
        curves = self.geometry.subdomains[doms]
        markers_curves = self.geometry.mesh_object["markers"][marks]
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

    def build_system(self):
        pass

    def prepare(self):
        pass

    def solve_system(self, direct=False):
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
        self.electric_field = -grad(self.potential)

    def solve(self, direct=False):
        self.weak_form()
        self.assemble()
        self.solve_system(direct=direct)


class Grating3DElstat(Grating2DElstat, Grating3D):
    def __init__(
        self, geometry, epsilon, degree=1, boundary_conditions={},
    ):
        Grating3D.__init__(
            self,
            geometry,
            epsilon,
            mu=None,
            lambda0=None,
            theta0=None,
            phi0=None,
            psi0=None,
            degree=degree,
            mat_degree=None,
            pml_stretch=1,
            boundary_conditions={},
            periodic_map_tol=1e-8,
        )
        del self.mu
        del self.lambda0
        del self.theta0
        del self.phi0
        del self.psi0
        del self.complex_space
        del self.real_space

        self.element = "CG"
        self.function_space = dolfin.FunctionSpace(
            self.mesh, self.element, self.degree, constrained_domain=self.periodic_bcs
        )

        self.dx = geometry.measure["dx"]
        self.ds = geometry.measure["ds"] if self.boundaries else None
        self.dS = geometry.measure["dS"] if self.boundaries else None

        self.unit_normal_vector = dolfin.FacetNormal(self.mesh)

    def _prepare_bcs(self, doms="surfaces", marks="triangle"):
        Grating2DElstat._prepare_bcs(self, doms=doms, marks=marks)
