#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.0.2
# License: MIT
# See the documentation at gyptis.gitlab.io

from .simulation import *


class Homogenization2D(Simulation):
    def __init__(
        self,
        geometry,
        epsilon=None,
        mu=None,
        boundary_conditions=None,
        degree=1,
        direction="x",
        direct=True,
        periodic=True,
        domain="everywhere",
    ):
        if boundary_conditions is None:
            boundary_conditions = {}
        # assert isinstance(geometry, Lattice2D)
        self.epsilon, self.mu = init_em_materials(geometry, epsilon, mu)
        self.domain = domain
        self.periodic_bcs = BiPeriodic2D(geometry) if periodic else None
        function_space = ComplexFunctionSpace(
            geometry.mesh, "CG", degree, constrained_domain=self.periodic_bcs
        )
        epsilon_coeff = Coefficient(self.epsilon, geometry, degree=degree)
        mu_coeff = Coefficient(self.mu, geometry, degree=degree)

        coefficients = epsilon_coeff, mu_coeff
        self.formulations = dict(epsilon={}, mu={})
        for case in ["epsilon", "mu"]:
            for direction in ["x", "y"]:
                form = TwoScale2D(
                    geometry,
                    coefficients,
                    function_space,
                    degree=degree,
                    boundary_conditions=boundary_conditions,
                    direction=direction,
                    case=case,
                )
                self.formulations[case][direction] = form
        self.formulation = self.formulations["epsilon"]["x"]
        super().__init__(geometry, self.formulation, direct=direct)
        self.solution = dict(epsilon={}, mu={})
        self.direct = direct
        self.direction = direction
        self.degree = degree
        self.cell_volume = np.cross(*geometry.vectors)
        # self.cell_volume = assemble(1 * geometry.measure["dx"])

    def solve_system(self, again=False):
        return super().solve_system(again=again, vector_function=False)

    def unit_cell_mean(self, f):
        return 1 / self.cell_volume * assemble(f * self.dx(self.domain))

    def solve_param(self, case, scalar=False):
        self.formulation = self.formulations[case]["x"]
        super().__init__(self.geometry, self.formulation, direct=self.direct)
        phi_x = self.solve()
        if not scalar:
            self.formulations[case]["y"].build_rhs()
            self.vector = assemble(self.formulations[case]["y"].rhs)
            phi_y = self.solve_system(again=True)
            self.solution[case] = dict(x=phi_x, y=phi_y)
        else:
            self.solution[case] = dict(x=phi_x)
        return self.solution

    def solve_all(self):
        phi_x = self.solve()
        self.formulation_y.build_rhs()
        self.vector = assemble(self.formulation_y.rhs)
        phi_y = self.solve_system(again=True)
        self.solution = dict(x=phi_x, y=phi_y)
        return self.solution

    def _get_effective_coeff(self, case, scalar=False):
        self.solve_param(case, scalar=scalar)
        xi = self.formulation.xi.as_subdomain()
        if xi.real.ufl_shape == (2, 2):
            if scalar:
                raise ValueError("scalar cannot be used with anisotropic materials")
            xi_mean = []
            for i in range(2):
                a = [self.unit_cell_mean(x).tocomplex() for x in xi[i]]

                xi_mean.append(a)
            xi_mean = np.array(xi_mean)
        else:
            xi_mean = self.unit_cell_mean(xi).tocomplex()
            if not scalar:
                xi_mean *= np.eye(2)
        if not scalar:
            A = []
            for phi in self.solution[case].values():
                integrand = xi * grad(phi)
                a = [self.unit_cell_mean(g).tocomplex() for g in integrand]
                A.append(a)
            xi_eff = xi_mean + np.array(A)
            param_eff_inplane = xi_eff.T / np.linalg.det(xi_eff)

        else:
            phi = self.solution[case]["x"]
            integrand = xi * grad(phi)
            a = self.unit_cell_mean(integrand[0]).tocomplex()
            xi_eff = xi_mean + a
            param_eff_inplane = np.eye(2) / xi_eff

        param_eff = np.zeros((3, 3), complex)
        param_eff[:2, :2] = param_eff_inplane
        i = 0 if case == "epsilon" else 1
        coeff = self.formulation.coefficients[i].as_subdomain()
        coeffzz = coeff[2][2] if coeff.real.ufl_shape == (3, 3) else coeff
        param_eff[2, 2] = self.unit_cell_mean(coeffzz).tocomplex()
        return param_eff

    def get_effective_permittivity(self, scalar=False):
        return self._get_effective_coeff("epsilon", scalar)

    def get_effective_permeability(self, scalar=False):
        return self._get_effective_coeff("mu", scalar)
