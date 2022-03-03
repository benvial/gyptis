#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# License: MIT
# See the documentation at gyptis.gitlab.io

from .simulation import *


class Homogenization2D(Simulation):
    def __init__(
        self,
        geometry,
        epsilon,
        mu,
        boundary_conditions={},
        degree=1,
        direction="x",
        direct=True,
    ):
        assert isinstance(geometry, Lattice2D)

        self.periodic_bcs = BiPeriodic2D(geometry)
        function_space = ComplexFunctionSpace(
            geometry.mesh, "CG", degree, constrained_domain=self.periodic_bcs
        )
        epsilon = {k: e + 1e-16j for k, e in epsilon.items()}
        mu = {k: m + 1e-16j for k, m in mu.items()}
        epsilon_coeff = Coefficient(epsilon, geometry, degree=degree)
        mu_coeff = Coefficient(mu, geometry, degree=degree)

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

    def solve_system(self, again=False):
        return super().solve_system(again=again, vector_function=False)

    def unit_cell_mean(self, f):
        return 1 / self.cell_volume * assemble(f * self.dx)

    def solve_param(self, case):
        self.formulation = self.formulations[case]["x"]
        super().__init__(self.geometry, self.formulation, direct=self.direct)
        phi_x = self.solve()
        self.formulations[case]["y"].build_rhs()
        self.vector = assemble(self.formulations[case]["y"].rhs)
        phi_y = self.solve_system(again=True)
        self.solution[case] = dict(x=phi_x, y=phi_y)
        return self.solution

    def solve_all(self):
        phi_x = self.solve()
        self.formulation_y.build_rhs()
        self.vector = assemble(self.formulation_y.rhs)
        phi_y = self.solve_system(again=True)
        self.solution = dict(x=phi_x, y=phi_y)
        return self.solution

    def _get_effective_coeff(self, case):
        self.solve_param(case)
        xi = self.formulation.xi.as_subdomain()
        if xi.real.ufl_shape == (2, 2):
            xi_mean = []
            for i in range(2):
                a = [self.unit_cell_mean(x) for x in xi[i]]
                a = [_.real + 1j * _.imag for _ in a]
                xi_mean.append(a)
            xi_mean = np.array(xi_mean)
        else:
            xi_mean = self.unit_cell_mean(xi)
            xi_mean = xi_mean.real + 1j * xi_mean.imag
            xi_mean *= np.eye(2)
        A = []
        for phi in self.solution[case].values():
            integrand = xi * grad(phi)
            a = [self.unit_cell_mean(g) for g in integrand]
            a = [_.real + 1j * _.imag for _ in a]
            A.append(a)
        xi_eff = xi_mean + np.array(A)
        param_eff_inplane = xi_eff.T / np.linalg.det(xi_eff)
        param_eff = np.zeros((3, 3), complex)
        param_eff[:2, :2] = param_eff_inplane
        i = 0 if case == "epsilon" else 1
        coeff = self.formulation.coefficients[i].as_subdomain()
        if coeff.real.ufl_shape == (3, 3):
            coeffzz = coeff[2][2]
        else:
            coeffzz = coeff
        czz = self.unit_cell_mean(coeffzz)
        param_eff[2, 2] = czz.real + 1j * czz.imag
        return param_eff

    def get_effective_permittivity(self):
        return self._get_effective_coeff("epsilon")

    def get_effective_permeability(self):
        return self._get_effective_coeff("mu")
