#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# License: MIT
# See the documentation at gyptis.gitlab.io

from .simulation import *


class Homogenization3D(Simulation):
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
        assert isinstance(geometry, Lattice3D)

        self.geometry = geometry
        self.periodic_bcs = Periodic3D(geometry)
        function_space = ComplexFunctionSpace(
            geometry.mesh, "CG", degree, constrained_domain=self.periodic_bcs
        )
        epsilon = {k: e + 1e-16j for k, e in epsilon.items()}
        mu = {k: m + 1e-16j for k, m in mu.items()}
        epsilon_coeff = Coefficient(epsilon, geometry, degree=degree, dim=3)
        mu_coeff = Coefficient(mu, geometry, degree=degree, dim=3)

        coefficients = epsilon_coeff, mu_coeff
        self.formulations = dict(epsilon={}, mu={})
        for case in ["epsilon", "mu"]:
            for direction in ["x", "y", "z"]:
                form = TwoScale3D(
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
        self.degree = degree
        self.solution = dict(epsilon={}, mu={})
        self.direct = direct
        self.direction = direction
        self.cell_volume = np.dot(
            np.cross(geometry.vectors[0], geometry.vectors[1]), geometry.vectors[2]
        )

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
        self.formulations[case]["z"].build_rhs()
        self.vector = assemble(self.formulations[case]["z"].rhs)
        phi_z = self.solve_system(again=True)
        self.solution[case] = dict(x=phi_x, y=phi_y, z=phi_z)
        return self.solution

    def _get_effective_param(self, case):
        self.solve_param(case)
        coeff = self.formulation.epsilon if case == "epsilon" else self.formulation.mu
        param = coeff.as_subdomain()
        if param.real.ufl_shape == (3, 3):
            param_mean = []
            for i in range(3):
                a = [self.unit_cell_mean(x) for x in param[i]]
                a = [_.real + 1j * _.imag for _ in a]
                param_mean.append(a)
            param_mean = np.array(param_mean)
        else:
            param_mean = self.unit_cell_mean(param)
            param_mean = param_mean.real + 1j * param_mean.imag
            param_mean *= np.eye(3)
        A = []
        for phi in self.solution[case].values():
            integrand = param * grad(phi)
            a = [self.unit_cell_mean(g) for g in integrand]
            a = [_.real + 1j * _.imag for _ in a]
            A.append(a)
        param_eff = param_mean - np.array(A)
        return param_eff

    def get_effective_permittivity(self):
        return self._get_effective_param("epsilon")

    def get_effective_permeability(self):
        return self._get_effective_param("mu")
