#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

from .bc import BiPeriodic2D
from .formulation import TwoScale2D
from .geometry import *
from .materials import *
from .phc2d import Lattice2D
from .simulation import Simulation


class Homogenization2D(Simulation):
    def __init__(
        self,
        geometry,
        epsilon,
        mu,
        boundary_conditions={},
        polarization="TM",
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

        self.formulation = TwoScale2D(
            geometry,
            coefficients,
            function_space,
            degree=degree,
            polarization=polarization,
            boundary_conditions=boundary_conditions,
            direction=direction,
        )

        self.formulation_x = TwoScale2D(
            geometry,
            coefficients,
            function_space,
            degree=degree,
            polarization=polarization,
            boundary_conditions=boundary_conditions,
            direction="x",
        )

        self.formulation_y = TwoScale2D(
            geometry,
            coefficients,
            function_space,
            degree=degree,
            polarization=polarization,
            boundary_conditions=boundary_conditions,
            direction="y",
        )

        super().__init__(geometry, self.formulation, direct=direct)
        self.degree = degree
        self.direction = direction
        self.cell_volume = np.cross(*geometry.vectors)

    def solve_system(self, again=False):
        return super().solve_system(again=again, vector_function=False)

    def unit_cell_mean(self, f):
        return 1 / self.cell_volume * assemble(f * self.dx)

    def solve_all(self):
        phi_x = self.solve()
        self.formulation_y.build_rhs()
        self.vector = assemble(self.formulation_y.rhs)
        phi_y = self.solve_system(again=True)
        self.solution = dict(x=phi_x, y=phi_y)
        return self.solution

    def get_effective_permittivity(self):
        if self.formulation.polarization == "TE":
            self.solve_all()
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
            for phi in self.solution.values():
                integrand = xi * grad(phi)
                a = [self.unit_cell_mean(g) for g in integrand]
                a = [_.real + 1j * _.imag for _ in a]
                A.append(a)
            xi_eff = xi_mean + np.array(A)
            return xi_eff.T / np.linalg.det(xi_eff)
        else:
            chi = self.formulation.chi.as_subdomain()
            return self.unit_cell_mean(chi)
