#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

from .bc import BiPeriodic2D, Periodic3D
from .formulation import TwoScale2D, TwoScale3D
from .geometry import *
from .materials import *
from .phc2d import Lattice2D
from .phc3d import Lattice3D
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

    def _get_effective_coeff(self, polarization):
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

    def get_effective_permittivity(self):
        return self._get_effective_coeff("TE")

    def get_effective_permeability(self):
        return self._get_effective_coeff("TM")


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
        assert np.all([m == 1 for m in mu.values()]), "mu must be unity"

        self.periodic_bcs = Periodic3D(geometry)
        function_space = ComplexFunctionSpace(
            geometry.mesh, "CG", degree, constrained_domain=self.periodic_bcs
        )

        epsilon = {k: e + 1e-16j for k, e in epsilon.items()}
        mu = {k: m + 1e-16j for k, m in mu.items()}
        epsilon_coeff = Coefficient(epsilon, geometry, degree=degree, dim=3)
        mu_coeff = Coefficient(mu, geometry, degree=degree, dim=3)

        coefficients = epsilon_coeff, mu_coeff

        self.formulation = TwoScale3D(
            geometry,
            coefficients,
            function_space,
            degree=degree,
            boundary_conditions=boundary_conditions,
            direction=direction,
            case="epsilon",
        )

        self.formulation_x = TwoScale3D(
            geometry,
            coefficients,
            function_space,
            degree=degree,
            boundary_conditions=boundary_conditions,
            direction="x",
            case="epsilon",
        )

        self.formulation_y = TwoScale3D(
            geometry,
            coefficients,
            function_space,
            degree=degree,
            boundary_conditions=boundary_conditions,
            direction="y",
            case="epsilon",
        )
        self.formulation_z = TwoScale3D(
            geometry,
            coefficients,
            function_space,
            degree=degree,
            boundary_conditions=boundary_conditions,
            direction="z",
            case="epsilon",
        )
        super().__init__(geometry, self.formulation, direct=direct)
        self.degree = degree
        self.direction = direction
        self.cell_volume = np.dot(
            np.cross(geometry.vectors[0], geometry.vectors[1]), geometry.vectors[2]
        )

    def solve_system(self, again=False):
        return super().solve_system(again=again, vector_function=False)

    def unit_cell_mean(self, f):
        return 1 / self.cell_volume * assemble(f * self.dx)

    def solve_all(self):
        phi_x = self.solve()
        self.formulation_y.build_rhs()
        self.vector = assemble(self.formulation_y.rhs)
        phi_y = self.solve_system(again=True)
        self.formulation_z.build_rhs()
        self.vector = assemble(self.formulation_z.rhs)
        phi_z = self.solve_system(again=True)
        self.solution = dict(x=phi_x, y=phi_y, z=phi_z)
        return self.solution

    def get_effective_permittivity(self):
        self.solve_all()
        epsilon = self.formulation.epsilon.as_subdomain()
        if epsilon.real.ufl_shape == (3, 3):
            epsilon_mean = []
            for i in range(3):
                a = [self.unit_cell_mean(x) for x in epsilon[i]]
                a = [_.real + 1j * _.imag for _ in a]
                epsilon_mean.append(a)
            epsilon_mean = np.array(epsilon_mean)
        else:
            epsilon_mean = self.unit_cell_mean(epsilon)
            epsilon_mean = epsilon_mean.real + 1j * epsilon_mean.imag
            epsilon_mean *= np.eye(3)
        A = []
        for phi in self.solution.values():
            integrand = -epsilon * grad(phi)
            a = [self.unit_cell_mean(g) for g in integrand]
            a = [_.real + 1j * _.imag for _ in a]
            A.append(a)
        print(A)
        print(epsilon_mean)
        epsilon_eff = epsilon_mean - np.array(A)
        return epsilon_eff
