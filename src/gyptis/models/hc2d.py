#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.0.2
# License: MIT
# See the documentation at gyptis.gitlab.io

from ..formulations import Maxwell2D
from .twoscale2d import *


class _EigenProblemInclusion2D(Simulation):
    def __init__(
        self, geometry, epsilon=None, mu=None, boundary_conditions=None, degree=1
    ):
        if boundary_conditions is None:
            boundary_conditions = {}
        assert isinstance(geometry, Geometry)
        self.epsilon, self.mu = init_em_materials(geometry, epsilon, mu)
        function_space = ComplexFunctionSpace(geometry.mesh, "CG", degree)
        epsilon_coeff = Coefficient(self.epsilon, geometry, degree=degree)
        mu_coeff = Coefficient(self.mu, geometry, degree=degree)

        coefficients = epsilon_coeff, mu_coeff
        formulation = Maxwell2D(
            geometry,
            coefficients,
            function_space,
            degree=degree,
            polarization="TM",
            boundary_conditions=boundary_conditions,
            modal=True,
        )
        super().__init__(geometry, formulation)

        self.degree = degree


class HighContrastHomogenization2D(Simulation):
    def __init__(
        self,
        background,
        inclusion,
        epsilon=None,
        mu=None,
        degree=1,
        direction="x",
        direct=True,
        inclusion_boundaries="inclusion_bnds",
        boundary_conditions=None,
    ):
        assert isinstance(background, Lattice2D)
        assert isinstance(inclusion, Geometry)
        self.epsilon, self.mu = init_em_materials(background, epsilon, mu)
        epsilon_inclusion, mu_inclusion = init_em_materials(inclusion, epsilon, mu)
        self.epsilon.update(epsilon_inclusion)
        self.mu.update(mu_inclusion)
        self.background = background
        self.inclusion = inclusion
        # self.epsilon_background = dict(background=self.epsilon["background"])
        # self.mu_background = dict(background=self.mu["background"])

        self.epsilon_background = self.epsilon.copy()
        self.epsilon_background.pop("inclusion")
        self.mu_background = self.mu.copy()
        self.mu_background.pop("inclusion")

        self.epsilon_inclusion = self.epsilon.copy()
        self.epsilon_inclusion.pop("background")
        self.mu_inclusion = self.mu.copy()
        self.mu_inclusion.pop("background")

        self.hom2scale = Homogenization2D(
            self.background, self.epsilon_background, self.mu_background
        )

        boundary_conditions = boundary_conditions or {inclusion_boundaries: "PEC"}

        self.epb = _EigenProblemInclusion2D(
            self.inclusion,
            self.epsilon_inclusion,
            self.mu_inclusion,
            boundary_conditions=boundary_conditions,
            degree=degree,
        )

    def get_effective_permittivity(self, **kwargs):
        return self.hom2scale.get_effective_permittivity(**kwargs)

    def get_effective_permeability(self, k, neigs=10, wavevector_target=0, tol=1e-12):
        self.eigs = self.epb.eigensolve(neigs, wavevector_target, tol, half=False)
        Es = self.eigs["eigenvalues"]
        psis = self.eigs["eigenvectors"]
        mu_eff = 1
        dx = self.epb.dx
        for E, psi in zip(Es, psis):
            if not np.allclose(E, 1) and E.imag > 0:
                eps_ = self.epb.formulation.epsilon.as_subdomain()
                norm = assemble(eps_ * psi * psi.conj * dx)
                alpha = assemble(psi * dx)
                alpha1 = assemble(eps_ * psi.conj * dx)
                mu_eff += -(k**2 * 1) / (k**2 * 1 - E**2) * alpha * alpha1 / norm
        return mu_eff
