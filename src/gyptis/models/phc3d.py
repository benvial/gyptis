#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.0.2
# License: MIT
# See the documentation at gyptis.gitlab.io

from .simulation import *


class PhotonicCrystal3D(Simulation):
    def __init__(
        self,
        geometry,
        epsilon=None,
        mu=None,
        propagation_vector=(0, 0, 0),
        boundary_conditions=None,
        degree=1,
    ):
        if boundary_conditions is None:
            boundary_conditions = {}
        assert isinstance(geometry, Lattice3D)
        self.epsilon, self.mu = init_em_materials(geometry, epsilon, mu)
        self.periodic_bcs = Periodic3D(geometry)
        function_space = ComplexFunctionSpace(
            geometry.mesh, "N1curl", degree, constrained_domain=self.periodic_bcs
        )
        epsilon_coeff = Coefficient(self.epsilon, geometry, degree=degree, dim=3)
        mu_coeff = Coefficient(self.mu, geometry, degree=degree, dim=3)

        coefficients = epsilon_coeff, mu_coeff
        formulation = Maxwell3DBands(
            geometry,
            coefficients,
            function_space,
            propagation_vector=propagation_vector,
            degree=degree,
            boundary_conditions=boundary_conditions,
        )

        super().__init__(geometry, formulation)

        self.degree = degree
        self.propagation_vector = propagation_vector

    def eigensolve(self, *args, **kwargs):
        sol = super().eigensolve(*args, **kwargs)
        self.solution["eigenvectors"] = [
            u * self.formulation.phasor for u in sol["eigenvectors"]
        ]
        return self.solution
