#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.0.2
# License: MIT
# See the documentation at gyptis.gitlab.io

from .simulation import *


class PhotonicCrystal2D(Simulation):
    """
    2D photonic crystal class

    Parameters
    ----------
    geometry : Lattice2D
        The geometry of the photonic crystal
    epsilon : dict
        Permittivity in various subdomains.
    mu : dict
        Permeability in various subdomains.
    propagation_vector : tuple of float
        The propagation vector of the mode
    boundary_conditions : dict or list of dict
        Boundary conditions of the simulation
    polarization : str
        Polarization of the mode
    degree : int
        The degree of the function space
    eps : float
        The tolerance for the periodic boundary conditions values
    map_tol : float
        The tolerance for the periodic boundary conditions mesh
    """

    def __init__(
        self,
        geometry,
        epsilon=None,
        mu=None,
        propagation_vector=(0, 0),
        boundary_conditions=None,
        polarization="TM",
        degree=1,
        eps=dolfin.DOLFIN_EPS,
        map_tol=1e-10,
    ):
        if boundary_conditions is None:
            boundary_conditions = {}
        assert isinstance(geometry, Lattice2D)
        self.epsilon, self.mu = init_em_materials(geometry, epsilon, mu)
        self.periodic_bcs = BiPeriodic2D(geometry, map_tol=map_tol, eps=eps)
        function_space = ComplexFunctionSpace(
            geometry.mesh, "CG", degree, constrained_domain=self.periodic_bcs
        )
        epsilon_coeff = Coefficient(self.epsilon, geometry, degree=degree)
        mu_coeff = Coefficient(self.mu, geometry, degree=degree)

        coefficients = epsilon_coeff, mu_coeff
        formulation = Maxwell2DBands(
            geometry,
            coefficients,
            function_space,
            propagation_vector=propagation_vector,
            degree=degree,
            polarization=polarization,
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
