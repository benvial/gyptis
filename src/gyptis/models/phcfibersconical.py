#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.0.2
# License: MIT
# See the documentation at gyptis.gitlab.io


from .simulation import *


class PHCFibersConical(Simulation):
    def __init__(
        self,
        geometry,
        epsilon=None,
        mu=None,
        source=None,
        boundary_conditions=None,
        degree=(2, 2),
        pml_stretch=1 - 1j,
        beta=0,
        propagation_vector=(0, 0),
        eps=dolfin.DOLFIN_EPS,
        map_tol=1e-10,
        type="E",
    ):
        if boundary_conditions is None:
            boundary_conditions = {}
        assert isinstance(geometry, Lattice2D)
        if source is not None:
            assert source.dim == 2

        self.beta = beta
        self.epsilon, self.mu = init_em_materials(geometry, epsilon, mu)

        self.propagation_vector = propagation_vector

        Element1 = dolfin.FiniteElement("N1curl", geometry.mesh.ufl_cell(), degree[0])
        Element2 = dolfin.FiniteElement("CG", geometry.mesh.ufl_cell(), degree[1])
        element = dolfin.MixedElement([Element1, Element2])
        # Element3 = dolfin.FiniteElement("R", geometry.mesh.ufl_cell(), 0)
        # element = dolfin.MixedElement([Element1, Element2, Element3])

        self.periodic_bcs = BiPeriodic2D(geometry, map_tol=map_tol, eps=eps)
        function_space = ComplexFunctionSpace(
            geometry.mesh, element, constrained_domain=self.periodic_bcs
        )
        fs = dolfin.FunctionSpace(geometry.mesh, "DG", 0)
        el = fs.ufl_element()
        epsilon_coeff = Coefficient(self.epsilon, geometry, degree=0, element=el)
        mu_coeff = Coefficient(self.mu, geometry, degree=0, element=el)

        coefficients = epsilon_coeff, mu_coeff
        no_source_domains = ["box"]
        source_domains = []
        formulation = MaxwellConicalBands(
            geometry,
            coefficients,
            function_space,
            source=source,
            source_domains=source_domains,
            reference="box",
            modal=True,
            boundary_conditions=boundary_conditions,
            beta=beta,
            propagation_vector=propagation_vector,
            type=type,
        )
        super().__init__(geometry, formulation)

        self.degree = degree

    def eigensolve(self, *args, **kwargs):
        sol = super().eigensolve(*args, **kwargs)
        phasor = self.formulation.phasor
        self.solution["eigenvectors"] = [u * phasor for u in sol["eigenvectors"]]
        return self.solution
