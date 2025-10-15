#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.0.3
# License: MIT
# See the documentation at gyptis.gitlab.io


from .simulation import *


class Waveguide(Simulation):
    def __init__(
        self,
        geometry,
        epsilon=None,
        mu=None,
        wavenumber=0,
        boundary_conditions=None,
        degree=(2, 2),
        pml_stretch=1 - 1j,
    ):
        if boundary_conditions is None:
            boundary_conditions = {}
        assert isinstance(geometry, LayeredBoxPML2D)
        source = None

        self.wavenumber = wavenumber
        self.epsilon, self.mu = init_em_materials(geometry, epsilon, mu)

        Element1 = dolfin.FiniteElement("N1curl", geometry.mesh.ufl_cell(), degree[0])
        Element2 = dolfin.FiniteElement("CG", geometry.mesh.ufl_cell(), degree[1])
        element = dolfin.MixedElement([Element1, Element2])

        function_space = ComplexFunctionSpace(geometry.mesh, element)
        names = list(geometry.layers.keys())

        pmls_list = []
        for name in names:
            pmlx = PML(
                "x",
                stretch=pml_stretch,
                matched_domain=name,
                applied_domain="pmlx_" + name,
            )
            pmls_list.append(pmlx)
        for name in [names[-1], names[0]]:
            pmly = PML(
                "y",
                stretch=pml_stretch,
                matched_domain=name,
                applied_domain="pmly_" + name,
            )
            pmls_list.append(pmly)
            pmlxy = PML(
                "xy",
                stretch=pml_stretch,
                matched_domain=name,
                applied_domain="pmlxy_" + name,
            )
            pmls_list.append(pmlxy)

        degree_mat = max(degree)

        epsilon_coeff = Coefficient(
            self.epsilon, geometry, pmls=pmls_list, degree=degree_mat
        )
        mu_coeff = Coefficient(self.mu, geometry, pmls=pmls_list, degree=degree_mat)

        coefficients = epsilon_coeff, mu_coeff
        no_source_domains = pmls_list
        source_domains = []
        formulation = MaxwellWaveguide(
            geometry,
            coefficients,
            function_space,
            source=source,
            source_domains=source_domains,
            boundary_conditions=boundary_conditions,
            wavenumber=wavenumber,
        )

        super().__init__(geometry, formulation)

        self.degree = max(degree)

    def solve(self, **kwargs):
        raise NotImplementedError

    def eigensolve(self, **kwargs):
        # kwargs["half"] = False
        self.solution = super().eigensolve(**kwargs)
        for i, (kz, v) in enumerate(
            zip(self.solution["eigenvalues"], self.solution["eigenvectors"])
        ):
            V0 = v[0] / Constant(kz)
            V1 = v[1] / Constant(kz)
            V2 = v[2] / Complex(0, 1)
            self.solution["eigenvectors"][i] = vector([V0, V1, V2])
        return self.solution
