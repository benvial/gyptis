#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.0.2
# License: MIT
# See the documentation at gyptis.gitlab.io


from .simulation import *


class FibersConical(Simulation):
    def __init__(
        self,
        geometry,
        epsilon=None,
        mu=None,
        source=None,
        boundary_conditions=None,
        modal=False,
        degree=(2, 2),
        pml_stretch=1 - 1j,
        beta=0,
    ):
        if boundary_conditions is None:
            boundary_conditions = {}
        assert isinstance(geometry, BoxPML2D)
        if source is not None:
            assert source.dim == 2

        self.beta = beta
        self.epsilon, self.mu = init_em_materials(geometry, epsilon, mu)

        Element1 = dolfin.FiniteElement("N1curl", geometry.mesh.ufl_cell(), degree[0])
        Element2 = dolfin.FiniteElement("CG", geometry.mesh.ufl_cell(), degree[1])
        element = dolfin.MixedElement([Element1, Element2])

        function_space = ComplexFunctionSpace(geometry.mesh, element)
        pmlx = PML(
            "x", stretch=pml_stretch, matched_domain="box", applied_domain="pmlx"
        )
        pmly = PML(
            "y", stretch=pml_stretch, matched_domain="box", applied_domain="pmly"
        )
        pmlxy = PML(
            "xy", stretch=pml_stretch, matched_domain="box", applied_domain="pmlxy"
        )

        epsilon_coeff = Coefficient(
            self.epsilon, geometry, pmls=[pmlx, pmly, pmlxy], degree=degree
        )
        mu_coeff = Coefficient(
            self.mu, geometry, pmls=[pmlx, pmly, pmlxy], degree=degree
        )

        coefficients = epsilon_coeff, mu_coeff
        no_source_domains = ["box", "pmlx", "pmly", "pmlxy"]
        if modal:
            source_domains = []
        else:
            source_domains = [
                dom for dom in geometry.domains if dom not in no_source_domains
            ]
        formulation = MaxwellConicalH(
            geometry,
            coefficients,
            function_space,
            source=source,
            source_domains=source_domains,
            reference="box",
            modal=modal,
            boundary_conditions=boundary_conditions,
            beta=beta,
        )

        super().__init__(geometry, formulation)

        self.degree = degree

    def solve_system(self, again=False):
        u = super().solve_system(again=again, vector_function=False)
        self.solution = {"diffracted": u, "total": u + self.source.expression}
        return u
