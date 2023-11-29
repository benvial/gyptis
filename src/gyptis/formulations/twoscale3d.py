#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.0.2
# License: MIT
# See the documentation at gyptis.gitlab.io


from .formulation import *


class TwoScale3D(Formulation):
    def __init__(
        self,
        geometry,
        coefficients,
        function_space,
        boundary_conditions=None,
        degree=1,
        direction="x",
        case="epsilon",
    ):
        if boundary_conditions is None:
            boundary_conditions = {}
        super().__init__(
            geometry,
            coefficients,
            function_space,
            boundary_conditions=boundary_conditions,
            degree=degree,
        )
        self.case = case
        self.epsilon, self.mu = self.coefficients
        self.direction = direction
        self.pec_boundaries = prepare_boundary_conditions(boundary_conditions)

    def poisson(self, u, v, coeff, domain="everywhere"):
        if domain == []:
            return 0
        if self.direction == "x":
            e = Constant((1, 0, 0))
        elif self.direction == "y":
            e = Constant((0, 1, 0))
        else:
            e = Constant((0, 0, 1))

        form = [inner(coeff * grad(u), grad(v))]
        form.append(dot(-coeff * e, grad(v)))
        return (form[0] + form[1]) * self.dx(domain)

    def _weak(self, u, v):
        coeff = self.epsilon if self.case == "epsilon" else self.mu
        coeff_sub = coeff.as_subdomain()
        coeff_dict = coeff.as_property()
        dom_func, dom_no_func = find_domains_function((coeff, coeff))
        form = self.poisson(u, v, coeff_sub, domain=dom_no_func)
        for dom in dom_func:
            form += self.poisson(u, v, coeff_dict[dom], domain=dom)

        return form.real + form.imag

    @property
    def weak(self):
        u = self.trial
        v = self.test
        return self._weak(u, v)

    def build_pec_boundary_conditions(self, applied_function):
        applied_function = project_iterative(applied_function, self.real_function_space)
        return build_pec_boundary_conditions(
            self.pec_boundaries,
            self.geometry,
            self.function_space,
            applied_function,
        )

    def build_boundary_conditions(self):
        applied_function = Constant(0)
        self._boundary_conditions = self.build_pec_boundary_conditions(applied_function)
        return self._boundary_conditions
