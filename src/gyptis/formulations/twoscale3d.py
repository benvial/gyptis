#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# License: MIT
# See the documentation at gyptis.gitlab.io


from .formulation import *


class TwoScale3D(Formulation):
    def __init__(
        self,
        geometry,
        coefficients,
        function_space,
        boundary_conditions={},
        degree=1,
        direction="x",
        case="epsilon",
    ):
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
        if self.direction == "x":
            e = Constant((1, 0, 0))
        elif self.direction == "y":
            e = Constant((0, 1, 0))
        else:
            e = Constant((0, 0, 1))

        form = []
        form.append(inner(coeff * grad(u), grad(v)))
        form.append(dot(-coeff * e, grad(v)))
        return (form[0] + form[1]) * self.dx(domain)

    def _weak(self, u, v):
        if self.case == "epsilon":
            coeff = self.epsilon
        else:
            coeff = self.mu

        coeff_sub = coeff.as_subdomain()
        coeff_dict = coeff.as_property()
        dom_func, dom_no_func = find_domains_function((coeff, coeff))
        form = self.poisson(u, v, coeff_sub, domain=dom_no_func)
        for dom in dom_func:
            form += self.poisson(u, v, coeff_dict[dom], domain=dom)

        weak = form.real + form.imag
        return weak

    @property
    def weak(self):
        u = self.trial
        v = self.test
        return self._weak(u, v)

    def build_pec_boundary_conditions(self, applied_function):
        applied_function = project_iterative(applied_function, self.real_function_space)
        _boundary_conditions = build_pec_boundary_conditions(
            self.pec_boundaries,
            self.geometry,
            self.function_space,
            applied_function,
        )
        return _boundary_conditions

    def build_boundary_conditions(self):
        applied_function = Constant(0)
        self._boundary_conditions = self.build_pec_boundary_conditions(applied_function)
        return self._boundary_conditions
