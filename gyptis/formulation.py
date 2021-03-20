#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

"""
Finite element weak formulations.
"""

from .complex import Constant, TestFunction, TrialFunction, dot, grad


class Formulation:
    def __init__(self, name, function_space, geometry):
        self.name = name
        self.function_space = function_space
        self.geometry = geometry
        self.trial = TrialFunction(self.function_space)
        self.test = TestFunction(self.function_space)
        self.lhs = dict(all={}, boundaries={})
        self.rhs = dict(all={}, boundaries={})


class Maxwell2D(Formulation):
    def __init__(
        self,
        function_space,
        domains,
        coefficients={},
        parameters={},
        polarization="TE",
    ):
        super().__init__("Maxwell 2D", function_space, domains)
        self.polarization = polarization
        self.xi = coefficients["xi"]
        self.chi = coefficients["chi"]
        self.source = coefficients["source"]
        self.wavenumber = parameters["wavenumber"]
        # self.build_lhs()
        # self.build_rhs()

    def lhs_expression(self, domain):
        return (
            -dot(self.xi[domain] * grad(self.trial), grad(self.test))
            + Constant(self.wavenumber ** 2) * self.chi[domain] * self.trial * self.test
        )

    def rhs_expression(self, domain):
        return (
            -dot(self.xi[domain] * grad(self.trial), grad(self.test))
            + Constant(self.wavenumber ** 2) * self.chi[domain] * self.trial * self.test
        )

    def build_lhs(self):
        for domain in self.domains["all"]:
            self.lhs["all"][domain] = self.lhs_expression(domain)

    def build_rhs(self):
        for domain in self.domains["sources"]:
            self.rhs["sources"][domain] = self.rhs_expression(domain)


class WeakFormBuilder:
    def __init__(self, formulation, measure):
        self.formulation = formulation
        self.measure = measure
        self.lhs = 0

    def build_lhs(self):
        dx = self.measure["dx"]
        for domain, term in self.formulation.lhs["all"].items():
            self.lhs += (term.real + term.imag) * dx(domain)
