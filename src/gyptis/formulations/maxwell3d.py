#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.0.2
# License: MIT
# See the documentation at gyptis.gitlab.io


from .formulation import *


class Maxwell3D(Formulation):
    def __init__(
        self,
        geometry,
        coefficients,
        function_space,
        source=None,
        boundary_conditions=None,
        source_domains=None,
        reference=None,
        modal=False,
        degree=1,
    ):
        if boundary_conditions is None:
            boundary_conditions = {}
        if source_domains is None:
            source_domains = []
        super().__init__(
            geometry,
            coefficients,
            function_space,
            source=source,
            boundary_conditions=boundary_conditions,
            modal=modal,
            degree=degree,
            dim=3,
        )

        self.source_domains = source_domains
        self.reference = reference
        self.epsilon, self.mu = self.coefficients
        self.pec_boundaries = prepare_boundary_conditions(boundary_conditions)

    def maxwell(self, u, v, epsilon, inv_mu, domain="everywhere"):
        if domain == []:
            return [0, 0] if self.modal else 0
        form = [-inner(inv_mu * curl(u), curl(v))]
        form.append(inner(epsilon * u, v))
        if self.modal:
            return [form[0] * self.dx(domain), -form[1] * self.dx(domain)]
        k0 = Constant(self.source.wavenumber)
        return (form[0] + k0**2 * form[1]) * self.dx(domain)

    def _weak(self, u, v, u1):
        epsilon = self.epsilon.as_subdomain()
        inv_mu = self.mu.invert().as_subdomain()

        epsilon_dict = self.epsilon.as_property()
        inv_mu_dict = self.mu.invert().as_property()

        dom_func, dom_no_func = find_domains_function((self.epsilon, self.mu))
        source_dom_func, source_dom_no_func = find_domains_function(
            (self.epsilon, self.mu), self.source_domains
        )
        form = self.maxwell(u, v, epsilon, inv_mu, domain=dom_no_func)

        for dom in dom_func:
            if self.modal:
                form_dom_func = self.maxwell(
                    u, v, xi_dict[dom], chi_dict[dom], domain=dom
                )
                form = [form[i] + form_dom_func[i] for i in range(2)]
            else:
                form += self.maxwell(
                    u, v, epsilon_dict[dom], inv_mu_dict[dom], domain=dom
                )

        if self.source_domains != []:
            epsilon_a = self.epsilon.build_annex(
                domains=self.source_domains, reference=self.reference
            ).as_subdomain()
            inv_mu_a = (
                self.mu.invert()
                .build_annex(domains=self.source_domains, reference=self.reference)
                .as_subdomain()
            )
            epsilon_a_dict = self.epsilon.build_annex(
                domains=self.source_domains, reference=self.reference
            ).as_property()
            inv_mu_a_dict = (
                self.mu.invert()
                .build_annex(domains=self.source_domains, reference=self.reference)
                .as_property()
            )
            if source_dom_no_func != []:
                form += self.maxwell(
                    u1,
                    v,
                    epsilon - epsilon_a,
                    inv_mu - inv_mu_a,
                    domain=source_dom_no_func,
                )
            for dom in source_dom_func:
                form += self.maxwell(
                    u1,
                    v,
                    epsilon_dict[dom] - epsilon_a_dict[dom],
                    inv_mu_dict[dom] - inv_mu_a_dict[dom],
                    domain=dom,
                )
        return [f.real + f.imag for f in form] if self.modal else form.real + form.imag

    def get_magnetic_field(self, E):
        omega = 1 if self.modal else self.source.pulsation
        j = Complex(0, 1)
        mu_inv = self.mu.invert().as_subdomain()
        return -mu_inv / (j * omega * mu_0) * curl(E)

    @property
    def weak(self):
        u1 = self.source.expression
        u = self.trial
        v = self.test
        return self._weak(u, v, u1)

    def build_pec_boundary_conditions(self, applied_function):
        if self.pec_boundaries != []:
            # FIXME: project is slow, avoid it.
            applied_function = project_iterative(
                applied_function, self.real_function_space
            )
            return build_pec_boundary_conditions(
                self.pec_boundaries,
                self.geometry,
                self.function_space,
                applied_function,
            )
        else:
            return []

    def build_boundary_conditions(self):
        applied_function = (
            Constant((0, 0, 0)) if self.modal else -self.source.expression
        )
        self._boundary_conditions = self.build_pec_boundary_conditions(applied_function)
        return self._boundary_conditions
