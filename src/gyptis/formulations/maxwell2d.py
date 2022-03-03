#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# License: MIT
# See the documentation at gyptis.gitlab.io


from .formulation import *


class Maxwell2D(Formulation):
    def __init__(
        self,
        geometry,
        coefficients,
        function_space,
        source=None,
        boundary_conditions={},
        polarization="TM",
        source_domains=[],
        reference=None,
        modal=False,
        degree=1,
    ):
        super().__init__(
            geometry,
            coefficients,
            function_space,
            source=source,
            boundary_conditions=boundary_conditions,
            modal=modal,
            degree=degree,
        )

        self.source_domains = source_domains
        self.reference = reference

        self.epsilon, self.mu = self.coefficients
        self.polarization = polarization

        if self.polarization == "TM":
            self.xi = self.mu.to_xi()
            self.chi = self.epsilon.to_chi()
        else:
            self.xi = self.epsilon.to_xi()
            self.chi = self.mu.to_chi()

        self.pec_boundaries = prepare_boundary_conditions(boundary_conditions)

    def maxwell(self, u, v, xi, chi, domain="everywhere"):

        form = []
        form.append(-inner(xi * grad(u), grad(v)))
        form.append(chi * u * v)
        if self.modal:
            return [form[0] * self.dx(domain), -form[1] * self.dx(domain)]
        else:
            k0 = Constant(self.source.wavenumber)
            return (form[0] + k0 ** 2 * form[1]) * self.dx(domain)

    def _weak(self, u, v, u1):
        xi = self.xi.as_subdomain()
        chi = self.chi.as_subdomain()

        xi_dict = self.xi.as_property()
        chi_dict = self.chi.as_property()

        dom_func, dom_no_func = find_domains_function((self.xi, self.chi))
        source_dom_func, source_dom_no_func = find_domains_function(
            (self.xi, self.chi), self.source_domains
        )

        form = self.maxwell(u, v, xi, chi, domain=dom_no_func)
        for dom in dom_func:
            if self.modal:
                form_dom_func = self.maxwell(
                    u, v, xi_dict[dom], chi_dict[dom], domain=dom
                )
                form = [form[i] + form_dom_func[i] for i in range(2)]
            else:
                form += self.maxwell(u, v, xi_dict[dom], chi_dict[dom], domain=dom)

        if self.source_domains != []:
            xi_a = self.xi.build_annex(
                domains=self.source_domains, reference=self.reference
            ).as_subdomain()
            chi_a = self.chi.build_annex(
                domains=self.source_domains, reference=self.reference
            ).as_subdomain()
            xi_a_dict = self.xi.build_annex(
                domains=self.source_domains, reference=self.reference
            ).as_property()
            chi_a_dict = self.chi.build_annex(
                domains=self.source_domains, reference=self.reference
            ).as_property()

            if source_dom_no_func != []:
                form += self.maxwell(
                    u1, v, xi - xi_a, chi - chi_a, domain=source_dom_no_func
                )
            for dom in source_dom_func:
                form += self.maxwell(
                    u1,
                    v,
                    xi_dict[dom] - xi_a_dict[dom],
                    chi_dict[dom] - chi_a_dict[dom],
                    domain=dom,
                )
        if self.polarization == "TE":
            for bnd in self.pec_boundaries:
                normal = self.geometry.unit_normal_vector
                form -= dot(grad(u1), normal) * v * self.ds(bnd)

        if self.modal:
            weak = [f.real + f.imag for f in form]
        else:
            weak = form.real + form.imag
        return weak

    @property
    def weak(self):
        u1 = self.source.expression if not self.modal else 0
        u = self.trial
        v = self.test
        return self._weak(u, v, u1)

    def build_pec_boundary_conditions(self, applied_function):
        if self.polarization == "TM" and self.pec_boundaries != []:
            ## FIXME: project is slow, avoid it.
            applied_function = project_iterative(
                applied_function, self.real_function_space
            )
            _boundary_conditions = build_pec_boundary_conditions(
                self.pec_boundaries,
                self.geometry,
                self.function_space,
                applied_function,
            )
        else:
            _boundary_conditions = []

        return _boundary_conditions

    def build_boundary_conditions(self):
        applied_function = Constant(0) if self.modal else -self.source.expression
        self._boundary_conditions = self.build_pec_boundary_conditions(applied_function)
        return self._boundary_conditions

    def get_dual(self, field, pulsation=None):
        pulsation = pulsation or self.source.pulsation
        coeff = (
            1j * pulsation * mu_0
            if self.polarization == "TM"
            else -1j * pulsation * epsilon_0
        )
        grad_field = grad(field)
        re = as_vector([grad_field[1].real, -grad_field[0].real])
        im = as_vector([grad_field[1].imag, -grad_field[0].imag])
        return self.xi.as_subdomain() / Constant(coeff) * Complex(re, im)
