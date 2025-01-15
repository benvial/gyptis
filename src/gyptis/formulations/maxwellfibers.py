#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.0.2
# License: MIT
# See the documentation at gyptis.gitlab.io


from .formulation import *

j = Complex(0, 1)
ez = vector([0, 0, 1])


def curl_t(u):
    return (Dx(u[1], 0) - Dx(u[0], 1)) * ez


def div_t(u):
    return Dx(u[0], 0) + Dx(u[1], 1)


def grad_t(u):
    du = grad(u[2])
    return vector([du[0], du[1], 0])


class MaxwellConical(Formulation):
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
        beta=0,
        degree=1,
        type="E",
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
        )
        self.beta = beta
        self.source_domains = source_domains
        self.reference = reference
        self.epsilon, self.mu = self.coefficients
        self.pec_boundaries = prepare_boundary_conditions(boundary_conditions)
        self.type = type

        if self.type == "H":
            _A, _B = self.epsilon, self.mu
        else:
            _A, _B = self.mu, self.epsilon
        self._A = _A.invert().as_property(dim=3)
        self._B = _B.as_property(dim=3)
        self._Asub = _A.invert().as_subdomain()
        self._Bsub = _B.as_subdomain()

    @property
    def phasor_z(self, z):
        return phase_shift_constant(self.beta * Constant(z))

    def curl_beta(self, u):
        Ht = vector([u[0], u[1], 0])
        return curl_t(u) + cross((grad_t(u) - j * Constant(self.beta) * Ht), ez)

    def div_beta(self, u):
        Ht = vector([u[0], u[1], 0])
        return div_t(u) + dot((grad_t(u) - j * Constant(self.beta) * Ht), ez)

    def maxwell(self, u, v, domain="everywhere"):
        beta = Constant(self.beta)
        if domain == []:
            return [0, 0] if self.modal else 0

        f0 = inner(self._A[domain] * self.curl_beta(u), self.curl_beta(v).conj)
        f1 = -inner(self._B[domain] * u, v.conj)
        form = [f0, f1]
        if self.modal:
            return [form[0] * self.dx(domain), -form[1] * self.dx(domain)]
        k0 = Constant(self.source.wavenumber)
        return (form[0] + k0**2 * form[1]) * self.dx(domain)

    def _weak(self, u, v, u1):

        formulation = 0
        for dom in self.geometry.domains.keys():
            form = self.maxwell(u, v, domain=dom)
            if formulation == 0:
                formulation = form
            else:
                if self.modal:
                    for i in range(2):
                        formulation[i] += form[i]
                else:
                    formulation += form

        return (
            [f.real + f.imag for f in formulation]
            if self.modal
            else formulation.real + formulation.imag
        )

    # def _weak(self, u, v, u1):
    #     form = self.maxwell(u, v)
    #     return [f.real + f.imag for f in form] if self.modal else form.real + form.imag

    def get_electric_field(self, H, omega=1):
        omega = omega if self.modal else self.source.pulsation
        j = Complex(0, 1)
        eps_inv = self.epsilon.invert().as_subdomain()
        return eps_inv / (j * omega * epsilon_0) * curl(H)

    @property
    def weak(self):
        u1 = Constant((0, 0, 0)) if self.modal else self.source.expression
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


class MaxwellConicalBands(MaxwellConical):

    def __init__(self, *args, propagation_vector=(0, 0), modal=True, **kwargs):
        super().__init__(*args, **kwargs, modal=modal)
        self.propagation_vector = propagation_vector

    @property
    def phasor(self):
        _phasor = phasor(
            self.propagation_vector[0],
            direction=0,
            degree=self.degree,
            domain=self.geometry.mesh,
        )
        _phasor *= phasor(
            self.propagation_vector[1],
            direction=1,
            degree=self.degree,
            domain=self.geometry.mesh,
        )
        return _phasor

    def curl_beta_k(self, u):
        return self.curl_beta(u * self.phasor) * self.phasor.conj

    def maxwell(self, u, v, domain="everywhere"):
        if domain == []:
            return [0, 0] if self.modal else 0
        f0 = inner(self._A[domain] * self.curl_beta_k(u), self.curl_beta_k(v).conj)
        f1 = -inner(self._B[domain] * u, v.conj)
        form = [f0, f1]
        if self.modal:
            return [form[0] * self.dx(domain), -form[1] * self.dx(domain)]
        k0 = Constant(self.source.wavenumber)
        return (form[0] + k0**2 * form[1]) * self.dx(domain)

    # def maxwell(self, u, v, domain="everywhere"):
    #     if domain == []:
    #         return [0, 0] if self.modal else 0
    #     f0 = inner(self._Asub * self.curl_beta_k(u), self.curl_beta_k(v).conj)
    #     f1 = -inner(self._Bsub * u, v.conj)
    #     form = [f0, f1]
    #     if self.modal:
    #         return [form[0] * self.dx(domain), -form[1] * self.dx(domain)]
    #     k0 = Constant(self.source.wavenumber)
    #     return (form[0] + k0**2 * form[1]) * self.dx(domain)
