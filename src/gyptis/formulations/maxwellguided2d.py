#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.1.0
# License: MIT
# See the documentation at gyptis.gitlab.io


from .formulation import *

j = Complex(0, 1)
unit_vec_z = vector([0, 0, 1])


def curl_t(u):
    return (Dx(u[1], 0) - Dx(u[0], 1)) * unit_vec_z


def grad_t(u):
    du = grad(u)
    return vector([du[0], du[1], 0])


class MaxwellWaveguide(Formulation):
    def __init__(
        self,
        geometry,
        coefficients,
        function_space,
        source=None,
        boundary_conditions=None,
        source_domains=None,
        reference=None,
        wavenumber=0,
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
            modal=True,
            degree=degree,
        )
        self.wavenumber = wavenumber
        self.source_domains = source_domains
        self.reference = reference
        self.epsilon, self.mu = self.coefficients
        self.pec_boundaries = prepare_boundary_conditions(boundary_conditions)
        self._mu_inv = self.mu.invert().as_property(dim=3)
        self._epsilon = self.epsilon.as_property(dim=3)
        self._epsilon_dom = self.epsilon.as_subdomain()
        self._mu_inv_dom = self.mu.invert().as_subdomain()

    def maxwell(self, u, v, domain="everywhere"):
        k0 = Constant(self.wavenumber)
        et, ez = vector([u[0], u[1], 0]), u[2]
        vt, vz = vector([v[0], v[1], 0]), v[2]
        if domain == []:
            return [0, 0] if self.modal else 0

        _eps = self._epsilon[domain]
        _mu_inv = self._mu_inv[domain]

        # _eps = self._epsilon_dom
        # _mu_inv = self._mu_inv_dom

        _ez = vector([0, 0, u[2]])
        _vz = vector([0, 0, v[2]])
        a_tt = inner(_mu_inv * curl_t(u), curl_t(v)) - (k0**2) * inner(_eps * et, vt)
        b_tt = inner(_mu_inv * et, vt)
        b_tz = inner(_mu_inv * et, grad_t(vz))
        b_zt = inner(_mu_inv * grad_t(ez), vt)
        b_zz = inner(_mu_inv * grad_t(ez), grad_t(vz)) - (k0**2) * inner(
            _eps * _ez, _vz
        )

        # a_tt = inner(_mu_inv * curl_t(u), curl_t(v)) - (k0**2) * inner(_eps * et, vt)
        # b_tt = inner(_mu_inv * et, vt)
        # b_tz = inner(_mu_inv * et, grad_t(vz))
        # b_zt = inner(_mu_inv * grad_t(ez), vt)
        # b_zz = inner(_mu_inv * grad_t(ez), grad_t(vz)) - (k0**2) * inner(
        #     _eps[2][2] * ez, vz
        # )

        f0 = a_tt
        f1 = b_tt + b_tz + b_zt + b_zz

        form = [f0, f1]
        if self.modal:
            return [form[0] * self.dx(domain), -form[1] * self.dx(domain)]
        raise NotImplementedError

    def _weak(self, u, v, u1):

        formulation = 0
        for dom in self.geometry.domains.keys():
            # for dom in ["everywhere"]:
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
