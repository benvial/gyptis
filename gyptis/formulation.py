#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

"""
Finite element weak formulations.
"""

from abc import ABC, abstractmethod

import numpy as np
from scipy.constants import epsilon_0, mu_0

from . import dolfin
from .bc import *
from .complex import *
from .source import PlaneWave
from .stack import make_stack


def _project_bc_function(applied_function, function_space):
    ## FIXME: project is slow, avoid it.
    return project(
        applied_function,
        function_space,
        solver_type="cg",
        preconditioner_type="jacobi",
    )


class Formulation(ABC):
    def __init__(
        self,
        geometry,
        coefficients,
        function_space,
        source=None,
        boundary_conditions={},
        modal=False,
    ):
        self.geometry = geometry
        self.coefficients = coefficients
        self.function_space = function_space
        self.source = source
        self.trial = TrialFunction(self.function_space)
        self.test = TestFunction(self.function_space)
        self.boundary_conditions = boundary_conditions
        self.modal = modal

        self.measure = geometry.measure
        self.dx = self.measure["dx"]
        self.ds = self.measure["ds"]
        self.dS = self.measure["dS"]

        self.element = self.function_space.split()[0].ufl_element()
        self.real_function_space = dolfin.FunctionSpace(
            self.geometry.mesh, self.element
        )

    def build_lhs(self):
        self.lhs = dolfin.lhs(self.weak)
        return self.lhs

    def build_rhs(self):
        self.rhs = dolfin.rhs(self.weak)
        if self.rhs.empty():
            if self.element.value_size() == 3:
                dummy_vect = as_vector(
                    [dolfin.DOLFIN_EPS, dolfin.DOLFIN_EPS, dolfin.DOLFIN_EPS]
                )
                dummy_form = dot(dummy_vect, self.trial) * self.dx
            else:
                dummy_form = dolfin.DOLFIN_EPS * self.trial * self.dx
            self.rhs = dummy_form.real + dummy_form.imag
        return self.rhs

    @abstractmethod
    def weak(self):
        pass

    # @abstractmethod
    # def build_lhs(self):
    #     pass
    #
    # @abstractmethod
    # def build_rhs(self):
    #     pass

    @abstractmethod
    def build_boundary_conditions(self):
        pass


def _is_dolfin_function(f):
    if iscomplex(f):
        out = hasattr(f.real, "ufl_shape") or hasattr(f.imag, "ufl_shape")
    else:
        out = hasattr(f, "ufl_shape")
    return out


def _find_domains_function(coeffs, list_domains=None):
    dom_function = []
    for coeff in coeffs:
        list_domains = list_domains or list(coeff.dict.keys())

        dom_function += [k for k, v in coeff.dict.items() if _is_dolfin_function(v)]
    dom_function = np.unique(dom_function).tolist()
    dom_no_function = [k for k in list_domains if k not in dom_function]
    return dom_function, dom_no_function


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
    ):
        super().__init__(
            geometry,
            coefficients,
            function_space,
            source=source,
            boundary_conditions=boundary_conditions,
            modal=modal,
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

        dom_func, dom_no_func = _find_domains_function((self.xi, self.chi))
        source_dom_func, source_dom_no_func = _find_domains_function(
            (self.xi, self.chi), self.source_domains
        )

        form = self.maxwell(u, v, xi, chi, domain=dom_no_func)
        for dom in dom_func:
            if self.modal:
                form_dom_func = self.maxwell(
                    u, v, xi_dict[dom], chi_dict[dom], domain=dom
                )
                form = [form[i] + form_extra[i] for i in range(2)]
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
        u1 = self.source.expression if self.source is not None else 0
        u = self.trial
        v = self.test
        return self._weak(u, v, u1)

    def build_pec_boundary_conditions(self, applied_function):
        if self.polarization == "TM" and self.pec_boundaries != []:

            applied_function = _project_bc_function(
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
        return self.xi.as_subdomain() / Constant(coeff) * grad(field)


class Maxwell2DBands(Maxwell2D):
    def __init__(self, *args, propagation_vector=(0, 0), degree=1, **kwargs):
        super().__init__(*args, **kwargs, modal=True)
        self.propagation_vector = propagation_vector
        self.degree = degree

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

    @property
    def weak(self):
        u = self.trial * self.phasor
        v = self.test * self.phasor.conj
        return super()._weak(u, v, Constant(0))


class Maxwell2DPeriodic(Maxwell2D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.propagation_vector = self.source.wavenumber * np.array(
            [np.cos(self.source.angle), np.sin(self.source.angle)]
        )
        self.phasor = phasor(
            self.propagation_vector[0],
            direction=0,
            degree=self.source.degree,
            domain=self.geometry.mesh,
        )
        self.annex_field = make_stack(
            self.geometry,
            self.coefficients,
            self.source,
            polarization=self.polarization,
            source_domains=self.source_domains,
            degree=self.source.degree,
            dim=2,
        )

    @property
    def weak(self):
        u1 = self.annex_field["as_subdomain"]["stack"]
        u = self.trial * self.phasor
        v = self.test * self.phasor.conj
        return super()._weak(u, v, u1)

    def build_boundary_conditions(self):
        applied_function = -self.annex_field["as_subdomain"]["stack"] * self.phasor.conj
        self._boundary_conditions = self.build_pec_boundary_conditions(applied_function)
        return self._boundary_conditions


class Maxwell3D(Formulation):
    def __init__(
        self,
        geometry,
        coefficients,
        function_space,
        source=None,
        boundary_conditions={},
        source_domains=[],
        reference=None,
    ):
        super().__init__(
            geometry,
            coefficients,
            function_space,
            source=source,
            boundary_conditions=boundary_conditions,
        )

        self.source_domains = source_domains
        self.reference = reference
        self.epsilon, self.mu = self.coefficients
        self.pec_boundaries = prepare_boundary_conditions(boundary_conditions)

    def maxwell(self, u, v, epsilon, inv_mu, domain="everywhere"):
        k0 = Constant(self.source.wavenumber)
        form = -inner(inv_mu * curl(u), curl(v)) + k0 ** 2 * inner(epsilon * u, v)
        return form * self.dx(domain)

    def _weak(self, u, v, u1):
        epsilon = self.epsilon.as_subdomain()
        inv_mu = self.mu.invert().as_subdomain()

        epsilon_a = self.epsilon.build_annex(
            domains=self.source_domains, reference=self.reference
        ).as_subdomain()
        inv_mu_a = (
            self.mu.invert()
            .build_annex(domains=self.source_domains, reference=self.reference)
            .as_subdomain()
        )
        form = self.maxwell(u, v, epsilon, inv_mu)
        if self.source_domains != []:
            form += self.maxwell(
                u1,
                v,
                epsilon - epsilon_a,
                inv_mu - inv_mu_a,
                domain=self.source_domains,
            )

        # for bnd in self.pec_boundaries:
        #     normal = self.geometry.unit_normal_vector
        #     form -= dot(xi * (grad(u1) * v), normal) * self.ds(bnd)
        weak = form.real + form.imag
        return weak

    @property
    def weak(self):
        u1 = self.source.expression
        u = self.trial
        v = self.test
        return self._weak(u, v, u1)

    def build_pec_boundary_conditions(self, applied_function):
        if self.pec_boundaries != []:
            applied_function = _project_bc_function(
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
        applied_function = -self.source.expression
        self._boundary_conditions = self.build_pec_boundary_conditions(applied_function)
        return self._boundary_conditions


class Maxwell3DPeriodic(Maxwell3D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        k0 = self.source.wavenumber
        theta0, phi0 = self.source.angle[0:2]
        alpha0 = k0 * np.cos(theta0) * np.cos(phi0)
        beta0 = k0 * np.cos(theta0) * np.sin(phi0)
        gamma0 = k0 * np.sin(theta0)
        self.propagation_vector = np.array([alpha0, beta0, gamma0])

        self.phasor_vect = [
            phasor(
                self.propagation_vector[i],
                direction=i,
                degree=self.source.degree,
                domain=self.geometry.mesh,
            )
            for i in range(3)
        ]
        self.phasor = self.phasor_vect[0] * self.phasor_vect[1]
        self.annex_field = make_stack(
            self.geometry,
            self.coefficients,
            self.source,
            source_domains=self.source_domains,
            degree=self.source.degree,
            dim=3,
        )

    @property
    def weak(self):
        u1 = self.annex_field["as_subdomain"]["stack"]
        u = self.trial * self.phasor
        v = self.test * self.phasor.conj
        return self._weak(u, v, u1)

    def build_boundary_conditions(self):
        applied_function = -self.annex_field["as_subdomain"]["stack"] * self.phasor.conj
        self._boundary_conditions = self.build_pec_boundary_conditions(applied_function)
        return self._boundary_conditions
