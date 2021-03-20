#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

from abc import ABC, abstractmethod

import numpy as np
from scipy.constants import c, epsilon_0, mu_0

from . import ADJOINT, dolfin
from .complex import *
from .geometry import *
from .materials import *
from .materials import _coefs, _invert_3by3_complex_matrix, _make_cst_mat
from .sources import *


class Simulation(ABC):
    @abstractmethod
    def prepare(self):
        pass

    @abstractmethod
    def weak_form(self):
        pass

    @abstractmethod
    def assemble(self):
        pass

    @abstractmethod
    def build_system(self):
        pass

    @abstractmethod
    def solve_system(self):
        pass

    def solve(self, direct=True):
        self.prepare()
        self.weak_form()
        self.assemble()
        self.build_system()
        self.solve_system(direct=direct)


#
# lu_params = {"report": True, "symmetric": False, "verbose": True}
#
#
# krylov_params = {
#     "absolute_tolerance": 1.0e-1,
#     "divergence_limit": 1000.0,
#     "error_on_nonconvergence": True,
#     "maximum_iterations": 500,
#     "monitor_convergence": True,
#     "nonzero_initial_guess": False,
#     "relative_tolerance": 1.0e-1,
#     "report": True,
# }


class Simulation2D(Simulation):
    """Base class for 2D simulations"""

    def __init__(
        self,
        geometry,
        degree=1,
        element="CG",
        boundary_conditions={},
    ):
        self.geometry = geometry
        self.dim = geometry.dim
        self.degree = degree
        self.element = element
        self.mesh_object = geometry.mesh_object
        self.mesh = geometry.mesh_object["mesh"]
        self.domains = geometry.subdomains["surfaces"]
        self.boundaries = geometry.subdomains["curves"]
        self.points = geometry.subdomains["points"]
        self.markers = geometry.mesh_object["markers"]["triangle"]
        self.boundary_markers = (
            geometry.mesh_object["markers"]["line"] if self.boundaries else []
        )

        self.dx = geometry.measure["dx"]
        self.ds = geometry.measure["ds"] if self.boundaries else None
        self.dS = geometry.measure["dS"] if self.boundaries else None
        self.boundary_conditions = boundary_conditions
        self.unit_normal_vector = dolfin.FacetNormal(self.mesh)
        self.source_domains = []


class ElectroMagneticSimulation2D(Simulation2D):
    """Base class for 2D electromagnetic simulations"""

    def __init__(
        self, geometry, epsilon, mu, lambda0=1, theta0=0, polarization="TE", **kwargs
    ):
        super().__init__(geometry, **kwargs)
        self.lambda0 = lambda0
        self.theta0 = theta0
        self.polarization = polarization
        self.epsilon = epsilon
        self.mu = mu

    @property
    def k0(self):
        return 2 * np.pi / self.lambda0

    @property
    def omega(self):
        return self.k0 * c

    def _make_subdomains(self, epsilon, mu):
        epsilon_coeff = Subdomain(
            self.markers, self.domains, epsilon, degree=self.degree
        )
        mu_coeff = Subdomain(self.markers, self.domains, mu, degree=self.degree)
        return epsilon_coeff, mu_coeff

    def _prepare_materials(self, ref_material, pmls=False):
        if pmls:
            self.epsilon_pml, self.mu_pml = self._make_pmls()
            self.epsilon.update(self.epsilon_pml)
            self.mu.update(self.mu_pml)
        self.epsilon_coeff, self.mu_coeff = self._make_subdomains(self.epsilon, self.mu)

        self.epsilon_annex, self.mu_annex = make_annex_materials(
            self.epsilon, self.mu, self.source_domains, ref_material
        )
        self.epsilon_coeff_annex, self.mu_coeff_annex = self._make_subdomains(
            self.epsilon_annex,
            self.mu_annex,
        )
        if self.polarization == "TE":
            self.xi, self.chi = _make_cst_mat(self.mu, self.epsilon)
            self.xi_annex, self.chi_annex = _make_cst_mat(
                self.mu_annex, self.epsilon_annex
            )
            self.xi_coeff, self.chi_coeff = _coefs(self.mu_coeff, self.epsilon_coeff)
            self.xi_coeff_annex, self.chi_coeff_annex = _coefs(
                self.mu_coeff_annex, self.epsilon_coeff_annex
            )
        else:
            self.xi, self.chi = _make_cst_mat(self.epsilon, self.mu)
            self.xi_annex, self.chi_annex = _make_cst_mat(
                self.epsilon_annex, self.mu_annex
            )
            self.xi_coeff, self.chi_coeff = _coefs(self.epsilon_coeff, self.mu_coeff)
            self.xi_coeff_annex, self.chi_coeff_annex = _coefs(
                self.epsilon_coeff_annex, self.mu_coeff_annex
            )


def build_lhs(utrial, utest, xi, chi, domains, unit_vect=None):
    lhs = {}
    for d in domains:
        L = []
        L.append(-dot(xi[d] * grad(utrial), grad(utest)))
        L.append(chi[d] * utrial * utest)
        if unit_vect:
            L.append(
                1j
                * (
                    dot(unit_vect, xi[d] * grad(utrial) * utest)
                    - dot(unit_vect, xi[d] * grad(utest) * utrial)
                )
            )
            L.append(-dot(xi[d] * unit_vect, unit_vect) * utrial * utest)
        lhs[d] = [t.real + t.imag for t in L]
    return lhs


def build_lhs_boundaries(
    utrial, utest, xi_coeff, boundaries, normal_vector, unit_vect=None
):
    lhs = {}
    for d in boundaries:
        L = []
        L.append(-1j * dot(xi_coeff * unit_vect, normal_vector) * utrial * utest)
        lhs[d] = [t.real + t.imag for t in L]
    return lhs


def build_rhs(
    usource,
    utest,
    xi,
    chi,
    xi_annex,
    chi_annex,
    domains,
    unit_vect=None,
    phasor=Complex(1, 0),
):
    rhs = {}
    for d in domains:
        if isinstance(usource, dict):
            usource_dom = usource[d]
        else:
            usource_dom = usource
        if xi[d].real.ufl_shape == (2, 2):
            xi_a = tensor_const(np.eye(2) * xi_annex[d], dim=2)
        else:
            xi_a = xi_annex[d]
        dxi = xi[d] - xi_a
        dchi = chi[d] - chi_annex[d]

        b = []
        b.append(-dot(dxi * grad(usource_dom), grad(utest)) * phasor.conj)
        b.append(dchi * usource_dom * utest * phasor.conj)
        if unit_vect:
            b.append(1j * dot(dxi * grad(usource_dom), unit_vect) * utest * phasor.conj)
        rhs[d] = [t.real + t.imag for t in b]
    return rhs


def build_rhs_boundaries(
    usource, utest, xi_coeff_annex, domains, normal_vector, phasor=Complex(1, 0)
):

    rhs = {}
    for d in domains:
        b = []
        # surface term for PEC in TM polarization
        b.append(
            dot(xi_coeff_annex * grad(usource), normal_vector) * utest * phasor.conj,
        )
        rhs[d] = [t.real + t.imag for t in b]
    return rhs


def make_system_matrix(domains, pec_bnds, Ahform, k0, alpha=0, boundary=False):
    for i, d in enumerate(domains):
        Ah_ = Ahform[d][0] + k0 ** 2 * Ahform[d][1]
        if ADJOINT:
            form_ = Ahform[d][0].form + k0 ** 2 * Ahform[d][1].form
        if i == 0:
            Ah = Ah_
            if ADJOINT:
                form = form_
        else:
            Ah += Ah_
            if ADJOINT:
                form += form_
        if alpha != 0:
            Ah += alpha * Ahform[d][2] + alpha ** 2 * Ahform[d][3]
            if ADJOINT:
                form += alpha * Ahform[d][2].form + alpha ** 2 * Ahform[d][3].form
    if boundary and alpha != 0:
        for d in pec_bnds:
            Ah += alpha * Ahform[d][0]
            if ADJOINT:
                form += alpha * Ahform[d][0].form
    if ADJOINT:
        Ah.form = form
    return Ah


def make_system_vector(source_domains, pec_bnds, bhform, k0, alpha=0, boundary=False):
    for i, d in enumerate(source_domains):
        bh_ = -bhform[d][0] - k0 ** 2 * bhform[d][1]
        if ADJOINT:
            form_ = -bhform[d][0].form - k0 ** 2 * bhform[d][1].form
        if i == 0:
            bh = bh_
            if ADJOINT:
                form = form_
        else:
            bh += bh_
            if ADJOINT:
                form += form_

        if alpha != 0:
            bh -= alpha * bhform[d][2]
            if ADJOINT:
                form -= alpha * bhform[d][2].form

    if boundary:
        for d in pec_bnds:
            bh += bhform[d][0]
            if ADJOINT:
                form += bhform[d][0].form
    if ADJOINT:
        bh.form = form
    return bh


def make_annex_materials(epsilon, mu, source_domains, reference_domain):
    mu_annex = mu.copy()
    eps_annex = epsilon.copy()
    for a in source_domains:
        eps_annex[a] = epsilon[reference_domain]
        mu_annex[a] = mu[reference_domain]
    return eps_annex, mu_annex


# solver.set_operator(Ah)
# # Create vector that spans the null space and normalize
# null_vec = dolfin.Vector(Efunc.vector())
# self.complex_space.dofmap().set(null_vec, 1.0)
# null_vec *= 1.0/null_vec.norm("l2")
#
# # Create null space basis object and attach to PETSc matrix
# null_space = dolfin.VectorSpaceBasis([null_vec])
# dolfin.as_backend_type(Ah).set_nullspace(null_space)
# null_space.orthogonalize(bh)
# solver.solve(Efunc.vector(), bh)


class Simulation3D(Simulation):
    """Base class for 3D simulations"""

    def __init__(
        self,
        geometry,
        degree=1,
        element="N1curl",
        boundary_conditions={},
    ):
        self.geometry = geometry
        self.dim = geometry.dim
        self.degree = degree
        self.element = element
        self.mesh_object = geometry.mesh_object
        self.mesh = geometry.mesh_object["mesh"]
        self.domains = geometry.subdomains["volumes"]
        self.boundaries = geometry.subdomains["surfaces"]
        self.lines = geometry.subdomains["curves"]
        self.points = geometry.subdomains["points"]
        self.markers = geometry.mesh_object["markers"]["tetra"]
        self.boundary_markers = (
            geometry.mesh_object["markers"]["triangle"] if self.boundaries else []
        )

        self.dx = geometry.measure["dx"]
        self.ds = geometry.measure["ds"] if self.boundaries else None
        self.dS = geometry.measure["dS"] if self.boundaries else None
        self.boundary_conditions = boundary_conditions
        self.unit_normal_vector = dolfin.FacetNormal(self.mesh)
        self.source_domains = []

        self.unit_vectors = (
            as_vector([1.0, 0.0, 0.0]),
            as_vector([0.0, 1.0, 0.0]),
            as_vector([0.0, 0.0, 1.0]),
        )


class ElectroMagneticSimulation3D(Simulation3D):
    """Base class for 3D electromagnetic simulations"""

    def __init__(
        self, geometry, epsilon, mu, lambda0=1, theta0=0, phi0=0, psi0=0, **kwargs
    ):
        super().__init__(geometry, **kwargs)
        self.lambda0 = lambda0
        self.theta0 = theta0
        self.phi0 = phi0
        self.psi0 = psi0
        self.epsilon = epsilon
        self.mu = mu

    @property
    def k0(self):
        return 2 * np.pi / self.lambda0

    @property
    def omega(self):
        return self.k0 * c

    def _make_subdomains(self, epsilon, mu):
        epsilon_coeff = Subdomain(
            self.markers, self.domains, epsilon, degree=self.mat_degree
        )
        mu_coeff = Subdomain(self.markers, self.domains, mu, degree=self.mat_degree)
        return epsilon_coeff, mu_coeff

    def _prepare_bcs(self):
        self._boundary_conditions = []
        self.pec_bnds = []
        for bnd, cond in self.boundary_conditions.items():
            if cond != "PEC":
                raise ValueError(f"unknown boundary condition {cond}")
            else:
                self.pec_bnds.append(bnd)
        for bnd in self.pec_bnds:
            Ebnd = -self.Estack_coeff * self.phasor.conj
            # ubnd = dolfin.as_vector((ubnd_vec.real, ubnd_vec.imag))
            Ebnd_proj = project(Ebnd, self.real_space)
            bc = DirichletBC(
                self.complex_space,
                Ebnd_proj,
                self.boundary_markers,
                bnd,
                self.boundaries,
            )
            [self._boundary_conditions.append(b) for b in bc]

    def _prepare_materials(self, ref_material, pmls=False):
        if pmls:
            self.epsilon_pml, self.mu_pml = self._make_pmls()
            self.epsilon.update(self.epsilon_pml)
            self.mu.update(self.mu_pml)
        self.epsilon_coeff, self.mu_coeff = self._make_subdomains(self.epsilon, self.mu)

        self._epsilon_annex, self._mu_annex = make_annex_materials(
            self.epsilon, self.mu, self.source_domains, ref_material
        )
        self.epsilon_coeff_annex, self.mu_coeff_annex = self._make_subdomains(
            self._epsilon_annex,
            self._mu_annex,
        )

        self.inv_mu_coeff = _invert_3by3_complex_matrix(self.mu_coeff)
        self.inv_mu_coeff_annex = _invert_3by3_complex_matrix(self.mu_coeff_annex)

        self.inv_mu = make_constant_property(self.mu, inv=True)
        self.eps = make_constant_property(self.epsilon)

        self.inv_mu_annex = make_constant_property(self._mu_annex, inv=True)
        self.eps_annex = make_constant_property(self._epsilon_annex)
