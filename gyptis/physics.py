#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT


from .complex import *
from .core import PML
from .geometry import *
from .helpers import _get_form
from .materials import *
from .sources import *


class Scatt2D(object):
    def __init__(
        self,
        geom,
        epsilon,
        mu,
        polarization="TE",
        lambda0=1,
        theta0=0,
        degree=1,
        pml_stretch=1 - 1j,
        boundary_conditions=[],
    ):
        self.geom = geom  # geometry model
        self.degree = degree
        self.lambda0 = lambda0
        self.theta0 = theta0
        self.polarization = polarization
        self.epsilon = epsilon
        self.mu = mu
        self.pml_stretch = pml_stretch

        self.mesh = geom.mesh_object["mesh"]
        self.markers = geom.mesh_object["markers"]["triangle"]
        self.domains = geom.subdomains["surfaces"]
        self.dx = geom.measure["dx"]
        self.boundary_conditions = boundary_conditions

        self._prepare_materials()

        self.complex_space = ComplexFunctionSpace(self.mesh, "CG", self.degree)
        self.real_space = df.FunctionSpace(self.mesh, "CG", self.degree)

    @property
    def k0(self):
        return 2 * np.pi / self.lambda0

    def _make_subdomains(self, epsilon, mu):
        epsilon_coeff = Subdomain(
            self.markers,
            self.domains,
            epsilon,
            degree=self.degree,
        )
        mu_coeff = Subdomain(self.markers, self.domains, mu, degree=self.degree)
        return epsilon_coeff, mu_coeff

    def _prepare_materials(self):
        epsilon = dict(box=1)
        mu = dict(box=1)
        epsilon.update(self.epsilon)
        mu.update(self.mu)
        self.epsilon_pml, self.mu_pml = self._make_pmls()
        self.epsilon.update(self.epsilon_pml)
        self.mu.update(self.mu_pml)
        self.epsilon_coeff, self.mu_coeff = self._make_subdomains(self.epsilon, self.mu)
        _no_source = ["box", "pmlx", "pmly", "pmlxy"]
        self.source_dom = [z for z in self.mu.keys() if z not in _no_source]

        mu_annex = self.mu.copy()
        eps_annex = self.epsilon.copy()
        for a in self.source_dom:
            mu_annex[a] = self.mu["box"]
            eps_annex[a] = self.epsilon["box"]
        self.epsilon_coeff_annex, self.mu_coeff_annex = self._make_subdomains(
            eps_annex, mu_annex
        )
        self._make_coefs()

    def _coefs(self, a, b):
        # xsi = det Q^T/det Q
        extract = lambda q: df.as_tensor([[q[0][0], q[1][0]], [q[0][1], q[1][1]]])
        det = lambda M: M[0][0] * M[1][1] - M[1][0] * M[0][1]
        a2 = Complex(extract(a.real), extract(a.imag))
        xi = a2 / det(a2)
        chi = b[2][2]
        return xi, chi

    def _make_coefs(self):
        if self.polarization == "TE":
            self.xi, self.chi = self._coefs(self.mu_coeff, self.epsilon_coeff)
            self.xi_annex, self.chi_annex = self._coefs(
                self.mu_coeff_annex, self.epsilon_coeff_annex
            )
        else:
            self.xi, self.chi = self._coefs(self.epsilon_coeff, self.mu_coeff)
            self.xi_annex, self.chi_annex = self._coefs(
                self.epsilon_coeff_annex, self.mu_coeff_annex
            )

    def _make_pmls(self):
        pmlx = PML("x", stretch=self.pml_stretch)
        pmly = PML("y", stretch=self.pml_stretch)
        pmlxy = PML("xy", stretch=self.pml_stretch)
        t = [np.array(pml.transformation_matrix()) for pml in [pmlx, pmly, pmlxy]]

        eps_pml_ = [(self.epsilon["box"] * t_).tolist() for t_ in t]
        mu_pml_ = [(self.mu["box"] * t_).tolist() for t_ in t]
        epsilon_pml = dict(pmlx=eps_pml_[0], pmly=eps_pml_[1], pmlxy=eps_pml_[2])
        mu_pml = dict(pmlx=mu_pml_[0], pmly=mu_pml_[1], pmlxy=mu_pml_[2])
        return epsilon_pml, mu_pml

    def weak_form(self):
        W = self.complex_space
        dx = self.dx

        k0 = self.k0
        self.u0, self.gradu0 = plane_wave_2D(
            self.lambda0, self.theta0, domain=self.mesh, grad=True
        )
        self.u = Function(W)
        utrial = TrialFunction(W)
        utest = TestFunction(W)
        dxi = self.xi - self.xi_annex
        dchi = self.chi - self.chi_annex
        L = (
            -inner(self.xi * grad(utrial), grad(utest)) * dx,
            self.chi * utrial * utest * dx,
        )
        b = (
            -dot(dxi * self.gradu0, grad(utest)) * dx(self.source_dom),
            dchi * self.u0 * utest * dx(self.source_dom),
        )
        self.lhs = [t.real + t.imag for t in L]
        self.rhs = [t.real + t.imag for t in b]

    def assemble_lhs(self):
        self.Ah = [assemble(A) for A in self.lhs]

    def assemble_rhs(self):
        self.bh = [assemble(b) for b in self.rhs]

    def assemble(self):
        self.assemble_lhs()
        self.assemble_rhs()

    def solve(self):
        ufunc = self.u.real
        # ufunc = df.Function(self.real_space)
        Ah = self.Ah[0] + self.k0 ** 2 * self.Ah[1]
        bh = self.bh[0] + self.k0 ** 2 * self.bh[1]
        for bc in self.boundary_conditions:
            bc.apply(Ah, bh)
        Ah.form = _get_form(self.Ah)
        bh.form = _get_form(self.bh)
        solver = df.LUSolver("mumps")
        solver.solve(Ah, ufunc.vector(), bh)

    def wavelength_sweep(self, wavelengths):

        wl_sweep = []
        for i, w in enumerate(wavelengths):
            t = -time.time()
            self.lambda0 = w
            self.u0, self.gradu0 = plane_wave_2D(
                self.lambda0, self.theta0, domain=self.mesh, grad=True
            )
            self.weak_form()
            if i == 0:
                self.assemble()

                df.list_timings(df.TimingClear.clear, [df.TimingType.wall])
            else:
                self.assemble_rhs()
            self.solve()

            df.list_timings(df.TimingClear.clear, [df.TimingType.wall])
            wl_sweep.append(self.u)
            t += time.time()
            print(f"iter {t:0.2f}s")
        return wl_sweep
