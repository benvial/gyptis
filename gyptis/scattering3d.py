#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT


from . import ADJOINT, dolfin
from .base import ElectroMagneticSimulation3D
from .bc import DirichletBC
from .complex import *
from .geometry import *
from .materials import *
from .source import *


class Scatt3D(ElectroMagneticSimulation3D):
    def __init__(
        self,
        geometry,
        epsilon,
        mu,
        pml_stretch=1 - 1j,
        mat_degree=None,
        **kwargs,
    ):
        super().__init__(geometry, epsilon, mu, **kwargs)

        self.mat_degree = mat_degree or self.degree
        self.epsilon = epsilon
        self.mu = mu
        self.pml_stretch = pml_stretch

        self.complex_space = ComplexFunctionSpace(self.mesh, self.element, self.degree)
        self.real_space = dolfin.FunctionSpace(self.mesh, self.element, self.degree)

        self.no_source_domains = [
            "box",
            "pmlx",
            "pmly",
            "pmlz",
            "pmlxy",
            "pmlxz",
            "pmlyz",
            "pmlxyz",
        ]
        self.source_domains = [
            z for z in self.epsilon.keys() if z not in self.no_source_domains
        ]

    def prepare(self):
        self._prepare_materials(ref_material="box", pmls=True)
        self.incident_field = plane_wave_3d(
            self.lambda0, self.theta0, self.phi0, self.psi0, domain=self.mesh
        )
        e0 = {k: np.zeros(3) for k in self.epsilon.keys()}
        e0 = {"box": list(self.incident_field)}
        for dom in self.source_domains:
            e0[dom] = e0["box"]
        self.E0_coeff = Subdomain(
            self.markers, self.domains, e0, degree=self.mat_degree, domain=self.mesh
        )
        self.alpha0 = self.k0 * np.sin(self.theta0) * np.cos(self.phi0)
        self.beta0 = self.k0 * np.sin(self.theta0) * np.sin(self.phi0)
        self.gamma0 = self.k0 * np.cos(self.theta0)

        self._prepare_bcs()

    def _make_pmls(self):
        epsilon_pml = dict()
        mu_pml = dict()
        for coord in ["x", "y", "z", "xy", "xz", "yz", "xyz"]:
            pml = PML(coord, stretch=self.pml_stretch)
            t = np.array(pml.transformation_matrix())
            epsilon_pml["pml" + coord] = (self.epsilon["box"] * t).tolist()
            mu_pml["pml" + coord] = (self.mu["box"] * t).tolist()
        return epsilon_pml, mu_pml

    def weak_form(self):
        W = self.complex_space
        dx = self.dx
        self.E = Function(W)
        Etrial = TrialFunction(W)
        Etest = TestFunction(W)

        self.lhs = {}
        self.rhs = {}

        for d in self.domains:
            L = (
                -inner(self.inv_mu[d] * curl(Etrial), curl(Etest)),
                inner(self.eps[d] * Etrial, Etest),
            )
            self.lhs[d] = [t.real + t.imag for t in L]
        for d in self.source_domains:
            if self.eps[d].real.ufl_shape == (3, 3):
                eps_annex = tensor_const(np.eye(3) * self._epsilon_annex[d])
            else:
                eps_annex = self._epsilon_annex[d]

            if self.inv_mu[d].real.ufl_shape == (3, 3):
                inv_mu_annex = tensor_const(np.eye(3) * 1 / self._mu_annex[d])
            else:
                inv_mu_annex = self.inv_mu_annex[d]

            delta_epsilon = self.eps[d] - eps_annex
            delta_inv_mu = self.inv_mu[d] - inv_mu_annex
            b = (
                inner(delta_inv_mu * curl(self.incident_field), curl(Etest)),
                -inner(delta_epsilon * self.incident_field, Etest),
            )

            self.rhs[d] = [t.real + t.imag for t in b]

    def assemble_lhs(self):
        self.Ah = {}
        for d in self.domains:
            self.Ah[d] = [assemble(A * self.dx(d)) for A in self.lhs[d]]

    def assemble_rhs(self):
        self.bh = {}
        for d in self.source_domains:
            self.bh[d] = [assemble(b * self.dx(d)) for b in self.rhs[d]]

    def assemble(self):
        self.assemble_lhs()
        self.assemble_rhs()

    def build_system(self):

        for i, d in enumerate(self.domains):
            Ah_ = self.Ah[d][0] + self.k0 ** 2 * self.Ah[d][1]
            if ADJOINT:
                form_ = self.Ah[d][0].form + self.k0 ** 2 * self.Ah[d][1].form
            if i == 0:
                Ah = Ah_
                if ADJOINT:
                    form = form_
            else:
                Ah += Ah_
                if ADJOINT:
                    form += form_

        if ADJOINT:
            Ah.form = form

        for i, d in enumerate(self.source_domains):
            bh_ = self.bh[d][0] + self.k0 ** 2 * self.bh[d][1]
            if ADJOINT:
                form_ = self.bh[d][0].form + self.k0 ** 2 * self.bh[d][1].form
            if i == 0:
                bh = bh_
                if ADJOINT:
                    form = form_
            else:
                bh += bh_
                if ADJOINT:
                    form += form_
        if ADJOINT:
            bh.form = form

        self.matrix = Ah
        self.vector = bh

    def solve_system(self, direct=True):
        self.E = dolfin.Function(self.complex_space)

        for bc in self._boundary_conditions:
            bc.apply(self.matrix, self.vector)

        if direct:
            # solver = dolfin.LUSolver(Ah) ### direct
            solver = dolfin.LUSolver("mumps")
            # solver.parameters.update(lu_params)
            solver.solve(self.matrix, self.E.vector(), self.vector)
        else:
            solver = dolfin.PETScKrylovSolver()  ## iterative
            # solver.parameters.update(krylov_params)
            solver.solve(self.matrix, self.E.vector(), self.vector)

        E = Complex(*self.E.split())
        Etot = E + self.incident_field
        self.solution = {}
        self.solution["diffracted"] = E
        self.solution["total"] = Etot
