#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

from collections import OrderedDict

import numpy as np
import pytest
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

from gyptis.complex import *
from gyptis.complex import _invert_3by3_complex_matrix
from gyptis.core import PML
from gyptis.geometry import *
from gyptis.helpers import BiPeriodicBoundary3D, DirichletBC
from gyptis.materials import *
from gyptis.sources import *
from gyptis.stack import *

from . import ADJOINT, dolfin

pi = np.pi


lu_params = {"report": True, "symmetric": False, "verbose": True}


krylov_params = {
    "absolute_tolerance": 1.0e-1,
    "divergence_limit": 1000.0,
    "error_on_nonconvergence": True,
    "maximum_iterations": 500,
    "monitor_convergence": True,
    "nonzero_initial_guess": False,
    "relative_tolerance": 1.0e-1,
    "report": True,
}


# dolfin.set_log_level(10)
# dolfin.parameters["form_compiler"]["quadrature_degree"] = 5  #


def translation_matrix(t):
    M = np.eye(4)
    M[:3, -1] = t
    return M


class Layered3D(Geometry):
    def __init__(
        self,
        period=(1, 1),
        thicknesses=None,
        model_name="3D grating",
        mesh_name="mesh.msh",
        data_dir=None,
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            mesh_name=mesh_name,
            data_dir=data_dir,
            dim=3,
            **kwargs,
        )
        self.period = period
        self.thicknesses = thicknesses or OrderedDict(
            {
                "pml_bottom": 1,
                "substrate": 1,
                "groove": 2,
                "superstrate": 1,
                "pml_top": 1,
            }
        )

        self.translation_x = translation_matrix([self.period[0], 0, 0]).ravel().tolist()
        self.translation_y = translation_matrix([0, self.period[1], 0]).ravel().tolist()

        self.total_thickness = sum(self.thicknesses.values())
        dx, dy = self.period
        self.z0 = -sum(list(self.thicknesses.values())[:2])

        z0 = self.z0
        self.layers = {}
        self.z_position = {}
        for id, thickness in self.thicknesses.items():
            layer = self.make_layer(z0, thickness)
            self.layers[id] = layer
            self.z_position[id] = z0
            self.add_physical(layer, id)
            z0 += thickness

        self.remove_all_duplicates()
        self.synchronize()
        for sub, num in self.subdomains["volumes"].items():
            self.add_physical(num, sub)

    def make_layer(self, z_position, thickness):
        dx, dy = self.period
        box = self.add_box(-dx / 2, -dy / 2, z_position, dx, dy, thickness)
        return box

    def set_periodic_mesh(self):
        s = self.get_periodic_bnds(self.z0, self.total_thickness)

        periodic_id = {}
        for k, v in s.items():
            periodic_id[k] = [S[-1] for S in v]
        gmsh.model.mesh.setPeriodic(
            2, periodic_id["+x"], periodic_id["-x"], self.translation_x
        )
        gmsh.model.mesh.setPeriodic(
            2, periodic_id["+y"], periodic_id["-y"], self.translation_y
        )

    def build(self, set_periodic=True, **kwargs):
        if set_periodic:
            self.set_periodic_mesh()
        super().build(**kwargs)

    def get_periodic_bnds(self, z_position, thickness, eps=1e-3):
        s = {}
        dx, dy = self.period

        pmin = -dx / 2 - eps, -dy / 2 - eps, z_position - eps
        pmax = -dx / 2 + eps, +dy / 2 + eps, thickness + eps
        s["-x"] = gmsh.model.getEntitiesInBoundingBox(*pmin, *pmax, 2)

        pmin = +dx / 2 - eps, -dy / 2 - eps, z_position - eps
        pmax = +dx / 2 + eps, +dy / 2 + eps, thickness + eps
        s["+x"] = gmsh.model.getEntitiesInBoundingBox(*pmin, *pmax, 2)

        pmin = -dx / 2 - eps, -dy / 2 - eps, z_position - eps
        pmax = +dx / 2 + eps, -dy / 2 + eps, thickness + eps
        s["-y"] = gmsh.model.getEntitiesInBoundingBox(*pmin, *pmax, 2)

        pmin = -dx / 2 - eps, +dy / 2 - eps, z_position - eps
        pmax = +dx / 2 + eps, +dy / 2 + eps, thickness + eps
        s["+y"] = gmsh.model.getEntitiesInBoundingBox(*pmin, *pmax, 2)
        return s


# Physics


class Grating3D(object):
    def __init__(
        self,
        geom,
        epsilon,
        mu,
        lambda0=1,
        theta0=0,
        phi0=0,
        psi0=0,
        degree=1,
        mat_degree=None,
        pml_stretch=1 - 1j,
        boundary_conditions=[],
    ):

        self.geom = geom  # geometry model
        self.degree = degree
        self.mat_degree = mat_degree or self.degree
        self.lambda0 = lambda0
        self.theta0 = theta0
        self.phi0 = phi0
        self.psi0 = psi0
        self.epsilon = epsilon
        self.mu = mu
        self.pml_stretch = pml_stretch
        self.period = self.geom.period

        self.mesh = geom.mesh_object["mesh"]
        self.markers = geom.mesh_object["markers"]["tetra"]
        self.domains = geom.subdomains["volumes"]
        self.boundaries = geom.subdomains["surfaces"]
        self.dx = geom.measure["dx"]
        self.boundary_conditions = boundary_conditions

        self.N_d_order = 0
        self.periodic_map_tol = 1e-10

        # self.E0 = plane_wave_3D(
        #     self.lambda0, self.theta0, self.phi0, self.psi0, domain=self.mesh
        # )

        self.periodic_bcs = BiPeriodicBoundary3D(
            self.geom.period, map_tol=self.periodic_map_tol
        )

        self.element = "N1curl"

        self.complex_space = ComplexFunctionSpace(
            self.mesh, self.element, self.degree, constrained_domain=self.periodic_bcs
        )
        self.real_space = dolfin.FunctionSpace(
            self.mesh, self.element, self.degree, constrained_domain=self.periodic_bcs
        )

    @property
    def k0(self):
        return 2 * np.pi / self.lambda0

    @property
    def omega(self):
        return self.k0 * c

    def _phasor(self, *args, **kwargs):
        phasor_re = dolfin.Expression("cos(alpha*x[0] + beta*x[1])", *args, **kwargs)
        phasor_im = dolfin.Expression("sin(alpha*x[0] + beta*x[1])", *args, **kwargs)
        return Complex(phasor_re, phasor_im)

    def _phasor_z(self, *args, **kwargs):
        phasor_re = dolfin.Expression("cos(gamma*x[2])", *args, **kwargs)
        phasor_im = dolfin.Expression("sin(gamma*x[2])", *args, **kwargs)
        return Complex(phasor_re, phasor_im)

    def _make_subdomains(self, epsilon, mu):
        epsilon_coeff = Subdomain(
            self.markers, self.domains, epsilon, degree=self.mat_degree
        )
        mu_coeff = Subdomain(self.markers, self.domains, mu, degree=self.mat_degree)
        return epsilon_coeff, mu_coeff

    def make_stack(self):

        config = OrderedDict(
            {
                "superstrate": {
                    "epsilon": self.epsilon["superstrate"],
                    "mu": self.mu["superstrate"],
                },
                "substrate": {
                    "epsilon": self.epsilon["substrate"],
                    "mu": self.mu["substrate"],
                },
            }
        )
        self.Phi, alpha0, beta0, gamma, self.Rstack, self.Tstack = get_coeffs_stack(
            config,
            self.lambda0,
            self.theta0,
            self.phi0,
            self.psi0,
        )
        self.Phi = [p[:6] for p in self.Phi]

        self.E_stack = [
            field_stack_3D(p, alpha0, beta0, g, domain=self.mesh)
            for p, g in zip(self.Phi, gamma)
        ]
        self.Phi0 = np.zeros_like(self.Phi[0])
        self.Phi0[::2] = self.Phi[0][::2]

        self.E_0 = field_stack_3D(self.Phi0, alpha0, beta0, gamma[0], domain=self.mesh)
        estack = {k: list(v) for k, v in zip(config.keys(), self.E_stack)}

        for dom in self.source_dom:
            estack[dom] = estack["superstrate"]
        estack["pml_bottom"] = estack["pml_top"] = np.zeros(3)

        e0 = {"superstrate": list(self.E_0)}
        for dom in self.source_dom:
            e0[dom] = e0["superstrate"]
        e0["substrate"] = e0["pml_bottom"] = e0["pml_top"] = np.zeros(3)

        self.Estack_coeff = Subdomain(
            self.markers, self.domains, estack, degree=self.mat_degree, domain=self.mesh
        )

        self.E0_coeff = Subdomain(
            self.markers, self.domains, e0, degree=self.mat_degree, domain=self.mesh
        )
        inc_field = {}
        stack_field = {}
        for dom in self.source_dom:
            inc_field[dom] = complex_vector(e0[dom])
            stack_field[dom] = complex_vector(estack[dom])
        self.annex_field = {"incident": inc_field, "stack": stack_field}

    def _prepare_materials(self):
        epsilon = dict(superstrate=1, substrate=1)
        mu = dict(superstrate=1, substrate=1)
        epsilon.update(self.epsilon)
        mu.update(self.mu)
        self.epsilon_pml, self.mu_pml = self._make_pmls()
        self.epsilon.update(self.epsilon_pml)
        self.mu.update(self.mu_pml)
        self.epsilon_coeff, self.mu_coeff = self._make_subdomains(self.epsilon, self.mu)

        self.no_source_dom = ["substrate", "pml_top", "pml_bottom", "superstrate"]
        self.source_dom = [
            z for z in self.epsilon.keys() if z not in self.no_source_dom
        ]
        mu_annex = self.mu.copy()
        eps_annex = self.epsilon.copy()
        for a in self.source_dom:
            mu_annex[a] = self.mu["superstrate"]
            eps_annex[a] = self.epsilon["superstrate"]
        self.epsilon_coeff_annex, self.mu_coeff_annex = self._make_subdomains(
            eps_annex, mu_annex
        )
        self._epsilon_annex = eps_annex
        self._mu_annex = eps_annex
        self.inv_mu_coeff = _invert_3by3_complex_matrix(self.mu_coeff)
        self.inv_mu_coeff_annex = _invert_3by3_complex_matrix(self.mu_coeff_annex)

        self.inv_mu = make_constant_property(self.mu, inv=True)
        self.eps = make_constant_property(self.epsilon)

        self.inv_mu_annex = make_constant_property(mu_annex, inv=True)
        self.eps_annex = make_constant_property(eps_annex)

    def _make_pmls(self):
        pml = PML("z", stretch=self.pml_stretch)
        t = pml.transformation_matrix()
        eps_pml_ = [
            (self.epsilon[d] * t).tolist() for d in ["substrate", "superstrate"]
        ]
        mu_pml_ = [(self.mu[d] * t).tolist() for d in ["substrate", "superstrate"]]
        epsilon_pml = dict(pml_bottom=eps_pml_[0], pml_top=eps_pml_[1])
        mu_pml = dict(pml_bottom=mu_pml_[0], pml_top=mu_pml_[1])
        return epsilon_pml, mu_pml

    def weak_form(self):
        self._prepare_materials()
        self.make_stack()
        W = self.complex_space
        dx = self.dx
        self.E = Function(W)
        Etrial = TrialFunction(W)
        Etest = TestFunction(W)

        self.alpha0 = self.k0 * np.sin(self.theta0) * np.cos(self.phi0)
        self.beta0 = self.k0 * np.sin(self.theta0) * np.sin(self.phi0)
        self.gamma0 = self.k0 * np.cos(self.theta0)

        self.phasor = self._phasor(
            degree=self.mat_degree,
            domain=self.mesh,
            alpha=self.alpha0,
            beta=self.beta0,
        )

        self.unit_vectors = (
            as_vector([1.0, 0.0, 0.0]),
            as_vector([0.0, 1.0, 0.0]),
            as_vector([0.0, 0.0, 1.0]),
        )
        self.k_para = (
            self.alpha0 * self.unit_vectors[0] + self.beta0 * self.unit_vectors[1]
        )

        self.lhs = {}
        self.rhs = {}

        for d in self.domains:
            L = (
                -dot(self.inv_mu[d] * curl(Etrial), curl(Etest)),
                -1j
                * (
                    dot(self.inv_mu[d] * cross(self.k_para, Etrial), curl(Etest))
                    - dot(self.inv_mu[d] * cross(self.k_para, Etest), curl(Etrial))
                ),
                -dot(
                    cross(self.inv_mu[d] * cross(self.k_para, Etrial), self.k_para),
                    Etest,
                ),
                dot(self.eps[d] * Etrial, Etest),
            )
            self.lhs[d] = [t.real + t.imag for t in L]
        for d in self.source_dom:
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
                dot(delta_inv_mu * curl(self.annex_field["stack"][d]), curl(Etest))
                * self.phasor.conj,
                -dot(delta_epsilon * self.annex_field["stack"][d], Etest)
                * self.phasor.conj,
            )

            self.rhs[d] = [t.real + t.imag for t in b]

    def assemble_lhs(self):
        self.Ah = {}
        for d in self.domains:
            self.Ah[d] = [assemble(A * self.dx(d)) for A in self.lhs[d]]

    def assemble_rhs(self):
        self.bh = {}
        for d in self.source_dom:
            self.bh[d] = [assemble(b * self.dx(d)) for b in self.rhs[d]]

    def assemble(self):
        self.assemble_lhs()
        self.assemble_rhs()

    def solve(self, direct=True):

        for i, d in enumerate(self.domains):
            Ah_ = self.Ah[d][0] + self.k0 ** 2 * self.Ah[d][3]
            if ADJOINT:
                form_ = self.Ah[d][0].form + self.k0 ** 2 * self.Ah[d][3].form
            if i == 0:
                Ah = Ah_
                if ADJOINT:
                    form = form_
            else:
                Ah += Ah_
                if ADJOINT:
                    form += form_
            if self.alpha0 ** 2 + self.beta0 ** 2 != 0:
                Ah += self.Ah[d][1] + self.Ah[d][2]
                if ADJOINT:
                    form += self.Ah[d][1].form + self.Ah[d][2].form
        if ADJOINT:
            Ah.form = form

        for i, d in enumerate(self.source_dom):
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

        self.E = dolfin.Function(self.complex_space)

        for bc in self.boundary_conditions:
            bc.apply(Ah, bh)

        if direct:
            # solver = dolfin.LUSolver(Ah) ### direct
            solver = dolfin.LUSolver("mumps")
            # solver.parameters.update(lu_params)
            solver.solve(Ah, self.E.vector(), bh)
        else:
            solver = dolfin.PETScKrylovSolver()  ## iterative
            # solver.parameters.update(krylov_params)
            solver.solve(Ah, self.E.vector(), bh)

        Eper = Complex(*self.E.split())
        E = Eper * self.phasor
        Etot = E + self.Estack_coeff
        self.solution = {}
        self.solution["periodic"] = Eper
        self.solution["diffracted"] = E
        self.solution["total"] = Etot

    def diffraction_efficiencies(
        self, cplx_effs=False, orders=False, subdomain_absorption=False, verbose=False
    ):
        orders_num = np.linspace(
            -self.N_d_order,
            self.N_d_order,
            2 * self.N_d_order + 1,
        )

        k, gamma = {}, {}
        for d in ["substrate", "superstrate"]:
            k[d] = self.k0 * np.sqrt(complex(self.epsilon[d] * self.mu[d]))
            gamma[d] = np.conj(np.sqrt(k[d] ** 2 - self.alpha0 ** 2 - self.beta0 ** 2))

        r_annex = self.Phi[0][1::2]
        t_annex = self.Phi[-1][::2]
        zpos = self.geom.z_position
        thickness = self.geom.thicknesses
        period_x, period_y = self.period
        eff_annex = dict(substrate=t_annex, superstrate=r_annex)
        Eper = self.solution["periodic"]
        effn = []
        effn_cplx = []
        for n in orders_num:
            effm = []
            effm_cplx = []
            for m in orders_num:
                if verbose:
                    print("*" * 55)
                    print(f"order ({n},{m})")
                    print("*" * 55)
                delta = 1 if n == m == 0 else 0
                qn = n * 2 * np.pi / period_x
                pm = m * 2 * np.pi / period_y
                alpha_n = self.alpha0 + qn
                beta_m = self.beta0 + pm
                efficiencies = {}
                efficiencies_complex = {}
                for d in ["substrate", "superstrate"]:
                    s = 1 if d == "superstrate" else -1
                    # s = 1 if d == "substrate" else -1
                    gamma_nm = np.sqrt(k[d] ** 2 - alpha_n ** 2 - beta_m ** 2)
                    ph_xy = self._phasor(
                        degree=self.degree, domain=self.mesh, alpha=-qn, beta=-pm
                    )
                    ph_z = self._phasor_z(
                        degree=self.degree, domain=self.mesh, gamma=s * gamma_nm.real
                    )
                    Jnm = []
                    for comp in range(3):
                        Jnm.append(
                            assemble(Eper[comp] * ph_xy * ph_z * self.dx(d))
                            / (period_x * period_y)
                        )
                    ph_pos = np.exp(-s * 1j * gamma_nm * zpos[d])
                    eff, sqnorm_eff = [], 0
                    for comp in range(3):
                        eff_ = (
                            delta * eff_annex[d][comp] + Jnm[comp] / thickness[d]
                        ) * ph_pos
                        sqnorm_eff += eff_ * eff_.conj
                        eff.append(eff_)
                    eff_nrj = sqnorm_eff * gamma_nm / (gamma["superstrate"])
                    efficiencies_complex[d] = eff
                    efficiencies[d] = eff_nrj

                effm.append(efficiencies)
                effm_cplx.append(efficiencies_complex)
            effn.append(effm)
            effn_cplx.append(effm_cplx)

        Q, Qdomains = self.compute_absorption(subdomain_absorption=subdomain_absorption)

        T_nm = [[e["substrate"].real for e in b] for b in effn]
        R_nm = [[e["superstrate"].real for e in b] for b in effn]

        t_nm = [[e["substrate"] for e in b] for b in effn]
        r_nm = [[e["superstrate"] for e in b] for b in effn]

        T = sum([sum(_) for _ in T_nm])
        R = sum([sum(_) for _ in R_nm])

        B = R + T + Q

        effs = dict()
        effs["R"] = r_nm if cplx_effs else (R_nm if orders else R)
        effs["T"] = t_nm if cplx_effs else (T_nm if orders else T)
        effs["Q"] = Qdomains if subdomain_absorption else Q
        effs["B"] = B

        if verbose:
            print(f"  Energy balance")
            print(f"  R = {R:0.6f}")
            print(f"  T = {T:0.6f}")
            print(f"  Q = {Q:0.6f}")
            print(f"  ------------------------")
            print(f"  B = {B:0.6f}")

        return effs

    def compute_absorption(self, subdomain_absorption=False):
        P0 = (
            self.period[0]
            * self.period[1]
            * (epsilon_0 / mu_0) ** 0.5
            / (2 * np.cos(self.theta0))
        )
        doms_no_pml = [
            z for z in self.epsilon.keys() if z not in ["pml_bottom", "pml_top"]
        ]
        Etot = self.solution["total"]
        # curl E = i ω μ_0 μ H

        Htot = self.inv_mu_coeff / (1j * self.omega * mu_0) * curl(Etot)
        Qelec, Qmag = {}, {}
        if subdomain_absorption:
            for d in doms_no_pml:
                if np.all(self.epsilon[d].imag) == 0:
                    Qelec[d] = 0
                else:
                    elec_nrj_dens = dot(self.epsilon[d] * Etot, Etot.conj)
                    Qelec[d] = (
                        -0.5
                        * epsilon_0
                        * self.omega
                        * assemble(elec_nrj_dens * self.dx(d))
                        / P0
                    ).imag
                if np.all(self.mu[d].imag) == 0:
                    Qmag[d] = 0
                else:
                    mag_nrj_dens = dot(self.mu[d] * Htot, Htot.conj)
                    Qmag[d] = (
                        -0.5
                        * mu_0
                        * self.omega
                        * assemble(mag_nrj_dens * self.dx(d))
                        / P0
                    ).imag
            Q = sum(Qelec.values()) + sum(Qmag.values())
        else:
            elec_nrj_dens = dot(self.epsilon_coeff * Etot, Etot.conj)
            Qelec = (
                -0.5
                * epsilon_0
                * self.omega
                * assemble(elec_nrj_dens * self.dx(doms_no_pml))
                / P0
            ).imag
            mag_nrj_dens = dot(self.mu_coeff * Htot, Htot.conj)
            Qmag = (
                -0.5
                * mu_0
                * self.omega
                * assemble(mag_nrj_dens * self.dx(doms_no_pml))
                / P0
            ).imag
            Q = Qelec + Qmag
        Qdomains = {"electric": Qelec, "magnetic": Qmag}
        self.Qtot = Q
        self.Qdomains = Qdomains
        return Q, Qdomains


if __name__ == "__main__":

    lambda0 = 22
    parmesh = 11
    parmesh_pml = parmesh * 2 / 3
    period = (20, 20)
    eps_island = 6
    eps_sub = 1
    degree = 1

    theta0 = 0 * pi / 180
    phi0 = 0 * pi / 180
    psi0 = 0 * pi / 180

    thicknesses = OrderedDict(
        {
            "pml_bottom": lambda0,
            "substrate": lambda0,
            "groove": 10,
            "superstrate": lambda0,
            "pml_top": lambda0,
        }
    )

    epsilon = dict(
        {"substrate": eps_sub, "groove": 1, "island": eps_island, "superstrate": 1}
    )
    mu = dict({"substrate": 1, "groove": 1, "island": 1, "superstrate": 1})

    index = dict()
    for e, m in zip(epsilon.items(), mu.items()):
        index[e[0]] = np.mean(
            (np.array(e[1]).real.max() * np.array(m[1]).real.max()) ** 0.5
        ).real

    model = Layered3D(period, thicknesses, kill=True)

    groove = model.layers["groove"]
    z0 = model.z_position["groove"]
    island_width_top = 10
    island_width_bottom = 14
    island_thickness = 8
    island_bottom = model.add_rectangle(
        -island_width_bottom / 2,
        -island_width_bottom / 2,
        z0,
        island_width_bottom,
        island_width_bottom,
    )
    island_top = model.add_rectangle(
        -island_width_top / 2,
        -island_width_top / 2,
        z0 + island_thickness,
        island_width_top,
        island_width_top,
    )

    island = model.addThruSections([island_bottom, island_top])
    island, groove = model.fragment(island[-1][-1], groove)
    model.remove_all_duplicates()
    model.synchronize()
    model.add_physical(groove, "groove")
    model.add_physical(island, "island")
    #
    sub = model.subdomains["volumes"]["substrate"]
    sup = model.subdomains["volumes"]["superstrate"]
    pmltop = model.subdomains["volumes"]["pml_top"]
    pmlbot = model.subdomains["volumes"]["pml_bottom"]

    model.set_size(sub, lambda0 / (index["substrate"] * parmesh))
    model.set_size(sup, lambda0 / (index["superstrate"] * parmesh))
    model.set_size(pmlbot, lambda0 / (index["substrate"] * parmesh_pml))
    model.set_size(pmltop, lambda0 / (index["superstrate"] * parmesh_pml))
    model.set_size(groove, lambda0 / (index["groove"] * parmesh))
    model.set_size(island, lambda0 / (index["island"] * 2 * parmesh))
    # face_top = model.get_boundaries(pmltop)[-1]
    # face_bottom = model.get_boundaries(pmlbot)[-2]

    # model.add_physical(face_top, "face_top", dim=2)
    # model.add_physical(face_bottom, "face_bottom", dim=2)

    # mesh_object = model.build(interactive=True, generate_mesh=True, write_mesh=True)

    mesh_object = model.build()

    # model.mesh_object = mesh_object

    g = Grating3D(
        model,
        epsilon,
        mu,
        lambda0=lambda0,
        theta0=theta0,
        phi0=phi0,
        psi0=psi0,
        degree=degree,
    )

    g.weak_form()
    g.assemble()
    g.solve(direct=True)
    # dolfin.list_timings(dolfin.TimingClear.clear, [dolfin.TimingType.wall])

    # rank = dolfin.MPI.rank(dolfin.MPI.comm_world)
    # if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:

    g.N_d_order = 0

    effs = g.diffraction_efficiencies()
    print(effs)
    print(g.Rstack, g.Tstack)
    W0 = dolfin.FunctionSpace(g.mesh, "CG", 1)
    # W0 = dolfin.FunctionSpace(g.mesh, "DG", 0)
    fplot = g.E[0].real + g.Estack_coeff[0].real
    # fplot = abs(g.Eper)
    dolfin.File("test.pvd") << project(fplot, W0)
    dolfin.File("markers.pvd") << g.markers

    # u_sol_gath = None
    # if rank == 0:
    #     u_sol_gath =  np.empty(numDataPerRank*size, dtype='f')
    #
    #
    # Eper_re = dolfin.MPI.comm_world.gather(g.Eper.real.vector().get_local(), root=0)
    # Eper_im = dolfin.MPI.comm_world.gather(g.Eper.imag.vector().get_local(), root=0)
    # # g.Eper = Complex(Eper_re,Eper_im)
    # size = dolfin.MPI.comm_world.Get_size()
    # if rank == 0:
    #     print("computing diffraction efficiencies")
    #     for i in range(size):
    #         g.Eper.real.vector()[:] = Eper_re[i]
    #         g.Eper.imag.vector()[:] = Eper_im[i]
    #
    #     effs = g.diffraction_efficiencies()
    #
    # print(effs)
    # print(g.Rstack, g.Tstack)
    # import sys
    #
