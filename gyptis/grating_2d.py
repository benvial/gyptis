#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT


from collections import OrderedDict

import dolfin as df
import numpy as np
import pytest
from scipy.constants import c, epsilon_0, mu_0

from gyptis.complex import *
from gyptis.complex import _invert_3by3_complex_matrix
from gyptis.core import PML
from gyptis.geometry import *
from gyptis.helpers import DirichletBC, PeriodicBoundary2DX
from gyptis.materials import *
from gyptis.sources import *
from gyptis.stack import *

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

df.set_log_level(20)


def _translation_matrix(t):
    M = np.eye(4)
    M[:3, -1] = t
    return M


class Layered2D(Model):
    def __init__(
        self,
        period=1,
        thicknesses=None,
        model_name="2D grating",
        mesh_name="mesh.msh",
        data_dir=None,
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            mesh_name=mesh_name,
            data_dir=data_dir,
            dim=2,
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

        self.translation_x = _translation_matrix([self.period, 0, 0]).ravel().tolist()

        self.total_thickness = sum(self.thicknesses.values())

        self.y0 = -sum(list(self.thicknesses.values())[:2])
        self.layers = {}
        self.y_position = {}
        y0 = self.y0
        self._phys_groups = []
        for id, thickness in self.thicknesses.items():
            layer = self.make_layer(y0, thickness)
            self.layers[id] = layer
            self.y_position[id] = y0
            self.add_physical(layer, id)
            self._phys_groups.append(layer)
            y0 += thickness

        # self.removeAllDuplicates()
        # self.synchronize()
        self.removeAllDuplicates()
        self.synchronize()
        # for id, layer in zip(self.thicknesses.keys(),self._phys_groups):
        #     self.add_physical(layer, id)

    def make_layer(self, y_position, thickness):
        box = self.addRectangle(-self.period / 2, y_position, 0, self.period, thickness)
        return box

    def build(self, **kwargs):

        s = self.get_periodic_bnds(self.y0, self.total_thickness)

        periodic_id = {}
        for k, v in s.items():
            periodic_id[k] = [S[-1] for S in v]
        gmsh.model.mesh.setPeriodic(
            1, periodic_id["+x"], periodic_id["-x"], self.translation_x
        )

        super().build(**kwargs)

    def get_periodic_bnds(self, y_position, thickness, eps=1e-3):
        s = {}

        pmin = -self.period / 2 - eps, -eps + y_position, -eps
        pmax = -self.period / 2 + eps, y_position + thickness + eps, eps
        s["-x"] = gmsh.model.getEntitiesInBoundingBox(*pmin, *pmax, 1)

        pmin = +self.period / 2 - eps, -eps + y_position, -eps
        pmax = +self.period / 2 + eps, y_position + thickness + eps, eps
        s["+x"] = gmsh.model.getEntitiesInBoundingBox(*pmin, *pmax, 1)

        return s


class Simulation2D(object):
    def __init__(
        self,
        geom,
        degree=1,
        boundary_conditions={},
    ):
        self.geom = geom
        self.degree = degree
        self.mesh = geom.mesh_object["mesh"]
        self.markers = geom.mesh_object["markers"]["triangle"]
        self.domains = geom.subdomains["surfaces"]
        self.surfaces = geom.subdomains["curves"]
        self.dx = geom.measure["dx"]
        self.ds = geom.measure["ds"]
        self.dS = geom.measure["dS"]
        self.boundary_conditions = boundary_conditions
        self.unit_normal_vector = df.FacetNormal(self.mesh)


class Grating2D(Simulation2D):
    def __init__(
        self,
        geom,
        epsilon,
        mu,
        lambda0=1,
        theta0=0,
        polarization="TE",
        degree=1,
        pml_stretch=1 - 1j,
        boundary_conditions={},
    ):

        super().__init__(geom, degree=degree, boundary_conditions=boundary_conditions)

        self.lambda0 = lambda0
        self.theta0 = theta0
        self.polarization = polarization
        self.epsilon = epsilon
        self.mu = mu
        self.pml_stretch = pml_stretch
        self.N_d_order = 0
        self.nb_slice = 20
        self.scan_dist_ratio = 5
        self.npt_integ = 401

        self.periodic_bcs = PeriodicBoundary2DX(self.geom.period)

        self.complex_space = ComplexFunctionSpace(
            self.mesh, "CG", self.degree, constrained_domain=self.periodic_bcs
        )
        self.real_space = df.FunctionSpace(
            self.mesh, "CG", self.degree, constrained_domain=self.periodic_bcs
        )

        self.no_source_dom = ["substrate", "pml_top", "pml_bottom", "superstrate"]
        self.source_dom = [
            z for z in self.epsilon.keys() if z not in self.no_source_dom
        ]

    def _prepare_bcs(self):
        self._boundary_conditions = []
        self.pec_bnds = []
        for bnd, cond in self.boundary_conditions.items():
            if cond != "PEC":
                raise ValueError(f"unknown boundary condition {cond}")
            else:
                self.pec_bnds.append(bnd)
        for bnd in self.pec_bnds:
            # if self.polarization == "TM":
            #     raise NotImplementedError(
            #         f"PEC not yet implemented for TM polarization"
            #     )
            #     # FIXME: implement PEC for TM polarization (Robin BC)
            # else:
            if self.polarization == "TE":
                curves = self.geom.subdomains["curves"]
                markers_curves = self.geom.mesh_object["markers"]["line"]
                ubnd = -self.ustack_coeff * self.phasor.conj
                # ubnd = df.as_vector((ubnd_vec.real, ubnd_vec.imag))
                ubnd_proj = project(ubnd, self.real_space)
                bc = DirichletBC(
                    self.complex_space, ubnd_proj, markers_curves, bnd, curves
                )
                [self._boundary_conditions.append(b) for b in bc]

    @property
    def k0(self):
        return 2 * np.pi / self.lambda0

    def get_N_d_order(self):
        a = self.geom.period / self.lambda0
        b = np.sin(self.theta0)
        index = np.array(
            [
                (np.sqrt(self.epsilon[d] * self.mu[d])).real
                for d in ["substrate", "superstrate"]
            ]
        )
        WRA = np.ravel([a * index + b, a * index - b])
        self._N_d_order = int(max(abs(WRA)))
        return self._N_d_order

    # @N_d_order.setter
    # def N_d_order(self, value):
    #     self._N_d_order = value

    def _make_subdomains(self, epsilon, mu):
        epsilon_coeff = Subdomain(
            self.markers, self.domains, epsilon, degree=self.degree
        )
        mu_coeff = Subdomain(self.markers, self.domains, mu, degree=self.degree)
        return epsilon_coeff, mu_coeff

    def _prepare_materials(self):
        epsilon = dict(superstrate=1, substrate=1)
        mu = dict(superstrate=1, substrate=1)
        epsilon.update(self.epsilon)
        mu.update(self.mu)
        self.epsilon_pml, self.mu_pml = self._make_pmls()
        self.epsilon.update(self.epsilon_pml)
        self.mu.update(self.mu_pml)
        self.epsilon_coeff, self.mu_coeff = self._make_subdomains(self.epsilon, self.mu)

        mu_annex = self.mu.copy()
        eps_annex = self.epsilon.copy()
        for a in self.source_dom:
            mu_annex[a] = self.mu["superstrate"]
            eps_annex[a] = self.epsilon["superstrate"]
        self.epsilon_coeff_annex, self.mu_coeff_annex = self._make_subdomains(
            eps_annex, mu_annex
        )

    def _coefs(self, a, b):
        # xsi = det Q^T/det Q
        extract = lambda q: df.as_tensor([[q[0][0], q[0][1]], [q[1][0], q[1][1]]])
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
        pml = PML("y", stretch=self.pml_stretch)
        t = pml.transformation_matrix()
        eps_pml_ = [
            (self.epsilon[d] * t).tolist() for d in ["substrate", "superstrate"]
        ]
        mu_pml_ = [(self.mu[d] * t).tolist() for d in ["substrate", "superstrate"]]
        epsilon_pml = dict(pml_bottom=eps_pml_[0], pml_top=eps_pml_[1])
        mu_pml = dict(pml_bottom=mu_pml_[0], pml_top=mu_pml_[1])
        return epsilon_pml, mu_pml

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
        if self.polarization == "TE":
            _psi = np.pi / 2
            _phi_ind = 2
        else:
            _psi = 0
            _phi_ind = 8
        phi_, alpha0, _, beta, self.Rstack, self.Tstack = get_coeffs_stack(
            config,
            self.lambda0,
            -self.theta0,
            0,
            _psi,
        )
        thick = [d["thickness"] for d in config.values() if "thickness" in d.keys()]
        self.phi = [[p[_phi_ind], p[_phi_ind + 1]] for p in phi_]
        self.phi = (np.array(self.phi) / self.phi[0][0]).tolist()

        self.u_stack = [
            field_stack_2D(p, alpha0, b, yshift=0, domain=self.mesh)
            for p, b in zip(self.phi, beta)
        ]
        self.phi0 = [self.phi[0][0], 0]
        self.u_0 = field_stack_2D(
            self.phi0, alpha0, beta[0], yshift=0, domain=self.mesh
        )
        estack = {k: v for k, v in zip(config.keys(), self.u_stack)}
        for dom in self.source_dom:
            estack[dom] = estack["superstrate"]
        estack["pml_bottom"] = estack["pml_top"] = 0
        e0 = {"superstrate": self.u_0}
        for dom in self.source_dom:
            e0[dom] = e0["superstrate"]
        e0["substrate"] = e0["pml_bottom"] = e0["pml_top"] = 0
        self.ustack_coeff = Subdomain(
            self.markers, self.domains, estack, degree=self.degree, domain=self.mesh
        )

        self.u0_coeff = Subdomain(
            self.markers, self.domains, e0, degree=self.degree, domain=self.mesh
        )

    def _phasor(self, *args, **kwargs):
        phasor_re = df.Expression("cos(alpha*x[0])", *args, **kwargs)
        phasor_im = df.Expression("sin(alpha*x[0])", *args, **kwargs)
        return Complex(phasor_re, phasor_im)

    def weak_form(self):
        self.alpha = -self.k0 * np.sin(self.theta0)
        self.phasor = self._phasor(
            degree=self.degree, domain=self.mesh, alpha=self.alpha
        )
        self._prepare_materials()
        self._make_coefs()
        self.make_stack()
        self._prepare_bcs()
        W = self.complex_space
        dx = self.dx
        ds = self.ds
        self.u = Function(W)
        utrial = TrialFunction(W)
        utest = TestFunction(W)
        dxi = self.xi - self.xi_annex
        dchi = self.chi - self.chi_annex
        n = self.unit_normal_vector

        # self.alpha_vect = Complex(df.as_vector([self.alpha, 0]), df.as_vector([0, 0]))

        # self.alpha_vect = as_vector([self.alpha, 0])
        self.ex = as_vector([1.0, 0.0])

        L = [
            -dot(self.xi * grad(utrial), grad(utest)) * dx,
            1j
            * (
                dot(self.ex, self.xi * grad(utrial) * utest)
                - dot(self.ex, self.xi * grad(utest) * utrial)
            )
            * dx,
            -dot(self.xi * self.ex, self.ex) * utrial * utest * dx,
            self.chi * utrial * utest * dx,
        ]

        b = [
            -dot(dxi * grad(self.ustack_coeff), grad(utest))
            * self.phasor.conj
            * dx(self.source_dom),
            1j
            * dot(dxi * grad(self.ustack_coeff), self.ex)
            * utest
            * self.phasor.conj
            * dx(self.source_dom),
            dchi * self.ustack_coeff * utest * self.phasor.conj * dx(self.source_dom),
        ]
        # surface term for PEC in TM polarization
        if self.polarization == "TM" and len(self.pec_bnds) > 0:
            L.append(
                -1j * dot(self.xi * self.ex, n) * utrial * utest * ds(self.pec_bnds)
            )
            b.append(
                dot(self.xi_annex * grad(self.ustack_coeff), n)
                * utest
                * self.phasor.conj
                * ds(self.pec_bnds),
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

    def solve(self, direct=True):
        ufunc = self.u.real
        Ah = self.Ah[0] + self.k0 ** 2 * self.Ah[3]
        bh = -self.bh[0] - self.k0 ** 2 * self.bh[2]
        if self.alpha != 0:
            Ah += self.alpha * self.Ah[1] + self.alpha ** 2 * self.Ah[2]
            bh -= self.alpha * self.bh[1]
            if self.polarization == "TM" and len(self.pec_bnds) > 0:
                Ah += self.alpha * self.Ah[4]
        if self.polarization == "TM" and len(self.pec_bnds) > 0:
            bh += self.bh[3]
        for bc in self._boundary_conditions:
            bc.apply(Ah, bh)

        if direct:
            # solver = df.LUSolver(Ah) ### direct
            solver = df.PETScLUSolver("mumps")
            solver.parameters.update(lu_params)
            solver.solve(Ah, ufunc.vector(), bh)
        else:
            solver = df.PETScKrylovSolver()  ## iterative
            solver.parameters.update(krylov_params)
            solver.solve(Ah, ufunc.vector(), bh)

            # solver.set_operator(Ah)
            # # Create vector that spans the null space and normalize
            # null_vec = df.Vector(Efunc.vector())
            # self.complex_space.dofmap().set(null_vec, 1.0)
            # null_vec *= 1.0/null_vec.norm("l2")
            #
            # # Create null space basis object and attach to PETSc matrix
            # null_space = df.VectorSpaceBasis([null_vec])
            # df.as_backend_type(Ah).set_nullspace(null_space)
            # null_space.orthogonalize(bh)
            # solver.solve(Efunc.vector(), bh)

        self.uper = self.u
        self.u *= self.phasor

    def diffraction_efficiencies(
        self, cplx_effs=False, orders=False, subdomain_absorption=False, verbose=False
    ):
        """Postprocess diffraction efficiencies.

        Parameters
        ----------
        cplx_effs : bool
            If `True`, return complex coefficients (amplitude reflection and transmission).
            If `False`, return real coefficients (power reflection and transmission)
        orders : bool
            If `True`, computes the transmission and reflection for all the propagating diffraction orders.
            If `False`, returns the sum of all the propagating diffraction orders.

        Returns
        -------
        dict
            A dictionary containing the diffraction efficiencies.

        """
        if self.polarization == "TE":
            nu = 1 / self.mu["substrate"]
        else:
            nu = 1 / self.epsilon["substrate"]
        order_shift = 0
        No_order = np.linspace(
            -self.N_d_order + order_shift,
            self.N_d_order + order_shift,
            2 * self.N_d_order + 1,
        )

        self.scan_dist = (
            min(
                self.geom.thicknesses["substrate"], self.geom.thicknesses["superstrate"]
            )
            / self.scan_dist_ratio
        )

        self.ycut = dict()
        for k in ["substrate", "superstrate"]:
            self.ycut[k] = (
                self.geom.y_position[k] + self.scan_dist,
                (self.geom.y_position[k] + self.geom.thicknesses[k] - self.scan_dist),
            )

        x_slice = np.linspace(
            -self.geom.period / 2, self.geom.period / 2, self.npt_integ
        )
        y_slice_t = np.linspace(*self.ycut["substrate"], self.nb_slice)
        y_slice_r = np.linspace(*self.ycut["superstrate"], self.nb_slice)
        k_sub = (
            2
            * np.pi
            * np.sqrt(complex(self.epsilon["substrate"] * self.mu["substrate"]))
            / self.lambda0
        )
        k_sup = (
            2
            * np.pi
            * np.sqrt(complex(self.epsilon["superstrate"] * self.mu["superstrate"]))
            / self.lambda0
        )
        alpha_sup = -k_sup * np.sin(self.theta0)
        beta_sup = np.sqrt(k_sup ** 2 - alpha_sup ** 2)
        # beta_sub = np.sqrt(k_sub ** 2 - alpha_sup ** 2)
        s_t = np.zeros((1, (2 * self.N_d_order + 1)), complex)[0, :]
        s_r = np.zeros((1, (2 * self.N_d_order + 1)), complex)[0, :]
        Aeff_t = np.zeros((self.nb_slice, 2 * self.N_d_order + 1), complex)
        Aeff_r = np.zeros((self.nb_slice, 2 * self.N_d_order + 1), complex)

        def get_line(u, y):
            re, im = [], []
            for x_ in x_slice:
                f = u(x_, y)
                re.append(f.real)
                im.append(f.imag)
            return np.array(re) + 1j * np.array(im)

        field_diff_t = []
        field_diff_r = []
        # u = project(self.u, self.real_space)

        for y_ in y_slice_t:
            l = get_line(self.uper, y_) * np.exp(1j * alpha_sup * x_slice)
            l += get_line(self.u_stack[-1], y_)
            field_diff_t.append(l)
        for y_ in y_slice_r:
            l = get_line(self.uper, y_) * np.exp(1j * alpha_sup * x_slice)
            l += get_line(self.u_stack[0], y_)
            l -= get_line(self.u_0, y_)
            field_diff_r.append(l)

        field_diff_t = np.array(field_diff_t)
        field_diff_r = np.array(field_diff_r)

        # field_diff_t = np.fliplr(field_diff_t)
        # field_diff_r = np.fliplr(field_diff_r)

        alphat_t = alpha_sup + 2 * np.pi / (self.geom.period) * No_order
        alphat_r = alpha_sup + 2 * np.pi / (self.geom.period) * No_order
        betat_sup = np.conj(np.sqrt(k_sup ** 2 - alphat_r ** 2))
        betat_sub = np.conj(np.sqrt(k_sub ** 2 - alphat_t ** 2))
        for m1 in range(0, self.nb_slice):
            slice_t = field_diff_t[m1, :]
            slice_r = field_diff_r[m1, :]

            for k in range(0, 2 * self.N_d_order + 1):
                expalpha_t = np.exp(-1j * alphat_t[k] * x_slice)
                expalpha_r = np.exp(-1j * alphat_r[k] * x_slice)
                s_t[k] = np.trapz(slice_t * expalpha_t, x=x_slice) / self.geom.period
                s_r[k] = np.trapz(slice_r * expalpha_r, x=x_slice) / self.geom.period

            Aeff_t[m1, :] = s_t * np.exp(-1j * betat_sub * (y_slice_t[m1]))
            Aeff_r[m1, :] = s_r * np.exp(
                1j
                * betat_sup
                * (y_slice_r[m1] - (self.ycut["substrate"][0] - self.scan_dist))
            )

        # Aeff_r = -np.conj(Aeff_r)

        Beff_t = (np.abs(Aeff_t)) ** 2 * betat_sub / beta_sup * nu
        Beff_r = (np.abs(Aeff_r)) ** 2 * betat_sup / beta_sup

        # print(Aeff_r)
        # print(Aeff_t)

        rcplx = np.mean(Aeff_r, axis=0)
        tcplx = np.mean(Aeff_t, axis=0)

        Rorders = np.mean(Beff_r.real, axis=0)
        Torders = np.mean(Beff_t.real, axis=0)
        R = np.sum(Rorders, axis=0)
        T = np.sum(Torders, axis=0)

        doms_no_pml = [
            z for z in self.epsilon.keys() if z not in ["pml_bottom", "pml_top"]
        ]
        omega = self.k0 * c

        if self.polarization == "TE":
            xi_0, chi_0 = mu_0, epsilon_0
        else:
            xi_0, chi_0 = epsilon_0, mu_0

        P0 = 0.5 * np.sqrt(chi_0 / xi_0) * np.cos(self.theta0) * self.geom.period

        u_tot = self.u + self.ustack_coeff

        nrj_chi_dens = (
            df.Constant(-0.5 * chi_0 * omega) * self.chi * abs(u_tot) ** 2
        ).imag

        nrj_xi_dens = (
            df.Constant(-0.5 * 1 / (omega * xi_0))
            * dot(grad(u_tot), (self.xi * grad(u_tot)).conj).imag
        )

        if subdomain_absorption:
            Qchi = {d: assemble(nrj_chi_dens * self.dx(d)) / P0 for d in doms_no_pml}
            Qxi = {d: assemble(nrj_xi_dens * self.dx(d)) / P0 for d in doms_no_pml}
            Q = sum(Qxi.values()) + sum(Qchi.values())
        else:
            Qchi = assemble(nrj_chi_dens * self.dx(doms_no_pml)) / P0
            Qxi = assemble(nrj_xi_dens * self.dx(doms_no_pml)) / P0
            Q = Qxi + Qchi

        if self.polarization == "TE":
            Q_domains = {"electric": Qchi, "magnetic": Qxi}
        else:
            Q_domains = {"electric": Qxi, "magnetic": Qchi}

        self.Qtot = Q

        B = T + R + Q  # energy balance

        if verbose:
            print("  Energy balance")
            print(
                "    R =",
                "%0.6f" % R,
                "    (std slice2slice =",
                "%0.6e" % np.std(np.sum(Beff_r.real, axis=1)),
                ")",
            )
            print(
                "    T =",
                "%0.6f" % T,
                "    (std slice2slice =",
                "%0.6e" % np.std(np.sum(Beff_t.real, axis=1)),
                ")",
            )
            print("    Q =", "%0.6f" % Q)
            print("    ------------------------")
            print("    B =", "%0.6f" % (B))

        if cplx_effs:
            R, T = rcplx, tcplx
        else:
            if orders:
                R, T = Rorders, Torders
        eff = dict()
        eff["R"] = R
        eff["T"] = T
        eff["Q"] = Q_domains if subdomain_absorption else Q
        eff["B"] = B
        return eff


if __name__ == "__main__":

    polarization = "TE"
    lambda0 = 40
    theta0 = 30 * np.pi / 180
    parmesh = 20
    parmesh_pml = parmesh * 2 / 3
    period = 20
    #
    eps_island = np.diag([6 - 1j, 3 - 0.7j, 3 - 0.6j])
    eps_off_diag = 6 - 0.1j
    eps_island[1, 0] = eps_island[0, 1] = eps_off_diag
    eps_island = 6 - 1j
    mu_island = np.diag([2 - 0.2j, 3 - 0.6j, 2 - 0.1j])
    mu_off_diag = 3.3 - 0.2j
    mu_island[1, 0] = mu_island[0, 1] = mu_off_diag
    mu_island = 1
    order = 2

    thicknesses = OrderedDict(
        {
            "pml_bottom": 1 * lambda0,
            "substrate": 1 * lambda0,
            "sublayer": 10,
            "groove": 10,
            "superstrate": 1 * lambda0,
            "pml_top": 1 * lambda0,
        }
    )
    #

    epsilon = dict(
        {
            "substrate": 5,
            "groove": 1,
            "sublayer": 3 - 0.1j,
            "island": eps_island,
            "superstrate": 1,
        }
    )
    mu = dict(
        {
            "substrate": 1,
            "groove": 1,
            "sublayer": 1,
            "island": mu_island,
            "superstrate": 1,
        }
    )

    index = dict()
    for e, m in zip(epsilon.items(), mu.items()):
        index[e[0]] = np.mean(
            (np.array(e[1]).real.max() * np.array(m[1]).real.max()) ** 0.5
        ).real

    model = Layered2D(period, thicknesses, kill=True)

    groove = model.layers["groove"]
    y0 = model.y_position["groove"]
    island_width_top = 10
    island_width_bottom = 10
    island_thickness = 5
    #
    #
    island = model.addRectangle(
        -island_width_bottom / 2, y0, 0, island_width_bottom, island_thickness
    )
    island, groove = model.fragmentize(island, groove)
    model.removeAllDuplicates()
    model.synchronize()
    model.add_physical(groove, "groove")
    model.add_physical(island, "island")
    #
    sub = model.subdomains["surfaces"]["substrate"]
    sup = model.subdomains["surfaces"]["superstrate"]
    pmltop = model.subdomains["surfaces"]["pml_top"]
    pmlbot = model.subdomains["surfaces"]["pml_bottom"]

    model.set_size(sub, lambda0 / (index["substrate"] * parmesh))
    model.set_size(sup, lambda0 / (index["superstrate"] * parmesh))
    model.set_size(pmlbot, lambda0 / (index["substrate"] * parmesh_pml))
    model.set_size(pmltop, lambda0 / (index["superstrate"] * parmesh_pml))
    model.set_size(groove, lambda0 / (index["groove"] * parmesh))
    model.set_size(island, lambda0 / (index["island"] * parmesh))
    face_top = model.get_boundaries(pmltop)[-2]
    face_bottom = model.get_boundaries(pmlbot)[0]

    model.add_physical(face_top, "face_top", dim=1)
    model.add_physical(face_bottom, "face_bottom", dim=1)

    # mesh_object = model.build(interactive=True, generate_mesh=True, write_mesh=True)
    mesh_object = model.build()

    # e = (3 * np.eye(3, dtype=complex)).tolist()
    # m = (np.eye(3, dtype=complex)).tolist()
    # epsilon = dict({"groove": m, "island": e})
    # mu = dict({"groove": m, "island": m})

    g = Grating2D(
        model,
        epsilon,
        mu,
        polarization=polarization,
        lambda0=lambda0,
        theta0=theta0,
        degree=order,
    )

    #
    # # df.File("test.pvd") << project(Estack[0].real[0], W0)
    # df.File("test.pvd") << project(test.real, W0)
    #
    # cds

    ### BCs
    # domains = model.subdomains["volumes"]
    # surfaces = model.subdomains["surfaces"]
    # markers_surf = model.mesh_object["markers"]["triangle"]
    # self.boundary_conditions = [
    #     DirichletBC(g.complex_space, [0] * 6, markers_surf, f, surfaces)
    #     for f in ["face_top", "face_bottom"]
    # ]

    from pprint import pprint

    import matplotlib.pyplot as plt

    plt.ion()

    g.weak_form()
    g.assemble()
    g.solve(direct=True)
    # g.solve(direct=False)
    # df.list_timings(df.TimingClear.clear, [df.TimingType.wall])

    # g.N_d_order=1

    g.N_d_order = 1
    effs = g.diffraction_efficiencies(orders=True, subdomain_absorption=True)
    pprint(effs)  # ,sort_dicts=False)
    print("Qtot", g.Qtot)

    g.polarization = "TM"
    g.weak_form()
    g.assemble()
    g.solve(direct=True)
    effs = g.diffraction_efficiencies(orders=True, subdomain_absorption=True)
    pprint(effs)  # ,sort_dicts=False)
    print("Qtot", g.Qtot)

    u_tot = g.u + g.ustack_coeff
    u_diff = u_tot - g.u0_coeff

    W0 = df.FunctionSpace(g.mesh, "DG", 0)
    W0 = df.FunctionSpace(g.mesh, "CG", 1)

    def plot_subdomains(alpha=0.3):
        a = df.plot(g.markers, cmap="binary", alpha=alpha, lw=0.00, edgecolor="face")
        return a
        # a.set_edgecolors((0.1, 0.2, 0.5, 0.))

    def plotcplx(toplot, ax, ref_cbar=False):

        test = project(toplot, W0)
        plt.sca(ax[0])
        p = df.plot(test.real, cmap="RdBu_r")
        cbar = plt.colorbar(p)
        plot_subdomains()
        if ref_cbar:
            v = test.real.vector().get_local()
            mn, mx = min(v), max(v)
            md = 0.5 * (mx + mn)
            cbar.set_ticks([mn, md, mx])
            lab = [f"{m:.2e}" for m in [mn, md, mx]]
            cbar.set_ticklabels(lab)
        plt.sca(ax[1])
        p = df.plot(test.imag, cmap="RdBu_r")
        cbar = plt.colorbar(p)
        plot_subdomains()
        if ref_cbar:
            v = test.imag.vector().get_local()
            mn, mx = min(v), max(v)
            md = 0.5 * (mx + mn)
            cbar.set_ticks([mn, md, mx])
            lab = [f"{m:.2e}" for m in [mn, md, mx]]
            cbar.set_ticklabels(lab)

    def plot(toplot):
        test = project(toplot, W0)
        p = df.plot(test, cmap="inferno")
        plt.colorbar(p)
        plot_subdomains()

    #
    plt.close("all")
    fig, ax = plt.subplots(1, 2)
    plotcplx(g.u + g.ustack_coeff, ax)
    [a.axis("off") for a in ax]
    [a.set_ylim((-10, 20)) for a in ax]
    plt.tight_layout()

    # no = abs(g.u + g.ustack_coeff-g.u0_coeff)
    no = abs(g.u)
    fig, ax = plt.subplots(1)
    plot(no)
    ax.axis("off")
    ax.set_ylim((-10, 20))
    plt.tight_layout()

    # plt.close("all")
    fig, ax = plt.subplots(1, 2)
    plotcplx(g.ustack_coeff, ax)
    print("    Rstack = ", g.Rstack)
    print("    Tstack = ", g.Tstack)

    # plot(abs(g.u))
    # plotcplx(g.u0_coeff)

    # fig, ax = plt.subplots(1, 2)
    # Nphi = 20
    # for i in range(Nphi):
    #     ax[0].clear()
    #     ax[1].clear()
    #     plotcplx(g.ustack_coeff * np.exp(1j * 2 * pi * i / Nphi), ax)
    #     plt.pause(0.1)

    #
    # toplot = g.ustack_coeff
    # test = project(toplot, W0)
    # plt.clf()
    # p = df.plot(test.imag, cmap="RdBu_r")
    # plt.colorbar(p)

    #
    #
    #
    # test = project(u_diff, W1)
    # plt.clf()
    # p = df.plot(test.real, cmap="RdBu_r")
    # plt.colorbar(p)

    # testplot = project(ustack_coeff, W0)
    # testplot1 = project(u0_coeff, W0)
    #
    # plt.clf()
    # p = df.plot(testplot.real -testplot1.real, cmap="RdBu_r")
    # plt.colorbar(p)
    #
    # ygyptis = np.linspace(0, g.geom.total_thickness, 1000)
    # y0 = sum(list(self.geom.thicknesses.values())[:2])
    # ygyptis -= y0
    # checkre = []
    # checkim = []
    # for y_ in ygyptis:
    #     f = testplot(0, y_)
    #     checkre.append(f.real)
    #     checkim.append(f.imag)
    #
    # plt.figure()
    # plt.plot(ygyptis, checkre)
    #
    #
    #
    # ############
    #
    # y = np.linspace(0, 1 * g.geom.total_thickness, 600)
    #
    #
    # msk = np.zeros_like(y)
    #
    # yi = 0
    # msk[y <= yi] = 1
    # mask = [msk]
    # for t in thick:
    #     msk = np.zeros_like(y)
    #     msk[(y > yi) & (y <= yi + t)] = 1
    #     mask.append(msk)
    #     yi += t
    # msk = np.zeros_like(y)
    # msk[(y > yi)] = 1
    # mask.append(msk)
    #
    # # mask=np.flipud(mask)
    #
    # [plt.plot(y, m, c="#d7d7d7") for m in mask]
    # y1 = y - y0
    # prop_plus = [np.exp(-1j * b * y1) for b in beta]
    # prop_minus = [np.exp(1j * b * y1) for b in beta]
    #
    # E = []
    # for pp, pm, p, m, e, b in zip(prop_plus, prop_minus, phi, mask, tcum, beta):
    #     dp, dm = np.exp(1j * b * e), np.exp(-1j * b * e)
    #     E.append((pp * p[0] * dp + pm * p[1] * dm) * m)
    #
    # E = np.fliplr(E)
    #
    # plt.clf()
    # # [plt.plot(z1, e) for e in Elay]
    # plt.plot(y1 + sum(thick), np.real(sum(E)), "r-")
    # # plt.plot(y, np.imag(sum(E)),"--")
    # # plt.plot(y, np.abs(sum(E)),"k")
    # plt.plot(ygyptis, checkre)
    #
    # xsas
