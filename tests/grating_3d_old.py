#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT


from collections import OrderedDict

import dolfin
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


class Layered3D(Model):
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

        self.removeAllDuplicates()
        self.synchronize()

    def make_layer(self, z_position, thickness):
        dx, dy = self.period
        box = self.addBox(-dx / 2, -dy / 2, z_position, dx, dy, thickness)
        return box

    def build(self, **kwargs):
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
        mat_degree=1,
        pml_stretch=1 - 1j,
        boundary_conditions=[],
    ):

        self.geom = geom  # geometry model
        self.degree = degree
        self.mat_degree = mat_degree
        self.lambda0 = lambda0
        self.theta0 = theta0
        self.phi0 = phi0
        self.psi0 = psi0
        self.epsilon = epsilon
        self.mu = mu
        self.pml_stretch = pml_stretch

        self.mesh = geom.mesh_object["mesh"]
        self.markers = geom.mesh_object["markers"]["tetra"]
        self.domains = geom.subdomains["volumes"]
        self.surfaces = geom.subdomains["surfaces"]
        self.dx = geom.measure["dx"]
        self.boundary_conditions = boundary_conditions

        self.N_d_order = 0
        self.ninterv_integ = 101
        self.scan_dist_ratio = 5
        self.nb_slice = 10
        self.periodic_map_tol = 1e-10

        # self.E0 = plane_wave_3D(
        #     self.lambda0, self.theta0, self.phi0, self.psi0, domain=self.mesh
        # )

        self.periodic_bcs = BiPeriodicBoundary3D(
            self.geom.period, map_tol=self.periodic_map_tol
        )

        self.complex_space = ComplexFunctionSpace(
            self.mesh, "N1curl", self.degree, constrained_domain=self.periodic_bcs
        )
        self.real_space = dolfin.FunctionSpace(
            self.mesh, "N1curl", self.degree, constrained_domain=self.periodic_bcs
        )

    @property
    def k0(self):
        return 2 * np.pi / self.lambda0

    def _phasor(self, *args, **kwargs):
        phasor_re = dolfin.Expression("cos(alpha*x[0] + beta*x[1])", *args, **kwargs)
        phasor_im = dolfin.Expression("sin(alpha*x[0] + beta*x[1])", *args, **kwargs)
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
        self.inv_mu_coeff = _invert_3by3_complex_matrix(self.mu_coeff)
        self.inv_mu_coeff_annex = _invert_3by3_complex_matrix(self.mu_coeff_annex)

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
        delta_epsilon = self.epsilon_coeff - self.epsilon_coeff_annex
        delta_inv_mu = self.inv_mu_coeff - self.inv_mu_coeff_annex

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

        L = (
            -dot(self.inv_mu_coeff * curl(Etrial), curl(Etest)) * dx,
            -1j
            * (
                dot(self.inv_mu_coeff * cross(self.k_para, Etrial), curl(Etest))
                - dot(self.inv_mu_coeff * cross(self.k_para, Etest), curl(Etrial))
            )
            * dx,
            -dot(
                cross(self.inv_mu_coeff * cross(self.k_para, Etrial), self.k_para),
                Etest,
            )
            * dx,
            dot(self.epsilon_coeff * Etrial, Etest) * dx,
        )
        b = (
            dot(delta_inv_mu * curl(self.Estack_coeff), curl(Etest))
            * self.phasor.conj
            * dx(self.source_dom),
            -dot(delta_epsilon * self.Estack_coeff, Etest)
            * self.phasor.conj
            * dx(self.source_dom),
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
        Efunc = self.E.real
        Ah = self.Ah[0] + self.k0 ** 2 * self.Ah[3]
        bh = self.bh[0] + self.k0 ** 2 * self.bh[1]
        if self.alpha0 ** 2 + self.beta0 ** 2 != 0:
            Ah += self.Ah[1] + self.Ah[2]
        for bc in self.boundary_conditions:
            bc.apply(Ah, bh)

        if direct:
            # solver = dolfin.LUSolver(Ah) ### direct
            solver = dolfin.PETScLUSolver("mumps")
            solver.parameters.update(lu_params)
            solver.solve(Ah, Efunc.vector(), bh)
        else:
            solver = dolfin.PETScKrylovSolver()  ## iterative
            solver.parameters.update(krylov_params)
            solver.solve(Ah, Efunc.vector(), bh)

        self.Eper = self.E
        self.E *= self.phasor

    def _get_slice(self, u, x_slice, y_slice, z):
        re, im = [], []
        for x_ in x_slice:
            re_y, im_y = [], []
            for y_ in y_slice:
                f = u((x_, y_, z))
                re_y.append(f.real)
                im_y.append(f.imag)
            re.append(re_y)
            im.append(im_y)
        return np.array(re) + 1j * np.array(im)

    def _interp_cplx_vect(self, u, method="linear"):
        if method == "nearest":
            Interp = NearestNDInterpolator
        else:
            Interp = LinearNDInterpolator
        # x, y, z = self.complex_space.tabulate_dof_coordinates().T
        # if type(u.real) == ufl.tensors.ListTensor:
        V_cg = dolfin.VectorFunctionSpace(self.mesh, "CG", self.degree)
        x, y, z = V_cg.tabulate_dof_coordinates().T
        points = x[::3], y[::3], z[::3]
        u = interpolate(u, V_cg)
        Enp_re = u.real.vector().get_local()
        Ex_re, Ey_re, Ez_re = Enp_re[::3], Enp_re[1::3], Enp_re[2::3]
        Enp_im = u.imag.vector().get_local()
        Ex_im, Ey_im, Ez_im = Enp_im[::3], Enp_im[1::3], Enp_im[2::3]

        interpolator = []
        for ere, eim in zip((Ex_re, Ey_re, Ez_re), (Ex_im, Ey_im, Ez_im)):
            interpolator.append(Interp(np.array(points).T, ere + 1j * eim))

        return interpolator

    def diffraction_efficiencies(self):
        npt_integ = self.ninterv_integ
        # print('gmsh cuts done !')
        period_x, period_y = self.geom.period
        N_d_order = self.N_d_order
        nb_slice = self.nb_slice
        x_t = x_r = np.linspace(-period_x / 2, period_x / 2, npt_integ)
        y_t = y_r = np.linspace(-period_y / 2, period_y / 2, npt_integ)
        order_shift = 0
        No_ordre = np.linspace(
            -self.N_d_order + order_shift,
            self.N_d_order + order_shift,
            2 * self.N_d_order + 1,
        )
        Nb_order = No_ordre.shape[0]
        alphat = self.alpha0 + 2 * np.pi / period_x * No_ordre
        betat = self.beta0 + 2 * np.pi / period_y * No_ordre
        gammatt = np.zeros((Nb_order, Nb_order), dtype=complex)
        gammatr = np.zeros((Nb_order, Nb_order), dtype=complex)
        AXsir = np.zeros((Nb_order, Nb_order, nb_slice), dtype=complex)
        AXsit = np.zeros((Nb_order, Nb_order, nb_slice), dtype=complex)

        self.scan_dist = (
            min(
                self.geom.thicknesses["substrate"], self.geom.thicknesses["superstrate"]
            )
            / self.scan_dist_ratio
        )

        self.zcut = dict()
        for k in ["substrate", "superstrate"]:
            self.zcut[k] = (
                self.geom.z_position[k] + self.scan_dist,
                (self.geom.z_position[k] + self.geom.thicknesses[k] - self.scan_dist),
            )

        z_slice_t = np.linspace(*self.zcut["substrate"], self.nb_slice)
        z_slice_r = np.linspace(*self.zcut["superstrate"], self.nb_slice)

        nb_layer_diopter = 2
        layer_diopter = []
        for k1 in range(0, nb_layer_diopter):
            layer_diopter.append({})
        layer_diopter[0]["epsilon"] = self.epsilon["superstrate"]
        layer_diopter[1]["epsilon"] = self.epsilon["substrate"]
        layer_diopter[0]["kp"] = self.k0 * np.sqrt(complex(layer_diopter[0]["epsilon"]))
        layer_diopter[1]["kp"] = self.k0 * np.sqrt(complex(layer_diopter[1]["epsilon"]))
        layer_diopter[0]["gamma"] = np.sqrt(
            layer_diopter[0]["kp"] ** 2 - self.alpha0 ** 2 - self.beta0 ** 2
        )
        layer_diopter[1]["gamma"] = np.sqrt(
            complex(layer_diopter[1]["kp"] ** 2 - self.alpha0 ** 2 - self.beta0 ** 2)
        )

        for nt in range(0, Nb_order):
            for mt in range(0, Nb_order):
                gammatt[nt, mt] = np.sqrt(
                    complex(
                        layer_diopter[-1]["kp"] ** 2 - alphat[nt] ** 2 - betat[mt] ** 2
                    )
                )
        for nr in range(0, Nb_order):
            for mr in range(0, Nb_order):
                gammatr[nr, mr] = np.sqrt(
                    complex(
                        layer_diopter[0]["kp"] ** 2 - alphat[nr] ** 2 - betat[mr] ** 2
                    )
                )

        Eper_interp = self._interp_cplx_vect(self.Eper)

        X_t, Y_t = np.meshgrid(x_t, y_t)
        phasor_slice = np.exp(1j * (self.alpha0 * X_t + self.beta0 * Y_t))

        Et = []

        for comp in range(3):
            eslice = []
            phiplus = self.Phi[-1][::2]
            phiminus = self.Phi[-1][1::2]
            for z in z_slice_t:
                points = X_t, Y_t, z * np.ones_like(X_t)
                prp, prm = (
                    np.exp(1j * layer_diopter[1]["gamma"] * z),
                    np.exp(-1j * layer_diopter[1]["gamma"] * z),
                )
                eper = Eper_interp[comp](points)
                p = phiplus[comp] * prp + phiminus[comp] * prm
                ecomp = (eper + p) * phasor_slice
                eslice.append(ecomp)
            Et.append(eslice)

        Er = []

        for comp in range(3):
            eslice = []
            phiplus = self.Phi[0][::2]
            phiminus = self.Phi[0][1::2]
            phi0 = self.Phi0[::2]
            for z in z_slice_r:
                points = X_t, Y_t, z * np.ones_like(X_t)
                prp, prm = (
                    np.exp(1j * layer_diopter[0]["gamma"] * z),
                    np.exp(-1j * layer_diopter[0]["gamma"] * z),
                )
                eper = Eper_interp[comp](points)
                p = phiplus[comp] * prp + phiminus[comp] * prm
                p0 = phi0[comp] * prp
                ecomp = (eper + p - p0) * phasor_slice
                eslice.append(ecomp)
            Er.append(eslice)

        Er = np.array(Er)
        Et = np.array(Et)
        # Ex_r2, Ey_r2, Ez_r2 = np.array(Er).reshape(3, npt_integ, npt_integ, nb_slice)
        # Ex_t2, Ey_t2, Ez_t2 = np.array(Et).reshape(3, npt_integ, npt_integ, nb_slice)

        for k11 in range(0, nb_slice):
            Ex_t3 = Et[0, k11, :, :].conj().T
            Ey_t3 = Et[1, k11, :, :].conj().T
            Ez_t3 = Et[2, k11, :, :].conj().T
            Ex_r3 = Er[0, k11, :, :].conj().T
            Ey_r3 = Er[1, k11, :, :].conj().T
            Ez_r3 = Er[2, k11, :, :].conj().T

            ex_nm_r_inter = np.zeros((1, npt_integ), dtype=complex)[0, :]
            ex_nm_t_inter = np.zeros((1, npt_integ), dtype=complex)[0, :]
            ey_nm_r_inter = np.zeros((1, npt_integ), dtype=complex)[0, :]
            ey_nm_t_inter = np.zeros((1, npt_integ), dtype=complex)[0, :]
            ez_nm_r_inter = np.zeros((1, npt_integ), dtype=complex)[0, :]
            ez_nm_t_inter = np.zeros((1, npt_integ), dtype=complex)[0, :]
            ex_nm_r = np.zeros((Nb_order, Nb_order), dtype=complex)
            ex_nm_t = np.zeros((Nb_order, Nb_order), dtype=complex)
            ey_nm_r = np.zeros((Nb_order, Nb_order), dtype=complex)
            ey_nm_t = np.zeros((Nb_order, Nb_order), dtype=complex)
            ez_nm_r = np.zeros((Nb_order, Nb_order), dtype=complex)
            ez_nm_t = np.zeros((Nb_order, Nb_order), dtype=complex)

            for n1 in range(0, Nb_order):
                for m1 in range(0, Nb_order):
                    for j1 in range(0, npt_integ):
                        expbeta = np.exp(1j * betat[m1] * y_r)
                        # ex_nm_r_inter[j1] = 1/period_y * np.trapz((Ex_r2[:,j1,k11])*expbeta,x=y_r)
                        ex_nm_r_inter[j1] = (
                            1 / period_y * np.trapz((Ex_r3[:, j1]) * expbeta, x=y_r)
                        )
                        # plt.plot np.trapz(y_t,(Ex_t[::-1,j1].transpose()*expbeta).conjugate()[::-1])
                    expalpha = np.exp(1j * alphat[n1] * x_t)
                    ex_nm_r[n1, m1] = (
                        1 / period_x * np.trapz(ex_nm_r_inter * expalpha, x=x_r)
                    )
            for n2 in range(0, Nb_order):
                for m2 in range(0, Nb_order):
                    for j1 in range(0, npt_integ):
                        expbeta = np.exp(1j * betat[m2] * y_t)
                        # ex_nm_t_inter[j1] = 1/period_y * np.trapz((Ex_t2[:,j1,k11])*expbeta,x=y_t)
                        ex_nm_t_inter[j1] = (
                            1 / period_y * np.trapz((Ex_t3[:, j1]) * expbeta, x=y_t)
                        )
                    expalpha = np.exp(1j * alphat[n2] * x_t)
                    ex_nm_t[n2, m2] = (
                        1 / period_x * np.trapz(ex_nm_t_inter * expalpha, x=x_t)
                    )
            for n3 in range(0, Nb_order):
                for m3 in range(0, Nb_order):
                    for j1 in range(0, npt_integ):
                        expbeta = np.exp(1j * betat[m3] * y_r)
                        # ey_nm_r_inter[j1] = 1/period_y * np.trapz((Ey_r2[:,j1,k11])*expbeta,x=y_r)
                        ey_nm_r_inter[j1] = (
                            1 / period_y * np.trapz((Ey_r3[:, j1]) * expbeta, x=y_r)
                        )
                    expalpha = np.exp(1j * alphat[n3] * x_t)
                    ey_nm_r[n3, m3] = (
                        1 / period_x * np.trapz(ey_nm_r_inter * expalpha, x=x_r)
                    )
            for n4 in range(0, Nb_order):
                for m4 in range(0, Nb_order):
                    for j1 in range(0, npt_integ):
                        expbeta = np.exp(1j * betat[m4] * y_t)
                        # ey_nm_t_inter[j1] = 1/period_y * np.trapz((Ey_t2[:,j1,k11])*expbeta,x=y_t)
                        ey_nm_t_inter[j1] = (
                            1 / period_y * np.trapz((Ey_t3[:, j1]) * expbeta, x=y_t)
                        )
                    expalpha = np.exp(1j * alphat[n4] * x_t)
                    ey_nm_t[n4, m4] = (
                        1 / period_x * np.trapz(ey_nm_t_inter * expalpha, x=x_t)
                    )
            for n6 in range(0, Nb_order):
                for m6 in range(0, Nb_order):
                    for j1 in range(0, npt_integ):
                        expbeta = np.exp(1j * betat[m6] * y_r)
                        # ez_nm_r_inter[j1] = 1/period_y * np.trapz((Ez_r2[:,j1,k11])*expbeta,x=y_r)
                        ez_nm_r_inter[j1] = (
                            1 / period_y * np.trapz((Ez_r3[:, j1]) * expbeta, x=y_r)
                        )
                    expalpha = np.exp(1j * alphat[n6] * x_t)
                    ez_nm_r[n6, m6] = (
                        1 / period_x * np.trapz(ez_nm_r_inter * expalpha, x=x_r)
                    )
            for n7 in range(0, Nb_order):
                for m7 in range(0, Nb_order):
                    for j1 in range(0, npt_integ):
                        expbeta = np.exp(1j * betat[m7] * y_t)
                        # ez_nm_t_inter[j1] = 1/period_y * np.trapz((Ez_t2[:,j1,k11])*expbeta,x=y_t)
                        ez_nm_t_inter[j1] = (
                            1 / period_y * np.trapz((Ez_t3[:, j1]) * expbeta, x=y_t)
                        )
                    expalpha = np.exp(1j * alphat[n7] * x_t)
                    ez_nm_t[n7, m7] = (
                        1 / period_x * np.trapz(ez_nm_t_inter * expalpha, x=x_t)
                    )
            ####################
            for n8 in range(0, Nb_order):
                for m8 in range(0, Nb_order):
                    AXsit[n8, m8, k11] = (
                        1
                        / (layer_diopter[0]["gamma"] * gammatt[n8, m8])
                        * (
                            +gammatt[n8, m8] ** 2 * np.abs(ex_nm_t[n8, m8]) ** 2
                            + gammatt[n8, m8] ** 2 * np.abs(ey_nm_t[n8, m8]) ** 2
                            + gammatt[n8, m8] ** 2 * np.abs(ez_nm_t[n8, m8]) ** 2
                        )
                    )
            for n9 in range(0, Nb_order):
                for m9 in range(0, Nb_order):
                    AXsir[n9, m9, k11] = (
                        1
                        / (layer_diopter[0]["gamma"] * gammatr[n9, m9])
                        * (
                            +gammatr[n9, m9] ** 2 * np.abs(ex_nm_r[n9, m9]) ** 2
                            + gammatr[n9, m9] ** 2 * np.abs(ey_nm_r[n9, m9]) ** 2
                            + gammatr[n9, m9] ** 2 * np.abs(ez_nm_r[n9, m9]) ** 2
                        )
                    )
        Q = self.postpro_absorption()
        Tnm = np.mean(AXsit, axis=2).real
        Rnm = np.mean(AXsir, axis=2).real
        # energy = dict([('trans', Tnm), ('refl', Rnm), ('abs1', Q),
        #                ('refl_slices', AXsir), ('trans_slices', AXsit)])
        balance = np.sum(np.sum(Tnm)) + np.sum(np.sum(Rnm)) + Q
        effs = dict([("T", Tnm), ("R", Rnm), ("Q", Q), ("B", balance)])
        return effs

    def postpro_absorption(self):
        return 0


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
    island_bottom = model.addRectangle(
        -island_width_bottom / 2,
        -island_width_bottom / 2,
        z0,
        island_width_bottom,
        island_width_bottom,
    )
    island_top = model.addRectangle(
        -island_width_top / 2,
        -island_width_top / 2,
        z0 + island_thickness,
        island_width_top,
        island_width_top,
    )

    island = model.addThruSections([island_bottom, island_top])
    island, groove = model.fragmentize(island[-1][-1], groove)
    model.removeAllDuplicates()
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
    fplot = g.E  # + g.Estack_coeff
    fplot = abs(g.Eper)
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
