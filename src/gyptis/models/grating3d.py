#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# License: MIT
# See the documentation at gyptis.gitlab.io

from .metaclasses import _GratingBase
from .simulation import *


class Grating3D(_GratingBase, Simulation):
    def __init__(
        self,
        geometry,
        epsilon,
        mu,
        source,
        boundary_conditions={},
        polarization=None,
        degree=1,
        pml_stretch=1 - 1j,
        periodic_map_tol=1e-8,
        eps_bc=1e-8,
    ):
        assert isinstance(geometry, Layered3D)
        assert source.dim == 3

        self.period = geometry.period
        self.epsilon = epsilon
        self.mu = mu
        self.degree = degree
        self.periodic_map_tol = periodic_map_tol
        self.eps_bc = eps_bc
        self.periodic_bcs = BiPeriodicBoundary3D(
            self.period,
            map_tol=self.periodic_map_tol,
            eps=self.eps_bc,
        )
        function_space = ComplexFunctionSpace(
            geometry.mesh, "N1curl", degree, constrained_domain=self.periodic_bcs
        )
        pml_bottom = PML(
            "z",
            stretch=pml_stretch,
            matched_domain="substrate",
            applied_domain="pml_bottom",
        )
        pml_top = PML(
            "z",
            stretch=pml_stretch,
            matched_domain="superstrate",
            applied_domain="pml_top",
        )

        epsilon_coeff = Coefficient(
            epsilon,
            geometry,
            pmls=[pml_bottom, pml_top],
            degree=degree,
            dim=3,
        )
        mu_coeff = Coefficient(
            mu, geometry, pmls=[pml_bottom, pml_top], degree=degree, dim=3
        )

        coefficients = epsilon_coeff, mu_coeff
        no_source_domains = ["substrate", "superstrate", "pml_bottom", "pml_top"]
        source_domains = [
            dom for dom in geometry.domains if dom not in no_source_domains
        ]

        formulation = Maxwell3DPeriodic(
            geometry,
            coefficients,
            function_space,
            source=source,
            source_domains=source_domains,
            reference="superstrate",
            boundary_conditions=boundary_conditions,
        )

        super().__init__(geometry, formulation)

    def solve_system(self, again=False):
        uper = super().solve_system(again=again, vector_function=False)
        u_annex = self.formulation.annex_field["as_subdomain"]["stack"]
        u = uper * self.formulation.phasor
        self.solution = {}
        self.solution["periodic"] = uper
        self.solution["diffracted"] = u
        self.solution["total"] = u + u_annex
        return u

    def diffraction_efficiencies(
        self,
        N_d_order=0,
        cplx_effs=False,
        orders=False,
        subdomain_absorption=False,
        verbose=False,
    ):
        orders_num = np.linspace(
            -N_d_order,
            N_d_order,
            2 * N_d_order + 1,
        )

        k, gamma = {}, {}
        for d in ["substrate", "superstrate"]:
            k[d] = self.source.wavenumber * np.sqrt(
                complex(self.epsilon[d] * self.mu[d])
            )
            gamma[d] = np.conj(
                np.sqrt(
                    k[d] ** 2
                    - self.formulation.propagation_vector[0] ** 2
                    - self.formulation.propagation_vector[1] ** 2
                )
            )

        Phi = self.formulation.annex_field["phi"]
        r_annex = Phi[0][1::2]
        t_annex = Phi[-1][::2]
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
                qn = n * 2 * np.pi / self.period[0]
                pm = m * 2 * np.pi / self.period[1]
                alpha_n = self.formulation.propagation_vector[0] + qn
                beta_m = self.formulation.propagation_vector[1] + pm
                efficiencies = {}
                efficiencies_complex = {}
                for d in ["substrate", "superstrate"]:
                    s = 1 if d == "superstrate" else -1
                    # s = 1 if d == "substrate" else -1
                    gamma_nm = np.sqrt(k[d] ** 2 - alpha_n ** 2 - beta_m ** 2)
                    ph_x = phasor(
                        -qn, direction=0, degree=self.degree, domain=self.mesh
                    )
                    ph_y = phasor(
                        -pm, direction=1, degree=self.degree, domain=self.mesh
                    )
                    ph_z = phasor(
                        s * gamma_nm.real,
                        direction=2,
                        degree=self.degree,
                        domain=self.mesh,
                    )
                    ph_xy = ph_x * ph_y
                    Jnm = []
                    for comp in range(3):
                        Jnm.append(
                            assemble(Eper[comp] * ph_xy * ph_z * self.dx(d))
                            / (self.period[0] * self.period[1])
                        )
                    ph_pos = np.exp(-s * 1j * gamma_nm * self.geometry.z_position[d])
                    eff, sqnorm_eff = [], 0
                    for comp in range(3):
                        eff_ = (
                            delta * eff_annex[d][comp]
                            + Jnm[comp] / self.geometry.thicknesses[d]
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

        t_nm = [[e["substrate"] for e in b] for b in effn_cplx]
        r_nm = [[e["superstrate"] for e in b] for b in effn_cplx]

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
            * (np.cos(self.source.angle[0]))
            / 2
        )
        doms_no_pml = [
            z for z in self.epsilon.keys() if z not in ["pml_bottom", "pml_top"]
        ]
        Etot = self.solution["total"]
        # curl E = i ω μ_0 μ H
        inv_mu = self.formulation.mu.invert().as_subdomain()
        Htot = inv_mu / (1j * self.source.pulsation * mu_0) * curl(Etot)
        Qelec, Qmag = {}, {}
        if subdomain_absorption:
            for d in doms_no_pml:
                if np.all(self.epsilon[d].imag) == 0:
                    Qelec[d] = 0
                else:
                    # Etot = (
                    #     self.formulation.annex_field["as_dict"]["stack"][d]
                    #     + self.solution["diffracted"]
                    # )
                    elec_nrj_dens = dolfin.Constant(
                        0.5 * epsilon_0 * self.source.pulsation
                    ) * dot(self.epsilon[d] * Etot, Etot.conj)
                    Qelec[d] = -assemble(elec_nrj_dens * self.dx(d)).imag / P0
                if np.all(self.mu[d].imag) == 0:
                    Qmag[d] = 0
                else:
                    mag_nrj_dens = dolfin.Constant(
                        0.5 * mu_0 * self.source.pulsation
                    ) * dot(self.mu[d] * Htot, Htot.conj)
                    Qmag[d] = -assemble(mag_nrj_dens * self.dx(d)).imag / P0
            Q = sum(Qelec.values()) + sum(Qmag.values())
        else:
            epsilon_coeff = self.formulation.epsilon.as_subdomain()
            mu_coeff = self.formulation.mu.as_subdomain()
            elec_nrj_dens = dot(epsilon_coeff * Etot, Etot.conj)
            Qelec = (
                -0.5
                * epsilon_0
                * self.source.pulsation
                * assemble(elec_nrj_dens * self.dx(doms_no_pml))
                / P0
            ).imag
            mag_nrj_dens = dot(mu_coeff * Htot, Htot.conj)
            Qmag = (
                -0.5
                * mu_0
                * self.source.pulsation
                * assemble(mag_nrj_dens * self.dx(doms_no_pml))
                / P0
            ).imag
            Q = Qelec + Qmag
        Qdomains = {"electric": Qelec, "magnetic": Qmag}
        self.Qtot = Q
        self.Qdomains = Qdomains
        return Q, Qdomains
