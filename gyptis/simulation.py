#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

from . import ADJOINT, dolfin
from .bc import *
from .complex import *
from .formulation import *
from .geometry import *
from .grating2d import Layered2D
from .grating3d import Layered3D
from .materials import *
from .source import *


class Simulation:
    def __init__(self, geometry, formulation=None):
        self.geometry = geometry
        self.formulation = formulation
        self.coefficients = formulation.coefficients
        self.function_space = formulation.function_space
        self._source = formulation.source
        self.boundary_conditions = formulation.boundary_conditions
        self.mesh = self.geometry.mesh
        self._boundary_conditions = []
        self.dx = formulation.dx

    @property
    def source(self):
        return self._source

    @source.setter
    def source(self, value):
        self._source = value

    def assemble_lhs(self, **kwargs):
        self.formulation.build_lhs(**kwargs)
        self.matrix = assemble(self.formulation.lhs)
        return self.matrix

    def assemble_rhs(self, **kwargs):
        self.formulation.build_rhs(**kwargs)
        self.vector = assemble(self.formulation.rhs)
        return self.vector

    def assemble(self, **kwargs):
        self.matrix = self.assemble_lhs(**kwargs)
        self.vector = self.assemble_rhs(**kwargs)
        return self.matrix, self.vector

    def apply_boundary_conditions(self):
        bcs = self.formulation.build_boundary_conditions()
        for bc in bcs:
            bc.apply(self.matrix, self.vector)

    def solve_system(self, again=False, vector_function=True):
        if vector_function:
            element = self.function_space.split()[0].ufl_element()
            V_vect = dolfin.VectorFunctionSpace(
                self.mesh, element.family(), element.degree()
            )
            u = dolfin.Function(V_vect)
        else:
            u = dolfin.Function(self.function_space)

        if not again:
            self.solver = dolfin.LUSolver(self.matrix, "mumps")

        self.solver.solve(u.vector(), self.vector)
        solution = Complex(*u.split())
        return solution

    def solve(self, again=False, **kwargs):
        self.assemble(**kwargs)
        self.apply_boundary_conditions()
        return self.solve_system(again=again, **kwargs)


class Scatt2D(Simulation):
    def __init__(
        self,
        geometry,
        epsilon,
        mu,
        source,
        boundary_conditions={},
        polarization="TE",
        degree=1,
        mat_degree=1,
        pml_stretch=1 - 1j,
    ):
        assert isinstance(geometry, BoxPML2D)
        assert source.dim == 2
        function_space = ComplexFunctionSpace(geometry.mesh, "CG", degree)
        pmlx = PML(
            "x", stretch=pml_stretch, matched_domain="box", applied_domain="pmlx"
        )
        pmly = PML(
            "y", stretch=pml_stretch, matched_domain="box", applied_domain="pmly"
        )
        pmlxy = PML(
            "xy", stretch=pml_stretch, matched_domain="box", applied_domain="pmlxy"
        )

        epsilon_coeff = Coefficient(
            epsilon, geometry, pmls=[pmlx, pmly, pmlxy], degree=mat_degree
        )
        mu_coeff = Coefficient(
            mu, geometry, pmls=[pmlx, pmly, pmlxy], degree=mat_degree
        )

        coefficients = epsilon_coeff, mu_coeff
        no_source_domains = ["box", "pmlx", "pmly", "pmlxy"]
        source_domains = [
            dom for dom in geometry.domains if dom not in no_source_domains
        ]

        formulation = Maxwell2D(
            geometry,
            coefficients,
            function_space,
            source=source,
            source_domains=source_domains,
            reference="box",
            polarization=polarization,
            boundary_conditions=boundary_conditions,
        )

        super().__init__(geometry, formulation)

    def solve_system(self, again=False):
        u = super().solve_system(again=again)
        self.solution = {}
        self.solution["diffracted"] = u
        self.solution["total"] = u + self.source.expression
        return u

    # def local_density_of_states(self, x, y):
    #     ldos = np.zeros((len(x), len(y)))
    #     for ix, x_ in enumerate(x):
    #         for iy, y_ in enumerate(y):
    #             print(x_, y_)
    #             self.source.position = x_, y_
    #             self.assemble_rhs()
    #             u = self.solve_system(again=True)
    #             ldos[ix, iy] = u(self.source.position).imag
    #     return ldos


class Scatt3D(Simulation):
    def __init__(
        self,
        geometry,
        epsilon,
        mu,
        source,
        boundary_conditions={},
        degree=1,
        mat_degree=1,
        pml_stretch=1 - 1j,
    ):
        assert isinstance(geometry, BoxPML3D)
        assert source.dim == 3
        function_space = ComplexFunctionSpace(geometry.mesh, "N1curl", degree)
        pmls = []
        pml_names = []
        for direction in ["x", "y", "z", "xy", "yz", "xz", "xyz"]:
            pml_name = f"pml{direction}"
            pml_names.append(pml_name)
            pmls.append(
                PML(
                    direction,
                    stretch=pml_stretch,
                    matched_domain="box",
                    applied_domain=pml_name,
                )
            )

        epsilon_coeff = Coefficient(
            epsilon, geometry, pmls=pmls, degree=mat_degree, dim=3,
        )
        mu_coeff = Coefficient(mu, geometry, pmls=pmls, degree=mat_degree, dim=3,)

        coefficients = epsilon_coeff, mu_coeff
        no_source_domains = ["box"] + pml_names
        source_domains = [
            dom for dom in geometry.domains if dom not in no_source_domains
        ]

        formulation = Maxwell3D(
            geometry,
            coefficients,
            function_space,
            source=source,
            source_domains=source_domains,
            reference="box",
            boundary_conditions=boundary_conditions,
        )

        super().__init__(geometry, formulation)

    def solve_system(self, again=False):
        E = super().solve_system(again=again, vector_function=False)
        self.solution = {}
        self.solution["diffracted"] = E
        self.solution["total"] = E + self.source.expression
        return E


class Grating2D(Simulation):
    def __init__(
        self,
        geometry,
        epsilon,
        mu,
        source,
        boundary_conditions={},
        polarization="TE",
        degree=1,
        mat_degree=1,
        pml_stretch=1 - 1j,
    ):
        assert isinstance(geometry, Layered2D)
        assert isinstance(source, PlaneWave)
        assert source.dim == 2
        self.epsilon = epsilon
        self.mu = mu
        self.degree = degree
        self.period = geometry.period
        self.periodic_bcs = PeriodicBoundary2DX(self.period)
        function_space = ComplexFunctionSpace(
            geometry.mesh, "CG", degree, constrained_domain=self.periodic_bcs
        )
        pml_top = PML(
            "y",
            stretch=pml_stretch,
            matched_domain="superstrate",
            applied_domain="pml_top",
        )
        pml_bottom = PML(
            "y",
            stretch=pml_stretch,
            matched_domain="substrate",
            applied_domain="pml_bottom",
        )
        epsilon_coeff = Coefficient(
            epsilon, geometry=geometry, pmls=[pml_top, pml_bottom]
        )
        mu_coeff = Coefficient(mu, geometry=geometry, pmls=[pml_top, pml_bottom])
        coefficients = epsilon_coeff, mu_coeff
        no_source_domains = ["substrate", "superstrate", "pml_bottom", "pml_top"]
        source_domains = [
            dom for dom in geometry.domains if dom not in no_source_domains
        ]

        formulation = Maxwell2DPeriodic(
            geometry,
            coefficients,
            function_space,
            source=source,
            source_domains=source_domains,
            reference="superstrate",
            polarization=polarization,
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
        if self.formulation.polarization == "TE":
            nu = 1 / self.mu["substrate"]
        else:
            nu = 1 / self.epsilon["substrate"]
        orders_num = np.linspace(-N_d_order, N_d_order, 2 * N_d_order + 1,)

        k, beta = {}, {}
        for d in ["substrate", "superstrate"]:
            k[d] = self.source.wavenumber * np.sqrt(
                complex(self.epsilon[d] * self.mu[d])
            )
            beta[d] = np.conj(
                np.sqrt(k[d] ** 2 - self.formulation.propagation_vector[0] ** 2)
            )

        phi_stack = self.formulation.annex_field["phi"]
        r_annex = phi_stack[0][-1]
        t_annex = phi_stack[-1][0]
        eff_annex = dict(substrate=t_annex, superstrate=r_annex)
        r_n, t_n = [], []
        R_n, T_n = [], []
        for n in orders_num:
            delta = 1 if n == 0 else 0
            qn = n * 2 * np.pi / self.period
            alpha_n = self.formulation.propagation_vector[0] + qn
            Jn, beta_n, eff = {}, {}, {}
            for d in ["substrate", "superstrate"]:
                s = 1 if d == "superstrate" else -1
                beta_n[d] = np.sqrt(k[d] ** 2 - alpha_n ** 2)
                ph_x = phasor(-qn, direction=0, degree=self.degree, domain=self.mesh)
                ph_y = phasor(
                    s * beta_n[d].real,
                    direction=1,
                    degree=self.degree,
                    domain=self.mesh,
                )
                Jn[d] = (
                    assemble(self.solution["periodic"] * ph_x * ph_y * self.dx(d))
                    / self.period
                )

                ph_pos = np.exp(-s * 1j * beta_n[d] * self.geometry.y_position[d])
                eff[d] = (
                    delta * eff_annex[d] + Jn[d] / self.geometry.thicknesses[d]
                ) * ph_pos
            r_, t_ = eff["superstrate"], eff["substrate"]
            sigma_r = beta_n["superstrate"] / beta["superstrate"]
            R_n_ = (r_ * r_.conj) * sigma_r
            sigma_t = beta_n["substrate"] / beta["superstrate"] * nu
            T_n_ = (t_ * t_.conj) * sigma_t

            r_n.append(r_)
            t_n.append(t_)
            R_n.append(R_n_.real)
            T_n.append(T_n_.real)

        R = np.sum(R_n, axis=0)
        T = np.sum(T_n, axis=0)

        Q, Qdomains = self.compute_absorption(subdomain_absorption=subdomain_absorption)

        B = T + R + Q  # energy balance

        if verbose:
            print(f"  Energy balance")
            print(f"  R = {R:0.6f}")
            print(f"  T = {T:0.6f}")
            print(f"  Q = {Q:0.6f}")
            print(f"  ------------------------")
            print(f"  B = {B:0.6f}")

        eff = dict()
        eff["R"] = r_n if cplx_effs else (R_n if orders else R)
        eff["T"] = t_n if cplx_effs else (T_n if orders else T)
        eff["Q"] = Qdomains if subdomain_absorption else Q
        eff["B"] = B
        self.efficiencies = eff
        return eff

    def compute_absorption(self, subdomain_absorption=False):
        omega = self.source.pulsation
        doms_no_pml = [
            z for z in self.epsilon.keys() if z not in ["pml_bottom", "pml_top"]
        ]

        xi_0, chi_0 = (
            (mu_0, epsilon_0)
            if self.formulation.polarization == "TE"
            else (epsilon_0, mu_0)
        )

        P0 = 0.5 * np.sqrt(chi_0 / xi_0) * np.sin(self.source.angle) * self.period

        Qchi = {}
        Qxi = {}
        u_tot = self.solution["total"]
        if subdomain_absorption:
            chi = self.formulation.chi.as_property()
            xi = self.formulation.xi.as_property()
            for d in doms_no_pml:

                nrj_chi_dens = (
                    dolfin.Constant(-0.5 * chi_0 * omega) * chi[d] * abs(u_tot) ** 2
                ).imag

                nrj_xi_dens = (
                    dolfin.Constant(-0.5 * 1 / (omega * xi_0))
                    * dot(grad(u_tot), (xi[d] * grad(u_tot)).conj).imag
                )

                Qchi[d] = assemble(nrj_chi_dens * self.dx(d)) / P0
                Qxi[d] = assemble(nrj_xi_dens * self.dx(d)) / P0
            Q = sum(Qxi.values()) + sum(Qchi.values())
        else:
            chi = self.formulation.chi.as_subdomain()
            xi = self.formulation.xi.as_subdomain()
            nrj_chi_dens = (
                dolfin.Constant(-0.5 * chi_0 * omega) * chi * abs(u_tot) ** 2
            ).imag

            nrj_xi_dens = (
                dolfin.Constant(-0.5 * 1 / (omega * xi_0))
                * dot(grad(u_tot), (xi * grad(u_tot)).conj).imag
            )
            Qchi = assemble(nrj_chi_dens * self.dx(doms_no_pml)) / P0
            Qxi = assemble(nrj_xi_dens * self.dx(doms_no_pml)) / P0
            Q = Qxi + Qchi

        if self.formulation.polarization == "TE":
            Qdomains = {"electric": Qchi, "magnetic": Qxi}
        else:
            Qdomains = {"electric": Qxi, "magnetic": Qchi}

        self.Qtot = Q
        self.Qdomains = Qdomains
        return Q, Qdomains

    def plot_geometry(self, nper=1, ax=None, **kwargs):
        from .plot import plot_subdomains, plt

        if ax == None:
            ax = plt.gca()
        domains = self.geometry.subdomains["surfaces"]
        scatt = []
        for d in domains:
            if d not in self.geometry.layers:
                scatt.append(d)
        scatt_ids = [domains[d] for d in scatt]
        scatt_lines = []
        if len(scatt_ids) > 0:
            for i in range(nper):
                s = plot_subdomains(
                    self.geometry.markers,
                    domain=scatt_ids,
                    shift=(i * self.period, 0),
                    **kwargs,
                )
                scatt_lines.append(s)
        yend = list(self.geometry.thicknesses.values())[-1]
        layers_lines = []
        for y0 in self.geometry.y_position.values():
            a = ax.axhline(y0, **kwargs)
            layers_lines.append(a)
        y0 += list(self.geometry.thicknesses.values())[-1]
        a = ax.axhline(y0, **kwargs)
        layers_lines.append(a)
        ax.set_aspect(1)
        return scatt_lines, layers_lines

    def plot_field(
        self,
        nper=1,
        ax=None,
        mincmap=None,
        maxcmap=None,
        fig=None,
        anim_phase=0,
        callback=None,
        **kwargs,
    ):

        from matplotlib.transforms import Affine2D

        from .plot import plt

        u = self.solution["total"]
        if ax == None:
            ax = plt.gca()
        if "cmap" not in kwargs:
            kwargs["cmap"] = "RdBu_r"
        per_plots = []
        ppmin, ppmax = [], []
        for i in range(nper):
            alpha = self.formulation.propagation_vector[0]
            t = Affine2D().translate(i * self.period, 0)
            f = u * phase_shift(
                i * alpha * self.period + anim_phase, degree=self.degree
            )
            fplot = f.real
            if ADJOINT:
                fplot = project(fplot, self.real_space)
            pp = dolfin.plot(fplot, transform=t + ax.transData, **kwargs)
            per_plots.append(pp)

            ppmax.append(pp.cvalues[-1])
            ppmin.append(pp.cvalues[0])
        ax.set_xlim(-self.period / 2, (nper - 1 / 2) * self.period)
        ax.set_aspect(1)
        mincmap = mincmap or min(ppmin)
        maxcmap = maxcmap or max(ppmax)
        for pp in per_plots:
            pp.set_clim(mincmap, maxcmap)

        cm = plt.cm.ScalarMappable(cmap=kwargs["cmap"])
        cm.set_clim(mincmap, maxcmap)

        fig = plt.gcf() if fig is None else fig
        cb = fig.colorbar(cm, ax=ax)

        if callback is not None:
            callback(**kwargs)

        return per_plots, cb


class Grating3D(Simulation):
    def __init__(
        self,
        geometry,
        epsilon,
        mu,
        source,
        boundary_conditions={},
        degree=1,
        mat_degree=1,
        pml_stretch=1 - 1j,
        periodic_map_tol=1e-8,
    ):
        assert isinstance(geometry, Layered3D)
        assert source.dim == 3

        self.period = geometry.period
        self.epsilon = epsilon
        self.mu = mu
        self.degree = degree
        self.periodic_map_tol = periodic_map_tol
        self.periodic_bcs = BiPeriodicBoundary3D(
            self.period, map_tol=self.periodic_map_tol,
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
            epsilon, geometry, pmls=[pml_bottom, pml_top], degree=mat_degree, dim=3,
        )
        mu_coeff = Coefficient(
            mu, geometry, pmls=[pml_bottom, pml_top], degree=mat_degree, dim=3
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
        orders_num = np.linspace(-N_d_order, N_d_order, 2 * N_d_order + 1,)

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
            * (np.sin(self.source.angle[0]))
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
