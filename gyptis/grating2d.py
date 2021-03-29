#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

__all__ = ["Layered2D", "Grating2D", "OrderedDict"]

import glob
import os
from collections import OrderedDict

from . import ADJOINT, dolfin
from .bc import PeriodicBoundary2DX
from .complex import *
from .formulation import Maxwell2DPeriodic
from .geometry import *
from .helpers import _translation_matrix
from .materials import *
from .simulation import Simulation
from .source import *


class Layered2D(Geometry):
    def __init__(
        self,
        period=1,
        thicknesses=OrderedDict(),
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
        # assert isinstance(self.thicknesses == OrderedDict)
        self.thicknesses = thicknesses

        self.layers = list(thicknesses.keys())

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

        self.remove_all_duplicates()
        self.synchronize()

        for sub, num in self.subdomains["surfaces"].items():
            self.add_physical(num, sub)

    @property
    def translation_x(self):
        return _translation_matrix([self.period, 0, 0])

    def make_layer(self, y_position, thickness):
        box = self.add_rectangle(
            -self.period / 2, y_position, 0, self.period, thickness
        )
        return box

    def build(self, *args, **kwargs):

        s = self.get_periodic_bnds(self.y0, self.total_thickness)
        periodic_id = {}
        for k, v in s.items():
            periodic_id[k] = [S[-1] for S in v]
        gmsh.model.mesh.setPeriodic(
            1, periodic_id["+x"], periodic_id["-x"], self.translation_x
        )
        super().build(*args, **kwargs)

    def get_periodic_bnds(self, y_position, thickness, eps=1e-3):
        s = {}

        pmin = -self.period / 2 - eps, -eps + y_position, -eps
        pmax = -self.period / 2 + eps, y_position + thickness + eps, eps
        s["-x"] = gmsh.model.getEntitiesInBoundingBox(*pmin, *pmax, 1)

        pmin = +self.period / 2 - eps, -eps + y_position, -eps
        pmax = +self.period / 2 + eps, y_position + thickness + eps, eps
        s["+x"] = gmsh.model.getEntitiesInBoundingBox(*pmin, *pmax, 1)

        return s


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
        orders_num = np.linspace(
            -N_d_order,
            N_d_order,
            2 * N_d_order + 1,
        )

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
        phase=0,
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
            f = u * phase_shift(i * alpha * self.period + phase, degree=self.degree)
            fplot = f.real
            if ADJOINT:
                fplot = project(fplot, self.formulation.real_function_space)
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

    def animate_field(self, n=11, filename="animation.gif", **kwargs):
        import tempfile

        from PIL import Image

        from .plot import plt

        anim = []
        tmpdir = tempfile.mkdtemp()
        fp_in = f"{tmpdir}/animation_tmp_*.png"
        phase = np.linspace(0, 2 * np.pi, n + 1)[:n]
        for iplot in range(n):
            number_str = str(iplot).zfill(4)
            pngname = f"{tmpdir}/animation_tmp_{number_str}.png"
            p = self.plot_field(phase=phase[iplot], **kwargs)
            fig = plt.gcf()
            fig.savefig(pngname)
            fig.clear()
            anim.append(p)

        plt.close(fig)

        img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
        img.save(
            fp=filename,
            format="GIF",
            append_images=imgs,
            save_all=True,
            duration=200,
            loop=0,
        )
        os.system(f"rm -f {tmpdir}/animation_tmp_*.png")
        return anim
