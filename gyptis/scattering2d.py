#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT


import glob
import os

from . import ADJOINT, dolfin
from .complex import *
from .formulation import Maxwell2D
from .geometry import *
from .materials import *
from .simulation import Simulation
from .source import *


class BoxPML2D(Geometry):
    def __init__(
        self,
        box_size=(1, 1),
        box_center=(0, 0),
        pml_width=(0.2, 0.2),
        Rcalc=0,
        model_name="2D box with PMLs",
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            dim=2,
            **kwargs,
        )
        self.box_size = box_size
        self.box_center = box_center
        self.pml_width = pml_width

        def _addrect_center(rect_size):
            corner = -np.array(rect_size) / 2
            corner = tuple(corner) + (0,)
            return self.add_rectangle(*corner, *rect_size)

        def _translate(tag, t):
            translation = tuple(t) + (0,)
            self.translate(self.dimtag(tag), *translation)

        def _add_pml(s, t):
            pml = _addrect_center(s)
            _translate(pml, t)
            return pml

        box = _addrect_center(self.box_size)
        s = (self.pml_width[0], self.box_size[1])
        t = np.array([self.pml_width[0] / 2 + self.box_size[0] / 2, 0])
        pmlxp = _add_pml(s, t)
        pmlxm = _add_pml(s, -t)
        s = (self.box_size[0], self.pml_width[1])
        t = np.array([0, self.pml_width[1] / 2 + self.box_size[1] / 2])
        pmlyp = _add_pml(s, t)
        pmlym = _add_pml(s, -t)

        s = (self.pml_width[0], self.pml_width[1])
        t = np.array(
            [
                self.pml_width[0] / 2 + self.box_size[0] / 2,
                self.pml_width[1] / 2 + self.box_size[1] / 2,
            ]
        )
        pmlxypp = _add_pml(s, t)
        pmlxymm = _add_pml(s, -t)
        pmlxypm = _add_pml(s, (-t[0], t[1]))
        pmlxymp = _add_pml(s, (t[0], -t[1]))

        all_dom = [
            box,
            pmlxp,
            pmlxm,
            pmlyp,
            pmlym,
            pmlxypp,
            pmlxypm,
            pmlxymm,
            pmlxymp,
        ]
        _translate(all_dom, self.box_center)

        self.box = box
        self.pmls = all_dom[1:]

        self.fragment(self.box, self.pmls)

        if Rcalc > 0:
            cyl_calc = self.add_circle(*self.box_center, 0, Rcalc)
            box, cyl_calc = self.fragment(box, cyl_calc)
            self.box = [box, cyl_calc]

        self.add_physical(box, "box")
        self.add_physical([pmlxp, pmlxm], "pmlx")
        self.add_physical([pmlyp, pmlym], "pmly")
        self.add_physical([pmlxypp, pmlxypm, pmlxymm, pmlxymp], "pmlxy")

        if Rcalc > 0:
            bnds = self.get_boundaries("box")
            self.calc_bnds = bnds[0]
            self.add_physical(self.calc_bnds, "calc_bnds", dim=1)


class Scatt2D(Simulation):
    def __init__(
        self,
        geometry,
        epsilon,
        mu,
        source=None,
        boundary_conditions={},
        polarization="TM",
        modal=False,
        degree=1,
        mat_degree=1,
        pml_stretch=1 - 1j,
    ):
        assert isinstance(geometry, BoxPML2D)
        if source is not None:
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
        if modal:
            source_domains = []
        else:
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
            modal=modal,
            boundary_conditions=boundary_conditions,
        )

        super().__init__(geometry, formulation)

        self.degree = degree

    def solve_system(self, again=False):
        u = super().solve_system(again=again, vector_function=False)
        self.solution = {}
        self.solution["diffracted"] = u
        self.solution["total"] = u + self.source.expression
        return u

    @property
    def time_average_incident_poynting_vector_norm(self):
        Z0 = np.sqrt(mu_0 / epsilon_0)
        S0 = 1 / (2 * Z0) if self.formulation.polarization == "TM" else 0.5 * Z0
        return S0

    def scattering_cross_section(self):
        uscatt = self.solution["diffracted"]
        vscatt = self.formulation.get_dual(uscatt)
        Sscatt = (Constant(0.5) * uscatt * vscatt.conj).real
        n = self.geometry.unit_normal_vector
        Ws = assemble(dot(n, Sscatt)("+") * self.dS("calc_bnds"))
        S0 = self.time_average_incident_poynting_vector_norm
        SCS = abs(Ws / S0)
        return SCS

    def absorption_cross_section(self):
        utot = self.solution["total"]
        vtot = self.formulation.get_dual(utot)
        Stot = (Constant(0.5) * utot * vtot.conj).real
        n = self.geometry.unit_normal_vector
        Wa = assemble(dot(n, Stot)("+") * self.dS("calc_bnds"))
        S0 = self.time_average_incident_poynting_vector_norm
        ACS = abs(Wa / S0)
        return ACS

    def extinction_cross_section(self):
        ui = self.source.expression
        vi = self.formulation.get_dual(ui)
        uscatt = self.solution["diffracted"]
        vscatt = self.formulation.get_dual(uscatt)
        Se = (Constant(0.5) * (uscatt * vi.conj + ui * vscatt.conj)).real
        n = self.geometry.unit_normal_vector
        We = assemble(dot(n, Se)("+") * self.dS("calc_bnds"))
        S0 = self.time_average_incident_poynting_vector_norm
        ECS = abs(We / S0)
        return ECS

    def get_cross_sections(self):
        scs = self.scattering_cross_section()
        acs = self.absorption_cross_section()
        ecs = self.extinction_cross_section()
        return dict(scattering=scs, absorption=acs, extinction=ecs)

    def local_density_of_states(self, x, y):
        self.source.position = x, y
        if hasattr(self, "solution"):
            self.assemble_rhs()
            self.solve_system(again=True)
        else:
            self.solve()
        u = self.solution["total"]
        delta = 1 + dolfin.DOLFIN_EPS_LARGE
        evalpoint = (
            max(dolfin.DOLFIN_EPS_LARGE, x * delta),
            max(dolfin.DOLFIN_EPS_LARGE, y * delta),
        )
        ldos = -2 * self.source.pulsation / (np.pi * c ** 2) * u(evalpoint).imag
        return ldos

    def plot_field(
        self,
        ax=None,
        mincmap=None,
        maxcmap=None,
        fig=None,
        phase=0,
        callback=None,
        **kwargs,
    ):

        import matplotlib as mpl

        from .plot import plt

        u = self.solution["total"]
        if ax == None:
            ax = plt.gca()
        if "cmap" not in kwargs:
            kwargs["cmap"] = "RdBu_r"

        f = u * phase_shift(phase, degree=self.degree)
        fplot = f.real
        fplot = project(
            fplot,
            self.formulation.real_function_space,
            solver_type="cg",
            preconditioner_type="jacobi",
        )
        pp = dolfin.plot(fplot, **kwargs)

        ppmax = pp.cvalues[-1]
        ppmin = pp.cvalues[0]
        ax.set_aspect(1)
        bs = self.geometry.box_size
        ax.set_xlim(-bs[0] / 2, bs[0] / 2)
        ax.set_ylim(-bs[1] / 2, bs[1] / 2)
        mincmap = mincmap or ppmin
        maxcmap = maxcmap or ppmax
        pp.set_clim(mincmap, maxcmap)

        cm = plt.cm.ScalarMappable(cmap=kwargs["cmap"])
        cm.set_clim(mincmap, maxcmap)

        fig = plt.gcf() if fig is None else fig
        cb = fig.colorbar(cm, ax=ax)

        if callback is not None:
            callback(**kwargs)

        return pp, cb

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
