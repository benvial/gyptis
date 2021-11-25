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
from .utils._meta import _ScatteringBase


class BoxPML2D(Geometry):
    def __init__(
        self,
        box_size=(1, 1),
        box_center=(0, 0),
        pml_width=(0.2, 0.2),
        Rcalc=0,
        **kwargs,
    ):
        super().__init__(
            dim=2,
            **kwargs,
        )
        self.box_size = box_size
        self.box_center = box_center
        self.pml_width = pml_width
        self.Rcalc = Rcalc

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

        self.pml_physical = [
            "pmlx",
            "pmly",
            "pmlxy",
        ]

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


class Scatt2D(_ScatteringBase, Simulation):
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
            epsilon, geometry, pmls=[pmlx, pmly, pmlxy], degree=degree
        )
        mu_coeff = Coefficient(mu, geometry, pmls=[pmlx, pmly, pmlxy], degree=degree)

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

    def _cross_section_helper(self, return_type="s", boundaries="calc_bnds"):
        uscatt = self.solution["diffracted"]
        vscatt = self.formulation.get_dual(uscatt)
        utot = self.solution["total"]
        vtot = self.formulation.get_dual(utot)
        ui = self.source.expression
        vi = self.formulation.get_dual(ui)

        parallel = dolfin.MPI.comm_world.size > 1

        ### normal vector is messing up in parallel so workaround here:
        if parallel:
            n_out = dolfin.Expression(
                (f"x[0]/Rcalc", f"x[1]/Rcalc"),
                degree=self.degree,
                Rcalc=self.geometry.Rcalc,
            )
        else:
            n_out = self.geometry.unit_normal_vector

        dcalc = self.dS(boundaries)

        if return_type == "s":
            Sscatt = _get_power_dens(uscatt, vscatt)
            W = assemble(dot(n_out, Sscatt)("+") * dcalc)
        elif return_type == "a":
            Stot = _get_power_dens(utot, vtot)
            W = assemble(dot(n_out, Stot)("+") * dcalc)
        else:
            Se = _get_power_dens(uscatt, vi) + _get_power_dens(ui, vscatt)
            W = assemble(dot(n_out, Se)("+") * dcalc)
        out = W / self.time_average_incident_poynting_vector_norm
        return out if out > 0 else -out

    def scattering_cross_section(self, **kwargs):
        return self._cross_section_helper("s", **kwargs)

    def extinction_cross_section(self, **kwargs):
        return self._cross_section_helper("e", **kwargs)

    def absorption_cross_section(self, **kwargs):
        return self._cross_section_helper("a", **kwargs)

    def local_density_of_states(self, x, y):
        """Compute the local density of state.

        Parameters
        ----------
        x : float
            x coordinate.
        y : float
            y coordinate.

        Returns
        -------
        float
            The local density of states.

        """
        # self.source =  LineSource(
        #     wavelength=self.source.wavelength, position=(x, y),domain=self.mesh, degree=self.degree,
        # )
        self.source.position = x, y
        if hasattr(self, "solution"):
            self.assemble_rhs()
            self.solve_system(again=True)
        else:
            self.solve()
        u = self.solution["total"]
        eps = dolfin.DOLFIN_EPS_LARGE
        delta = 1 + eps
        evalpoint = x * delta, y * delta
        if evalpoint[0] == 0:
            evalpoint = eps, evalpoint[1]
        if evalpoint[1] == 0:
            evalpoint = evalpoint[0], eps
        ldos = -2 * self.source.pulsation / (np.pi * c ** 2) * u(evalpoint).imag
        return ldos

    def check_optical_theorem(cs, rtol=1e-12):
        np.allclose(cs["extinction"], cs["scattering"] + cs["absorption"], rtol=rtol)

    def plot_field(
        self,
        type="real",
        field="total",
        ax=None,
        mincmap=None,
        maxcmap=None,
        fig=None,
        phase=0,
        callback=None,
        **kwargs,
    ):

        import matplotlib as mpl

        from .plot import _check_plot_type, plt

        u = self.solution[field]
        if ax == None:
            ax = plt.gca()
        if "cmap" not in kwargs:
            kwargs["cmap"] = "RdBu_r"

        f = u * phase_shift(phase, degree=self.degree)

        fplot = _check_plot_type(type, f)
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


def _get_power_dens(u, v):
    re = as_vector([v[1].real, -v[0].real])
    im = as_vector([v[1].imag, -v[0].imag])
    EcrossHstar = u * Complex(re, -im)
    return (Constant(0.5) * EcrossHstar).real
