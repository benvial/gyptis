#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT


import glob
import os

from PIL import Image

from .base import *

# from .helpers import _get_form


class Scatt2D(ElectroMagneticSimulation2D):
    def __init__(
        self,
        geometry,
        epsilon,
        mu,
        pml_stretch=1 - 1j,
        source="PW",
        xs=(0, 0),
        **kwargs,
    ):

        super().__init__(geometry, epsilon, mu, **kwargs)
        self.pml_stretch = pml_stretch
        self.complex_space = ComplexFunctionSpace(self.mesh, self.element, self.degree)
        self.real_space = dolfin.FunctionSpace(self.mesh, self.element, self.degree)

        self.no_source_domains = ["box", "pmlx", "pmly", "pmlxy"]
        self.source_domains = [
            z for z in self.domains if z not in self.no_source_domains
        ]
        self.pec_bnds = []
        self._boundary_conditions = self.boundary_conditions
        self.Ah = {}
        self.bh = {}

        self.utrial = TrialFunction(self.complex_space)
        self.utest = TestFunction(self.complex_space)
        self.source = source
        assert self.source in ["PW", "LS"]

        self.xs = xs

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

    def prepare(self):
        self._prepare_materials(ref_material="box", pmls=True)
        if self.source == "PW":
            self.u0 = plane_wave_2d(
                self.lambda0, self.theta0, degree=self.degree, domain=self.mesh,
            )

        else:
            self.u0 = green_function_2d(
                self.lambda0,
                self.xs[0],
                self.xs[1],
                degree=self.degree,
                domain=self.mesh,
            )
        self.gradu0 = grad(self.u0)

    def build_rhs(self):
        return build_rhs(
            self.u0,
            # self.annex_field["stack"],
            self.utest,
            self.xi,
            self.chi,
            self.xi_annex,
            self.chi_annex,
            self.source_domains,
        )

    def build_lhs_boundaries(self):
        return build_lhs_boundaries(
            self.utrial,
            self.utest,
            self.xi_coeff,
            self.pec_bnds,
            self.unit_normal_vector,
        )

    def build_lhs(self):
        return build_lhs(self.utrial, self.utest, self.xi, self.chi, self.domains)

    def build_rhs_boundaries(self):
        return build_rhs_boundaries(
            self.u0,
            self.utest,
            self.xi_coeff_annex,
            self.pec_bnds,
            self.unit_normal_vector,
        )

    def weak_form(self):

        self.lhs = self.build_lhs()

        if self.polarization == "TM":
            lhs_bnds = self.build_lhs_boundaries()
            self.lhs.update(lhs_bnds)

        self.rhs = self.build_rhs()

        if self.polarization == "TM":
            rhs_bnds = self.build_rhs_boundaries()
            self.rhs.update(rhs_bnds)

    def assemble_lhs(self, domains=None, pec_bnds=None):
        domains = self.domains if domains is None else domains
        pec_bnds = self.pec_bnds if pec_bnds is None else pec_bnds
        Ah = {}
        for d in domains:
            Ah[d] = [assemble(A * self.dx(d)) for A in self.lhs[d]]

        if self.polarization == "TM":
            for d in pec_bnds:
                Ah[d] = [assemble(A * self.ds(d)) for A in self.lhs[d]]
        self.Ah.update(Ah)

    def assemble_rhs(self, source_domains=None, pec_bnds=None):
        source_domains = (
            self.source_domains if source_domains is None else source_domains
        )
        pec_bnds = self.pec_bnds if pec_bnds is None else pec_bnds
        bh = {}
        for d in source_domains:
            bh[d] = [assemble(b * self.dx(d)) for b in self.rhs[d]]
        if self.polarization == "TM":
            for d in pec_bnds:
                bh[d] = [assemble(b * self.ds(d)) for b in self.rhs[d]]
        self.bh.update(bh)

    def assemble(
        self, domains=None, source_domains=None, pec_bnds=None,
    ):
        domains = self.domains if domains is None else domains
        source_domains = (
            self.source_domains if source_domains is None else source_domains
        )
        pec_bnds = self.pec_bnds if pec_bnds is None else pec_bnds
        self.assemble_lhs(domains=domains, pec_bnds=pec_bnds)
        self.assemble_rhs(source_domains=source_domains, pec_bnds=pec_bnds)

    def build_system(self):
        self.matrix = make_system_matrix(
            self.domains,
            self.pec_bnds,
            self.Ah,
            self.k0,
            boundary=(self.polarization == "TM"),
        )
        self.vector = make_system_vector(
            self.source_domains,
            self.pec_bnds,
            self.bh,
            self.k0,
            boundary=(self.polarization == "TM"),
        )
        # Ah.form = _get_form(self.Ah)
        # bh.form = _get_form(self.bh)

    def solve_system(self, direct=True, again=False):

        for bc in self._boundary_conditions:
            bc.apply(self.matrix, self.vector)

        VVect = dolfin.VectorFunctionSpace(self.mesh, self.element, self.degree)
        u = dolfin.Function(VVect)

        if not again:
            if direct:
                self.solver = dolfin.LUSolver(self.matrix, "mumps")
            else:
                self.solver = dolfin.PETScKrylovSolver("cg", "petsc_amg")  ## iterative
                self.solver.parameters["absolute_tolerance"] = 1e-7
                self.solver.parameters["relative_tolerance"] = 1e-12
                self.solver.parameters["maximum_iterations"] = 1000
                self.solver.parameters["monitor_convergence"] = True
                self.solver.parameters["nonzero_initial_guess"] = False
                self.solver.parameters["report"] = True

        self.solver.solve(u.vector(), self.vector)

        self.u = Complex(*u.split())
        utot = self.u + self.u0
        self.solution = {}
        self.solution["diffracted"] = self.u
        self.solution["total"] = utot

    def solve(self, direct=True, again=False):
        self.prepare()
        self.weak_form()
        self.assemble()
        self.build_system()
        self.solve_system(direct=direct, again=again)

    def wavelength_sweep(self, wavelengths):

        wl_sweep = []
        for i, w in enumerate(wavelengths):
            t = -time.time()
            self.lambda0 = w
            self.u0 = plane_wave_2d(
                self.lambda0, self.theta0, domain=self.mesh, degree=self.degree,
            )
            self.gradu0 = grad(self.u0)
            self.weak_form()
            if i == 0:
                self.assemble()

                dolfin.list_timings(dolfin.TimingClear.clear, [dolfin.TimingType.wall])
            else:
                self.assemble_rhs()
            self.solve()

            dolfin.list_timings(dolfin.TimingClear.clear, [dolfin.TimingType.wall])
            wl_sweep.append(self.u)
            t += time.time()
            print(f"iter {t:0.2f}s")
        return wl_sweep

    def plot_geometry(self, ax=None, **kwargs):
        from .plot import plot_subdomains, plt

        if ax == None:
            ax = plt.gca()

        domains = self.geometry.subdomains["surfaces"]
        scatt = []
        for d in domains:
            scatt.append(d)
        scatt_ids = [domains[d] for d in scatt]
        scatt_lines = []
        for id in scatt_ids:
            s = plot_subdomains(self.markers, domain=id, **kwargs)
            scatt_lines.append(s)
        ax.set_aspect(1)

        return scatt_lines

    def _phase_shift(self, phase):
        phasor_re = dolfin.Expression(f"cos(phase)", phase=phase, degree=self.degree)
        phasor_im = dolfin.Expression(f"sin(phase)", phase=phase, degree=self.degree)
        return Complex(phasor_re, phasor_im)

    def plot_field(
        self,
        ax=None,
        mincmap=None,
        maxcmap=None,
        fig=None,
        anim_phase=0,
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
        per_plots = []
        f = u * self._phase_shift(anim_phase)
        fplot = f.real
        if ADJOINT:
            fplot = project(fplot, self.real_space)
        pp = dolfin.plot(fplot, **kwargs)

        ppmax = pp.cvalues[-1]
        ppmin = pp.cvalues[0]
        ax.set_aspect(1)
        mincmap = mincmap or ppmin
        maxcmap = maxcmap or ppmax
        pp.set_clim(mincmap, maxcmap)

        cm = plt.cm.ScalarMappable(cmap=kwargs["cmap"])
        cm.set_clim(mincmap, maxcmap)

        fig = plt.gcf() if fig is None else fig
        cb = fig.colorbar(cm, ax=ax)

        if callback is not None:
            callback(**kwargs)

        return per_plots, cb

    def animate_field(self, n=11, filename="animation.gif", **kwargs):
        from .plot import plt

        anim = []
        tmpdir = tempfile.mkdtemp()
        fp_in = f"{tmpdir}/animation_tmp_*.png"
        phase = np.linspace(0, 2 * np.pi, n + 1)[:n]
        for iplot in range(n):
            number_str = str(iplot).zfill(4)
            pngname = f"{tmpdir}/animation_tmp_{number_str}.png"
            p = self.plot_field(anim_phase=phase[iplot], **kwargs)
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
