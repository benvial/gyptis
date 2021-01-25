#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT


from .base import *
from .base import _coefs, _make_cst_mat
from .helpers import _get_form


class Scatt2D(ElectroMagneticSimulation2D):
    def __init__(self, geom, epsilon, mu, pml_stretch=1 - 1j, **kwargs):

        super().__init__(geom, epsilon, mu, **kwargs)
        self.pml_stretch = pml_stretch
        self.complex_space = ComplexFunctionSpace(self.mesh, self.element, self.degree)
        self.real_space = dolfin.FunctionSpace(self.mesh, self.element, self.degree)

        self.no_source_domains = ["box", "pmlx", "pmly", "pmlxy"]
        self.source_domains = [
            z for z in self.domains if z not in self.no_source_domains
        ]
        self.pec_bnds = []
        self._boundary_conditions = self.boundary_conditions

        self.utrial = TrialFunction(self.complex_space)
        self.utest = TestFunction(self.complex_space)

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
        self._make_coefs()
        self.u0, self.gradu0 = plane_wave_2D(
            self.lambda0, self.theta0, domain=self.mesh, grad=True
        )

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

    def assemble_lhs(self):
        self.Ah = {}
        for d in self.domains:
            self.Ah[d] = [assemble(A * self.dx(d)) for A in self.lhs[d]]

        if self.polarization == "TM":
            for d in self.pec_bnds:
                self.Ah[d] = [assemble(A * self.ds(d)) for A in self.lhs[d]]

    def assemble_rhs(self):
        self.bh = {}
        for d in self.source_domains:
            self.bh[d] = [assemble(b * self.dx(d)) for b in self.rhs[d]]
        if self.polarization == "TM":
            for d in self.pec_bnds:
                self.bh[d] = [assemble(b * self.ds(d)) for b in self.rhs[d]]

    def assemble(self):
        self.assemble_lhs()
        self.assemble_rhs()

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

    def solve(self, direct=True):

        for bc in self._boundary_conditions:
            bc.apply(self.matrix, self.vector)

        # ufunc = self.u.real
        VVect = dolfin.VectorFunctionSpace(self.mesh, self.element, self.degree)
        u = dolfin.Function(VVect)
        # self.u = Function(self.complex_space)
        # ufunc = self.u.real

        if direct:
            # solver = dolfin.LUSolver(Ah) ### direct
            # solver = dolfin.PETScLUSolver("mumps")  ## doesnt work for adjoint
            solver = dolfin.LUSolver("mumps")
            # solver.parameters.update(lu_params)
            solver.solve(self.matrix, u.vector(), self.vector)
        else:
            solver = dolfin.PETScKrylovSolver()  ## iterative
            # solver.parameters.update(krylov_params)
            solver.solve(self.matrix, u.vector(), self.vector)

        self.u = Complex(*u.split())

    # def solve(self):
    #     VVect = dolfin.VectorFunctionSpace(self.mesh, self.element, self.degree)
    #     Ah = self.Ah[0] + self.k0 ** 2 * self.Ah[1]
    #     bh = self.bh[0] + self.k0 ** 2 * self.bh[1]
    #     for bc in self.boundary_conditions:
    #         bc.apply(Ah, bh)
    #     Ah.form = _get_form(self.Ah)
    #     bh.form = _get_form(self.bh)
    #
    #     self.u = dolfin.Function(VVect)
    #     # ufunc = self.u.real
    #     ufunc = self.u
    #     solver = dolfin.LUSolver("mumps")
    #     solver.solve(Ah, ufunc.vector(), bh)
    #
    #     self.u = Complex(*self.u.split())

    def wavelength_sweep(self, wavelengths):

        wl_sweep = []
        for i, w in enumerate(wavelengths):
            t = -time.time()
            self.lambda0 = w
            self.u0, self.gradu0 = plane_wave_2D(
                self.lambda0, self.theta0, domain=self.mesh, grad=True
            )
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
