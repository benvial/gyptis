#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT


from collections import OrderedDict

from .base import *
from .base import _coefs, _make_cst_mat
from .helpers import DirichletBC, PeriodicBoundary2DX
from .stack import *

# dolfin.set_log_level(20)


def _translation_matrix(t):
    M = np.eye(4)
    M[:3, -1] = t
    return M.ravel().tolist()


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

        self.translation_x = _translation_matrix([self.period, 0, 0])

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

        self.removeAllDuplicates()
        self.synchronize()

        for sub, num in self.subdomains["surfaces"].items():
            self.add_physical(num, sub)

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


class Grating2D(ElectroMagneticSimulation2D):
    def __init__(self, geom, epsilon, mu, pml_stretch=1 - 1j, **kwargs):

        super().__init__(geom, epsilon, mu, **kwargs)

        self.period = self.geom.period
        self.pml_stretch = pml_stretch
        self.N_d_order = 0
        self.nb_slice = 20
        self.scan_dist_ratio = 5
        self.npt_integ = 401

        self.ex = as_vector([1.0, 0.0])

        self.periodic_bcs = PeriodicBoundary2DX(self.period)

        self.complex_space = ComplexFunctionSpace(
            self.mesh, "CG", self.degree, constrained_domain=self.periodic_bcs
        )
        self.real_space = dolfin.FunctionSpace(
            self.mesh, "CG", self.degree, constrained_domain=self.periodic_bcs
        )

        self.no_source_domains = ["substrate", "pml_top", "pml_bottom", "superstrate"]
        self.source_domains = [
            z for z in self.domains if z not in self.no_source_domains
        ]

        self.utrial = TrialFunction(self.complex_space)
        self.utest = TestFunction(self.complex_space)

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
                ubnd = -self.ustack_coeff * self.phasor.conj
                # ubnd = dolfin.as_vector((ubnd_vec.real, ubnd_vec.imag))
                ubnd_proj = project(ubnd, self.real_space)
                bc = DirichletBC(
                    self.complex_space,
                    ubnd_proj,
                    self.boundary_markers,
                    bnd,
                    self.boundaries,
                )
                [self._boundary_conditions.append(b) for b in bc]

    def get_N_d_order(self):
        a = self.period / self.lambda0
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

    def _make_pmls(self):
        pml_domains = ["substrate", "superstrate"]
        pml = PML("y", stretch=self.pml_stretch)
        t = pml.transformation_matrix()
        eps_pml_ = [(self.epsilon[d] * t).tolist() for d in pml_domains]
        mu_pml_ = [(self.mu[d] * t).tolist() for d in pml_domains]
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
        for dom in self.source_domains:
            estack[dom] = estack["superstrate"]
        # estack["pml_bottom"] = estack["pml_top"] = Complex(0,0)
        estack["pml_bottom"] = estack["substrate"]
        estack["pml_top"] = estack["superstrate"]
        e0 = {"superstrate": self.u_0}
        for dom in self.source_domains:
            e0[dom] = e0["superstrate"]
        e0["substrate"] = e0["pml_bottom"] = e0["pml_top"] = Complex(
            0, 0
        )  # Constant(0)
        self.ustack_coeff = Subdomain(
            self.markers, self.domains, estack, degree=self.degree, domain=self.mesh
        )

        self.u0_coeff = Subdomain(
            self.markers, self.domains, e0, degree=self.degree, domain=self.mesh
        )

        inc_field = {}
        stack_field = {}
        for dom in self.domains:
            inc_field[dom] = e0[dom]
            stack_field[dom] = estack[dom]
        self.annex_field = {"incident": inc_field, "stack": stack_field}

    def _phasor(self, *args, i=0, **kwargs):
        phasor_re = dolfin.Expression(f"cos(alpha*x[{i}])", *args, **kwargs)
        phasor_im = dolfin.Expression(f"sin(alpha*x[{i}])", *args, **kwargs)
        return Complex(phasor_re, phasor_im)

    def prepare(self):
        self.alpha = -self.k0 * np.sin(self.theta0)
        self.phasor = self._phasor(
            degree=self.degree, domain=self.mesh, alpha=self.alpha
        )
        self._prepare_materials(ref_material="superstrate", pmls=True)
        self._make_coefs()
        self.make_stack()
        self._prepare_bcs()

    def build_rhs(self):
        return build_rhs(
            self.ustack_coeff,
            # self.annex_field["stack"],
            self.utest,
            self.xi,
            self.chi,
            self.xi_annex,
            self.chi_annex,
            self.source_domains,
            unit_vect=self.ex,
            phasor=self.phasor,
        )

    def build_rhs_boundaries(self):
        return build_rhs_boundaries(
            self.ustack_coeff,
            self.utest,
            self.xi_coeff_annex,
            self.pec_bnds,
            self.unit_normal_vector,
            phasor=self.phasor,
        )

    def build_lhs_boundaries(self):
        return build_lhs_boundaries(
            self.utrial,
            self.utest,
            self.xi_coeff,
            self.pec_bnds,
            self.unit_normal_vector,
            unit_vect=self.ex,
        )

    def build_lhs(self):
        return build_lhs(
            self.utrial, self.utest, self.xi, self.chi, self.domains, unit_vect=self.ex
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
            alpha=self.alpha,
            boundary=(self.polarization == "TM"),
        )
        self.vector = make_system_vector(
            self.source_domains,
            self.pec_bnds,
            self.bh,
            self.k0,
            alpha=self.alpha,
            boundary=(self.polarization == "TM"),
        )
        # Ah.form = _get_form(self.Ah)
        # bh.form = _get_form(self.bh)

    def solve(self, direct=True):

        for bc in self._boundary_conditions:
            bc.apply(self.matrix, self.vector)

        # ufunc = self.u.real
        VVect = dolfin.VectorFunctionSpace(
            self.mesh, self.element, self.degree, constrained_domain=self.periodic_bcs
        )
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

        self.uper = Complex(*u.split())
        self.u = self.uper * self.phasor

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
        orders_num = np.linspace(
            -self.N_d_order,
            self.N_d_order,
            2 * self.N_d_order + 1,
        )

        k, beta = {}, {}
        for d in ["substrate", "superstrate"]:
            k[d] = self.k0 * np.sqrt(complex(self.epsilon[d] * self.mu[d]))
            beta[d] = np.conj(np.sqrt(k[d] ** 2 - self.alpha ** 2))

        def K(beta, y0, h):
            return (
                np.exp(1j * beta * y0) * (np.exp(1j * beta * h) - 1) / (1j * beta * h)
            )

        r_annex = self.phi[0][-1]
        t_annex = self.phi[-1][0]
        eff_annex = dict(substrate=t_annex, superstrate=r_annex)
        ypos = self.geom.y_position
        thickness = self.geom.thicknesses
        r_n, t_n = [], []
        R_n, T_n = [], []
        for n in orders_num:
            delta = 1 if n == 0 else 0
            qn = n * 2 * np.pi / self.period
            alpha_n = self.alpha + qn
            Jn, beta_n, eff = {}, {}, {}
            for d in ["substrate", "superstrate"]:
                s = 1 if d == "superstrate" else -1
                beta_n[d] = np.sqrt(k[d] ** 2 - alpha_n ** 2)
                ph_x = self._phasor(degree=self.degree, domain=self.mesh, alpha=-qn)
                ph_y = self._phasor(
                    degree=self.degree, domain=self.mesh, alpha=s * beta_n[d].real, i=1
                )
                Jn[d] = assemble(self.uper * ph_x * ph_y * self.dx(d)) / self.period

                ph_pos = np.exp(-s * 1j * beta_n[d] * ypos[d])
                eff[d] = (delta * eff_annex[d] + Jn[d] / thickness[d]) * ph_pos
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
        return eff

    def compute_absorption(self, subdomain_absorption=False):
        ##### absorption

        doms_no_pml = [
            z for z in self.epsilon.keys() if z not in ["pml_bottom", "pml_top"]
        ]
        omega = self.k0 * c

        if self.polarization == "TE":
            xi_0, chi_0 = mu_0, epsilon_0
        else:
            xi_0, chi_0 = epsilon_0, mu_0

        P0 = 0.5 * np.sqrt(chi_0 / xi_0) * np.cos(self.theta0) * self.period

        u_tot = self.u + self.ustack_coeff

        Qchi = {}
        Qxi = {}
        if subdomain_absorption:
            for d in doms_no_pml:

                # u_tot = self.u + self.annex_field["stack"][d]
                nrj_chi_dens = (
                    dolfin.Constant(-0.5 * chi_0 * omega)
                    * self.chi[d]
                    * abs(u_tot) ** 2
                ).imag

                nrj_xi_dens = (
                    dolfin.Constant(-0.5 * 1 / (omega * xi_0))
                    * dot(grad(u_tot), (self.xi[d] * grad(u_tot)).conj).imag
                )

                Qchi[d] = assemble(nrj_chi_dens * self.dx(d)) / P0
                Qxi[d] = assemble(nrj_xi_dens * self.dx(d)) / P0
            Q = sum(Qxi.values()) + sum(Qchi.values())
        else:

            nrj_chi_dens = (
                dolfin.Constant(-0.5 * chi_0 * omega) * self.chi_coeff * abs(u_tot) ** 2
            ).imag

            nrj_xi_dens = (
                dolfin.Constant(-0.5 * 1 / (omega * xi_0))
                * dot(grad(u_tot), (self.xi_coeff * grad(u_tot)).conj).imag
            )
            Qchi = assemble(nrj_chi_dens * self.dx(doms_no_pml)) / P0
            Qxi = assemble(nrj_xi_dens * self.dx(doms_no_pml)) / P0
            Q = Qxi + Qchi

        if self.polarization == "TE":
            Qdomains = {"electric": Qchi, "magnetic": Qxi}
        else:
            Qdomains = {"electric": Qxi, "magnetic": Qchi}

        self.Qtot = Q
        self.Qdomains = Qdomains
        return Q, Qdomains
