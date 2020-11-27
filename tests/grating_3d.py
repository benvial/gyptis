#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT


from collections import OrderedDict

import dolfin as df
import numpy as np
import pytest

from gyptis.complex import *
from gyptis.complex import _invert_3by3_complex_matrix
from gyptis.core import PML
from gyptis.geometry import *
from gyptis.helpers import BiPeriodicBoundary3D, DirichletBC
from gyptis.materials import *
from gyptis.sources import *

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

df.set_log_level(10)


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

        z0 = 0
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
        s = self.get_periodic_bnds(0, self.total_thickness)

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

        pmin = -dx / 2 - eps, -dy / 2 - eps, 0 - eps
        pmax = -dx / 2 + eps, +dy / 2 + eps, thickness + eps
        s["-x"] = gmsh.model.getEntitiesInBoundingBox(*pmin, *pmax, 2)

        pmin = +dx / 2 - eps, -dy / 2 - eps, 0 - eps
        pmax = +dx / 2 + eps, +dy / 2 + eps, thickness + eps
        s["+x"] = gmsh.model.getEntitiesInBoundingBox(*pmin, *pmax, 2)

        pmin = -dx / 2 - eps, -dy / 2 - eps, 0 - eps
        pmax = +dx / 2 + eps, -dy / 2 + eps, thickness + eps
        s["-y"] = gmsh.model.getEntitiesInBoundingBox(*pmin, *pmax, 2)

        pmin = -dx / 2 - eps, +dy / 2 - eps, 0 - eps
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
        pml_stretch=1 - 1j,
        boundary_conditions=[],
    ):

        self.geom = geom  # geometry model
        self.degree = degree
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

        self.E0 = plane_wave_3D(
            self.lambda0, self.theta0, self.phi0, self.psi0, domain=self.mesh
        )
        self._prepare_materials()
        self.inv_mu_coeff = _invert_3by3_complex_matrix(self.mu_coeff)
        self.inv_mu_coeff_annex = _invert_3by3_complex_matrix(self.mu_coeff_annex)

        self.periodic_bcs = BiPeriodicBoundary3D(self.geom.period)

        self.complex_space = ComplexFunctionSpace(
            self.mesh, "N1curl", self.degree, constrained_domain=self.periodic_bcs
        )
        self.real_space = df.FunctionSpace(
            self.mesh, "N1curl", self.degree, constrained_domain=self.periodic_bcs
        )

    @property
    def k0(self):
        return 2 * np.pi / self.lambda0

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
        _no_source = ["substrate", "pml_top", "groove", "pml_bottom", "superstrate"]

        self.source_dom = [z for z in self.mu.keys() if z not in _no_source]

        mu_annex = self.mu.copy()
        eps_annex = self.epsilon.copy()
        for a in self.source_dom:
            mu_annex[a] = self.mu["superstrate"]
            eps_annex[a] = self.epsilon["superstrate"]
        self.epsilon_coeff_annex, self.mu_coeff_annex = self._make_subdomains(
            eps_annex, mu_annex
        )

    def _make_pmls(self):
        pml = PML("z", stretch=self.pml_stretch)
        t = pml.transformation_matrix()
        eps_pml_ = [(epsilon[d] * t).tolist() for d in ["substrate", "superstrate"]]
        mu_pml_ = [(mu[d] * t).tolist() for d in ["substrate", "superstrate"]]
        epsilon_pml = dict(pml_bottom=eps_pml_[0], pml_top=eps_pml_[1])
        mu_pml = dict(pml_bottom=mu_pml_[0], pml_top=mu_pml_[1])
        return epsilon_pml, mu_pml

    def weak_form(self):
        W = self.complex_space
        dx = self.dx
        self.E = Function(W)
        Etrial = TrialFunction(W)
        Etest = TestFunction(W)
        delta_epsilon = self.epsilon_coeff - self.epsilon_coeff_annex
        delta_inv_mu = self.inv_mu_coeff - self.inv_mu_coeff_annex

        L = (
            -inner(self.inv_mu_coeff * curl(Etrial), curl(Etest)) * dx,
            inner(self.epsilon_coeff * Etrial, Etest) * dx,
        )
        b = (
            dot(delta_inv_mu * curl(self.E0), curl(Etest)) * dx(self.source_dom),
            -dot(delta_epsilon * self.E0, Etest) * dx(self.source_dom),
        )
        self.lhs = [t.real + t.imag for t in L]
        self.rhs = [t.real + t.imag for t in b]

    def assemble(self):
        self.Ah = [assemble(A) for A in self.lhs]
        self.bh = [assemble(b) for b in self.rhs]

    def solve(self, direct=True):
        Efunc = self.E[0].real.ufl_operands[0]
        Ah = self.Ah[0] + self.k0 ** 2 * self.Ah[1]
        bh = self.bh[0] + self.k0 ** 2 * self.bh[1]
        for bc in self.boundary_conditions:
            bc.apply(Ah, bh)

        print("##############################")
        print("SOLVING")
        print("##############################")

        if direct:
            # solver = df.LUSolver(Ah) ### direct
            solver = df.PETScLUSolver("mumps")
            solver.parameters.update(lu_params)
            solver.solve(Ah, Efunc.vector(), bh)
        else:
            solver = df.PETScKrylovSolver()  ## iterative
            solver.parameters.update(krylov_params)
            solver.solve(Ah, Efunc.vector(), bh)

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

        print("##############################")
        print("END SOLVING")
        print("##############################")


lambda0 = 40
parmesh = 10
period = (20, 20)


thicknesses = OrderedDict(
    {
        "pml_bottom": lambda0,
        "substrate": lambda0,
        "groove": 10,
        "superstrate": lambda0,
        "pml_top": lambda0,
    }
)

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


model.set_size(sub, parmesh)
model.set_size(sup, parmesh)
model.set_size(pmltop, lambda0 / (parmesh))
model.set_size(pmlbot, lambda0 / (parmesh))
model.set_size(groove, lambda0 / (1 * parmesh))
model.set_size(island, lambda0 / (1.5 * 50 ** 0.5 * parmesh))
face_top = model.get_boundaries(pmltop)[-1]
face_bottom = model.get_boundaries(pmlbot)[-2]

model.add_physical(face_top, "face_top", dim=2)
model.add_physical(face_bottom, "face_bottom", dim=2)

# mesh_object = model.build(interactive=True, generate_mesh=True, write_mesh=True)

mesh_object = model.build()

# e = (3 * np.eye(3, dtype=complex)).tolist()
# m = (np.eye(3, dtype=complex)).tolist()
# epsilon = dict({"groove": m, "island": e})
# mu = dict({"groove": m, "island": m})
#

epsilon = dict({"substrate": 2, "groove": 1, "island": 50, "superstrate": 1})
mu = dict({"substrate": 1, "groove": 1, "island": 1, "superstrate": 1})


g = Grating3D(model, epsilon, mu, lambda0=lambda0)

from stack import *

config = OrderedDict(
    {
        "superstrate": {"epsilon": epsilon["superstrate"], "mu": mu["superstrate"]},
        "groove": {
            "epsilon": epsilon["groove"],
            "mu": mu["groove"],
            "thickness": thicknesses["groove"],
        },
        "substrate": {"epsilon": epsilon["substrate"], "mu": mu["substrate"]},
    }
)

phi, alpha0, beta0, gamma = get_coeffs_stack(
    config, g.lambda0, g.theta0, g.phi0, g.psi0
)

zshift = -thicknesses["pml_bottom"] - thicknesses["substrate"]

Estack = [
    field_stack_3D(p, alpha0, beta0, g, zshift=zshift) for p, g in zip(phi, gamma)
]

W0 = df.FunctionSpace(g.mesh, "DG", 0)
W1 = df.FunctionSpace(g.mesh, "CG", 1)
estack = {k: project(v[0], W0) for k, v in zip(config.keys(), Estack)}
estack["pml_bottom"] = 0
estack["pml_top"] = 0

test = Subdomain(g.markers, g.domains, estack, degree=g.degree)


# df.File("test.pvd") << project(Estack[0].real[0], W0)
df.File("test.pvd") << project(test.real, W0)

cds


### BCs
# domains = model.subdomains["volumes"]
# surfaces = model.subdomains["surfaces"]
# markers_surf = model.mesh_object["markers"]["triangle"]
# g.boundary_conditions = [
#     DirichletBC(g.complex_space, [0] * 6, markers_surf, f, surfaces)
#     for f in ["face_top", "face_bottom"]
# ]

g.weak_form()
g.assemble()
g.solve(direct=True)
df.list_timings(df.TimingClear.clear, [df.TimingType.wall])


# N = abs(g.E[0]) ** 2 + abs(g.E[1]) ** 2 + abs(g.E[2]) ** 2
N = df.sqrt(dot(g.E, g.E.conj).real)
#
W0 = df.FunctionSpace(g.mesh, "CG", 1)
W0 = df.FunctionSpace(g.mesh, "DG", 0)

df.File("norm.pvd") << project(N, W0)
marks = g.geom.mesh_object["markers"]["tetra"]
df.File("markers.pvd") << marks
df.File("reEx.pvd") << project(g.E[0].real, W0)
# df.File("test.pvd") << project(g.E[0].imag, W0)

# df.File("test.pvd") << project(g.inv_mu_coeff.real, W0)


# df.File("test.pvd") << project(g.E[3], W0)


# df.File("test.pvd") << project(g.E0.real, W0)


# e = g.epsilon_coeff.imag[1][1]
#
# df.File("test.pvd") << project(e, W0)
#
# epsilon=g.epsilon_coeff[2,2].real
# df.File("test.pvd") << project(epsilon, W0)


# import matplotlib.pyplot as plt
# a = g.Ah[0].array() + g.k0 ** 2 * g.Ah[1].array()
# # a= Ah.array()
# c = plt.imshow(a,cmap="RdBu")
# plt.colorbar(c)
# plt.show()
# plt.savefig("test.png")


#
#
# class Periodic3D(object):
#     def __init__(self, model):
#         pass


#
#
# class Thickness(OrderedDict):
#     def insert(self, key_before, key, value):
#         self[key] = value
#         index = [i for i,k in enumerate(list(self.keys())) if k ==key_before][0]
#         for ii, k in enumerate(list(self.keys())):
#             if ii > index and k != key:
#                 self.move_to_end(k)
#
#
# thicknesses = Thickness(
#     {"pml_bottom": 1, "substrate": 1, "groove": 2, "superstrate": 2, "pml_top": 1}
# )
# print(thicknesses)
# thicknesses.insert("groove","test",100)
# print(thicknesses)
