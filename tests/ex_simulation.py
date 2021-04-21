#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT


from pprint import pprint

import pytest

import gyptis
from gyptis.geometry import *
from gyptis.helpers import list_time
from gyptis.plot import *
from gyptis.simulation import *
from gyptis.source import *

plt.ion()


# @pytest.mark.parametrize("degree", [1, 2])
# def test_scatt3d(degree):
#
# degree = 1
# wavelength = 1
# epsilon = dict(box=1, sphere=3)
# mu = dict(box=1, sphere=1)
#
# lmin = wavelength / 1

# geom = BoxPML(
#     dim=3,
#     box_size=(2 * wavelength, 2 * wavelength, 2 * wavelength),
#     pml_width=(wavelength, wavelength, wavelength),
# )
# sphere = geom.add_sphere(0, 0, 0, wavelength / 2)
# sphere, box = geom.fragment(sphere, geom.box)
# geom.add_physical(box, "box")
# geom.add_physical(sphere, "sphere")
# [geom.set_size(pml, lmin) for pml in geom.pmls]
# geom.set_size("box", lmin)
# geom.set_size("sphere", lmin)
# geom.build()
#
# mesh = geom.mesh
#
# pw = PlaneWave(
#     wavelength=wavelength, angle=(0, 0, 0), dim=3, domain=mesh, degree=degree
# )
# bcs={}
# s = Scatt3D(geom, epsilon, mu, pw, boundary_conditions=bcs)
#
# s.solve()
# list_time()
#
#
### PEC
#
#
# geom = BoxPML(
#     dim=3,
#     box_size=(2 * wavelength, 2 * wavelength, 2 * wavelength),
#     pml_width=(wavelength, wavelength, wavelength),
# )
# sphere = geom.add_sphere(0, 0, 0, wavelength / 2)
# box = geom.cut(geom.box,sphere)
# geom.add_physical(box, "box")
# [geom.set_size(pml, lmin) for pml in geom.pmls]
# geom.set_size("box", lmin)
# bnd_sphere = geom.get_boundaries("box")[-1]
# geom.add_physical(bnd_sphere, "bnd_sphere", dim=2)
#
# geom.build(0)
#
# epsilon = dict(box=1)
# mu = dict(box=1)
#
#
# mesh = geom.mesh
#
# pw = PlaneWave(
#     wavelength=wavelength, angle=(0, 0, 0), dim=3, domain=mesh, degree=degree
# )
# bcs={"bnd_sphere": "PEC"}
# s = Scatt3D(geom, epsilon, mu, pw, boundary_conditions=bcs)
#
# s.solve()
# list_time()


####### Grating 3D


##  ---------- incident wave ----------

lambda0 = 1
theta0 = 0 * np.pi / 180
phi0 = 0 * np.pi / 180
psi0 = 0 * np.pi / 180

##  ---------- geometry ----------

period = (0.3, 0.3)
R, a, b = 0.15 / 2, 0.1, 0.05
grooove_thickness = 6 * b

thicknesses = OrderedDict(
    {
        "pml_bottom": lambda0,
        "substrate": lambda0 / 1,
        "groove": grooove_thickness,
        "superstrate": lambda0 / 1,
        "pml_top": lambda0,
    }
)

##  ---------- mesh ----------
parmesh = 5

degree = 2

parmesh_hole = parmesh * 1

parmesh_groove = parmesh
parmesh_pml = parmesh * 2 / 3

mesh_params = dict(
    {
        "pml_bottom": parmesh_pml,
        "substrate": parmesh,
        "groove": parmesh,
        "torus": parmesh,
        "superstrate": parmesh,
        "pml_top": parmesh_pml,
    }
)

##  ---------- materials ----------
eps_torus = -21 - 20j
eps_substrate = 1.5 ** 2

epsilon = dict(
    {
        "substrate": eps_substrate,
        "groove": 1,
        "torus": eps_torus,
        "superstrate": 1,
    }
)
mu = dict({"substrate": 1, "groove": 1, "torus": 1, "superstrate": 1})

index = dict()
for e, m in zip(epsilon.items(), mu.items()):
    index[e[0]] = np.mean(
        (np.abs(np.array(e[1]).real).max() * np.abs(np.array(m[1]).real).max()) ** 0.5
    ).real

##  ---------- build geometry ----------
geom = Layered3D(period, thicknesses, finalize=False)

groove = geom.layers["groove"]
substrate = geom.layers["substrate"]
superstrate = geom.layers["superstrate"]
z0 = geom.z_position["groove"]

ellipse = geom.add_ellipse(R, 0, z0 + 1 * b, a / 2, b / 2, surface=True)
geom.rotate(ellipse, (0, 0, 0), (1, 0, 0), np.pi / 2, dim=1)

# line = geom.add_circle(0, 0, z0 + 1* b, R, surface=False)
# line = geom.add_curve_loop([line])

nturns = 1.0
npts = 20
p = []
for i in range(0, npts):
    theta = i * 2 * np.pi / (npts)
    gmsh.model.occ.addPoint(
        R * np.cos(theta), R * np.sin(theta), b / 2 + z0, 1, 1000 + i
    )
    p.append(1000 + i)
gmsh.model.occ.addSpline(p, 1000)

# A wire is like a curve loop, but open:
line = gmsh.model.occ.addWire([1000], 1000)


torus = geom.add_pipe([(2, ellipse)], line)[0][-1]


geom.remove([(2, ellipse)])


# geom.translate([(1,ellipse)], R,0,0)
# rot = np.linspace(0,2*np.pi,6)[1:-2]
# sections = []
# for angle in rot:
#     ell = geom.copy([(1,ellipse)])
#
#     geom.rotate(ell[0][-1],(0,0,0),(0,0,1),angle,dim=1)
#     ell = geom.add_curve_loop([ell[0][-1]])
#     sections.append(ell)
#
# torus = geom.add_thru_sections(sections)[0][-1]
# gmsh.model.occ.addCurveLoop([13], 13)
# gmsh.model.occ.addThruSections([11, 12, 13], 11, True, True)


torus, *groove = geom.fragment(torus, groove)
# hole, groove = geom.fragment(hole, groove)
geom.add_physical(groove, "groove")
geom.add_physical(torus, "torus")
# geom.add_physical(substrate, "substrate")
# geom.add_physical(superstrate, "superstrate")

index["pml_top"] = index["substrate"]
index["pml_bottom"] = index["substrate"]
pmesh = {k: lambda0 / (index[k] * mesh_params[k]) for k, p in mesh_params.items()}
geom.set_mesh_size(pmesh)
mesh_object = geom.build(interactive=True)
xs


E1 = s.formulation.annex_field["as_subdomain"]["stack"]
# E = s.solution["total"]
# E = project(E,s.formulation.real_function_space)
# dolfin.File("test.pvd") << E.real
#
# W = dolfin.FunctionSpace(geom.mesh, "CG", 2)
# dolfin.File("Ex.pvd") << project(E1[0].real, W)
# dolfin.File("Ey.pvd") << project(E1[1].real, W)
# dolfin.File("Ez.pvd") << project(E1[2].real, W)


#
# ###### PEC
# plt.close("all")
#
# wavelength = 600
# period = 800
# h = 8
# w = 600
# theta0 = 20
# lmin = wavelength / 3
# lmin_pec = h / 8
# degree = 2
# polarization = "TE"
#
# from gyptis.grating2d import Layered2D, OrderedDict
#
# thicknesses = OrderedDict(
#     {
#         "pml_bottom": wavelength,
#         "substrate": wavelength,
#         "groove": wavelength,
#         "superstrate": wavelength,
#         "pml_top": wavelength,
#     }
# )
# from scipy.spatial.transform import Rotation
#
# rot = Rotation.from_euler("zyx", [20, 0, 0], degrees=True)
# rmat = rot.as_matrix()
# eps_groove = 1#rmat @ np.diag((4 - 0.01j, 3 - 0.01j, 2 - 0.02j)) @ rmat.T
# mu_groove = 1#rmat @ np.diag((2 - 0.01j, 4 - 0.01j, 3 - 0.02j)) @ rmat.T
#
# epsilon = dict({"substrate": 1., "groove": eps_groove, "superstrate": 1})
# mu = dict({"substrate": 1., "groove": mu_groove, "superstrate": 1})
#
# geom = Layered2D(period, thicknesses)
#
# yc = geom.y_position["groove"] + thicknesses["groove"] / 2
# diff = geom.add_ellipse(0, yc, 0, w / 2, h / 2)
# groove = geom.cut(geom.layers["groove"], diff)
#
# geom.add_physical(groove, "groove")
# bnds = geom.get_boundaries("groove")
# geom.add_physical(bnds[-1], "hole", dim=1)
#
# for dom in ["substrate", "superstrate", "pml_bottom", "pml_top", "groove"]:
#     geom.set_size(dom, lmin)
#
# geom.set_size("hole", lmin_pec, dim=1)
#
# geom.build()
#
#
# angle = np.pi / 2 - theta0 * np.pi / 180
#
# pw = PlaneWave(wavelength=wavelength, angle=angle, dim=2, domain=mesh, degree=degree)
#
# boundary_conditions = {"hole": "PEC"}
#
#
# s = Grating2D(
#     geom,
#     epsilon,
#     mu,
#     source=pw,
#     degree=degree,
#     polarization=polarization,
#     boundary_conditions=boundary_conditions,
# )
# #
# #
# # u = s.solve()
# # # list_time()
# #
# #
# #
# # # print(s.diffraction_efficiencies(1, orders=True))
# # pprint(s.diffraction_efficiencies(1, orders=True, subdomain_absorption=True))
# # #
# #
# # # # s.plot_geometry(c="k")
# # #
# # plt.clf()
# #
# # s.plot_field(nper=3)
# # s.plot_geometry(nper=3, c="k")
# plt.ion()
# f = s.formulation
#
#
# # xi = xi_a
# # chi = chi_a
#
# dx = f.dx
# ds = f.ds
#
# k0 = Constant(f.source.wavenumber)
# normal = f.geometry.unit_normal_vector
#
#
# def a(u, v, xi, where="everywhere"):
#     form = -inner(xi * grad(u), grad(v))
#     return form * dx(where)
#
#
# def b(u, v, xi, where="everywhere"):
#     form = 0.5 * dot(xi * (grad(u) * v - u * grad(v)), normal)
#     # form =  dot(xi * (grad(u) * v ), normal)
#     return form * ds(where)
#
#
# def c(u, v, chi, where="everywhere"):
#     form = k0 ** 2 * chi * u * v
#     return form * dx(where)
#
#
# def F(u, v, xi, chi, where="everywhere"):
#     return a(u, v, xi, where=where) + c(u, v, chi, where=where)
#
#
# u = f.trial * f.phasor
# v = f.test * f.phasor.conj
#
# ### --------
# xi = f.xi.as_subdomain()
# chi = f.chi.as_subdomain()
# xi_a = f.xi.build_annex(domains=f.souce_domains, reference=f.reference).as_subdomain()
# chi_a = f.chi.build_annex(domains=f.souce_domains, reference=f.reference).as_subdomain()
#
# list_time()
# u1 = f.annex_field["as_subdomain"]["stack"]
# F0 = F(u, v, xi, chi) + F(u1, v, xi - xi_a, chi - chi_a, where=f.souce_domains)
# if f.polarization == "TM":
#     for bnd in f.pec_boundaries:
#         # F0 += b( u + u1, v, chi , where=bnd)
#         F0 -= dot(xi * (grad(u1) * v ), normal) *ds(bnd)
#         # F0 += b(u1, v, chi , where=bnd)
# ### --------
# # xi = f.xi.as_property()
# # chi = f.chi.as_property()
# # xi_a = f.xi.build_annex(domains=f.souce_domains, reference=f.reference).as_property()
# # chi_a = f.chi.build_annex(domains=f.souce_domains, reference=f.reference).as_property()
#
# # u1 = f.annex_field["as_dict"]["stack"]
# # F0 = 0
# # for dom in geom.domains:
# #     F0 += F(u, v, xi[dom], chi[dom], dom)
# #
# #
# # for dom in f.souce_domains:
# #     F0 += F(u1[dom], v, xi[dom] - xi_a[dom], chi[dom] - chi_a[dom], dom)
#
# F0 = (F0.real + F0.imag)
# ### --------
# s.matrix = assemble(dolfin.lhs(F0))
# s.vector = assemble(dolfin.rhs(F0))
# list_time()
# s.apply_boundary_conditions()
# list_time()
# s.solve_system()
# list_time()
# effs = s.diffraction_efficiencies(2, orders=True, subdomain_absorption=True)
# list_time()
# pprint(effs)
# #
# # element = f.function_space.split()[0].ufl_element()
# # V_vect = dolfin.VectorFunctionSpace(
# #     f.geometry.mesh, element.family(), element.degree()
# # )
# # u = dolfin.Function(V_vect)
# # u = dolfin.Function(f.function_space)
# # solver.solve(u.vector(), rhs)
# # u = Complex(*u)
# # utot = u * f.phasor + u1_
# # plotcplx(f.phasor * u, mesh=geom.mesh)
# u = s.solution["periodic"]
# utot = s.solution["total"]
# plotcplx(u)
# plotcplx(utot)
#
# # plotcplx(u1, mesh=geom.mesh)
