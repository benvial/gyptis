#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT


import pytest
from pprint import pprint
import gyptis
from gyptis.geometry import *
from gyptis.helpers import list_time
from gyptis.plot import *
from gyptis.simulation import *
from gyptis.source import *

plt.ion()

polarization = "TE"
degree = 2
wavelength = 0.3
pmesh = 8
lmin = wavelength / pmesh

geom = BoxPML(
    dim=2, box_size=(4 * wavelength, 4 * wavelength), pml_width=(wavelength, wavelength)
)
cyl = geom.add_circle(0, 0, 0, 0.2)
cyl, box = geom.fragment(cyl, geom.box)
geom.add_physical(box, "box")
geom.add_physical(cyl, "cyl")
[geom.set_size(pml, lmin) for pml in geom.pmls]
geom.set_size("box", lmin)
geom.set_size("cyl", lmin)
geom.build()
mesh = geom.mesh_object["mesh"]
markers = geom.mesh_object["markers"]["triangle"]
mapping = geom.subdomains["surfaces"]

epsilon = dict(box=1, cyl=3)
mu = dict(box=1, cyl=1)


@pytest.mark.parametrize(
    "degree,polarization", [(1, "TE"), (2, "TE"), (1, "TM"), (2, "TM")]
)
def test_scatt2d_pw(degree, polarization):
    pw = PlaneWave(wavelength=wavelength, angle=0, dim=2, domain=mesh, degree=degree)

    s = Scatt2D(geom, epsilon, mu, pw, degree=degree, polarization=polarization)
    u = s.solve()
    list_time()
    print(assemble(u * s.formulation.dx))
    plt.ion()
    plotcplx(u)
    if gyptis.ADJOINT:
        eps_max, eps_min = 3, 1
        Actrl = dolfin.FunctionSpace(mesh, "DG", 0)
        ctrl0 = dolfin.Expression("0.1", degree=2)
        ctrl = project(ctrl0, Actrl)
        eps_lens_func = ctrl * (eps_max - eps_min) + eps_min
        dolfin.set_working_tape(dolfin.Tape())
        h = dolfin.Function(Actrl)
        h.vector()[:] = 1e-2 * np.random.rand(Actrl.dim())

        epsilon["cyl"] = eps_lens_func
        s = Scatt2D(geom, epsilon, mu, pw, degree=degree, polarization=polarization)
        field = s.solve()
        J = -assemble(inner(field, field.conj) * s.formulation.dx("box")).real
        Jhat = dolfin.ReducedFunctional(J, dolfin.Control(ctrl))
        conv_rate = dolfin.taylor_test(Jhat, ctrl, h)
        print("convergence rate = ", conv_rate)
        assert abs(conv_rate - 2) < 1e-2

        plotcplx(field)
    #

    s.source.angle = np.pi / 2
    s.assemble_rhs()
    u = s.solve_system(again=True)
    list_time()
    plotcplx(u)


@pytest.mark.parametrize(
    "degree,polarization", [(1, "TE"), (2, "TE"), (1, "TM"), (2, "TM")]
)
def test_scatt2d_ls(degree, polarization):
    gf = LineSource(wavelength, (-wavelength, 0), domain=mesh, degree=degree)
    s = Scatt2D(geom, epsilon, mu, gf, degree=degree, polarization=polarization)
    u = s.solve()
    list_time()
    tot = gf.expression + u
    s.source.position = (0, -wavelength)
    s.assemble_rhs()
    u1 = s.solve_system(again=True)
    list_time()
    tot1 = gf.expression + u1

    P = dolfin.FunctionSpace(mesh, "CG", degree)
    plotcplx(project(tot, P))
    plotcplx(project(tot1, P))

    #
    # x = np.linspace(-wavelength, wavelength, 3)
    # y = np.linspace(-wavelength, wavelength, 3)
    #
    #
    # ldos = s.local_density_of_states(x, y)
    #
    # plt.figure()
    # plt.imshow(ldos)
    # plt.axis("scaled")
    # plt.tight_layout()


@pytest.mark.parametrize("polarization", ["TE", "TM"])
def test_scatt2d_pec(polarization):

    geom = BoxPML(
        dim=2,
        box_size=(4 * wavelength, 4 * wavelength),
        pml_width=(wavelength, wavelength),
    )
    cyl = geom.add_circle(0, 0, 0, 0.2)
    box = geom.cut(geom.box, cyl)
    geom.add_physical(box, "box")
    cyl_bnds = geom.get_boundaries("box")[-1]
    geom.add_physical(cyl_bnds, "cyl_bnds", dim=1)
    [geom.set_size(pml, lmin) for pml in geom.pmls]
    geom.set_size("box", lmin)
    geom.build(0)
    mesh = geom.mesh_object["mesh"]
    markers = geom.mesh_object["markers"]["triangle"]
    mapping = geom.subdomains["surfaces"]
    epsilon = dict(box=1)
    mu = dict(box=1)

    pw = PlaneWave(wavelength=wavelength, angle=0, dim=2, domain=mesh, degree=degree)

    bcs = {"cyl_bnds": "PEC"}
    s = Scatt2D(
        geom,
        epsilon,
        mu,
        pw,
        degree=degree,
        polarization=polarization,
        boundary_conditions=bcs,
    )
    

    u = s.solve()
    list_time()
    print(assemble(u * s.formulation.dx))
    plotcplx(u)


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
# 


@pytest.mark.parametrize("polarization", ["TE","TM"])
def test_grating2d( polarization):

    from gyptis.grating2d import Layered2D, OrderedDict
    from scipy.spatial.transform import Rotation

    wavelength = 1
    lmin = wavelength / 15
    degree= 2
    period = 0.8

    thicknesses = OrderedDict(
        {
            "pml_bottom": wavelength,
            "substrate": wavelength,
            "groove": wavelength,
            "superstrate": wavelength,
            "pml_top": wavelength,
        }
    )

    rot = Rotation.from_euler("zyx", [10, 0, 0], degrees=True)
    rmat = rot.as_matrix()
    eps_diff = rmat @ np.diag((8 - 0.1j, 3 - 0.1j, 4 - 0.2j)) @ rmat.T


    mu_groove = rmat @ np.diag((2 - 0.1j, 9 - 0.1j, 3 - 0.2j)) @ rmat.T

    epsilon = dict({"substrate": 2.1, "groove": 1, "superstrate": 1, "diff": eps_diff})
    mu = dict({"substrate": 1.6, "groove": mu_groove, "superstrate": 1, "diff": 1})

    geom = Layered2D(period, thicknesses)

    R = period / 3

    yc = geom.y_position["groove"] + thicknesses["groove"] / 2
    diff = geom.add_circle(0, yc, 0, R)
    diff, groove = geom.fragment(diff, geom.layers["groove"])
    geom.add_physical(groove, "groove")
    geom.add_physical(diff, "diff")

    [geom.set_size(pml, lmin) for pml in ["pml_bottom", "pml_top"]]
    geom.set_size("groove", lmin)
    geom.set_size("diff", lmin)

    geom.build()

    theta0 = 10

    angle = np.pi / 2 - theta0 * np.pi / 180

    pw = PlaneWave(wavelength=wavelength, angle=angle, dim=2, domain=mesh, degree=degree)
    s = Grating2D(geom, epsilon, mu, source=pw, degree=degree, polarization=polarization)


    u = s.solve()
    list_time()
    print(assemble(u * s.formulation.dx))
    plt.ion()
    plotcplx(s.solution["total"])

    effs = s.diffraction_efficiencies(1, orders=True, subdomain_absorption=True)
    print(effs)
    assert abs(effs["B"] - 1) < 1e-3, "Unsatified energy balance"

    plt.ion()
    # s.plot_geometry(c="k")

    plt.clf()

    s.plot_field(nper=3)
    s.plot_geometry(nper=3, c="k")



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
