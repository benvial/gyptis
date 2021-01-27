#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

import sys
import time
from pprint import pprint

import matplotlib.pyplot as plt

import gyptis
from gyptis.grating_2d import *
from gyptis.plotting import *

plt.ion()


# def test_grating_2d():

if __name__ == "__main__":

    polarization = "TE"
    lambda0 = 1
    theta0 = 0 * np.pi / 180

    order = 2
    parmesh = 10
    parmesh_pml = parmesh * 2 / 3
    period = 0.7
    island_width = period * 3 / 4
    island_thickness = 2 * period

    tr = pi / 5
    rot = lambda t: np.array(
        [[np.sin(t), -np.cos(t), 0], [np.cos(t), np.sin(t), 0], [0, 0, 1]]
    )

    R = rot(tr)
    eps_island = np.diag([6 - 0.02j, 4 - 0.021j, 3 - 0.021j])
    eps_island = R.T @ eps_island @ R

    mu_island = np.diag([5 - 0.03j, 3 - 0.02j, 2 - 0.01j])
    mu_island = R.T @ mu_island @ R

    eps_island = 8
    mu_island = 1
    eps_substrate = 4
    mu_substrate = 1
    eps_sublayer = 1

    epsilon = dict(
        {
            "substrate": eps_substrate,
            "groove": 1,
            "sublayer": eps_sublayer,
            "island": eps_island,
            "superstrate": 1,
        }
    )
    mu = dict(
        {
            "substrate": 1,
            "groove": 1,
            "sublayer": 1,
            "island": mu_island,
            "superstrate": 1,
        }
    )

    index = dict()
    for (d, e), (d, m) in zip(epsilon.items(), mu.items()):
        index[d] = np.mean(
            (np.array(e).real.max() * np.array(m).real.max()) ** 0.5
        ).real
    index["pml_bottom"] = index["substrate"]
    index["pml_top"] = index["superstrate"]

    thicknesses = OrderedDict(
        {
            "pml_bottom": lambda0,
            "substrate": 1.4*lambda0,
            "sublayer": lambda0,
            "groove": island_thickness + lambda0,
            "superstrate": 1.4*lambda0,
            "pml_top": lambda0,
        }
    )

    model = Layered2D(period, thicknesses, kill=False)
    #
    groove = model.layers["groove"]
    sublayer = model.layers["sublayer"]
    y0 = model.y_position["groove"]
    island = model.addRectangle(
        -island_width / 2, y0, 0, island_width, island_thickness
    )
    island, groove, sublayer = model.fragmentize(island, [groove, sublayer])
    # island, sublayer = model.fragmentize(island, sublayer)
    # groove, sublayer = model.fragmentize(groove, sublayer)
    model.add_physical(groove, "groove")
    model.add_physical(island, "island")
    model.add_physical(sublayer, "sublayer")
    mesh_size = dict(
        {
            "pml_bottom": parmesh_pml,
            "substrate": parmesh,
            "sublayer": parmesh,
            "groove": parmesh,
            "superstrate": parmesh,
            "island": parmesh,
            "pml_top": parmesh_pml,
        }
    )
    m = {
        d: lambda0 / (s * i) for (d, s), (d, i) in zip(mesh_size.items(), index.items())
    }

    model.set_mesh_size(m)
    #
    face_top = model.get_boundaries("pml_top")[-2]
    face_bottom = model.get_boundaries("pml_bottom")[0]
    model.add_physical(face_top, "face_top", dim=1)
    model.add_physical(face_bottom, "face_bottom", dim=1)

    # mesh_object = model.build(
    #     interactive=True, generate_mesh=True, read_info=False, write_mesh=False
    # )
    mesh_object = model.build()
    # sys.exit(0)

    mesh = model.mesh_object["mesh"]

    g = Grating2D(
        model,
        epsilon,
        mu,
        polarization=polarization,
        lambda0=lambda0,
        theta0=theta0,
        degree=order,
    )

    ctrl0 = dolfin.Expression("1", degree=2)
    Actrl = dolfin.FunctionSpace(mesh, "DG", 0)
    ctrl = dolfin.project(ctrl0, Actrl)
    eps = Complex(4, 0) * ctrl

    epsilon["sublayer"] = 1  # eps

    t = -time.time()
    g.prepare()
    g.weak_form()
    g.assemble()
    g.build_system()
    g.solve()

    field = g.u
    J = assemble(inner(field, field.conj) * g.dx("substrate")).real

    if gyptis.ADJOINT:
        dJdx = dolfin.compute_gradient(J, dolfin.Control(ctrl))

    t += time.time()
    # dolfin.list_timings(dolfin.TimingClear.clear, [dolfin.TimingType.wall])
    print("-" * 60)
    print(f"solution time {t:.4f}s")
    print("-" * 60)

    t = -time.time()
    # g.solve(direct=False)
    g.N_d_order = 0
    effsTE = g.diffraction_efficiencies(orders=True, subdomain_absorption=True)
    pprint(effsTE)  # ,sort_dicts=False)
    print("Qtot", g.Qtot)
    
    xs
    t += time.time()
    dolfin.list_timings(dolfin.TimingClear.clear, [dolfin.TimingType.wall])
    print("-" * 60)
    print(f"postpro time {t:.4f}s")
    print("-" * 60)

    # if gyptis.ADJOINT:
    #     J = effsTE["R"][1]
    #     dJdx = dolfin.compute_gradient(J, dolfin.Control(ctrl))
    #

    # cb = dolfin.plot(g.u.real + g.ustack_coeff.real, mesh=g.mesh)
    fig, ax = plt.subplots(1, 2)
    W0 = dolfin.FunctionSpace(g.mesh, "CG", 2)
    plotcplx(g.u, ax=ax, W0=W0)

    g.polarization = "TM"
    t = -time.time()
    g.prepare()
    g.weak_form()
    g.assemble()
    g.build_system()
    g.solve()
    t += time.time()
    dolfin.list_timings(dolfin.TimingClear.clear, [dolfin.TimingType.wall])
    print("-" * 60)
    print(f"solution time {t:.4f}s")
    print("-" * 60)

    t = -time.time()
    g.N_d_order = 0
    effsTM = g.diffraction_efficiencies(orders=True, subdomain_absorption=True)
    pprint(effsTM)
    print("Qtot", g.Qtot)
    t += time.time()
    print("-" * 60)
    print(f"postpro time {t:.4f}s")
    print("-" * 60)
    #
    #
    assert abs(effsTE["B"] - 1) < 1e-3
    assert abs(effsTM["B"] - 1) < 1e-3

    # plt.show()
    #
    # plt.figure()
    #
    # for d in g.domains:
    #     delt = g.annex_field["stack"][d] - g.ustack_coeff
    #     test = assemble(delt * g.dx(d))
    #     q = assemble(g.ustack_coeff * g.dx(d))
    #     if q != 0:
    #         test /= q
    #     print(f">>> {d} : {test}")
    #
    # for d in g.domains:
    #     cb = dolfin.plot(g.annex_field["stack"][d].imag - g.ustack_coeff.imag, mesh=g.mesh)
    #     plt.colorbar(cb)
    #
    # d = "pml_top"
    #
    #
    # plt.figure()
    #
    # cb = dolfin.plot(g.annex_field["stack"][d].real, mesh=g.mesh)
    # plt.colorbar(cb)

    #
    #
    #
    # for d in g.domains:
    #     if d in ["pml_top","pml_bottom"]:
    #         print("PMLS")
    #     else:
    #         cb = dolfin.plot(g.annex_field["stack"][d].real- g.ustack_coeff.real,mesh=g.mesh)
    #         plt.colorbar(cb)
    #


#
#     cb = dolfin.plot(g.u.real)
#     plt.colorbar(cb)
#     plt.show()
#
#     ex = as_tensor([1 + 0j, 0 + 0j])
#     ey = as_tensor([0 + 0j, 1 + 0j])
#     [assemble(dot(g.xi * ex, ex) * g.dx(d)) for d in g.domains]
#     [assemble(dot(g.xi * ey, ey) * g.dx(d)) for d in g.domains]
#     [assemble(dot(g.xi * ex, ey) * g.dx(d)) for d in g.domains]
#     [assemble(dot(g.xi * ey, ex) * g.dx(d)) for d in g.domains]
#
#     ex = as_tensor([1 + 0j, 0 + 0j])
#     ey = as_tensor([0 + 0j, 1 + 0j])
#     [assemble(dot(g.xi[d] * ex, ex) * g.dx(d)) for d in g.domains]
#     [assemble(dot(g.xi[d] * ey, ey) * g.dx(d)) for d in g.domains]
#     [assemble(dot(g.xi[d] * ex, ey) * g.dx(d)) for d in g.domains]
#     [assemble(dot(g.xi[d] * ey, ex) * g.dx(d)) for d in g.domains]
#
#     [assemble(g.ustack_coeff * g.dx(d)) for d in g.domains]
#
#     [assemble(g.annex_field["stack"][d] * g.dx(d)) for d in g.domains]
#     [assemble(g.annex_field["incident"][d] * g.dx(d)) for d in g.domains]
#
#     g.lambda0 *= 1.1
#     g.weak_form()
#     g.assemble_rhs()
#
#     g.solve(direct=True)
#
#     dolfin.list_timings(dolfin.TimingClear.clear, [dolfin.TimingType.wall])
#
#     # g.N_d_order=1
#
#     g.polarization = "TM"
#     g.weak_form()
#     g.assemble()
#     g.solve(direct=True)
#     dolfin.list_timings(dolfin.TimingClear.clear, [dolfin.TimingType.wall])
#     effs = g.diffraction_efficiencies(orders=True, subdomain_absorption=True)
#     pprint(effs)  # ,sort_dicts=False)
#     print("Qtot", g.Qtot)
#
#     b = 2 * np.pi / g.lambda0
#     t = {}
#     for dom in g.domains:
#         phi = np.random.rand(2) + 1j * np.random.rand(2)
#         t[dom] = field_stack_2D(phi, 0, b, yshift=0, domain=g.mesh)
#         # t[dom] = Complex(np.random.rand(1)[0] , np.random.rand(1)[0])
#
#     a1 = [assemble(t[d] * g.dx(d)) for d in g.domains]
#
#     a1 = np.array([a.real + 1j * a.imag for a in a1])
#
#     test = Subdomain(g.markers, g.domains, t, degree=2, domain=g.mesh)
#
#     a2 = [assemble(test * g.dx(d)) for d in g.domains]
#     a2 = np.array([a.real + 1j * a.imag for a in a2])
#
#     print(a1)
#     print(a2)
#     err = np.abs(a1 - a2) / np.abs(a1)
#     print(err)
#
#     err_re = (a1 - a2).real / (a1).real
#     err_im = (a1 - a2).imag / (a1).imag
#     for i, d in enumerate(g.domains):
#         print(f">>> {d} : {err[i]}")
#         print(f">>> {d} : {err_re[i]} (real)")
#         print(f">>> {d} : {err_im[i]} (imag)")
#
#
# # q = [assemble(g.annex_field["stack"][d] * g.dx(d)) for d in g.domains]
# # q = np.array([a.real + 1j * a.imag for a in q])
# #
# # np.sum(q)
# # assemble(g.ustack_coeff * g.dx)
# #
# # q = [assemble(g.ustack_coeff * g.dx(d)) for d in g.domains]
# # q = np.array([a.real + 1j * a.imag for a in q])
# # np.sum(q)

#
# # dolfin.File("test.pvd") << project(Estack[0].real[0], W0)
# dolfin.File("test.pvd") << project(test.real, W0)
