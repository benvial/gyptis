#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

import sys
import time
from pprint import pprint

from gyptis.grating_2d import *


def test_grating_2d():

    # if __name__ == "__main__":

    polarization = "TE"
    lambda0 = 40
    theta0 = 0 * np.pi / 180
    parmesh = 20
    parmesh_pml = parmesh * 2 / 3
    period = 20

    tr = 0  # pi / 5
    rot = lambda t: np.array(
        [[np.sin(t), -np.cos(t), 0], [np.cos(t), np.sin(t), 0], [0, 0, 1]]
    )

    R = rot(tr)
    eps_island = np.diag([6 - 0.2j, 4 - 0.21j, 3 - 0.21j])
    eps_island = R.T @ eps_island @ R
    # eps_off_diag = 3 - 10.1j
    # eps_island[0, 1] = eps_off_diag
    # eps_island[1, 0] = np.conj(eps_off_diag)
    # eps_island = 12

    mu_island = np.diag([5 - 0.0j, 3 - 0.0j, 2 - 0.0j])
    mu_island = R.T @ mu_island @ R
    # mu_off_diag = 0  # 3.3 - 1.2j
    # mu_island[1, 0] = mu_island[0, 1] = mu_off_diag
    mu_island = 1

    order = 2

    thicknesses = OrderedDict(
        {
            "pml_bottom": 1.32 * lambda0,
            "substrate": 1.3456 * lambda0,
            "sublayer": 10,
            "groove": 10,
            "superstrate": 1.3456 * lambda0,
            "pml_top": 1.22 * lambda0,
        }
    )
    #

    eps_sublayer = df.Expression(
        "3 + exp(-pow(x[0]/r,2) - pow(x[1]/r,2))", degree=0, r=period / 6
    )
    eps_sublayer = 1
    eps_substrate = 1

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

    model = Layered2D(period, thicknesses, kill=False)
    #
    groove = model.layers["groove"]
    sublayer = model.layers["sublayer"]
    y0 = model.y_position["groove"]
    island_width_top = 10
    island_width_bottom = 10
    island_thickness = 5
    island = model.addRectangle(
        -island_width_bottom / 2, y0, 0, island_width_bottom, island_thickness
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

    g = Grating2D(
        model,
        epsilon,
        mu,
        polarization=polarization,
        lambda0=lambda0,
        theta0=theta0,
        degree=order,
    )

    #
    # # df.File("test.pvd") << project(Estack[0].real[0], W0)
    # df.File("test.pvd") << project(test.real, W0)
    #
    # cds

    ### BCs
    # domains = model.subdomains["volumes"]
    # surfaces = model.subdomains["surfaces"]
    # markers_surf = model.mesh_object["markers"]["triangle"]
    # self.boundary_conditions = [
    #     DirichletBC(g.complex_space, [0] * 6, markers_surf, f, surfaces)
    #     for f in ["face_top", "face_bottom"]
    # ]
    t = -time.time()
    g.weak_form()
    #
    # df.list_timings(df.TimingClear.clear, [df.TimingType.wall])

    # g.assemble_rhs()
    # df.list_timings(df.TimingClear.clear, [df.TimingType.wall])
    # g.assemble_lhs()
    # df.list_timings(df.TimingClear.clear, [df.TimingType.wall])
    #
    g.assemble()
    g.solve(direct=True)

    t += time.time()

    df.list_timings(df.TimingClear.clear, [df.TimingType.wall])

    print(t)

    t = -time.time()
    # g.solve(direct=False)
    g.N_d_order = 0
    effs = g.diffraction_efficiencies(orders=True, subdomain_absorption=True)
    pprint(effs)  # ,sort_dicts=False)
    print("Qtot", g.Qtot)
    t += time.time()

    df.list_timings(df.TimingClear.clear, [df.TimingType.wall])
    print(t)

    import matplotlib.pyplot as plt

    plt.ion()
    # cb = df.plot(g.u.real + g.ustack_coeff.real, mesh=g.mesh)
    cb = df.plot(g.u.real, mesh=g.mesh)
    plt.colorbar(cb)
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
    #     cb = df.plot(g.annex_field["stack"][d].imag - g.ustack_coeff.imag, mesh=g.mesh)
    #     plt.colorbar(cb)
    #
    # d = "pml_top"
    #
    #
    # plt.figure()
    #
    # cb = df.plot(g.annex_field["stack"][d].real, mesh=g.mesh)
    # plt.colorbar(cb)

    #
    #
    #
    # for d in g.domains:
    #     if d in ["pml_top","pml_bottom"]:
    #         print("PMLS")
    #     else:
    #         cb = df.plot(g.annex_field["stack"][d].real- g.ustack_coeff.real,mesh=g.mesh)
    #         plt.colorbar(cb)
    #


#
#     cb = df.plot(g.u.real)
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
#     df.list_timings(df.TimingClear.clear, [df.TimingType.wall])
#
#     # g.N_d_order=1
#
#     g.polarization = "TM"
#     g.weak_form()
#     g.assemble()
#     g.solve(direct=True)
#     df.list_timings(df.TimingClear.clear, [df.TimingType.wall])
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
