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
from gyptis.helpers import list_time, rot_matrix_2d
from gyptis.plotting import *

plt.ion()


def test_grating_2d(parmesh=10):

    # if __name__ == "__main__":

    polarization = "TE"
    lambda0 = 1
    theta0 = 30 * np.pi / 180

    order = 2
    # parmesh = 10
    parmesh_pml = parmesh * 2 / 3
    period = 0.8
    island_width = period * 3 / 4
    island_thickness = 0.4 * period

    tr = pi / 5

    R = rot_matrix_2d(tr)
    eps_island = np.diag([6 - 0.02j, 4 - 0.021j, 3 - 0.021j])
    eps_island = R.T @ eps_island @ R

    mu_island = np.diag([5 - 0.03j, 3 - 0.02j, 2 - 0.01j])
    mu_island = R.T @ mu_island @ R

    # eps_island = 8
    # mu_island = 1
    eps_substrate = 2
    mu_substrate = 1.35
    eps_sublayer = 4

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
            "substrate": 1 * lambda0,
            "sublayer": 0.5 * lambda0,
            "groove": island_thickness * 1.3,
            "superstrate": 1 * lambda0,
            "pml_top": lambda0,
        }
    )

    model = Layered2D(period, thicknesses, kill=False)
    #
    groove = model.layers["groove"]
    sublayer = model.layers["sublayer"]
    y0 = model.y_position["groove"]
    island = model.add_rectangle(
        -island_width / 2, y0, 0, island_width, island_thickness
    )
    island, sublayer, groove = model.fragment(island, [groove, sublayer])
    # island, sublayer = model.fragment(island, sublayer)
    # groove, sublayer = model.fragment(groove, sublayer)
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

    epsilon["sublayer"] = eps

    t = -time.time()
    g.prepare()
    g.weak_form()
    g.assemble()
    g.build_system()
    g.solve()

    field = g.solution["total"]
    J = assemble(inner(field, field.conj) * g.dx("substrate")).real

    if gyptis.ADJOINT:
        dJdx = dolfin.compute_gradient(J, dolfin.Control(ctrl))

    t += time.time()
    list_time()
    print("-" * 60)
    print(f"solution time {t:.4f}s")
    print("-" * 60)

    t = -time.time()

    g.N_d_order = 2
    effsTE = g.diffraction_efficiencies(orders=True, subdomain_absorption=True)
    pprint(effsTE)

    print("Qtot", g.Qtot)

    if gyptis.ADJOINT:
        dRdx = dolfin.compute_gradient(effsTE["R"][g.N_d_order], dolfin.Control(ctrl))
        plot(dRdx, markers=g.markers)

    t += time.time()
    #
    print("-" * 60)
    print(f"postpro time {t:.4f}s")
    print("-" * 60)

    fig, ax = plt.subplots(1, 2)
    W0 = dolfin.FunctionSpace(g.mesh, "CG", 2)
    plotcplx(g.solution["diffracted"], ax=ax, W0=W0)

    ### TM polarization ####

    g.polarization = "TM"
    t = -time.time()
    g.prepare()
    g.weak_form()
    g.assemble()
    g.build_system()
    g.solve()
    t += time.time()
    list_time()
    print("-" * 60)
    print(f"solution time {t:.4f}s")
    print("-" * 60)

    t = -time.time()
    effsTM = g.diffraction_efficiencies(orders=True, subdomain_absorption=True)
    pprint(effsTM)
    print("Qtot", g.Qtot)
    t += time.time()
    print("-" * 60)
    print(f"postpro time {t:.4f}s")
    print("-" * 60)

    assert abs(effsTE["B"] - 1) < 1e-3
    assert abs(effsTM["B"] - 1) < 1e-3
    
    return g


if __name__ == "__main__":
    g = test_grating_2d(int(sys.argv[1]))
