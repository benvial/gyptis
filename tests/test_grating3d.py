#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

from pprint import pprint

import numpy as np

from gyptis import dolfin
from gyptis.grating3d import Grating3D, Layered3D, OrderedDict
from gyptis.helpers import list_time
from gyptis.source import PlaneWave


def test_grating3d(degree=1):
    dolfin.parameters["form_compiler"]["quadrature_degree"] = 2

    ##  ---------- incident wave ----------
    p = 1000

    lambda0 = p * 3
    theta0 = 0 * np.pi / 180
    phi0 = 0 * np.pi / 180
    psi0 = 0 * np.pi / 180

    ##  ---------- geometry ----------

    period = (p, p)
    grooove_thickness = p / 20
    hole_radius = p / 4
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
    parmesh = 4

    parmesh_hole = parmesh * 1

    parmesh_groove = parmesh
    parmesh_pml = parmesh * 2 / 3

    mesh_params = dict(
        {
            "pml_bottom": parmesh_pml,
            "substrate": parmesh,
            "groove": parmesh_groove * 2,
            "hole": parmesh_hole * 2,
            "superstrate": parmesh,
            "pml_top": parmesh_pml,
        }
    )

    ##  ---------- materials ----------
    eps_groove = (1.75 - 1.5j) ** 2
    eps_hole = 1
    eps_substrate = 1.5 ** 2

    epsilon = dict(
        {
            "substrate": eps_substrate,
            "groove": eps_groove,
            "hole": eps_hole,
            "superstrate": 1,
        }
    )
    mu = dict({"substrate": 1, "groove": 1, "hole": 1, "superstrate": 1})

    index = dict()
    for e, m in zip(epsilon.items(), mu.items()):
        index[e[0]] = np.mean(
            (np.array(e[1]).real.max() * np.array(m[1]).real.max()) ** 0.5
        ).real

    ##  ---------- build geometry ----------
    geom = Layered3D(period, thicknesses, finalize=False)

    groove = geom.layers["groove"]
    substrate = geom.layers["substrate"]
    superstrate = geom.layers["superstrate"]
    z0 = geom.z_position["groove"]

    hole = geom.add_cylinder(
        0,
        0,
        z0,
        0,
        0,
        z0 + grooove_thickness,
        hole_radius,
    )

    superstrate, substrate, hole, groove = geom.fragment(
        [superstrate, substrate, groove], hole
    )
    # hole, groove = geom.fragment(hole, groove)
    geom.add_physical(groove, "groove")
    geom.add_physical(hole, "hole")
    geom.add_physical(substrate, "substrate")
    geom.add_physical(superstrate, "superstrate")

    index["pml_top"] = index["substrate"]
    index["pml_bottom"] = index["substrate"]
    pmesh = {k: lambda0 / (index[k] * mesh_params[k]) for k, p in mesh_params.items()}
    geom.set_mesh_size(pmesh)
    mesh_object = geom.build()

    pw = PlaneWave(
        wavelength=lambda0,
        angle=(np.pi / 2, 0, 0),
        dim=3,
        domain=geom.mesh,
        degree=degree,
    )
    bcs = {}
    s = Grating3D(
        geom,
        epsilon,
        mu,
        pw,
        boundary_conditions=bcs,
        degree=degree,
        periodic_map_tol=1e-6,
    )

    s.solve()
    list_time()
    print("  >> computing diffraction efficiencies")
    print("---------------------------------------")

    effs = s.diffraction_efficiencies(2, subdomain_absorption=True, orders=True)
    # effs = s.diffraction_efficiencies(2, subdomain_absorption=False, orders=True)
    pprint(effs)
    R = np.sum(effs["R"])
    T = np.sum(effs["T"])
    Q = sum([q for t in effs["Q"].values() for q in t.values()])
    print("ΣR = ", R)
    print("ΣT = ", T)
    print("ΣQ = ", Q)
    print("B  = ", effs["B"])
    list_time()
