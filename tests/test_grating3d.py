#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

from pprint import pprint

from gyptis.grating3d import *

# dolfin.parameters["form_compiler"]["quadrature_degree"] = 5


def test_grating3d():
    # if __name__ == "__main__":

    ##  ---------- incident wave ----------
    lambda0 = 500
    theta0 = 0 * pi / 180
    phi0 = 0 * pi / 180
    psi0 = 0 * pi / 180

    ##  ---------- geometry ----------
    grooove_thickness = 50
    hole_radius = 500 / 2
    period = (1000, 1000)
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

    N_d_order = 1
    degree = 2

    parmesh_hole = parmesh * 1
    parmesh_pml = parmesh * 2 / 3

    mesh_params = dict(
        {
            "pml_bottom": parmesh_pml,
            "substrate": parmesh,
            "groove": parmesh,
            "hole": parmesh_hole,
            "superstrate": parmesh,
            "pml_top": parmesh_pml,
        }
    )

    ##  ---------- materials ----------
    eps_groove = (1.75 - 1.5j) ** 2
    eps_substrate = 1.5 ** 2

    epsilon = dict(
        {"substrate": eps_substrate, "groove": eps_groove, "hole": 1, "superstrate": 1,}
    )
    mu = dict({"substrate": 1, "groove": 1, "hole": 1, "superstrate": 1})

    index = dict()
    for e, m in zip(epsilon.items(), mu.items()):
        index[e[0]] = np.mean(
            (np.array(e[1]).real.max() * np.array(m[1]).real.max()) ** 0.5
        ).real

    ##  ---------- build geometry ----------
    model = Layered3D(period, thicknesses, kill=False)

    groove = model.layers["groove"]
    substrate = model.layers["substrate"]
    superstrate = model.layers["superstrate"]
    z0 = model.z_position["groove"]

    hole = model.add_cylinder(0, 0, z0, 0, 0, z0 + grooove_thickness, hole_radius,)

    superstrate, substrate, hole, groove = model.fragment(
        [superstrate, substrate, groove], hole
    )
    # hole, groove = model.fragment(hole, groove)
    model.add_physical(groove, "groove")
    model.add_physical(hole, "hole")
    model.add_physical(substrate, "substrate")
    model.add_physical(superstrate, "superstrate")

    model.set_size(
        "substrate", lambda0 / (index["substrate"] * mesh_params["substrate"])
    )
    model.set_size(
        "superstrate", lambda0 / (index["superstrate"] * mesh_params["superstrate"])
    )
    model.set_size(
        "pml_bottom", lambda0 / (index["substrate"] * mesh_params["pml_bottom"])
    )
    model.set_size("pml_top", lambda0 / (index["superstrate"] * mesh_params["pml_top"]))
    model.set_size("groove", lambda0 / (index["groove"] * mesh_params["groove"]))
    model.set_size("hole", lambda0 / (index["hole"] * mesh_params["hole"]))

    mesh_object = model.build()

    ##  ---------- grating ----------

    g = Grating3D(
        model,
        epsilon,
        mu,
        lambda0=lambda0,
        theta0=theta0,
        phi0=phi0,
        psi0=psi0,
        degree=degree,
    )

    g.mat_degree = degree
    g.solve(direct=True)
    g.N_d_order = N_d_order

    print("  >> computing diffraction efficiencies")
    print("---------------------------------------")

    effs = g.diffraction_efficiencies(subdomain_absorption=True)
    print(effs)
