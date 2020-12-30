#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

from gyptis.grating_2d import *

# def test_grating_2d():


if __name__ == "__main__":

    polarization = "TE"
    lambda0 = 40
    theta0 = 20 * np.pi / 180
    parmesh = 8
    parmesh_pml = parmesh * 2 / 3
    period = 20

    tr = pi / 5
    rot = lambda t: np.array(
        [[np.sin(t), -np.cos(t), 0], [np.cos(t), np.sin(t), 0], [0, 0, 1]]
    )

    R = rot(tr)
    eps_island = np.diag([6 - 0.01j, 4 - 0.02j, 3 - 0.03j])
    eps_island = R.T @ eps_island @ R
    # eps_off_diag = 3 - 10.1j
    # eps_island[0, 1] = eps_off_diag
    # eps_island[1, 0] = np.conj(eps_off_diag)
    # eps_island = 6 - 1j
    mu_island = np.diag([5 - 0.01j, 3 - 0.02j, 2 - 0.01j])
    mu_island = R.T @ mu_island @ R
    # mu_off_diag = 0  # 3.3 - 1.2j
    # mu_island[1, 0] = mu_island[0, 1] = mu_off_diag
    # mu_island = 1

    order = 2

    thicknesses = OrderedDict(
        {
            "pml_bottom": 1 * lambda0,
            "substrate": 1 * lambda0,
            "sublayer": 10,
            "groove": 10,
            "superstrate": 1 * lambda0,
            "pml_top": 1 * lambda0,
        }
    )
    #

    eps_sublayer = df.Expression(
        "3 + exp(-pow(x[0]/r,2) - pow(x[1]/r,2))", degree=0, r=period / 6
    )
    eps_sublayer = 2

    epsilon = dict(
        {
            "substrate": 3,
            "groove": 1,
            "sublayer": eps_sublayer,
            "island": eps_island,
            "superstrate": 1,
        }
    )
    mu = dict(
        {
            "substrate": 2,
            "groove": 1,
            "sublayer": 1,
            "island": mu_island,
            "superstrate": 1,
        }
    )

    index = dict()
    for e, m in zip(epsilon.items(), mu.items()):
        try:
            index[e[0]] = np.mean(
                (np.array(e[1]).real.max() * np.array(m[1]).real.max()) ** 0.5
            ).real
        except:
            index[e[0]] = 3

    model = Layered2D(period, thicknesses, kill=False)

    # model.removeAllDuplicates()
    # model.synchronize()

    # pg = gmsh.model.getPhysicalGroups()
    # print(pg)
    # # print(model._phys_groups)
    #
    # pgnames = [gmsh.model.getPhysicalName(*p) for p in pg]
    # print(pgnames)
    #
    groove = model.layers["groove"]
    y0 = model.y_position["groove"]
    island_width_top = 10
    island_width_bottom = 10
    island_thickness = 5
    #
    #
    island = model.addRectangle(
        -island_width_bottom / 2, y0, 0, island_width_bottom, island_thickness
    )
    island, groove = model.fragmentize(island, groove)

    model.removeAllDuplicates()
    # model.healShapes()
    model.synchronize()
    model.add_physical(groove, "groove")
    model.add_physical(island, "island")
    # for id, layer in zip(model.thicknesses.keys(),model._phys_groups):
    # model.add_physical(layer, id)

    #
    sub = model.subdomains["surfaces"]["substrate"]
    sup = model.subdomains["surfaces"]["superstrate"]
    pmltop = model.subdomains["surfaces"]["pml_top"]
    pmlbot = model.subdomains["surfaces"]["pml_bottom"]
    sublayer = model.subdomains["surfaces"]["sublayer"]

    model.set_size(sub, lambda0 / (index["substrate"] * parmesh))
    model.set_size(sup, lambda0 / (index["superstrate"] * parmesh))
    model.set_size(pmlbot, lambda0 / (index["substrate"] * parmesh_pml))
    model.set_size(pmltop, lambda0 / (index["superstrate"] * parmesh_pml))
    model.set_size(groove, lambda0 / (index["groove"] * parmesh))
    model.set_size(island, lambda0 / (index["island"] * parmesh))
    model.set_size(sublayer, lambda0 / (index["sublayer"] * parmesh))
    face_top = model.get_boundaries(pmltop)[-2]
    face_bottom = model.get_boundaries(pmlbot)[0]

    gmsh.model.removePhysicalGroups()

    model.add_physical(face_top, "face_top", dim=1)
    model.add_physical(face_bottom, "face_bottom", dim=1)

    for sub, num in model.subdomains["surfaces"].items():
        model.add_physical(num, sub, dim=2)

    # model.synchronize()
    # ent = model.getEntities(dim=2)
    # print("getEntities")
    # print(ent)
    # pg = gmsh.model.getPhysicalGroups()
    # print("getPhysicalGroups")
    # print(pg)
    # # print(model._phys_groups)
    # pgnames = [gmsh.model.getPhysicalName(*p) for p in pg]
    # print("getPhysicalName")
    # print(pgnames)
    # print('model.subdomains["surfaces"]')
    #
    # print(model.subdomains["surfaces"])

    # mesh_object = model.build(
    #     interactive=True, generate_mesh=True, read_info=False, write_mesh=False
    # )
    mesh_object = model.build()

    # e = (3 * np.eye(3, dtype=complex)).tolist()
    # m = (np.eye(3, dtype=complex)).tolist()
    # epsilon = dict({"groove": m, "island": e})
    # mu = dict({"groove": m, "island": m})

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

    from pprint import pprint

    g.weak_form()
    g.assemble()
    g.solve(direct=True)
    # g.solve(direct=False)
    df.list_timings(df.TimingClear.clear, [df.TimingType.wall])

    # g.N_d_order=1

    g.N_d_order = 1
    effs = g.diffraction_efficiencies(orders=True, subdomain_absorption=True)
    pprint(effs)  # ,sort_dicts=False)
    print("Qtot", g.Qtot)

    g.polarization = "TM"
    g.weak_form()
    g.assemble()
    g.solve(direct=True)
    effs = g.diffraction_efficiencies(orders=True, subdomain_absorption=True)
    pprint(effs)  # ,sort_dicts=False)
    print("Qtot", g.Qtot)
