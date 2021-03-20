# -*- coding: utf-8 -*-
"""
3D Grating
==========

An example of a bi-periodic diffraction grating.
"""

import sys
from pprint import pprint

from gyptis import dolfin
from gyptis.grating3d import *
from gyptis.helpers import list_time, mpi_print
from gyptis.stack import *

# dolfin.set_log_level(1)

N_d_order = 0
degree = 2
rd = 30
rs = 1  # /rd

##  ---------- incident wave ----------
lambda0 = 500
theta0 = 40 * pi / 180
phi0 = 0 * pi / 180
psi0 = 0 * pi / 180


##  ---------- geometry ----------
grooove_thickness = 50

period = (lambda0 / rd, lambda0 / rd)
hole_radius = period[0] / 4

# hole_radius = 500 / 2
# period = (1000, 1000)


thicknesses = OrderedDict(
    {
        "pml_bottom": lambda0,
        "substrate": lambda0 / rs,
        "groove": grooove_thickness,
        "superstrate": lambda0 / rs,
        "pml_top": lambda0,
    }
)

##  ---------- materials ----------
eps_groove = (1.75 - 1.5j) ** 2
eps_hole = eps_groove
eps_substrate = 1.5 ** 2

epsilon = dict(
    {
        "substrate": eps_substrate,
        "groove": eps_groove,
        # "hole": eps_hole,
        "superstrate": 1,
    }
)
mu = {d: 1 for d in epsilon.keys()}


#### stack

config = OrderedDict(
    {
        "superstrate": {"epsilon": 1, "mu": 1},
        "groove": {"epsilon": eps_groove, "mu": 1, "thickness": grooove_thickness},
        "substrate": {"epsilon": eps_substrate, "mu": 1.0},
    }
)


_, _, _, _, Rstack, Tstack = get_coeffs_stack(config, lambda0, theta0, phi0, psi0)


def main(parmesh):
    ##  ---------- mesh ----------

    # parmesh *=10

    parmesh_hole = parmesh * 1.0
    parmesh_groove = parmesh_hole * 1.0
    parmesh_pml = parmesh * 1.0

    mesh_params = dict(
        {
            "pml_bottom": parmesh_pml,
            "substrate": parmesh,
            "groove": parmesh_groove,
            # "hole": parmesh_hole,
            "superstrate": parmesh,
            "pml_top": parmesh_pml,
        }
    )

    index = dict()
    for e, m in zip(epsilon.items(), mu.items()):
        index[e[0]] = np.mean(
            (np.array(e[1]).real.max() * np.array(m[1]).real.max()) ** 0.5
        ).real
    index["pml_top"] = index["superstrate"]
    index["pml_bottom"] = index["substrate"]

    ##  ---------- build geometry ----------

    mpi_print("-----------------------------")
    mpi_print(">> Building geometry and mesh")
    mpi_print("-----------------------------")

    model = Layered3D(period, thicknesses, kill=False)

    groove = model.layers["groove"]
    substrate = model.layers["substrate"]
    superstrate = model.layers["superstrate"]
    z0 = model.z_position["groove"]

    # hole = model.add_cylinder(0, 0, z0, 0, 0, z0 + grooove_thickness, hole_radius)
    #
    # superstrate, substrate, hole, groove = model.fragment(
    #     [superstrate, substrate, groove], hole
    # )

    # hole, groove = model.fragment(hole, groove)
    # model.add_physical(groove, "groove")
    # # model.add_physical(hole, "hole")
    # model.add_physical(substrate, "substrate")
    # model.add_physical(superstrate, "superstrate")

    mesh_sizes = {d: lambda0 / (index[d] * mesh_params[d]) for d in mesh_params.keys()}
    model.set_mesh_size(mesh_sizes)
    mesh_object = model.build(interactive=0)

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

    g.mat_degree = 2

    g.prepare()

    g.weak_form()

    mpi_print("-------------")
    mpi_print(">> Assembling")
    mpi_print("-------------")

    g.assemble()

    mpi_print("----------------------")
    mpi_print(">> Computing solution")
    mpi_print("----------------------")

    g.solve()

    # list_time()
    g.N_d_order = N_d_order

    mpi_print("-------------------------------------")
    mpi_print(">> Computing diffraction efficiencies")
    mpi_print("-------------------------------------")

    effs = g.diffraction_efficiencies(
        orders=True, subdomain_absorption=True, verbose=False
    )

    # list_time()
    print("diffraction efficiencies")
    print("------------------------")
    pprint(effs)
    print("R00", effs["R"][N_d_order][N_d_order])
    print("Σ R = ", np.sum(effs["R"]))
    print("Σ T = ", np.sum(effs["T"]))
    Q = sum(effs["Q"]["electric"].values()) + sum(effs["Q"]["magnetic"].values())
    print("Q   = ", Q)
    print("B   = ", effs["B"])

    # W0 = dolfin.FunctionSpace(g.mesh, "CG", 1)
    # fplot = g.solution["total"][0].real
    #
    # dolfin.File("test.pvd") << dolfin.project(fplot, W0)
    # dolfin.File("markers.pvd") << g.markers

    return effs


if __name__ == "__main__":
    #
    # try:
    #     parmesh = int(sys.argv[1])
    # except:
    #     parmesh = 5
    #
    import matplotlib.pyplot as plt

    plt.ion()
    # plt.clf()
    cv = []
    p = range(2, 20, 4)
    for ip, parmesh in enumerate(p):
        print(f"mesh refinement: {parmesh}")

        effs = main(parmesh)
        cv.append(effs)
        plt.cla()
        R00 = [effs["R"][N_d_order][N_d_order] for effs in cv]
        T00 = [effs["T"][N_d_order][N_d_order] for effs in cv]
        Q = [
            sum(effs["Q"]["electric"].values()) + sum(effs["Q"]["magnetic"].values())
            for effs in cv
        ]
        B = [effs["B"] for effs in cv]
        dr = np.log10(np.abs(R00 - Rstack))
        dt = np.log10(np.abs(T00 - Tstack))
        Qstack = 1 - Rstack - Tstack
        dq = np.log10(np.abs(Q - Qstack))
        db = np.log10(np.abs(1 - np.array(B)))

        plt.plot(p[0 : ip + 1], dr, "--ob")
        plt.plot(p[0 : ip + 1], dt, "--or")
        plt.plot(p[0 : ip + 1], dq, "--og")
        plt.plot(p[0 : ip + 1], db, "--ok")
        plt.xlabel("$N_m$")
        plt.ylabel("$R_{00}$")

        plt.pause(0.2)

    print("Rstack", Rstack)
    print("Tstack", Tstack)

    # np.savez("cv.npz", cv=cv, p=p)
    #
    # R00 = [effs["R"][N_d_order][N_d_order] for effs in cv]
    #
    # plt.plot(p, R00, "--o")
    # plt.xlabel("$N_m$")
    # plt.ylabel("$R_{00}$")
    # plt.show()
