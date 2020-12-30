#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

from pprint import pprint

from gyptis.grating_3d import *

df.set_log_level(10)


if __name__ == "__main__":

    ##  ---------- incident wave ----------
    lambda0 = 1
    theta0 = 0 * pi / 180
    phi0 = 0 * pi / 180
    psi0 = 0 * pi / 180

    ##  ---------- geometry ----------
    island_params = dict(width=1.25 * lambda0, thickness=lambda0)
    period = (1.25 * lambda0 * 2, 1.25 * lambda0 * 2)
    thicknesses = OrderedDict(
        {
            "pml_bottom": lambda0,
            "substrate": lambda0 / 1,
            "groove": island_params["thickness"],
            "superstrate": lambda0 / 1,
            "pml_top": lambda0,
        }
    )

    ##  ---------- mesh ----------
    parmesh = 5
    parmesh_ferro = parmesh * 1
    parmesh_pml = parmesh * 2 / 3

    mesh_params = dict(
        {
            "pml_bottom": parmesh_pml,
            "substrate": parmesh,
            "groove": parmesh,
            "island": parmesh_ferro,
            "film": parmesh_ferro,
            "superstrate": parmesh,
            "pml_top": parmesh_pml,
        }
    )

    ##  ---------- materials ----------
    eps_island = 2.25
    # eps_island = 5*np.eye(3)

    epsilon = dict(
        {
            "substrate": 1,
            "groove": 1,
            "island": 1,
            "superstrate": eps_island,
        }
    )
    mu = dict({"substrate": 1, "groove": 1, "island": 1, "superstrate": 1})

    index = dict()
    for e, m in zip(epsilon.items(), mu.items()):
        index[e[0]] = np.mean(
            (np.array(e[1]).real.max() * np.array(m[1]).real.max()) ** 0.5
        ).real

    ##  ---------- build geometry ----------

    # def build_geometry(period,thicknesses):

    model = Layered3D(period, thicknesses, kill=True)

    groove = model.layers["groove"]
    z0 = model.z_position["groove"]

    island1 = model.addBox(
        -period[0] / 2,
        -period[1] / 2,
        z0,
        island_params["width"],
        island_params["width"],
        z0 + island_params["thickness"],
    )
    island2 = model.addBox(
        0,
        0,
        z0,
        island_params["width"],
        island_params["width"],
        z0 + island_params["thickness"],
    )

    out = model.fragmentize([island1, island2], groove)
    island, groove = out[0:2], out[2:]
    model.removeAllDuplicates()
    model.synchronize()
    model.add_physical(groove, "groove")
    model.add_physical(island, "island")
    #
    sub = model.subdomains["volumes"]["substrate"]
    sup = model.subdomains["volumes"]["superstrate"]
    pmltop = model.subdomains["volumes"]["pml_top"]
    pmlbot = model.subdomains["volumes"]["pml_bottom"]
    # film = model.subdomains["volumes"]["film"]

    model.set_size(sub, lambda0 / (index["substrate"] * mesh_params["substrate"]))
    model.set_size(sup, lambda0 / (index["superstrate"] * mesh_params["superstrate"]))
    model.set_size(pmlbot, lambda0 / (index["substrate"] * mesh_params["pml_bottom"]))
    model.set_size(pmltop, lambda0 / (index["superstrate"] * mesh_params["pml_top"]))
    model.set_size(groove, lambda0 / (index["groove"] * mesh_params["groove"]))
    model.set_size(island, lambda0 / (index["island"] * mesh_params["island"]))
    # model.set_size(film, lambda0 / (index["film"] * mesh_params["film"]))

    s = model.get_periodic_bnds(model.z0, model.total_thickness)
    a = s["+x"][2]
    b = s["+x"][-1]
    s["+x"][-1] = a
    s["+x"][2] = b
    a = s["+y"][2]
    b = s["+y"][-1]
    s["+y"][-1] = a
    s["+y"][2] = b
    # # s=s_tmp

    periodic_id = {}
    for k, v in s.items():
        periodic_id[k] = [S[-1] for S in v]
    gmsh.model.mesh.setPeriodic(
        2, periodic_id["+x"], periodic_id["-x"], model.translation_x
    )
    gmsh.model.mesh.setPeriodic(
        2, periodic_id["+y"], periodic_id["-y"], model.translation_y
    )

    # mesh_object = model.build(
    #     interactive=True, generate_mesh=True, write_mesh=True, set_periodic=False
    # )
    mesh_object = model.build(set_periodic=False)
    # mesh_object = model.build()

    # model.mesh_object = mesh_object

    ##  ---------- grating ----------

    degree = 1

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

    g.weak_form()
    g.assemble()
    g.solve(direct=True)

    #
    # g.lambda0 = 60
    # g.weak_form()
    # # g.assemble_rhs()
    # g.bh[1] =assemble( g.rhs[1])
    # g.solve(direct=True)
    #

    df.list_timings(df.TimingClear.clear, [df.TimingType.wall])

    # rank = df.MPI.rank(df.MPI.comm_world)
    # if df.MPI.rank(df.MPI.comm_world) == 0:

    g.N_d_order = 2
    g.ninterv_integ = 101
    g.nb_slice = 5
    #

    print("  >> computing diffraction efficiencies")
    print("---------------------------------------")

    effs = g.diffraction_efficiencies()

    print("diffraction efficiencies")
    print("------------------------")
    pprint(effs)
    print("fresnel coefficients")
    print("---------------------")
    effs_stack = {"R": g.Rstack, "T": g.Tstack}
    pprint(effs_stack)

    W0 = df.FunctionSpace(g.mesh, "CG", 1)
    # W0 = df.FunctionSpace(g.mesh, "DG", 0)
    # fplot = g.E[0].real  # + g.Estack_coeff
    # # fplot = abs(g.E+ g.Estack_coeff)
    # # fplot = abs(g.E+ g.Estack_coeff)
    # df.File("test.pvd") << g.Eper.real#interpolate(fplot, W0)
    # df.File("markers.pvd") << g.markers
    #
    # W = g.complex_space
    # dx = g.dx
    # g.E = Function(W)
    # Etrial = TrialFunction(W)
    # Etest = TestFunction(W)
    # delta_epsilon = g.epsilon_coeff - g.epsilon_coeff_annex
    #
    # t = dot(g.Estack_coeff, Etest) * dx
    # q = t.real + t.imag
    # assemble(q)
    #
    # t = dot(g.E_stack[-1], Etest) * dx
    # q = t.real + t.imag
    # assemble(q)

    # # g.E_stack[-1]
    #
    # for d in g.domains:
    #     print(d)
    #     print("----")
    #     vec = g.Estack_coeff
    #     # vec = as_tensor([g.phasor,g.phasor,g.phasor])
    #     coef = as_tensor(np.eye(3))
    #     coef = as_tensor(np.diag((Constant(5 + 0j), 3, 4)))
    #
    #     k0 = 1 + 1 * 1j
    #     # k0 = Constant(k0)
    #     q = np.diag((k0, k0, k0))
    #     coef = tensor_const(q)
    #
    #     # coef = Constant(64)
    #
    #     # coef = g.inv_mu_coeff
    #     # vec = g.E_stack[-1]
    #     t = dot(vec, vec) * dx(d)
    #     t = dot(coef * curl(Etrial), curl(Etest)) * dx(d)
    #     q = t.real + t.imag
    #     a = assemble(q)
    #     print(a)
    #
    #
    #
    #
    #
    # k0 = 23 + 43 * 1j
    # # k0 = Constant(k0)
    # q = np.diag((k0, k0, k0))
    # coef = tensor_const(q)
    # t = dot(coef * curl(Etrial), curl(Etest)) * dx(d)
    # q = t.real + t.imag
    # a = assemble(q)
    #
    #
    #
    #
    # t = dot(invmu[d] * curl(Etrial), curl(Etest)) * dx(d)
    # q = t.real + t.imag
    # a = assemble(q)

    n, m = 0, 0

    p_n = n * 2 * np.pi / g.geom.period[0]
    q_m = m * 2 * np.pi / g.geom.period[1]
    alpha_n = g.alpha0 + p_n
    beta_m = g.beta0 + q_m
    gamma_n = {}
    gamma = {}
    for d in ["superstrate", "substrate"]:
        gamma_n[d] = np.sqrt(
            g.k0 ** 2 * g.epsilon[d] * g.mu[d] - alpha_n ** 2 - beta_m ** 2
        )
        gamma[d] = np.sqrt(
            g.k0 ** 2 * g.epsilon[d] * g.mu[d] - g.alpha0 ** 2 - g.beta0 ** 2
        )

    def _propa_z(*args, **kwargs):
        phasor_re = df.Expression("cos(gamma*x[2])", *args, **kwargs)
        phasor_im = df.Expression("sin(gamma*x[2])", *args, **kwargs)
        return Complex(phasor_re, phasor_im)

    dx, dy = g.geom.period

    ph_nm = g._phasor(degree=g.mat_degree, domain=g.mesh, alpha=-p_n, beta=-q_m)

    pp = _propa_z(degree=g.mat_degree, domain=g.mesh, gamma=-gamma_n["superstrate"])

    integ = (g.Eper) * ph_nm * pp

    q = 1j * gamma_n["superstrate"] * g.geom.thicknesses["superstrate"]
    az = (np.exp(q) - 1) / q

    # Vsup = assemble(1 * g.dx("superstrate"))
    Vsup = dx * dy * g.geom.thicknesses["substrate"]
    enm = [assemble(i * g.dx("superstrate")) for i in integ]
    enm = np.array([e.real + 1j * e.imag for e in enm]) / (Vsup * az)
    enm += g.Phi[0][1::2]

    Rnm = (gamma_n["superstrate"] / g.gamma0 * enm @ enm.conj()).real
    print(Rnm)

    ##### ----------

    pm = _propa_z(degree=g.mat_degree, domain=g.mesh, gamma=gamma_n["substrate"])
    integ = (g.Eper + g.Phi[1][0::2]) * ph_nm * pm

    # a = assemble(pm * g.dx("substrate")) / g.geom.thicknesses["substrate"]
    # a = a.real + 1j * a.imag
    # a /= period[1]*period[0]

    q = -1j * gamma_n["substrate"] * g.geom.thicknesses["substrate"]
    az = (np.exp(q) - 1) / q

    Vsub = dx * dy * g.geom.thicknesses["substrate"]
    enm = [assemble(i * g.dx("substrate")) for i in integ]

    enm = np.array([e.real + 1j * e.imag for e in enm]) / (Vsub * az)

    enm += g.Phi[1][0::2]  # *ax*ay

    Tnm = (gamma_n["substrate"] / g.gamma0 * enm @ enm.conj()).real
    print(Tnm)

    print(Tnm + Rnm)
