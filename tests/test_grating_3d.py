#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

from pprint import pprint

from gyptis.grating_3d import *

# dolfin.set_log_level(10)


# def test_grating_3d():
if __name__ == "__main__":

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
    parmesh = 6

    N_d_order = 1
    ninterv_integ = 41
    nb_slice = 10
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
        {
            "substrate": eps_substrate,
            "groove": eps_groove,
            "hole": 1,
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

    # def build_geometry(period,thicknesses):

    model = Layered3D(period, thicknesses, kill=False)

    groove = model.layers["groove"]
    substrate = model.layers["substrate"]
    superstrate = model.layers["superstrate"]
    z0 = model.z_position["groove"]

    hole = model.addCylinder(
        0,
        0,
        z0,
        0,
        0,
        z0 + grooove_thickness,
        hole_radius,
    )

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

    # mesh_object = model.build(
    #     interactive=True,
    #     generate_mesh=True,
    #     write_mesh=False,
    #     read_info=False,
    #     set_periodic=True,
    # )
    # xsa
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

    g.mat_degree = 2

    g.weak_form()
    g.assemble()
    g.solve(direct=True)

    dolfin.list_timings(dolfin.TimingClear.clear, [dolfin.TimingType.wall])

    g.N_d_order = N_d_order
    g.ninterv_integ = ninterv_integ
    g.nb_slice = nb_slice

    print("  >> computing diffraction efficiencies")
    print("---------------------------------------")

    effs = g.diffraction_efficiencies()

    print("diffraction efficiencies")
    print("------------------------")
    pprint(effs)
    print("R00", effs["R"][1, 1])
    print("Σ R", np.sum(effs["R"]))
    print("Σ T", np.sum(effs["T"]))
    np.sum(effs["T"])
    print("fresnel coefficients")
    print("---------------------")
    effs_stack = {"R": g.Rstack, "T": g.Tstack}
    pprint(effs_stack)

    # W0 = dolfin.FunctionSpace(g.mesh, "CG", 1)
    # W0 = dolfin.FunctionSpace(g.mesh, "DG", 0)
    # fplot = g.E[0].real  # + g.Estack_coeff
    # # fplot = abs(g.E+ g.Estack_coeff)
    # # fplot = abs(g.E+ g.Estack_coeff)
    # dolfin.File("test.pvd") << g.Eper.real#interpolate(fplot, W0)
    # dolfin.File("markers.pvd") << g.markers
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
    #
    # n, m = 0, 0
    #
    # p_n = n * 2 * np.pi / g.geom.period[0]
    # q_m = m * 2 * np.pi / g.geom.period[1]
    # alpha_n = g.alpha0 + p_n
    # beta_m = g.beta0 + q_m
    # gamma_n = {}
    # gamma = {}
    # for d in ["superstrate", "substrate"]:
    #     gamma_n[d] = np.sqrt(
    #         g.k0 ** 2 * g.epsilon[d] * g.mu[d] - alpha_n ** 2 - beta_m ** 2
    #     )
    #     gamma[d] = np.sqrt(
    #         g.k0 ** 2 * g.epsilon[d] * g.mu[d] - g.alpha0 ** 2 - g.beta0 ** 2
    #     )
    #
    # def _propa_z(*args, **kwargs):
    #     phasor_re = dolfin.Expression("cos(gamma*x[2])", *args, **kwargs)
    #     phasor_im = dolfin.Expression("sin(gamma*x[2])", *args, **kwargs)
    #     return Complex(phasor_re, phasor_im)
    #
    # dx, dy = g.geom.period
    #
    # ph_nm = g._phasor(degree=g.mat_degree, domain=g.mesh, alpha=-p_n, beta=-q_m)
    #
    # pp = _propa_z(degree=g.mat_degree, domain=g.mesh, gamma=-gamma_n["superstrate"])
    #
    # integ = (g.Eper) * ph_nm * pp
    #
    # q = 1j * gamma_n["superstrate"] * g.geom.thicknesses["superstrate"]
    # az = (np.exp(q) - 1) / q
    #
    # # Vsup = assemble(1 * g.dx("superstrate"))
    # Vsup = dx * dy * g.geom.thicknesses["substrate"]
    # enm = [assemble(i * g.dx("superstrate")) for i in integ]
    # enm = np.array([e.real + 1j * e.imag for e in enm]) / (Vsup * az)
    # enm += g.Phi[0][1::2]
    #
    # Rnm = (gamma_n["superstrate"] / g.gamma0 * enm @ enm.conj()).real
    # print(Rnm)
    #
    # ##### ----------
    #
    # pm = _propa_z(degree=g.mat_degree, domain=g.mesh, gamma=gamma_n["substrate"])
    # integ = (g.Eper + g.Phi[1][0::2]) * ph_nm * pm
    #
    # # a = assemble(pm * g.dx("substrate")) / g.geom.thicknesses["substrate"]
    # # a = a.real + 1j * a.imag
    # # a /= period[1]*period[0]
    #
    # q = -1j * gamma_n["substrate"] * g.geom.thicknesses["substrate"]
    # az = (np.exp(q) - 1) / q
    #
    # Vsub = dx * dy * g.geom.thicknesses["substrate"]
    # enm = [assemble(i * g.dx("substrate")) for i in integ]
    #
    # enm = np.array([e.real + 1j * e.imag for e in enm]) / (Vsub * az)
    #
    # enm += g.Phi[1][0::2]  # *ax*ay
    #
    # Tnm = (gamma_n["substrate"] / g.gamma0 * enm @ enm.conj()).real
    # print(Tnm)
    #
    # print(Tnm + Rnm)
