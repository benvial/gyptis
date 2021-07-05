#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT
"""
Introduction to electromagnetic scattering: tutorial
https://www.osapublishing.org/josaa/fulltext.cfm?uri=josaa-35-1-163&id=380136
"""

import numpy as np

from gyptis import c, dolfin, epsilon_0, mu_0
from gyptis.complex import *
from gyptis.plot import *
from gyptis.scattering3d import BoxPML3D, Scatt3D
from gyptis.source import PlaneWave

plt.ion()
plt.close("all")


shared_datadir = "../../tests/data"
## needed for surface integral in parallel (mpi)
dolfin.parameters["ghost_mode"] = "shared_facet"
dolfin.parameters["form_compiler"]["quadrature_degree"] = 2

scs_file = shared_datadir + "/sphere_diel.csv"
benchmark = np.loadtxt(scs_file, delimiter=",")


eps_sphere = 4
eps_bg = 1
a = 0.25
S_sphere = a ** 2 * np.pi

### analytical

import scipy.special as sp


def get_ana_cs(k0, Nmax=30):
    k1 = k0 * (eps_bg) ** 0.5
    k2 = k0 * (eps_sphere) ** 0.5
    chi = k2 / k1

    def coeffs(n):
        q1 = sp.jvp(n, k1 * a) * sp.jv(n, k2 * a) - chi * sp.jv(n, k1 * a) * sp.jvp(
            n, k2 * a
        )
        q2 = sp.h1vp(n, k1 * a) * sp.jv(n, k2 * a) - chi * sp.hankel1(
            n, k1 * a
        ) * sp.jvp(n, k2 * a)

        c_over_a = -q1 / q2

        q1 = sp.jv(n, k1 * a) * sp.jvp(n, k2 * a) - chi * sp.jvp(n, k1 * a) * sp.jv(
            n, k2 * a
        )
        q2 = sp.hankel1(n, k1 * a) * sp.jvp(n, k2 * a) - chi * sp.h1vp(
            n, k1 * a
        ) * sp.jv(n, k2 * a)

        d_over_b = -q1 / q2
        return c_over_a, d_over_b

    Csana = 0

    for n in range(1, Nmax):
        A, B = coeffs(n)
        # print(np.mean((np.abs(A)**2 + np.abs(B)**2)))
        Csana += (2 * n + 1) * (np.abs(A) ** 2 + np.abs(B) ** 2)

    Csana *= 2 * np.pi / k1 ** 2
    return Csana


GAMMA = benchmark[:, 0]
CS_NORMA_REF = benchmark[:, 1]
circ = 2 * np.pi * a


Gamma = np.linspace(0.1, 5, 500)
lambda0 = circ / Gamma
ks = 2 * np.pi / lambda0

CSANA = get_ana_cs(ks)
CSANA_NORMA_MIE = np.array(CSANA) / S_sphere

plt.plot(GAMMA, CS_NORMA_REF, "--", c="#525252", label="Ref")
plt.plot(Gamma, CSANA_NORMA_MIE, c="#545cc7", label="Mie")
plt.legend()
plt.xlabel(r"circumfenrence/wavelength $k_0 a$")
plt.ylabel(r"normalized scattering cross section $\sigma_s / S$")
plt.tight_layout()


def compute_scs(lambda0, pmesh=2, degree=1):

    pmesh_scatt = 1 * pmesh

    b = a * 2 * 1.8
    box_size = (b, b, b)
    pml_width = (lambda0, lambda0, lambda0)

    Rcalc = (min(box_size) / 2 + a) / 2
    # Rcalc=0.9*b/2

    g = BoxPML3D(box_size=box_size, pml_width=pml_width, Rcalc=Rcalc)

    box = g.box
    sphere = g.add_sphere(0, 0, 0, a)
    sphere, sphere_cross_sections, box = g.fragment(sphere, box)

    g.add_physical([box, sphere_cross_sections], "box")
    g.add_physical(sphere, "sphere")
    surf = g.get_boundaries(sphere_cross_sections, physical=False)[0]
    smin = a / 2
    s = lambda0 / pmesh  # min(lambda0 / pmesh, smin)
    print(s)
    smin_pml = lambda0 / (0.7 * pmesh)

    for coord in ["x", "y", "z", "xy", "xz", "yz", "xyz"]:
        g.set_mesh_size({"pml" + coord: smin_pml})

    g.set_size(box, s)
    g.set_size(sphere_cross_sections, s)
    g.set_size(surf, s, dim=2)
    s = min(lambda0 / (eps_sphere ** 0.5 * pmesh_scatt), smin)
    s = lambda0 / (eps_sphere ** 0.5 * pmesh_scatt)
    g.set_size(sphere, s)
    g.build()

    epsilon = dict(sphere=eps_sphere, box=eps_bg)
    mu = dict(sphere=1, box=1)

    pw = PlaneWave(
        wavelength=lambda0, angle=(0, 0, 0), dim=3, domain=g.mesh, degree=degree
    )
    bcs = {}
    s = Scatt3D(
        g,
        epsilon,
        mu,
        pw,
        boundary_conditions=bcs,
        degree=degree,
    )

    s.solve()
    Z0 = np.sqrt(mu_0 / epsilon_0)
    S0 = 1 / (2 * Z0)
    n_out = g.unit_normal_vector
    Es = s.solution["diffracted"]
    inv_mu_coeff = s.coefficients[1].invert().as_subdomain()
    omega = s.source.pulsation
    Hs = inv_mu_coeff / Complex(0, dolfin.Constant(omega * mu_0)) * curl(Es)
    Ss = dolfin.Constant(0.5) * cross(Es, Hs.conj).real
    Ws = assemble(dot(n_out, Ss)("+") * s.dS("calc_bnds"))
    Sigma_s = Ws / S0
    return Sigma_s


#
#
#
# SCSN = []
# Gamma = np.linspace(0.5, 5, 10)


# for gamma in Gamma:
#     lambda0 = circ / gamma
#     Sigma_s = compute_scs(lambda0)
#     Sigma_s_norm = Sigma_s / S_sphere
#     print(Sigma_s_norm)
#     SCSN.append(Sigma_s_norm)
#     plt.plot(gamma, Sigma_s_norm, "o", c="#c64545",label="gyptis")
#     plt.pause(0.1)

plt.clf()
gamma = 1
degree = 2
PMESH = [3, 5, 7, 10]
SCSN = []
P = []
for pmesh in PMESH:
    lambda0 = circ / gamma
    Sigma_s = compute_scs(lambda0, pmesh=pmesh, degree=degree)
    Sigma_s_norm = Sigma_s / S_sphere
    print(Sigma_s_norm)
    SCSN.append(Sigma_s_norm)
    P.append(pmesh)
    # plt.plot(pmesh, Sigma_s_norm, "o", c="#c64545", label="gyptis")
    plt.plot(P, SCSN, "-o", c="#c64545", label="gyptis")
    plt.pause(0.1)
