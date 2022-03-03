#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# License: MIT
# See the documentation at gyptis.gitlab.io

from collections import OrderedDict

import numpy as np
from numpy.linalg import inv
from scipy.constants import c, epsilon_0, mu_0

from ..materials import Subdomain, complex_vector
from .source import *


def field_stack_2D(phi, alpha, beta, yshift=0, degree=1, domain=None):
    alpha0_re, alpha0_im, beta0_re, beta0_im = sp.symbols(
        "alpha0_re,alpha0_im,beta0_re,beta0_im", real=True
    )
    alpha0 = alpha0_re + 1j * alpha0_im
    beta0 = beta0_re + 1j * beta0_im
    Kplus = vector((alpha0, beta0, 0))
    Kminus = vector((alpha0, -beta0, 0))
    deltaY = vector((0, sp.symbols("yshift", real=True), 0))
    pw = lambda K: sp.exp(1j * K.dot(X - deltaY))
    phi_plus_re, phi_plus_im = sp.symbols("phi_plus_re,phi_plus_im", real=True)
    phi_minus_re, phi_minus_im = sp.symbols("phi_minus_re,phi_minus_im", real=True)
    phi_plus = phi_plus_re + 1j * phi_plus_im
    phi_minus = phi_minus_re + 1j * phi_minus_im
    field = phi_plus * pw(Kplus) + phi_minus * pw(Kminus)

    expr = expression2complex_2d(
        field,
        alpha0_re=alpha.real,
        alpha0_im=alpha.imag,
        beta0_re=beta.real,
        beta0_im=beta.imag,
        phi_plus_re=phi[0].real,
        phi_plus_im=phi[0].imag,
        phi_minus_re=phi[1].real,
        phi_minus_im=phi[1].imag,
        yshift=yshift,
        degree=degree,
        domain=domain,
    )
    return expr


def field_stack_3D(phi, alpha, beta, gamma, zshift=0, degree=1, domain=None):
    alpha0_re, alpha0_im, beta0_re, beta0_im, gamma0_re, gamma0_im = sp.symbols(
        "alpha0_re, alpha0_im, beta0_re, beta0_im, gamma0_re, gamma0_im", real=True
    )
    alpha0 = alpha0_re + 1j * alpha0_im
    beta0 = beta0_re + 1j * beta0_im
    gamma0 = gamma0_re + 1j * gamma0_im
    Kplus = vector((alpha0, beta0, gamma0))
    Kminus = vector((alpha0, beta0, -gamma0))
    deltaZ = vector((0, 0, sp.symbols("zshift", real=True)))
    pw = lambda K: sp.exp(1j * K.dot(X - deltaZ))
    fields = []
    for comp in ["x", "y", "z"]:
        phi_plus_re, phi_plus_im = sp.symbols(
            f"phi_plus_{comp}_re,phi_plus_{comp}_im", real=True
        )
        phi_minus_re, phi_minus_im = sp.symbols(
            f"phi_minus_{comp}_re,phi_minus_{comp}_im", real=True
        )
        phi_plus = phi_plus_re + 1j * phi_plus_im
        phi_minus = phi_minus_re + 1j * phi_minus_im
        field = phi_plus * pw(Kplus) + phi_minus * pw(Kminus)
        fields.append(field)
    code = [[sp.printing.ccode(p) for p in f.as_real_imag()] for f in fields]
    code = np.ravel(code).tolist()
    expr = [
        dolfin.Expression(
            c,
            alpha0_re=alpha.real,
            alpha0_im=alpha.imag,
            beta0_re=beta.real,
            beta0_im=beta.imag,
            gamma0_re=gamma.real,
            gamma0_im=gamma.imag,
            phi_plus_x_re=phi[0].real,
            phi_plus_x_im=phi[0].imag,
            phi_minus_x_re=phi[1].real,
            phi_minus_x_im=phi[1].imag,
            phi_plus_y_re=phi[2].real,
            phi_plus_y_im=phi[2].imag,
            phi_minus_y_re=phi[3].real,
            phi_minus_y_im=phi[3].imag,
            phi_plus_z_re=phi[4].real,
            phi_plus_z_im=phi[4].imag,
            phi_minus_z_re=phi[5].real,
            phi_minus_z_im=phi[5].imag,
            zshift=zshift,
            degree=degree,
            domain=domain,
        )
        for c in code
    ]

    return Complex(
        as_tensor([expr[0], expr[2], expr[4]]), as_tensor([expr[1], expr[3], expr[5]])
    )


pi = np.pi


def get_coeffs_stack(config, lambda0, theta0, phi0, psi0):

    k0 = 2 * pi / lambda0
    omega = k0 * c

    eps = [d["epsilon"] for d in config.values()]
    mu = [d["mu"] for d in config.values()]
    thicknesses = [d["thickness"] for d in config.values() if "thickness" in d.keys()]

    #
    alpha0 = -k0 * np.sin(theta0) * np.cos(phi0)
    beta0 = -k0 * np.sin(theta0) * np.sin(phi0)
    gamma0 = -k0 * np.cos(theta0)
    Ex0 = np.cos(psi0) * np.cos(theta0) * np.cos(phi0) - np.sin(psi0) * np.sin(phi0)
    Ey0 = np.cos(psi0) * np.cos(theta0) * np.sin(phi0) + np.sin(psi0) * np.cos(phi0)
    Ez0 = -np.cos(psi0) * np.sin(theta0)

    def _matrix_pi(M, gamma):
        q = gamma * M
        return np.array(
            [
                [1, 1, 0, 0],
                [0, 0, 1, 1],
                [q[0, 1], -q[0, 1], -q[0, 0], q[0, 0]],
                [q[1, 1], -q[1, 1], -q[1, 0], q[1, 0]],
            ],
            dtype=complex,
        )

    def _matrix_t(gamma, e):
        t = np.zeros((4, 4), dtype=complex)
        t[0, 0] = t[2, 2] = np.exp(1j * gamma * e)
        t[1, 1] = t[3, 3] = np.exp(-1j * gamma * e)
        return t

    def _matrix_B(eps, mu):
        eps *= epsilon_0
        mu *= mu_0
        return np.array(
            [
                [omega * mu, 0, beta0],
                [0, omega * mu, -alpha0],
                [-beta0, alpha0, -omega * eps],
            ]
        )

    gamma = [
        np.sqrt(k0 ** 2 * e * m - alpha0 ** 2 - beta0 ** 2) for e, m in zip(eps, mu)
    ]
    B = [_matrix_B(e, m) for e, m in zip(eps, mu)]
    M = [inv(b) for b in B]
    Pi = [_matrix_pi(m, g) for m, g in zip(M, gamma)]
    T_ = [_matrix_t(g, e) for g, e in zip(gamma[1:-1], thicknesses)]
    Tr = T_ + [np.eye(4)]

    M1 = np.eye(4)
    p_prev = Pi[0]
    for p, t in zip(Pi[1:], Tr):
        M1 = inv(t) @ inv(p) @ p_prev @ M1
        p_prev = p

    K = inv(M1)
    Q = np.array(
        [
            [K[0, 0], 0, K[0, 2], 0],
            [K[1, 0], -1, K[1, 2], 0],
            [K[2, 0], 0, K[2, 2], 0],
            [K[3, 0], 0, K[3, 2], -1],
        ],
        dtype=complex,
    )

    ## solve
    U0 = np.array([Ex0, 0, Ey0, 0], dtype=complex)
    sol = inv(Q) @ U0

    # get coefficients by recurrence
    phi_0 = np.array([Ex0, sol[1], Ey0, sol[3]])
    phi_end = np.array([sol[0], 0, sol[2], 0])
    p_prev = Pi[0]
    phi_prev = phi_0
    phi = [phi_0]

    for p, t in zip(Pi[1:], Tr):
        phi_j = inv(t) @ inv(p) @ p_prev @ phi_prev
        phi.append(phi_j)
        p_prev = p
        phi_prev = phi_j

    # assert np.all(np.abs(phi_j - phi_end)<1e-14)

    ## Ez
    for i, (p, g, m) in enumerate(zip(phi.copy(), gamma, M)):
        phiz_plus = (m[2, 0] * p[2] - m[2, 1] * p[0]) * g
        phiz_minus = (m[2, 1] * p[1] - m[2, 0] * p[3]) * g
        phixh_plus = (m[0, 0] * p[2] - m[0, 1] * p[0]) * g
        phixh_minus = (m[0, 1] * p[1] - m[0, 0] * p[3]) * g
        phiyh_plus = (m[1, 0] * p[2] - m[1, 1] * p[0]) * g
        phiyh_minus = (m[1, 1] * p[1] - m[1, 0] * p[3]) * g
        phizh_plus = (m[2, 1] * phixh_plus - m[2, 0] * phiyh_plus) * g
        phizh_minus = (m[2, 0] * phiyh_minus - m[2, 1] * phixh_minus) * g
        phi[i] = np.append(
            phi[i],
            [
                phiz_plus,
                phiz_minus,
                phixh_plus,
                phixh_minus,
                phiyh_plus,
                phiyh_minus,
                phizh_plus,
                phizh_minus,
            ],
        )

    R = (
        1.0
        / gamma[0] ** 2
        * (
            (gamma[0] ** 2 + alpha0 ** 2) * abs(phi[0][1]) ** 2
            + (gamma[0] ** 2 + beta0 ** 2) * abs(phi[0][3]) ** 2
            + 2 * alpha0 * beta0 * np.real(phi[0][1] * phi[0][3].conjugate())
        )
    )
    T = (
        1.0
        / (gamma[0] * gamma[-1] * mu[-1])
        * (
            (gamma[-1] ** 2 + alpha0 ** 2) * abs(phi[-1][0]) ** 2
            + (gamma[-1] ** 2 + beta0 ** 2) * abs(phi[-1][2]) ** 2
            + 2 * alpha0 * beta0 * np.real(phi[-1][0] * phi[-1][2].conjugate())
        )
    )

    losses = np.array(eps).imag

    P0 = 0.5 * np.sqrt(epsilon_0 / mu_0) * np.cos(theta0)

    th = [0] + thicknesses + [0]
    tcum = np.cumsum([0] + thicknesses).tolist()
    tcum += [tcum[-1]]

    q = [
        l * e * sum(abs(p) ** 2)
        + 2
        * l
        * np.real(
            p[0]
            * p[1].conj()
            # * np.exp(-2j * g * zi)
            * (np.exp(-2j * g * zi) - 1)
            / (2j * g)
        )
        for l, p, e, zi, g in zip(losses, phi, th, tcum, gamma)
    ]

    Q_ = [-0.5 * epsilon_0 * omega / P0 * q_ for q_ in q]
    Q = sum(Q_)
    propagation_constants = alpha0, beta0, gamma
    efficiencies = dict(R=R, T=T, Q=Q)

    return phi, propagation_constants, efficiencies


def make_stack(
    geometry,
    coefficients,
    plane_wave,
    polarization="TM",
    source_domains=[],
    degree=1,
    dim=2,
):
    epsilon, mu = coefficients
    config = OrderedDict(
        {
            "superstrate": {
                "epsilon": epsilon.dict["superstrate"],
                "mu": mu.dict["superstrate"],
            },
            "substrate": {
                "epsilon": epsilon.dict["substrate"],
                "mu": mu.dict["substrate"],
            },
        }
    )
    if dim == 2:
        if polarization == "TM":
            _psi = np.pi / 2
            _phi_ind = 2
        else:
            _psi = 0
            _phi_ind = 8

        stack_output = get_coeffs_stack(
            config,
            plane_wave.wavelength,
            plane_wave.angle,
            0,
            _psi,
        )
        phi_, propagation_constants, efficiencies_stack = stack_output
        (
            alpha0,
            _,
            beta,
        ) = propagation_constants
        phi = [[p[_phi_ind], p[_phi_ind + 1]] for p in phi_]
        phi = (np.array(phi) / phi[0][0]).tolist()

        u_stack = [
            field_stack_2D(p, alpha0, b, yshift=0, domain=geometry.mesh)
            for p, b in zip(phi, beta)
        ]
        phi0 = [phi[0][0], 0]
        u_0 = field_stack_2D(phi0, alpha0, beta[0], yshift=0, domain=geometry.mesh)
        estack = {k: v for k, v in zip(config.keys(), u_stack)}

        for dom in source_domains:
            estack[dom] = estack["superstrate"]

        estack["pml_bottom"] = estack["substrate"]
        estack["pml_top"] = estack["superstrate"]

        # estack["pml_bottom"] = estack["pml_top"]= Complex(0, 0)
        e0 = {"superstrate": u_0}
        for dom in source_domains:
            e0[dom] = e0["superstrate"]
        e0["substrate"] = e0["pml_bottom"] = e0["pml_top"] = Complex(0, 0)

        ustack_coeff = Subdomain(
            geometry.markers,
            geometry.domains,
            estack,
            degree=degree,
            domain=geometry.mesh,
        )
        u0_coeff = Subdomain(
            geometry.markers, geometry.domains, e0, degree=degree, domain=geometry.mesh
        )

        inc_field = {}
        stack_field = {}
        for dom in epsilon.dict.keys():
            inc_field[dom] = e0[dom]
            stack_field[dom] = estack[dom]
    else:
        stack_output = get_coeffs_stack(
            config,
            plane_wave.wavelength,
            plane_wave.angle[0],
            plane_wave.angle[1],
            plane_wave.angle[2],
        )
        phi_, propagation_constants, efficiencies_stack = stack_output
        (
            alpha0,
            beta0,
            gamma,
        ) = propagation_constants
        phi = [p[:6] for p in phi_]

        u_stack = [
            field_stack_3D(p, alpha0, beta0, g, domain=geometry.mesh)
            for p, g in zip(phi, gamma)
        ]
        phi0 = np.zeros_like(phi[0])
        phi0[::2] = phi[0][::2]
        u_0 = field_stack_3D(phi0, alpha0, beta0, gamma[0], domain=geometry.mesh)

        ustack_coeff = []
        u0_coeff = []
        estack_list = []
        e0_list = []
        for comp in range(3):
            estack = {k: v[comp] for k, v in zip(config.keys(), u_stack)}
            for dom in source_domains:
                estack[dom] = estack["superstrate"]

            estack["pml_bottom"] = estack["substrate"]
            estack["pml_top"] = estack["superstrate"]

            # estack["pml_bottom"] = estack["pml_top"]= Complex(0, 0)
            e0 = {"superstrate": u_0}
            for dom in source_domains:
                e0[dom] = e0["superstrate"]
            e0["substrate"] = e0["pml_bottom"] = e0["pml_top"] = Complex(0, 0)

            _ustack_coeff = Subdomain(
                geometry.markers,
                geometry.domains,
                estack,
                degree=degree,
                domain=geometry.mesh,
            )
            _u0_coeff = Subdomain(
                geometry.markers,
                geometry.domains,
                e0,
                degree=degree,
                domain=geometry.mesh,
            )
            ustack_coeff.append(_ustack_coeff)
            u0_coeff.append(_u0_coeff)
            estack_list.append(estack)
            e0_list.append(e0)

        ustack_coeff = complex_vector(ustack_coeff)
        u0_coeff = complex_vector(u0_coeff)

        inc_field = {}
        stack_field = {}
        for dom in epsilon.dict.keys():
            inc_field[dom] = complex_vector([e0[dom] for e0 in e0_list])
            stack_field[dom] = complex_vector([estack[dom] for estack in estack_list])

    annex_field_dict = {"incident": inc_field, "stack": stack_field}
    annex_field_coeff = {"incident": u0_coeff, "stack": ustack_coeff}
    annex_field = dict(as_subdomain=annex_field_coeff, as_dict=annex_field_dict)
    annex_field["stack_output"] = stack_output
    annex_field["phi"] = phi
    return annex_field
