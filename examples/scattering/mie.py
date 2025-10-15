#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.1.2
# License: MIT
# See the documentation at gyptis.gitlab.io

"""
Mie theory functions
"""


import numpy as np
import scipy.special as sp


def sj(n, q):
    return sp.spherical_jn(n, q)


def dsj(n, q):
    return sp.spherical_jn(n, q, derivative=True)


def sy(n, q):
    return sp.spherical_yn(n, q)


def dsy(n, q):
    return sp.spherical_yn(n, q, derivative=True)


def sh1(n, q):
    return sj(n, q) + 1j * sy(n, q)


def dsh1(n, q):
    return dsj(n, q) + 1j * dsy(n, q)


def fdot(n, q, f, df):
    return 1 / q * (f(n, q) + q * df(n, q))


def sjdot(n, q):
    return fdot(n, q, sj, dsj)


def sh1dot(n, q):
    return fdot(n, q, sh1, dsh1)


def get_cross_sections_analytical(k0, a, eps_sphere=4, eps_bg=1, Nmax=25):
    k1 = k0 * (eps_bg.conjugate()) ** 0.5
    k2 = k0 * (eps_sphere.conjugate()) ** 0.5
    chi = k2 / k1

    def coeffs(n):
        q1 = sjdot(n, k1 * a) * sj(n, k2 * a) - chi * sj(n, k1 * a) * sjdot(n, k2 * a)
        q2 = sh1dot(n, k1 * a) * sj(n, k2 * a) - chi * sh1(n, k1 * a) * sjdot(n, k2 * a)

        c_over_a = -q1 / q2

        q1 = sj(n, k1 * a) * sjdot(n, k2 * a) - chi * sjdot(n, k1 * a) * sj(n, k2 * a)
        q2 = sh1(n, k1 * a) * sjdot(n, k2 * a) - chi * sh1dot(n, k1 * a) * sj(n, k2 * a)

        d_over_b = -q1 / q2
        return c_over_a, d_over_b

    Cs, Ce = 0, 0

    for n in range(1, Nmax):
        A, B = coeffs(n)
        # print(np.mean((np.abs(A)**2 + np.abs(B)**2)))
        Cs += (2 * n + 1) * (np.abs(A) ** 2 + np.abs(B) ** 2)
        Ce += -(2 * n + 1) * ((A) + (B)).real

    Cs *= 2 * np.pi / k1**2
    Ce *= 2 * np.pi / k1**2
    Ca = Ce - Cs
    return Cs, Ce, Ca
