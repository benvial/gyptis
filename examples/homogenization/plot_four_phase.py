#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.0.3
# License: MIT
# See the documentation at gyptis.gitlab.io
"""
Four phase material composite
=============================

Calculating the effective conductivity/resistivity.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import ellipk, ellipkm1, lpmv

import gyptis as gy


def _ellipk(m):
    return ellipkm1(m) if np.abs(m - 1) < 1e-6 else ellipk(m)


logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.WARNING)
np.set_printoptions(precision=4, suppress=True)

##############################################################################
# Results are compared with :cite:p:`Craster2001`.


class FourPhaseComposite:
    def __init__(self, l, h, rho):
        self.h = h
        self.l = l

        self.rho = np.array(rho)

        self.rho_dict = dict(
            W1=rho[0],
            W2=rho[1],
            W3=rho[2],
            W4=rho[3],
        )

        sigma1 = np.sum(rho)
        sigma2 = rho[0] * rho[2] - rho[1] * rho[3]
        sigma3 = (
            rho[0] * rho[1] * rho[2]
            + rho[0] * rho[1] * rho[3]
            + rho[0] * rho[2] * rho[3]
            + rho[1] * rho[2] * rho[3]
        )
        self.sigma = np.array([sigma1, sigma2, sigma3])

        self.Delta2 = sigma2**2 / (sigma1 * sigma3 + sigma2**2)

        self.lambda_ = np.arccos(1 - 2 * self.Delta2) / np.pi

    def build_geometry(self, lmin=0.05):
        logging.info("Building geometry and mesh")
        ax, ay = l, h  # lattice constant
        hx, hy = 0.5 * ax, 0.5 * ay
        lattice = gy.Lattice(dim=2, vectors=((ax, 0), (0, ay)))
        cell = lattice.cell
        sq1 = lattice.add_rectangle(hx, hy, 0, ax - hx, ay - hy)
        sq1, cell = lattice.fragment(sq1, cell)
        sq2 = lattice.add_rectangle(hx, 0, 0, ax - hx, hy)
        sq2, cell = lattice.fragment(sq2, cell)
        sq3 = lattice.add_rectangle(0, 0, 0, hx, hy)
        sq3, cell = lattice.fragment(sq3, cell)
        sq4 = lattice.add_rectangle(0, hy, 0, hx, ay - hy)
        sq4 = lattice.fragment(sq4, cell)
        lattice.remove_all_duplicates()
        lattice.add_physical(sq1, "W1")
        lattice.add_physical(sq2, "W2")
        lattice.add_physical(sq3, "W3")
        lattice.add_physical(sq4, "W4")
        lattice.set_size("W1", lmin)
        lattice.set_size("W2", lmin)
        lattice.set_size("W3", lmin)
        lattice.set_size("W4", lmin)
        lattice.build()
        self.lattice = lattice

    def _homogenize_numerical(self):
        logging.info("Computing homogenization problem with FEM")
        hom = gy.Homogenization2D(
            self.lattice,
            self.rho_dict,
            degree=2,
        )
        eps_eff = hom.get_effective_permittivity()[:2, :2].real
        rho_y = eps_eff[0, 0]
        rho_x = eps_eff[1, 1]
        self.model = hom
        return rho_x, rho_y

    def _homogenize_analytical(self):
        logging.info("Computing homogenization problem analytically")
        m = self.compute_m()
        self.m = m
        ratio_sigma = (
            self.l
            / self.h
            * lpmv(0, 0.5 * (self.lambda_ - 1), 2 * m - 1)
            / lpmv(0, 0.5 * (self.lambda_ - 1), 1 - 2 * m)
        )
        Q = (
            ((rho[1] + rho[2]) * (rho[3] + rho[0]))
            / ((rho[0] + rho[1]) * (rho[2] + rho[3]))
        ) ** 0.5
        p = (self.sigma[2] / self.sigma[0]) ** 0.5
        rho_x_ana = 1 / ratio_sigma * Q * p
        rho_y_ana = ratio_sigma / Q * p
        return rho_x_ana, rho_y_ana

    def check_product(self, rho, rtol=1e-3):
        rho_x, rho_y = rho
        assert np.allclose(rho_x * rho_y, self.sigma[2] / self.sigma[0], rtol=rtol)

    def compute_m(self):
        logging.info("Computing parameter m")

        def objective(m):
            # m = m[0]
            obj = np.abs(_ellipk(m) / _ellipk(1 - m) - self.l / self.h)
            logging.info(f"m = {m},  objective = {obj}")
            return obj

        logging.info("  Root finding for m")

        opt = minimize_scalar(objective, bounds=(0, 1), method="bounded")
        logging.info(opt)

        m = opt.x

        return m

    def homogenize(self, method):
        if method in ["numerical", "analytical"]:
            return (
                self._homogenize_numerical()
                if method == "numerical"
                else self._homogenize_analytical()
            )
        else:
            raise ValueError("Wrong method: choose between analytical or numerical.")


##############################################################################
# Loop over lattice size along x:

L = np.linspace(0.3, 1, 21)
h = 1
rho = [1, 4, 5, 10]

num = []
ana = []
for l in L:
    prob = FourPhaseComposite(l, h, rho)
    prob.build_geometry(lmin=0.05)
    rho_x, rho_y = prob.homogenize("numerical")
    rho_x_ana, rho_y_ana = prob.homogenize("analytical")
    prob.check_product([rho_x, rho_y], rtol=1e-2)

    print(rho_x, rho_y)
    print(rho_x_ana, rho_y_ana)
    assert np.allclose(rho_x, rho_x_ana, rtol=1e-2)
    assert np.allclose(rho_y, rho_y_ana, rtol=1e-2)

    num.append([rho_x, rho_y])
    ana.append([rho_x_ana, rho_y_ana])

num = np.array(num).T
ana = np.array(ana).T

fig, ax = plt.subplots(1, 2)
ax[0].plot(L, ana[0], label="analytical")
ax[0].plot(L, num[0], "o", label="numerical")
ax[1].plot(L, ana[1])
ax[1].plot(L, num[1], "o")
ax[0].set_xlabel(r"$l/h$")
ax[1].set_xlabel(r"$l/h$")
ax[0].set_ylabel(r"$\rho_x$")
ax[1].set_ylabel(r"$\rho_y$")

ax[0].legend()
plt.tight_layout()
