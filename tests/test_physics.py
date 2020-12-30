#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT


import dolfin as df
import numpy as np
import pytest

from gyptis.physics import *

df.set_log_level(20)


def test_scatt2D():

    pi = np.pi

    parmesh = 16
    lambda0 = 1
    eps_cyl = 3
    eps_box = 1

    geom = BoxPML(dim=2, box_size=(6, 6), pml_width=(1, 1))
    r = 0.9
    cyl = geom.addDisk(0, 0, 0, r, r)
    cyl, geom.box = geom.fragmentize(cyl, geom.box)
    geom.add_physical(geom.box, "box")
    geom.add_physical(cyl, "cyl")
    geom.set_size(geom.pmls, lambda0 / parmesh * 0.7)
    geom.set_size(geom.box, lambda0 / parmesh)
    geom.set_size(cyl, lambda0 / (parmesh * eps_cyl ** 0.5))
    geom.build(interactive=False)

    epsilon = dict(cyl=eps_cyl, box=eps_box)
    mu = dict(cyl=1, box=1)

    def ellipse(Rinclx, Rincly, rot_incl, x0, y0):
        c, s = np.cos(rot_incl), np.sin(rot_incl)
        Rot = np.array([[c, -s], [s, c]])
        nt = 360
        theta = np.linspace(-pi, pi, nt)
        x = Rinclx * np.sin(theta)
        y = Rincly * np.cos(theta)
        x, y = np.linalg.linalg.dot(Rot, np.array([x, y]))
        points = x + x0, y + y0
        return points

    s = Scatt2D(geom, epsilon, mu, polarization="TE")
    s.weak_form()
    #
    # t = -time.time()
    # # s.assemble()
    # s.bh = assemble(s.rhs)
    # t += time.time()
    # print(f"assembly RHS {t:0.2f}s")
    #
    # t = -time.time()
    # s.Ah = assemble(s.lhs)
    # t += time.time()
    # print(f"assembly LHS {t:0.2f}s")
    #
    # t = -time.time()
    # s.solve()
    # t += time.time()
    # print(f"solution {t:0.2f}s")

    s.assemble()
    s.solve()
    wavelengths = np.linspace(0.9, 1.5, 2)
    wl_sweep = s.wavelength_sweep(wavelengths)

    # import matplotlib.pyplot as plt
    #
    # plt.ion()
    #
    # field = (s.u).real
    #
    # nlev = 25
    # plt.figure()
    # cm = df.plot(field, cmap="RdBu_r", levels=nlev)
    # plt.colorbar(cm)
    # rod = ellipse(r, r, 0, 0, 0)
    # plt.plot(rod[0], rod[1], "w")
    #
    # plt.xlim(-geom.box_size[0] / 2, geom.box_size[0] / 2)
    # plt.ylim(-geom.box_size[1] / 2, geom.box_size[1] / 2)
    # plt.title(f"u Field (real part) fenics {s.polarization}")
    # plt.tight_layout()
    #
    #
    s = Scatt2D(geom, epsilon, mu, polarization="TM")
    s.weak_form()
    s.assemble()
    s.solve()

    #
    #

    #
    # field = (s.u).real
    #
    # plt.figure()
    # cm = df.plot(field, cmap="RdBu_r", levels=nlev)
    # plt.colorbar(cm)
    # rod = ellipse(r, r, 0, 0, 0)
    # plt.plot(rod[0], rod[1], "w")
    #
    # plt.xlim(-geom.box_size[0] / 2, geom.box_size[0] / 2)
    # plt.ylim(-geom.box_size[1] / 2, geom.box_size[1] / 2)
    # plt.title(f"u Field (real part) fenics {s.polarization}")
    # plt.tight_layout()
