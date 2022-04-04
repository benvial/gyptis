#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# License: MIT
# See the documentation at gyptis.gitlab.io

import matplotlib.pyplot as plt
import numpy as np

import gyptis as gy

wl = 1
geom = gy.BoxPML(dim=2, box_size=(4 * wl, 4 * wl), pml_width=(wl, wl))
box = geom.box
scatt = geom.add_circle(0, 0, 0, wl / 2)
scatt, box = geom.fragment(scatt, geom.box)
geom.add_physical(box, "box")
geom.add_physical(scatt, "scatt")
geom.set_pml_mesh_size(wl / 5)
geom.set_size("box", wl / 6)
geom.set_size("scatt", wl / 20)
geom.build()

pw = gy.PlaneWave(wavelength=wl, angle=0, dim=2, domain=geom.mesh, degree=2)
epsilon = dict(box=1, scatt=3)
mu = dict(box=1, scatt=1)

s = gy.Scattering(
    geom,
    epsilon,
    mu,
    pw,
    degree=2,
    polarization="TE",
)
s.solve()
s.plot_field()
# geom.plot_mesh(lw=0.1)
geom_lines = geom.plot_subdomains()
plt.xlabel(r"$x$ (nm)")
plt.ylabel(r"$y$ (nm)")
plt.tight_layout()
