#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

import pytest
from gyptis import BoxPML
from gyptis.complex import ComplexFunctionSpace
from gyptis.formulation import *
from gyptis.geometry import *
from gyptis.materials import *
from gyptis.plot import *
from gyptis import Scattering
from scipy.constants import c

plt.ion()

dolfin.set_log_level(1)


degree = 2
wavelength = 14
pmesh = 3
lmin = wavelength / pmesh

Rx, Ry = 8, 12
eps_cyl = 41

geom = BoxPML(
    dim=2,
    box_size=(2 * wavelength, 2 * wavelength),
    pml_width=(2 * wavelength, 2 * wavelength),
)
cyl = geom.add_ellipse(0, 0, 0, Rx, Ry)
cyl, box = geom.fragment(cyl, geom.box)
geom.add_physical(box, "box")
geom.add_physical(cyl, "cyl")
[geom.set_size(pml, lmin) for pml in geom.pmls]
geom.set_size("box", lmin)
geom.set_size("cyl", lmin / eps_cyl ** 0.5)
geom.build()

epsilon = dict(box=1, cyl=eps_cyl)
mu = dict(box=1, cyl=1)

s = Scattering(
    geom,
    epsilon,
    mu,
    modal=True,
    polarization="TM",
)

wavelength_target = 428
n_eig = 16


k_target = 2 * np.pi / wavelength_target

solution = s.eigensolve(n_eig, k_target)

KNs = solution["eigenvalues"]
eig_vects = solution["eigenvectors"]

KNs = np.array(KNs)


plt.close("all")
plt.figure()

plt.plot(KNs.real, KNs.imag, "o")

for mode, eval in zip(eig_vects, KNs):
    Q = -eval.real / eval.imag * 0.5
    kre = eval.real
    f = kre * c / (2 * np.pi) * 1e-6
    if eval.imag < 0 and Q > 1.5:
        print(f)
        print(Q)
        plot(mode.real, cmap="RdBu_r")
        plt.title(fr"$f = {f:0.3f}\,$THz, $Q={Q:0.1f}$")
        H = s.formulation.get_dual(mode, 1)
        Vvect = dolfin.VectorFunctionSpace(geom.mesh, "CG", 2)
        H = project(
            H,
            Vvect,
            solver_type="cg",
            preconditioner_type="jacobi",
        )
        dolfin.plot(H.real, cmap="Greys")
        geom.plot_subdomains()
        plt.xlim(-geom.box_size[0] / 2, geom.box_size[0] / 2)
        plt.ylim(-geom.box_size[1] / 2, geom.box_size[1] / 2)
