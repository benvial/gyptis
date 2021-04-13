# -*- coding: utf-8 -*-
"""
Local density of states
=======================

Calculation of the LDOS 2D in photonic crystals.
"""


# sphinx_gallery_thumbnail_number = 2

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c

from gyptis import BoxPML, Scattering
from gyptis.source import LineSource
from gyptis.helpers import list_time
import dolfin as df
from gyptis.complex import project


##############################################################################
# Reference results are taken from [Asatryan2001]_.


formu = 0

plt.ion()

pmesh = 7
wavelength = 3.5
n_cyl = 3
eps_cyl = n_cyl ** 2

a = 0.3


def create_geometry(wavelength, pml_width):
    lmin = wavelength / pmesh

    geom = BoxPML(dim=2, box_size=(16, 16), pml_width=(pml_width, pml_width),)
    box = geom.box
    cylinders = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            cyl = geom.add_circle(i, j, 0, a)
            cylinders.append(cyl)
    for i in [-4, 4]:
        for j in range(-3, 4):
            cyl = geom.add_circle(i, j, 0, a)
            cylinders.append(cyl)

    for j in [-4, 4]:
        for i in range(-3, 4):
            cyl = geom.add_circle(i, j, 0, a)
            cylinders.append(cyl)

    for i in [-5, 5]:
        cyl = geom.add_circle(i, 0, 0, a)
        cylinders.append(cyl)
        cyl = geom.add_circle(0, i, 0, a)
        cylinders.append(cyl)

    cyl = geom.add_circle(0, 0, 0, a)
    *cylinders, box = geom.fragment(cylinders, box)
    geom.add_physical(box, "box")
    [geom.set_size(pml, lmin * 1) for pml in geom.pmls]
    geom.set_size("box", lmin)
    if formu == 1:
        for i, cyl in enumerate(cylinders):
            geom.add_physical(cyl, f"cylinder_{i}")
            geom.set_size(f"cylinder_{i}", lmin / n_cyl)
    else:
        geom.add_physical(cylinders, f"cylinders")
        geom.set_size(f"cylinders", lmin / n_cyl)

    geom.build(0)
    return geom


geom = create_geometry(wavelength, pml_width=wavelength)


##############################################################################
# Define the incident plane wave and materials

ls = LineSource(wavelength=wavelength, position=(0, 7.3), domain=geom.mesh, degree=2)

epsilon = {d: eps_cyl for d in geom.domains}
epsilon["box"] = 1
mu = {d: 1 for d in geom.domains}


##############################################################################
# Scattering problem

s = Scattering(geom, epsilon, mu, ls, degree=2, polarization="TE",)
s.solve()

G = s.solution["total"]

list_time()


v = df.ln(abs(G)) / df.ln(10)
vplot = project(
    v,
    s.formulation.real_function_space,
    solver_type="cg",
    preconditioner_type="jacobi",
)


plt.clf()

m = df.plot(vplot, mode="contour", cmap="jet", levels=52)
plt.colorbar(m)

plt.xlim(-8, 8)
plt.ylim(-8, 8)

geom_lines = geom.plot_subdomains()
plt.xlabel(r"$x/d$")
plt.ylabel(r"$y/d$")
plt.tight_layout()

list_time()


nx, ny = 20, 20
X = np.linspace(0, 8, nx)
Y = np.linspace(0, 8, ny)

ldos = np.zeros((nx, ny))


for j, y in enumerate(Y):
    for i, x in enumerate(X):
        if j <= i:
            ldos[i, j] = s.local_density_of_states(x, y)
        else:
            ldos[i, j] = ldos[j, i]

        list_time()

X = np.linspace(-8, 8, 2 * nx - 1)
Y = np.linspace(-8, 8, 2 * ny -1)

LX = np.vstack([np.flipud(ldos[1:, :]), ldos])
LDOS = np.hstack([np.fliplr(LX[:, 1:]), LX])
plt.pcolor(LDOS)

v = np.log10(LDOS * np.pi * c ** 2 / (2 * ls.pulsation))

plt.figure()
m = plt.contour(X, Y, v, cmap="jet", levels=52)
plt.colorbar(m)
plt.axis("square")
plt.xlabel(r"$x/d$")
plt.ylabel(r"$y/d$")
plt.xlim(-8, 8)
plt.ylim(-8, 8)
plt.tight_layout()

######################################################################
#
# .. [Asatryan2001] A. A. Asatryan et al.,
#   Two-dimensional local density of states in two-dimensional photonic crystals
#   Opt. Express, vol. 8, no. 3, pp. 191–196, (2001).
#   `<https://www.doi.org/110.1364/OE.8.000191>`_
