# -*- coding: utf-8 -*-
"""
Metamaterial with elliptical inclusions
=======================================

Calculating the effective permittivity.
"""

import matplotlib.pyplot as plt
import numpy as np

import gyptis as gy

##############################################################################
# Results are compared with :cite:p:`PopovGratingBook`.

a = 1  # lattice constant
Rx = 0.3 * a  # ellipse semi-axis x
Ry = 0.4 * a  # ellipse semi-axis y
lmin = 0.05  # minimum mesh size
eps_incl = 4 - 3j

##############################################################################
# Build the geometry


lattice = gy.Lattice(dim=2, vectors=((a, 0), (0, a)))
circ = lattice.add_ellipse(a / 2, a / 2, 0, Rx, Ry)
circ, cell = lattice.fragment(circ, lattice.cell)
lattice.add_physical(cell, "background")
lattice.add_physical(circ, "inclusion")
lattice.set_size("background", lmin)
lattice.set_size("inclusion", lmin)
lattice.build()

######################################################################
# Materials

epsilon = dict(background=1.25, inclusion=eps_incl)
mu = dict(background=1, inclusion=1)


######################################################################
# Homogenization problem

hom = gy.Homogenization2D(
    lattice,
    epsilon,
    mu,
    polarization="TE",
    degree=2,
)

######################################################################
# Calculate the effective permittivity

eps_eff = hom.get_effective_permittivity()

print("Effective permittivity")
print("----------------------")
print(eps_eff)


######################################################################
# Let's visualize the potentials solutions of the two cell problems

fig, ax = plt.subplots(1, 2, figsize=(5, 2))
gy.plot(hom.solution["x"].real, geometry=lattice, ax=ax[0])
gy.plot(hom.solution["y"].real, geometry=lattice, ax=ax[1])
[a.set_axis_off() for a in ax]
ax[0].set_title("$V_x$")
ax[1].set_title("$V_y$")
plt.tight_layout()
fig.show()
