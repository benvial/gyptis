#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# License: MIT
# See the documentation at gyptis.gitlab.io


"""
Tutorial: a scattering simulation in 2D
---------------------------------------



"""

############################################################################
# We first need to import the Python packages

import matplotlib.pyplot as plt
import numpy as np

import gyptis as gy

plt.ion()
plt.close("all")

############################################################################
# Define the wavelength of operation. This can be choosen arbitrarily, but
# it is best for numerical stability not to use too big/small numbers
# (e.g. if we work in optics, it is better to assume the
# units are in microns amd use ``wavelength = 0.8`` rather than considering
# meters and ``wavelength = 800e-9``).

wavelength = 0.8


############################################################################
# We now define the geometry using the class :class:`~gyptis.BoxPML`.
# This is a rectangular box in 2D (with the argument `dim=2`) centered at the
# origin by default and surrounded by Cartesian PErfectly Matched Layers.
# The important arguments are its size `box_size` and  the witdh of the
# PMLs along :math:`x` and :math:`y`


geom = gy.BoxPML(
    dim=2,
    box_size=(4 * wavelength, 4 * wavelength),
    pml_width=(wavelength, wavelength),
)

############################################################################
# We can now build and mesh the geometry. The method
# :meth:`~gyptis.BoxPML.build` takes an ``interactive`` boolean argument
# to open and wisualize the geometry in ``gmsh`` (usefull for debugging).

geom.build(finalize=False)

############################################################################
# Let's plot the geometry and mesh.

fig, ax = plt.subplots(figsize=(3, 3))
geom.plot_subdomains(ax=ax)
geom.plot_mesh(ax=ax, color="red")
plt.axis("equal")
plt.xlabel("$x$ (μm)")
plt.ylabel("$y$ (μm)")
plt.tight_layout()

############################################################################
# .. attention::
#       A geometry object cannot be modified after the method
#       :meth:`~gyptis.BoxPML.build` has been called. We need to create a new object,
#       define the geometry and set mesh parameters before building.


############################################################################
# Now we add a circular rod.

scatt = geom.add_circle(0, 0, 0, wavelength / 2)

############################################################################
# We use the boolean operation :meth:`~gyptis.BoxPML.fragment` to substract
# the rod from the box and get the remaining entities:

scatt, box = geom.fragment(scatt, geom.box)


############################################################################
# Add physical domains:

geom.add_physical(box, "box")
geom.add_physical(scatt, "rod")

############################################################################
# And set the mesh sizes:

geom.set_pml_mesh_size(wavelength / 5)
geom.set_size("box", wavelength / 6)
geom.set_size("rod", wavelength / 10)

############################################################################
# Now we can build it:

geom.build()

############################################################################
# Visualize the mesh:

fig, ax = plt.subplots(figsize=(3, 3))
geom.plot_subdomains(ax=ax)
geom.plot_mesh(ax=ax, color="red")
plt.axis("equal")
plt.xlabel("$x$ (μm)")
plt.ylabel("$y$ (μm)")
plt.tight_layout()

############################################################################
# Visualize the subdomains:


fig, ax = plt.subplots(figsize=(3, 2.3))
out = geom.plot_subdomains(markers=True, ax=ax)
plt.axis("scaled")
plt.xlabel("$x$ (μm)")
plt.ylabel("$y$ (μm)")
plt.tight_layout()


# pw = gy.PlaneWave(wavelength=wl, angle=0, dim=2, domain=geom.mesh, degree=2)
# epsilon = dict(box=1, scatt=3)
# mu = dict(box=1, scatt=1)
#
# s = gy.Scattering(
#     geom,
#     epsilon,
#     mu,
#     pw,
#     degree=2,
#     polarization="TE",
# )
# s.solve()
# s.plot_field()
# # geom.plot_mesh(lw=0.1)
# geom_lines = geom.plot_subdomains()
# plt.xlabel(r"$x$ (nm)")
# plt.ylabel(r"$y$ (nm)")
# plt.tight_layout()
