#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# License: MIT
# See the documentation at gyptis.gitlab.io


"""
A scattering simulation in 2D
-----------------------------

In this example we will study the diffraction of a plane wave by a nanorod

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
# Silicon permittivity at this wavelength:

epsilon_rod = 13.646 - 1j * 0.048


############################################################################
# We now define the geometry using the class :class:`~gyptis.BoxPML` a
# subclass of :class:`~gyptis.Geometry`.
# This is a rectangular box in 2D (with the argument `dim=2`) centered at the
# origin by default and surrounded by Cartesian Perfectly Matched Layers.
# The important arguments are its size `box_size` and  the witdh of the
# PMLs along :math:`x` and :math:`y`.
# And optinal argument ``Rcalc`` can be used
# to build a circular boundary containing the scatterer(s) in order to compute
# cross sections.


geom = gy.BoxPML(
    dim=2,
    box_size=(5 * wavelength, 5 * wavelength),
    pml_width=(wavelength, wavelength),
    Rcalc=2.4 * wavelength,
)


############################################################################
# Now we add a circular rod.
radius = wavelength * 0.5
rod = geom.add_circle(0, 0, 0, radius)

############################################################################
# We use the boolean operation :meth:`~gyptis.BoxPML.fragment` to substract
# the rod from the box and get the remaining entities:

rod, *box = geom.fragment(rod, geom.box)


############################################################################
# Add physical domains:

geom.add_physical(box, "box")
geom.add_physical(rod, "rod")

############################################################################
# And set the mesh sizes. A good practice is to have a mesh size that
# is smaller than the wavelength in the media to resolve the field
# so ``size = wavelength / (n*pmesh)``, with ``n`` the refractive index.

pmesh = 10
geom.set_pml_mesh_size(wavelength / pmesh)
geom.set_size("box", wavelength / pmesh)
geom.set_size("rod", wavelength / (pmesh * np.real(epsilon_rod) ** 0.5))

############################################################################
# We can now build and mesh the geometry. The method
# :meth:`~gyptis.BoxPML.build` takes an ``interactive`` boolean argument
# to open and wisualize the geometry in ``gmsh`` (usefull for debugging).

geom.build()

############################################################################
# .. attention::
#       A geometry object cannot be modified after the method
#       :meth:`~gyptis.BoxPML.build` has been called: you should create a new object,
#       define the geometry and set mesh parameters before building.


############################################################################
# Visualize the mesh:

fig, ax = plt.subplots(figsize=(2, 2))
geom.plot_subdomains(ax=ax)
geom.plot_mesh(ax=ax, color="red", lw=0.2)
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


############################################################################
# We define the incident plane wave and plot it. The angle is in radian and
# ``theta=0`` corresponds to a wave travelling from the bottom.


pw = gy.PlaneWave(wavelength=wavelength, angle=0, dim=2, domain=geom.mesh, degree=2)

fig, ax = plt.subplots(1, 2, figsize=(4, 1.8))
fig, ax, maps, cb = pw.plot(ax=ax)
plt.axis("scaled")
for a in ax[:2]:
    geom.plot_subdomains(ax=a)
    a.set_xlabel("$x$ (μm)")
    a.set_ylabel("$y$ (μm)")
plt.tight_layout()


############################################################################
# Initialize the simulation. By default, materials properties (``epsilon`` and
# ``mu`` arguments are ``None``, and they will be initialized to unity in all
# subdomains). The values for the PMLs are constructed automatically.

epsilon = dict(box=1, rod=epsilon_rod)

simulation = gy.Scattering(
    geom,
    epsilon=epsilon,
    source=pw,
    degree=2,
    polarization="TE",
)

############################################################################
# We are now ready to solve the problem with the FEM:

simulation.solve()

############################################################################
# The attribute ``simulation.solution`` is a dictionary with keys
# ``diffracted`` and ``total`` containing the diffracted and total fields.

print(simulation.solution)

############################################################################
# Lets plot the results. By default, we visualize the real part of the total field:

fig, ax = plt.subplots(1, figsize=(2.5, 2))
simulation.plot_field(ax=ax)
geom_lines = geom.plot_subdomains()
plt.xlabel(r"$x$ (μm)")
plt.ylabel(r"$y$ (μm)")
plt.tight_layout()


############################################################################
# But we can switch to plot the module of the diffracted field:

fig, ax = plt.subplots(1, figsize=(2.5, 2))
simulation.plot_field(ax=ax, field="diffracted", type="module", cmap="inferno")
geom_lines = geom.plot_subdomains(color="white")
plt.xlabel(r"$x$ (μm)")
plt.ylabel(r"$y$ (μm)")
plt.tight_layout()


##############################################################################
# Compute cross sections (in 2D those are rather scattering width *etc.*,
# or cross sections per unit length) and check energy conservation (optical theorem).


cs = simulation.get_cross_sections()
print(f'scattering width = {cs["scattering"]:0.4f}μm')
print(f'absorption width = {cs["absorption"]:0.4f}μm')
print(f'extinction width = {cs["extinction"]:0.4f}μm')
balance = (cs["scattering"] + cs["absorption"]) / cs["extinction"]
print(f"    balance      = {balance}")
assert np.allclose(cs["extinction"], cs["scattering"] + cs["absorption"], rtol=1e-3)


##############################################################################
# Lets recalculate for TM polarization

simulationTM = gy.Scattering(
    geom,
    epsilon=epsilon,
    source=pw,
    degree=2,
    polarization="TM",
)

simulationTM.solve()
fig, ax = plt.subplots(1, figsize=(2.5, 2))
simulationTM.plot_field(ax=ax, field="diffracted", type="module", cmap="inferno")
geom_lines = geom.plot_subdomains(color="white")
plt.xlabel(r"$x$ (μm)")
plt.ylabel(r"$y$ (μm)")
plt.tight_layout()


cs = simulationTM.get_cross_sections()
print(f'scattering width = {cs["scattering"]:0.4f}μm')
print(f'absorption width = {cs["absorption"]:0.4f}μm')
print(f'extinction width = {cs["extinction"]:0.4f}μm')
balance = (cs["scattering"] + cs["absorption"]) / cs["extinction"]
print(f"    balance      = {balance}")
assert np.allclose(cs["extinction"], cs["scattering"] + cs["absorption"], rtol=1e-3)
