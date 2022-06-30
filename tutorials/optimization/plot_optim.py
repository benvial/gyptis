#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# License: MIT
# See the documentation at gyptis.gitlab.io


"""
Optimizing a lens
-----------------

In this tutorial, we will perform a topology optimization of a lens

"""


# sphinx_gallery_thumbnail_number = -1

############################################################################
# We first need to import the Python packages

import matplotlib.colors as mplcolors
import matplotlib.pyplot as plt
import numpy as np

import gyptis as gy

plt.ion()
plt.close("all")


############################################################################
# Importantly, we enable automatic differentiation:

gy.use_adjoint(True)

import gyptis.optimize as go
import gyptis.utils as gu

############################################################################
# Define EM and geometrical parameters

wavelength = 550  # wavelength of operation (in mm)
Lx, Ly = 4800, 1816.2  # box size
lx, ly = 3000, 250  # lens size
pml_dist = 250
f = 726.5
rtarget = wavelength / 10
yf = -Ly / 2 + pml_dist + ly + f

waist = 1500
position = 0, pml_dist + lx / 2

pmesh = 6

epsilon_min = 1
epsilon_max = 3.48**2
################################################################
# Optimization parameters

filtering_type = "density"
# filtering_type = "sensitivity"
rfilt = ly / 5
maxiter = 17
threshold = (0, 7)


############################################################################
# We now define the geometry:

geom = gy.BoxPML(
    dim=2,
    box_size=(Lx, Ly),
    pml_width=(wavelength, wavelength),
)


############################################################################
# Now we add a recangular design domain where the permittivity will be optimized

design = geom.add_rectangle(-lx / 2, +pml_dist - Ly / 2, 0, lx, ly)
design, box = geom.fragment(design, geom.box)
# sub = geom.add_rectangle(-Lx / 2, -Ly / 2, 0, Lx, pml_dist)
# sub, box = geom.fragment(sub, box)
target = geom.add_circle(0, yf, 0, rtarget)
target, box = geom.fragment(target, box)
############################################################################
# Add physical domains:

geom.add_physical(box, "box")
geom.add_physical(design, "design")
# geom.add_physical(sub, "sub")
geom.add_physical(target, "target")

############################################################################
# And set the mesh sizes. A good practice is to have a mesh size that
# is smaller than the wavelength in the media to resolve the field
# so ``size = wavelength / (n*pmesh)``, with ``n`` the refractive index.

geom.set_pml_mesh_size(wavelength / pmesh)
geom.set_size("box", wavelength / pmesh)
geom.set_size("target", wavelength / pmesh)
# geom.set_size("sub", wavelength / (pmesh * np.real(epsilon_max) ** 0.5))
geom.set_size("design", wavelength / (1.2 * pmesh * np.real(epsilon_max) ** 0.5))

geom.build()


############################################################################
# We define the incident plane wave. The angle is in radian and
# ``theta=0`` corresponds to a wave travelling from the bottom.


# gb = gy.GaussianBeam(
#     wavelength=wavelength,
#     angle=0,
#     waist=waist,
#     position=position,
#     Npw=21,
#     dim=2,
#     domain=geom.mesh,
#     degree=2,
# )
gb = gy.PlaneWave(
    wavelength=wavelength,
    angle=0,
    dim=2,
    domain=geom.mesh,
    degree=2,
)

############################################################################
# Define the objective function


def objective_function(epsilon_design):
    global simulation

    epsilon = dict(box=1, target=1, design=epsilon_design)

    simulation = gy.Scattering(
        geom,
        epsilon=epsilon,
        source=gb,
        degree=2,
        polarization="TE",
    )

    simulation.solve()

    Hz = simulation.solution["total"]
    nrj = gy.assemble((Hz * Hz.conj).real * simulation.dx("target")) / (
        gy.pi * rtarget**2
    )
    return -nrj


################################################################
# Function to filter and project density


def density_proj_filt(self, density, proj, filt, filtering_type):
    if filtering_type == "density":
        density_f = self.filter.apply(density) if filt else density
    else:
        density_f = density
    density_fp = (
        go.projection(density_f, beta=2**self.proj_level) if proj else density_f
    )
    fs_plot = gy.dolfin.FunctionSpace(self.submesh, "DG", 0)
    return gu.project_iterative(density_fp, fs_plot)


################################################################
# Define callback function

jopt = 0


def callback(self):
    global jopt
    proj = self.proj_level is not None
    filt = self.filter != 0
    density_fp = density_proj_filt(self, self.density, proj, filt, filtering_type)

    Hz = simulation.solution["total"]
    projspace = optimizer.fs_ctrl
    # projspace = simulation.real_function_space
    fieldplot = gu.project_iterative((Hz * Hz.conj).real, projspace)
    fig = plt.figure(figsize=(4.5, 3))
    plt.clf()
    ax = []
    gs = fig.add_gridspec(3, 1)
    ax.append(fig.add_subplot(gs[0:2, :]))

    gy.plot(
        fieldplot,
        ax=ax[0],
        cmap="inferno",
        edgecolors="face",
        vmin=0,
        vmax=12,
        # norm=mplcolors.LogNorm(vmin=1e-2, vmax=None),
    )
    geom_lines = geom.plot_subdomains(color="white", ax=ax[0])
    gy.dolfin.plot(density_fp, vmin=0, vmax=1, cmap="Greys", alpha=0.7)
    plt.xlabel(r"$x$ (μm)")
    plt.ylabel(r"$y$ (μm)")
    plt.axis("off")
    # plt.tight_layout()
    ax.append(fig.add_subplot(gs[2, :]))
    plt.sca(ax[1])

    dplot, cb2 = gy.plot(density_fp, ax=ax[1], vmin=0, vmax=1, cmap="Reds", alpha=1)
    plt.xlim(-lx / 2, lx / 2)
    y0 = -Ly / 2 + pml_dist
    plt.ylim(y0, y0 + ly)
    plt.axis("off")
    plt.suptitle(f"iteration {jopt}, objective = {-self.objective:.5f}")
    plt.tight_layout()
    gy.pause(0.1)

    jopt += 1
    return self.objective


################################################################
# Initialize the optimizer

optimizer = go.TopologyOptimizer(
    objective_function,
    geom,
    eps_bounds=(epsilon_min, epsilon_max),
    rfilt=rfilt,
    filtering_type=filtering_type,
    callback=callback,
    verbose=True,
    ftol_rel=1e-6,
    xtol_rel=1e-12,
    maxiter=maxiter,
    threshold=threshold,
)
################################################################
# Initial value

x0 = np.ones(optimizer.nvar) * 0.5

################################################################
# Optimize!

optimizer.minimize(x0)
