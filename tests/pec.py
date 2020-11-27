#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT
from pprint import pprint

from gyptis.grating_2d import *

lambda0 = 40
theta0 = 0 * np.pi / 180
parmesh = 25
parmesh_pml = parmesh * 2 / 3
period = 20
eps_island = 20  # np.diag([6 - 1j,1-0.1j,3-0.6j])
degree = 2

thicknesses = OrderedDict(
    {
        "pml_bottom": 1 * lambda0,
        "substrate": 1 * lambda0,
        "groove": 10,
        "superstrate": 1 * lambda0,
        "pml_top": 1 * lambda0,
    }
)

epsilon = dict(
    {
        "substrate": 3,
        "groove": 1,
        "superstrate": 1,
    }
)
mu = dict(
    {
        "substrate": 1,
        "groove": 1,
        "superstrate": 1,
    }
)


model = Layered2D(period, thicknesses, kill=True)


groove = model.layers["groove"]
y0 = model.y_position["groove"]
island_width_top = 5
island_width_bottom = 5
island_thickness = 1

island = model.addRectangle(
    -island_width_bottom / 2, y0, 0, island_width_bottom, island_thickness
)
# island, groove = model.fragmentize(island, groove)
groove = model.cut(
    model.dimtag(groove), model.dimtag(island), removeObject=False, removeTool=False
)
groove = groove[0][-1][-1]
model.removeAllDuplicates()
model.synchronize()
model.add_physical(groove, "groove")

bnds_island = model.get_boundaries(island)

model.add_physical(bnds_island, "PEC", 1)
model.remove(model.dimtag(island))

index = dict()
for e, m in zip(epsilon.items(), mu.items()):
    index[e[0]] = np.mean((np.array(e[1]) * np.array(m[1])) ** 0.5).real

sub = model.subdomains["surfaces"]["substrate"]
sup = model.subdomains["surfaces"]["superstrate"]
pmltop = model.subdomains["surfaces"]["pml_top"]
pmlbot = model.subdomains["surfaces"]["pml_bottom"]
model.set_size(sub, lambda0 / (index["substrate"] * parmesh))
model.set_size(sup, lambda0 / (index["superstrate"] * parmesh))
model.set_size(pmlbot, lambda0 / (index["substrate"] * parmesh_pml))
model.set_size(pmltop, lambda0 / (index["superstrate"] * parmesh_pml))
model.set_size(groove, lambda0 / (index["groove"] * parmesh))

# mesh_object = model.build(interactive=True, generate_mesh=True, write_mesh=True)

model.build()

g = Grating2D(
    model,
    epsilon,
    mu,
    polarization="TE",
    lambda0=lambda0,
    theta0=theta0,
    degree=degree,
)

curves = model.subdomains["curves"]
markers_curves = model.mesh_object["markers"]["line"]


ubnd = df.as_vector((-g.ustack_coeff.real, -g.ustack_coeff.imag))

g.boundary_conditions = [
    DirichletBC(g.complex_space, ubnd, markers_curves, "PEC", curves)
]


import matplotlib.pyplot as plt

plt.ion()

g.weak_form()
g.assemble()
g.solve(direct=True)

effs = g.diffraction_efficiencies(orders=True)
pprint(effs)

W0 = df.FunctionSpace(g.mesh, "DG", 0)


def plotcplx(toplot, ax):

    test = project(toplot, W0)
    plt.sca(ax[0])
    p = df.plot(test.real, cmap="RdBu_r")
    cbar = plt.colorbar(p)
    v = test.real.vector().get_local()
    mn, mx = min(v), max(v)
    md = 0.5 * (mx + mn)
    cbar.set_ticks([mn, md, mx])
    cbar.set_ticklabels([mn, md, mx])
    plt.sca(ax[1])
    p = df.plot(test.imag, cmap="RdBu_r")
    cbar = plt.colorbar(p)
    v = test.imag.vector().get_local()
    mn, mx = min(v), max(v)
    md = 0.5 * (mx + mn)
    cbar.set_ticks([mn, md, mx])
    cbar.set_ticklabels([mn, md, mx])


def plot(toplot):
    test = project(toplot, W0)
    p = df.plot(test, cmap="inferno")
    plt.colorbar(p)


# plotcplx(g.ustack_coeff)

plt.close("all")
fig, ax = plt.subplots(1, 2)
plotcplx(g.u, ax)
