# -*- coding: utf-8 -*-
"""
Superscatterer
==============

Topology optimization.
"""

import matplotlib.pyplot as plt
import numpy as np

import gyptis as gy
import gyptis.optimize as go

# from gyptis import dolfin as df

plt.close("all")
plt.ion()

pmesh = 8
ref_des = 1

degree = 2
eps_min = 1
eps_max = 9

wavelength = 1
R = wavelength * 1
polarization = "TM"
lmin = wavelength / pmesh
Rcalc = R + 1 * R
lbox = Rcalc * 2 * 1.1
geom = gy.BoxPML(
    dim=2,
    box_size=(lbox, lbox),
    pml_width=(wavelength, wavelength),
    Rcalc=Rcalc,
)
cyl = geom.add_circle(0, 0, 0, R)
cyl, *box = geom.fragment(cyl, geom.box)
geom.add_physical(box, "box")
geom.add_physical(cyl, "design")
bnds = geom.get_boundaries("box")
geom.set_pml_mesh_size(lmin * 0.7)
geom.set_size("box", lmin)
geom.set_size("design", lmin / eps_max ** 0.5 / ref_des)
geom.build()

pw = gy.PlaneWave(
    wavelength=wavelength, angle=gy.pi / 2, dim=2, domain=geom.mesh, degree=degree
)

epsilon = dict(box=1, design=2)
mu = dict(box=1, design=1)


def em_simulation(epsilon_design):
    epsilon["design"] = epsilon_design
    sim = gy.Scattering(
        geom,
        epsilon,
        mu,
        source=pw,
        degree=degree,
        polarization=polarization,
    )
    return sim


#
# sim = em_simulation(2)
# sim.solve()
# SCS0 = sim.scattering_cross_section()


def objfun(epsilon_design):
    sim = em_simulation(epsilon_design)
    sim.solve()
    SCS = sim.scattering_cross_section()
    return -SCS / (2 * R)


def callback(self):
    proj = self.proj_level is not None
    filt = self.filter != 0
    density = self.density
    density_f = self.filter.apply(density) if filt else density
    density_fp = (
        go.projection(density_f, beta=2 ** self.proj_level) if proj else density_f
    )
    density_fp_plot = gy.utils.project_iterative(density_fp, self.fs_sub)
    plt.clf()
    gy.plot(density_fp_plot, ax=plt.gca())
    plt.xlim(-R, R)
    plt.ylim(-R, R)
    plt.axis("off")
    plt.tight_layout()
    gy.pause(0.1)


filtering_type = "sensitivity"
filtering_type = "density"
rfilt = R / 15
opt = go.TopologyOptimizer(
    objfun,
    geom,
    eps_bounds=(eps_min, eps_max),
    threshold=(4, 8),
    rfilt=rfilt,
    filtering_type=filtering_type,
    callback=callback,
)

x0 = np.ones(opt.nvar) * 0.5
opt.minimize(x0)

xopt = opt.xopt

xopt[xopt < 0.5] = 0
xopt[xopt >= 0.5] = 1
density = gy.utils.array2function(xopt, opt.fs_sub)
epsilon_design = go.simp(density, opt.eps_min, opt.eps_max, opt.p) * gy.Complex(1, 0)

sim = em_simulation(epsilon_design)
sim.solve()
SCS = sim.scattering_cross_section()
obj_final = -SCS / (2 * R)

sim.plot_field()

density_fp_plot = gy.utils.project_iterative(density, opt.fs_sub)
plt.clf()
gy.plot(density_fp_plot, ax=plt.gca())
plt.xlim(-R, R)
plt.ylim(-R, R)
plt.axis("off")
plt.tight_layout()
gy.pause(0.1)
