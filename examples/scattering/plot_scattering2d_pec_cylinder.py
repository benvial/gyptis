# -*- coding: utf-8 -*-
"""
PEC cylinder
============

An example of scattering from a perfectly conducting cylinder
"""

import matplotlib.pyplot as plt
import numpy as np

from gyptis import BoxPML, Scattering
from gyptis.source import PlaneWave

##############################################################################
# Reference results are taken from :cite:p:`Ruppin2006`.

pmesh = 10
degree = 2

R = 1
kR = np.linspace(0.09, 10, 15)
wl = 2 * np.pi * R / kR

scs_gyptis = dict()
for polarization in ["TM", "TE"]:
    scsnorm = []

    for wavelength in wl:
        lmin = wavelength / pmesh
        Rcalc = R + 1 * R
        lbox = Rcalc * 2 * 1.1
        geom = BoxPML(
            dim=2,
            box_size=(lbox, lbox),
            pml_width=(wavelength, wavelength),
            Rcalc=Rcalc,
        )
        box = geom.box

        cyl = geom.add_circle(0, 0, 0, R)
        box = geom.cut(box, cyl)
        geom.add_physical(box, "box")
        bnds = geom.get_boundaries("box")
        cyl_bnds = bnds[1]
        geom.add_physical(cyl_bnds, "cyl_bnds", dim=1)
        [geom.set_size(pml, lmin * 0.7) for pml in geom.pmls]
        geom.set_size("box", lmin)
        geom.build()

        pw = PlaneWave(
            wavelength=wavelength, angle=0, dim=2, domain=geom.mesh, degree=degree
        )

        bcs = {"cyl_bnds": "PEC"}
        epsilon = dict(box=1)
        mu = dict(box=1)

        s = Scattering(
            geom,
            epsilon,
            mu,
            pw,
            degree=degree,
            polarization=polarization,
            boundary_conditions=bcs,
        )

        s.solve()
        SCS = s.scattering_cross_section()
        SCS_norma = SCS / (2 * R)
        scsnorm.append(SCS_norma)
    scs_gyptis[polarization] = scsnorm


color = dict(TM="#da8555", TE="#4fb4a5")
for polarization in ["TM", "TE"]:
    scs_file = f"scs_pec_{polarization}.csv"
    benchmark = np.loadtxt(scs_file, delimiter=",")
    plt.plot(
        benchmark[:, 0],
        benchmark[:, 1],
        c=color[polarization],
        label=f"ref. {polarization}",
    )
    plt.plot(
        kR,
        scs_gyptis[polarization],
        "o",
        c=color[polarization],
        label=f"gyptis {polarization}",
    )

plt.xlabel(r"$kR$")
plt.ylabel(r"$\sigma_{\rm s}/2R$")
plt.legend()
plt.tight_layout()
