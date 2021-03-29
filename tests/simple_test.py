#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT


from gyptis import BoxPML, Scattering
from gyptis.source import PlaneWave


def test_simple():
    polarization = "TE"
    degree = 1
    wavelength = 0.3
    geom = BoxPML(
        dim=2,
        box_size=(4 * wavelength, 4 * wavelength),
        pml_width=(wavelength, wavelength),
    )
    cyl = geom.add_circle(0, 0, 0, 0.2)
    cyl, box = geom.fragment(cyl, geom.box)
    geom.add_physical(box, "box")
    geom.add_physical(cyl, "cyl")
    geom.build()
    epsilon = dict(box=1, cyl=3)
    mu = dict(box=1, cyl=1)
    pw = PlaneWave(
        wavelength=wavelength, angle=0, dim=2, domain=geom.mesh, degree=degree
    )
    s = Scattering(geom, epsilon, mu, pw, degree=degree, polarization=polarization)
    u = s.solve()
