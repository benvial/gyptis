#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT


from test_scattering2d import *


def test_simple():
    degree, polarization = 1, "TE"
    pw = PlaneWave(wavelength=wavelength, angle=0, dim=2, domain=mesh, degree=degree)
    s = Scattering(geom, epsilon, mu, pw, degree=degree, polarization=polarization)
    u = s.solve()
