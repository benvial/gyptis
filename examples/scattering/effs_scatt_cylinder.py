#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.1.1
# License: MIT
# See the documentation at gyptis.gitlab.io
import numpy as np
from scipy.special import h2vp, hankel2, jv, jvp


def compute_a(n, m, kr) -> float:
    J_nu_alpha = jv(n, kr)
    J_nu_malpha = jv(n, m * kr)
    J_nu_alpha_p = jvp(n, kr, 1)
    J_nu_malpha_p = jvp(n, m * kr, 1)

    H_nu_alpha = hankel2(n, kr)
    H_nu_alpha_p = h2vp(n, kr, 1)

    a_nu_num = J_nu_alpha * J_nu_malpha_p - m * J_nu_malpha * J_nu_alpha_p
    a_nu_den = H_nu_alpha * J_nu_malpha_p - m * J_nu_malpha * H_nu_alpha_p
    return a_nu_num / a_nu_den


def calculate_analytical_cross_sections(eps_cyl, eps_bg, wavelength, radius, N=50):
    n_bg = np.sqrt((eps_bg))
    m = np.sqrt((eps_cyl)) / n_bg
    kr = 2 * np.pi * radius / wavelength * n_bg
    c = 2 / kr
    q_ext = c * np.real(compute_a(0, m, kr))
    q_sca = c * np.abs(compute_a(0, m, kr)) ** 2
    for n in range(1, N + 1):
        q_ext += c * 2 * np.real(compute_a(n, m, kr))
        q_sca += c * 2 * np.abs(compute_a(n, m, kr)) ** 2
    return dict(absorption=(q_ext - q_sca), scattering=(q_sca), extinction=q_ext)
