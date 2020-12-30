#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

import pytest

from gyptis.stack import *


def test_stack():

    lambda0 = 1.1
    theta0 = -0.0 * pi
    phi0 = 0 * pi / 3
    psi0 = 0 * pi

    config = OrderedDict(
        {
            "superstrate": {"epsilon": 1, "mu": 1},
            "layer 1": {"epsilon": 2.3 - 0.2j, "mu": 2, "thickness": 1.7},
            "layer 2": {"epsilon": 3.5, "mu": 1, "thickness": 0.5},
            "layer 3": {"epsilon": 8.6 - 0.3j, "mu": 1, "thickness": 0.1},
            "substrate": {"epsilon": 3, "mu": 1.0},
        }
    )

    phi, alpha0, beta0, gamma, R, T = get_coeffs_stack(
        config, lambda0, theta0, phi0, psi0
    )
    thicknesses = [d["thickness"] for d in config.values() if "thickness" in d.keys()]

    tcum = np.cumsum([0] + thicknesses).tolist()
    tcum += [tcum[-1]]

    z = np.linspace(-3, 1, 100000)
    zshift = 0  # -sum(thicknesses)
    z1 = -z
    prop_plus = [np.exp(-1j * g * z1) for g in gamma]
    prop_minus = [np.exp(1j * g * z1) for g in gamma]

    print(R + T)
