#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.0.2
# License: MIT
# See the documentation at gyptis.gitlab.io

import pytest

from gyptis.sources.stack import *


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

    phi_, propagation_constants, efficiencies_stack = get_coeffs_stack(
        config, lambda0, theta0, phi0, psi0
    )
    print(efficiencies_stack["R"] + efficiencies_stack["T"])
