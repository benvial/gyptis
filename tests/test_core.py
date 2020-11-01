#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT


from gyptis.core import Material, Box, PML
import pytest
import numpy as np

air = Material(1, 1)
eps = [
    [1.1 - 0.1233j, 2.11233134, 3.11233134],
    [4.11233134, 5.11233134, 6.11233134],
    [1.11233134, 2.11233134, 3.11233134],
]
aniso = Material(epsilon=eps, mu=12 - 2j, name="dielectric")
print(repr(aniso))


def test_material():
    assert aniso.is_isotropic() == (False, True)
    with pytest.raises(ValueError):
        Material(epsilon=[1, 2]).is_isotropic()
    assert aniso.is_isotropic() == (False, True)
    assert np.all(aniso.is_isotropic()) == False
    assert Material(1).is_isotropic()[0], "anisotropic"
    assert Material([1]).is_isotropic()[0], "anisotropic"
    assert Material(np.array([1])).is_isotropic()[0], "anisotropic"
    assert Material(12 * np.eye(3)).is_isotropic()[0], "anisotropic"

def test_box():
    box = Box(material=air)
    boxa = Box(material=aniso)

def test_pml():
    pml = PML()
