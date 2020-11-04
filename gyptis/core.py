#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

__all__ = ["Material", "Box", "PML"]

"""
Finite Element model for solving Maxwell's equations in 2D media

"""

import dolfin as df
import meshio
import numpy as np
import pygmsh as pg

from .geom import _init_pmls

# from collections import namedtupless


def _matfmt(mat, fmt=".4g", extraspace=[0, 0, 0]):
    mat = np.asarray(mat)
    col_maxes = [max([len(("{:" + fmt + "}").format(x)) for x in col]) for col in mat.T]
    s = ""
    for i, x in enumerate(mat):
        s += extraspace[i] * " "
        for j, y in enumerate(x):
            b = "|" if j == 0 else ""
            s += ("{} {:" + str(col_maxes[j]) + fmt + "}  ").format(b, y)
        s += "|\n"
    return s


def _is_isotropic(mat):
    mat = np.asarray(mat)
    if np.isscalar(mat) or np.asarray(mat).shape == () or np.asarray(mat).shape == (1,):
        return True
    elif np.all(mat == np.eye(len(mat)) * mat[0, 0]):
        return True
    else:
        return False


def _is_diag(mat):
    return np.all(np.asarray(mat) == np.diag(np.diagonal(mat)))


def _is_diag_iso(mat):
    if not _is_isotropic(mat):
        mat = np.asarray(mat)
        return
    else:
        return False


def _make_iso(mat):
    if not _is_isotropic(mat):
        return mat
    else:
        if np.asarray(mat).shape != ():
            if np.asarray(mat).shape == (1,):
                return np.asarray(mat)[0]
            else:
                return np.asarray(mat)[0, 0]
        else:
            return mat


def _is_reciprocal(mat):
    mat = np.asarray(mat)
    return np.all(mat.T == mat)


def _check_mat(mat):
    mat = np.asarray(mat)
    error = False
    if len(mat.shape) > 2:
        error = True
    else:
        if len(mat.shape) == 1 and mat.shape[0] != 1:
            error = True
        elif len(mat.shape) == 2 and mat.shape != (3, 3):
            error = True
    if error:
        raise ValueError(f"Material tensor must be a scalar or 3x3 matrix")


def _carac_length(lambda_mesh, parmesh, material):
    pseudo_index = np.sqrt(
        np.abs(np.max(material.epsilon.real)) * np.abs(np.max(material.mu.real))
    )
    return lambda_mesh / (parmesh * pseudo_index)


class Material(object):
    """Define material with permittivity and permeability"""

    def __init__(self, epsilon=1, mu=1, name="material"):
        _check_mat(epsilon)
        _check_mat(mu)
        self.epsilon = _make_iso(epsilon)
        self.mu = _make_iso(mu)
        self.name = name

    def __str__(self):
        return str(self.name)

    def __repr__(self):
        iso = self.is_isotropic()
        reprmat = lambda mat, iso: mat if iso else _matfmt(mat, extraspace=[0, 21, 21])

        return (
            f"Material({self.name}) with\n"
            f"   permittivity: ϵ = {reprmat(self.epsilon,iso[0])}\n"
            f"   permeability: μ = {reprmat(self.mu,iso[1])}"
        )

    def is_isotropic(self):
        return _is_isotropic(self.epsilon), _is_isotropic(self.mu)


class Box(object):
    """The computational domain"""

    vacuum = Material(epsilon=1, mu=1, name="vacuum")

    def __init__(
        self,
        width=(1, 1),
        lcar=0.1,
        material=vacuum,
        bcs="dirichlet",
        pml_width=(1, 1),
        name="computational domain",
    ):
        self.name = name
        self.width = width
        self.bcs = bcs
        # bcs = {"top": None, "bottom": None, "left": None, "right": None}
        # dictionary or string (if string, then applies to all)
        self.material = material
        self.lcar = lcar
        self.pml_width = pml_width  # tuple or dict of tuples
        self.pml_dict, self.pml_positions = _init_pmls(self)

    def __str__(self):
        return str(self.name)

    def __repr__(self):
        return f"Box({self.name}) with size {self.width}"

    # this should go in simulation object
    # @property
    # def lcar(self):
    #     return _carac_length(self.lambda_mesh, self.parmesh, self.material)


class PML(object):
    def __init__(
        self,
        direction="x",
        width=(1, 1),
        corner=(0, 0),
        stretch=1 - 1j,
        lcar=0.1,
        name="pml",
    ):
        self.direction = direction  ## "x", "y", or "xy"
        self.width = width
        self.corner = corner  # lower left corner
        self.stretch = stretch
        self.lcar = lcar
        self.name = name

    @property
    def direction(self):
        return self._direction

    @direction.setter
    def direction(self, value):
        if value not in ["x", "y", "xy"]:
            raise ValueError(f'Unkown direction {value}, must be "x", "y", or "xy".')

        self._direction = value

    def __str__(self):
        return str(self.name)

    def __repr__(self):
        return f"PML({self.name} of size {self.width} with stretch {self.stretch} along {self.direction}"

    def jacobian_matrix(self):
        if self.direction == "x":
            sx, sy = self.stretch, 1
        elif self.direction == "y":
            sx, sy = 1, self.stretch
        else:
            sx, sy = self.stretch, self.stretch
        return np.diag([sx, sy, 1])

    def transformation_matrix(self):
        J = self.jacobian_matrix()
        return np.matmul(J, J.T) / np.linalg.det(J)


#
# class Excitation(object):
#     def __init__(self, name="excitation"):
#         pass
#
#     def __str__(self):
#         return str(self.name)
#
#     def __repr__(self):
#         return f"Excitation({self.name})"
#
#
# class Simulation(object):
#     def __init__(self, lambda0=1, cbox=None, name="Maxwell's equations"):
#         self.name = name
#         self.lambda0 = lambda0
#         self.cbox = cbox
#
#     def __str__(self):
#         return str(self.name)
#
#     def __repr__(self):
#         return f"Simulation({self.name}) with wavelength {self.lambda0}"
#

############### ideas ####################
## plot(markers) to visualize subdomains
## markers subdomaid maps id to name
## label in UserExpression t = XsiReal(markers,label="whatevs") then t.label()

if __name__ == "__main__":

    air = Material(1, 1)
    eps = [
        [1.1 - 0.1233j, 2.11233134, 3.11233134],
        [4.11233134, 5.11233134, 6.11233134],
        [1.11233134, 2.11233134, 3.11233134],
    ]
    aniso = Material(epsilon=eps, mu=12 - 2j, name="dielectric")

    print(repr(aniso))
    box = Box(material=air)
    boxa = Box(material=aniso)
    # broken = Material(epsilon=[1, 2])
    assert aniso.is_isotropic() == (False, True)
    assert np.all(aniso.is_isotropic()) == False
    assert _is_isotropic(1), "anisotropic"
    assert _is_isotropic([1]), "anisotropic"
    assert _is_isotropic(np.array([1])), "anisotropic"
    assert _is_isotropic(12 * np.eye(3)), "anisotropic"

    pml = PML()

    ###
