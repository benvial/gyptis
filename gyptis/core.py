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


class PML(object):
    def __init__(
        self,
        direction="x",
        stretch=1 - 1j,
    ):
        self.direction = direction
        self.stretch = stretch

    @property
    def direction(self):
        return self._direction

    @direction.setter
    def direction(self, value):
        valid = ["x", "y", "z", "xy", "xz", "yz", "xyz"]
        if value not in valid:
            raise ValueError(f"Unkown direction {value}, must be one of {valid}")

        self._direction = value

    def jacobian_matrix(self):
        if self.direction == "x":
            s = self.stretch, 1, 1
        elif self.direction == "y":
            s = 1, self.stretch, 1
        elif self.direction == "z":
            s = 1, 1, self.stretch
        elif self.direction == "xy":
            s = self.stretch, self.stretch, 1
        elif self.direction == "xz":
            s = self.stretch, 1, self.stretch
        elif self.direction == "yz":
            s = 1, self.stretch, self.stretch
        else:
            s = self.stretch, self.stretch, self.stretch
        return np.diag(s)

    def transformation_matrix(self):
        J = self.jacobian_matrix()
        invJ = np.linalg.inv(J)
        return invJ @ invJ.T * np.linalg.det(J)


############### ideas ####################
## plot(markers) to visualize subdomains
## markers subdomaid maps id to name
## label in UserExpression t = XsiReal(markers,label="whatevs") then t.label()
