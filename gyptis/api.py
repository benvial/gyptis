#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT


__all__ = [
    "Complex",
    "Layered",
    "Lattice",
    "Scattering",
    "Grating",
    "PhotonicCrystal",
    "PlaneWave",
    "LineSource",
    "BoxPML",
    "Layered",
    "Geometry",
]

from ._meta import _GratingBase, _ScatteringBase

#
# from . import complex
from .complex import Complex
from .geometry import Geometry
from .grating2d import Grating2D, Layered2D
from .grating3d import Grating3D, Layered3D
from .photonic_crystal import Lattice2D, PhotonicCrystal2D
from .scattering2d import BoxPML2D, Scatt2D
from .scattering3d import BoxPML3D, Scatt3D
from .simulation import Simulation
from .source import LineSource, PlaneWave


def _check_dimension(dim):
    if dim not in [2, 3]:
        raise ValueError("dimension must be 2 or 3")


class BoxPML(Geometry):
    """BoxPML(dim, box_size, box_center, pml_width, Rcalc=0, model_name="Box with PMLs", **kwargs)
    Computational domain with Perfectly Matched Layers (PMLs).

    Parameters
    ----------
    dim : int
        Geometric dimension (either 2 or 3, the default is 3).
    box_size : tuple of floats of length `dim`
        Size of the box: `(lx,ly)` if `dim=2` or `(lx, ly, lz)` if `dim=3`.
    box_center : tuple of floats of length `dim`
        Size of the box: `(cx,cy)` if `dim=2` or `(cx, cy, cz)` if `dim=3`.
    pml_width : tuple of floats of length `dim`
        Size of the PMLs: `(hx,hy)` if `dim=2` or `(hx, hy, hz)`` if `dim=3`.
    model_name : str
        Name of the model.
    **kwargs : dictionary
        Additional parameters. See the parent class :class:`~gyptis.Geometry`.

    """

    def __new__(self, dim=3, *args, **kwargs):
        if dim not in [2, 3]:
            raise ValueError("dimension must be 2 or 3")
        if dim == 3:
            return BoxPML3D(*args, **kwargs)
        else:
            return BoxPML2D(*args, **kwargs)


class Layered(Geometry):
    """Layered media."""

    def __new__(self, dim=3, *args, **kwargs):
        _check_dimension(dim)
        if dim == 3:
            return Layered3D(*args, **kwargs)
        else:
            return Layered2D(*args, **kwargs)


class Lattice(Geometry):
    """Unit cell for periodic problems."""

    def __new__(self, dim=3, *args, **kwargs):
        _check_dimension(dim)
        if dim == 3:
            raise NotImplementedError
        else:
            return Lattice2D(*args, **kwargs)


class Scattering(_ScatteringBase, Simulation):
    """Scattering(geometry, epsilon, mu, source=None, boundary_conditions={}, polarization="TM", modal=False, degree=1, pml_stretch=1 - 1j)
    Scattering problem.

    Parameters
    ----------
    geometry : :class:`~gyptis.Geometry`.
        Description of parameter `geometry`.
    epsilon : dict
        Permittivity in various subdomains.
    mu : dict
        Permeability in various subdomains.
    source : :class:`~gyptis.source.Source`.
        Excitation (the default is None).
    boundary_conditions : dict
        Boundary conditions {"boundary": "condition"} (the default is {}).
        Valid condition is only "PEC".
    polarization : str
        Polarization case (only makes sense for 2D problems, the default is "TM").
    modal : str
        Perform modal analysis (the default is False).
    degree : int
        Degree of finite elements interpolation (the default is 1).
    pml_stretch : complex
        Complex coordinate stretch for te PMLs (the default is 1 - 1j).

    Returns
    -------
    type
        Description of returned object.

    """

    def __new__(self, *args, **kwargs):
        geom = kwargs.get("geometry") or args[0]
        _check_dimension(geom.dim)
        if geom.dim == 3:
            return Scatt3D(*args, **kwargs)
        else:
            return Scatt2D(*args, **kwargs)


class Grating(_GratingBase, Simulation):
    """Grating(geometry, epsilon, mu, source, boundary_conditions={}, polarization="TM", degree=1, pml_stretch=1 - 1j, periodic_map_tol=1e-8)
    Grating problem.

    Parameters
    ----------
    geometry : type
        Description of parameter `geometry`.
    epsilon : type
        Description of parameter `epsilon`.
    mu : type
        Description of parameter `mu`.
    source : type
        Description of parameter `source`.
    boundary_conditions : type
        Description of parameter `boundary_conditions` (the default is {}).
    polarization : type
        Description of parameter `polarization` (the default is "TM").
    degree : type
        Description of parameter `degree` (the default is 1).
    pml_stretch : type
        Description of parameter `pml_stretch` (the default is 1 - 1j).
    periodic_map_tol : type
        Description of parameter `periodic_map_tol` (the default is 1e-8).

    Returns
    -------
    type
        Description of returned object.

    """

    def __new__(self, *args, **kwargs):
        geom = kwargs.get("geometry") or args[0]
        _check_dimension(geom.dim)
        if geom.dim == 3:
            return Grating3D(*args, **kwargs)
        else:
            return Grating2D(*args, **kwargs)


class PhotonicCrystal:
    def __new__(self, *args, **kwargs):
        geom = kwargs.get("geometry") or args[0]
        _check_dimension(geom.dim)
        if geom.dim == 3:
            raise NotImplementedError
        else:
            return PhotonicCrystal2D(*args, **kwargs)
