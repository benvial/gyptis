#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# License: MIT
# See the documentation at gyptis.gitlab.io


__all__ = [
    "Layered",
    "Lattice",
    "Scattering",
    "Grating",
    "PhotonicCrystal",
    "PlaneWave",
    "LineSource",
    "GaussianBeam",
    "BoxPML",
    "Layered",
    "Geometry",
    "Homogenization2D",
]


from .geometry import *
from .models import *
from .models.metaclasses import _GratingBase, _ScatteringBase
from .sources import *


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
        Size of the box: :math:`(l_x,l_y)` if `dim=2` or :math:`(l_x, l_y, l_z)` if `dim=3`.
    box_center : tuple of floats of length `dim`
        Size of the box: :math:`(c_x,c_y)` if `dim=2` or :math:`(c_x, c_y, c_z)` if `dim=3`.
    pml_width : tuple of floats of length `dim`
        Size of the PMLs: :math:`(h_x,h_y)` if `dim=2` or :math:`(h_x, h_y, h_z)` if `dim=3`.
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
    """Layered(dim, period, thicknesses, **kwargs)
    Layered media for diffraction problems, defining the periodic unit cell
    for mono or bi periodic gratings.

    Parameters
    ----------
    dim : int
        Geometric dimension (either 2 or 3, the default is 3).
    period : float or tuple
        In 2D, periodicity of the grating :math:`d` along :math:`x` (float).
        In 3D, periodicity of the grating :math:`(d_x,d_y)` along :math:`x`
        and :math:`y` (tuple of floats of lenght 2).
    thicknesses : :class:`~collections.OrderedDict`
        Dictionary containing physical names and thicknesses from top to bottom.
        (``thicknesses["phyiscal_name"]=thickness_value``)
    **kwargs : dictionary
        Additional parameters. See the parent class :class:`~gyptis.Geometry`.


    Examples
    --------

    >>> from collections import OrderedDict
    >>> from gyptis import Layered
    >>> t = OrderedDict(pml_top=1, slab=3, pml_bot=1)
    >>> lays = Layered(dim=2, period=1.3, thicknesses=t)
    >>> lays.build()

    """

    def __new__(self, dim=3, *args, **kwargs):
        _check_dimension(dim)
        if dim == 3:
            return Layered3D(*args, **kwargs)
        else:
            return Layered2D(*args, **kwargs)


class Lattice(Geometry):
    """Lattice(vectors, **kwargs)
    Unit cell for periodic problems.

    Parameters
    ----------
    dim : int
        Geometric dimension (either 2 or 3, the default is 3).
    vectors : tuple
        In 2D, a tuple of lengh 2 with the :math:`(x,y)` coordinates of 2 basis vectors.
        In 3D, a tuple of lengh 3 with the :math:`(x,y,z)` coordinates of 3 basis vectors.
    **kwargs : dictionary
        Additional parameters. See the parent class :class:`~gyptis.Geometry`.
    """

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
    geometry : :class:`~gyptis.Geometry`
        The meshed geometry
    epsilon : dict
        Permittivity in various subdomains.
    mu : dict
        Permeability in various subdomains.
    source : :class:`~gyptis.source.Source`
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

    """

    def __new__(self, *args, **kwargs):
        geom = kwargs.get("geometry") or args[0]
        _check_dimension(geom.dim)
        if geom.dim == 3:
            return Scatt3D(*args, **kwargs)
        else:
            return Scatt2D(*args, **kwargs)


class Grating(_GratingBase, Simulation):
    """Grating(geometry, epsilon, mu, source, boundary_conditions={}, polarization="TM", degree=1, pml_stretch=1 - 1j, periodic_map_tol=1e-8, propagation_constant=0.0)
    Grating problem.

    Parameters
    ----------
    geometry : :class:`~gyptis.Geometry`
        The meshed geometry
    epsilon : dict
        Permittivity in various subdomains.
    mu : dict
        Permeability in various subdomains.
    source : :class:`~gyptis.source.Source`
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
    periodic_map_tol : float
        Tolerance for mapping boundaries (the default is 1e-8).
    propagation_constant : float
        Propagation constant along the periodicity. Only
        makes sense for modal analysis (the default is 0.0).


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
