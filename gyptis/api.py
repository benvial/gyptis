#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT


__all__ = ["BoxPML", "Layered", "Scattering", "Grating"]

from .grating2d import Grating2D, Layered2D
from .grating3d import Grating3D, Layered3D
from .scattering2d import BoxPML2D, Scatt2D
from .scattering3d import BoxPML3D, Scatt3D


class BoxPML:
    def __new__(self, dim=3, *args, **kwargs):
        if dim not in [2, 3]:
            raise ValueError("dimension must be 2 or 3")
        if dim == 3:
            return BoxPML3D(*args, **kwargs)
        else:
            return BoxPML2D(*args, **kwargs)


class Layered:
    def __new__(self, dim=3, *args, **kwargs):
        if dim not in [2, 3]:
            raise ValueError("dimension must be 2 or 3")
        if dim == 3:
            return Layered3D(*args, **kwargs)
        else:
            return Layered2D(*args, **kwargs)


class Scattering:
    def __new__(self, *args, **kwargs):
        geom = kwargs.get("geometry") or args[0]
        if geom.dim not in [2, 3]:
            raise ValueError("dimension must be 2 or 3")
        if geom.dim == 3:
            return Scatt3D(*args, **kwargs)
        else:
            return Scatt2D(*args, **kwargs)


class Grating:
    def __new__(self, *args, **kwargs):
        geom = kwargs.get("geometry") or args[0]
        if geom.dim not in [2, 3]:
            raise ValueError("dimension must be 2 or 3")
        if geom.dim == 3:
            return Grating3D(*args, **kwargs)
        else:
            return Grating2D(*args, **kwargs)
