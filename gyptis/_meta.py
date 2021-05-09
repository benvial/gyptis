#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT


"""Metaclasses"""


from abc import ABC, abstractmethod


class _ScatteringBase(ABC):
    """Scattering problem."""

    # def __init__(self,*args,**kwargs):
    #     pass

    @abstractmethod
    def scattering_cross_section(self):
        """Compute the scattering cross section.

        Returns
        -------
        float
            Scattering cross section.

        """
        pass

    @abstractmethod
    def absorption_cross_section(self):
        """Compute the absorption cross section.

        Returns
        -------
        float
            Absorption cross section.

        """
        pass

    @abstractmethod
    def extinction_cross_section(self):
        """Compute the extinction cross section.

        Returns
        -------
        float
            Extinction cross section.

        """
        pass

    def get_cross_sections(self):
        """Compute cross sections.

        Returns
        -------
        dict
            A dictionary containing scattering, absorption and extinction
            cross sections.

        """

        scs = self.scattering_cross_section()
        acs = self.absorption_cross_section()
        ecs = self.extinction_cross_section()
        return dict(scattering=scs, absorption=acs, extinction=ecs)
