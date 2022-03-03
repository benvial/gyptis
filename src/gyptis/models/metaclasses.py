#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# License: MIT
# See the documentation at gyptis.gitlab.io


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

    def get_cross_sections(self, **kwargs):
        """Compute cross sections.

        Returns
        -------
        dict
            A dictionary containing scattering, absorption and extinction
            cross sections.

        """

        scs = self.scattering_cross_section(**kwargs)
        acs = self.absorption_cross_section(**kwargs)
        ecs = self.extinction_cross_section(**kwargs)
        return dict(scattering=scs, absorption=acs, extinction=ecs)


class _GratingBase(ABC):
    """Base class for grating problems."""

    @abstractmethod
    def diffraction_efficiencies(
        self,
        N_order=0,
        cplx_effs=False,
        orders=False,
        subdomain_absorption=False,
        verbose=False,
    ):
        """Compute the diffraction efficiencies.

        Parameters
        ----------
        N_order : int
            Number of diffraction orders (the default is 0). This will include negative
            and positive orders. In 2D, it will calculate `2 * N_order + 1` coefficients,
            and `(2 * N_order + 1) ** 2` in 3D.
        cplx_effs : bool
            If `True`, return complex coefficients (amplitude reflection and transmission).
            If `False`, return real coefficients (power reflection and transmission)
        orders : bool
            If `True`, computes the transmission and reflection per diffraction orders.
            If `False`, returns the sum of the diffraction orders.
            (the default is False).
        subdomain_absorption : bool
            If `True`, computes the absorption per subdomains and splits the electric and magnetic contributions.
            If `False`, returns the total absorption.
            (the default is False).
        verbose : bool
            If `True`, prints the energy balance (the default is False).

        Returns
        -------
        dict {`R`, `T`, `Q`, `B`}
            A dictionary containing reflection `R`, transmission, `T`, absorption, `Q`, and energy balance `B`.

        """

        pass

    @abstractmethod
    def compute_absorption(self, subdomain_absorption=False):
        """Computes the absorption.

        Parameters
        ----------
        subdomain_absorption : bool
            If `True`, computes the absorption per subdomain and splits the electric and magnetic contributions.
            If `False`, returns the total absorption.
            (the default is False).

        Returns
        -------
        Q_tot : float
            Total absorption.
        Q_domains : dict or dict of dict
            Detailled absorption with electric and magnetic contributions.
            Returns aborption per subdomain if `subdomain_absorption=True`.

        """
        pass
