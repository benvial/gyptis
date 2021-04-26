# -*- coding: utf-8 -*-
"""
Core shell nanorod grating
==========================

Diffraction by a grating of silver-coated circular nanocylinders on a dielectric substrate.
"""


from collections import OrderedDict
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c

from gyptis import Grating, Layered, PlaneWave

##############################################################################
# Reference results are taken from [Jandieri2015]_.
# We first define a function for the Drude Lorentz model of silver permittivity.


def epsilon_silver(omega):
    eps_inf = 3.91
    omega_D = 13420e12
    gamma_D = 84e12
    Omega_L = 6870e12
    Gamma_L = 12340e12
    delta_eps = 0.76
    epsilon = (
        eps_inf
        - omega_D ** 2 / (omega * (omega + 1j * gamma_D))
        - delta_eps
        * Omega_L ** 2
        / ((omega ** 2 - Omega_L ** 2) + 1j * Gamma_L * omega)
    )
    return np.conj(epsilon)


wavelength_min = 250
wavelength_max = 800

##############################################################################
# Now we create the geometry and mesh

pmesh = 20
wavelength = 452
eps_core = 2
eps_substrate = 2.5


period = 130
R1 = 60
R2 = 30
thicknesses = OrderedDict(
    {
        "pml_bottom": 400,
        "substrate": 400,
        "groove": 2 * R1,
        "superstrate": 400,
        "pml_top": 400,
    }
)

lmin = wavelength / pmesh
omega = 2 * np.pi * c / (wavelength * 1e-9)
epsAg = epsilon_silver(omega)

nAg = abs(epsAg.real) ** 0.5
ncore = (eps_core.real) ** 0.5
nsubstrate = (eps_substrate.real) ** 0.5

geom = Layered(2, period, thicknesses)
groove = geom.layers["groove"]
substrate = geom.layers["substrate"]
superstrate = geom.layers["superstrate"]
y0 = geom.y_position["groove"]
shell = geom.add_circle(0, y0 + R1, 0, R1)
out = geom.fragment(shell, [groove, substrate, superstrate])
groove = out[3:]
substrate = out[1]
superstrate = out[2]
shell = out[0]
core = geom.add_circle(0, y0 + R1, 0, R2)
core, shell = geom.fragment(core, shell)
geom.add_physical(superstrate, "superstrate")
geom.add_physical(substrate, "substrate")
geom.add_physical(groove, "groove")
geom.add_physical(core, "core")
geom.add_physical(shell, "shell")
[geom.set_size(pml, lmin * 0.75) for pml in ["pml_bottom", "pml_top"]]
geom.set_size("superstrate", lmin)
geom.set_size("groove", lmin)
geom.set_size("substrate", lmin / nsubstrate)
geom.set_size("core", 0.5 * lmin / ncore)
geom.set_size("shell", 0.5 * lmin / nAg)
geom.build(0)


##############################################################################
# Define the incident plane wave and materials and the diffraction problem


def solve(wavelength):

    pw = PlaneWave(
        wavelength=wavelength, angle=np.pi / 2, dim=2, domain=geom.mesh, degree=2
    )
    omega = 2 * np.pi * c / (wavelength * 1e-9)
    epsilon = dict(
        core=eps_core,
        shell=epsilon_silver(omega),
        substrate=eps_substrate,
        groove=1,
        superstrate=1,
    )
    mu = {d: 1 for d in epsilon.keys()}
    g = Grating(
        geom,
        epsilon,
        mu,
        pw,
        degree=2,
        polarization="TE",
    )
    g.solve()

    return g


fig, axes = plt.subplots(1, 2, figsize=(5, 2))
nper = 5


effs_ref = {
    "316": dict(R=0.245963, T=0.024110, Q=0.729925, B=1.000000),
    "452": dict(R=0.519389, T=0.045008, Q=0.435601, B=1.000000),
}


for ax, wavelength in zip(axes, [316, 452]):
    g = solve(wavelength)
    g.plot_field(nper=nper, type="module", cmap="hot", ax=ax)
    scatt_lines, layers_lines = g.plot_geometry(nper=nper, c="w", ax=ax)
    ax.set_ylim(-300, 400)
    ax.set_xlim(-period / 2, nper * period - period / 2)
    [layers_lines[i].remove() for i in range(6)]
    ax.set_axis_off()
    ax.annotate(
        fr"$\lambda_0={wavelength}$ nm",
        (0.6, 0.2),
        c="w",
        xycoords="axes fraction",
        weight="medium",
    )
    ax.annotate(
        fr"$\Lambda={period}$ nm",
        (0.6, 0.1),
        c="w",
        xycoords="axes fraction",
        weight="medium",
    )
    fig.tight_layout()

    # Compute diffraction efficiencies

    effs = g.diffraction_efficiencies()

    print(" ")
    print(f"diffraction efficiencies, wavelength = {wavelength} nm")
    print("-------------------------------------------------------")

    print("gyptis")
    pprint(effs, width=30)
    print("reference")
    pprint(effs_ref[f"{wavelength}"], width=30)


######################################################################
#
# .. [Jandieri2015] V. Jandieri, P. Meng, K. Yasumoto, and Y. Liu,
#   Scattering of light by gratings of metal-coated nanocylinders on dielectric substrate.
#   Journal of the Optical Society of America A, vol. 32, p. 1384, (2015).
#   `<https://www.doi.org/10.1364/JOSAA.32.001384>`_
