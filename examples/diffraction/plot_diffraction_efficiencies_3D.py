# -*- coding: utf-8 -*-
"""
3D Grating
==========

An example of a bi-periodic diffraction grating.
"""

import sys
from pprint import pprint

from gyptis.grating_3d import *
from gyptis.helpers import list_time, mpi_print

##############################################################################
# The diffracted field :math:`{\mathbf E}^d` can be decomposed in a Rayley series:
#
# .. math::
#
#    {\mathbf E}^d(x,y,z) = \sum_{(n,m) \in \mathbb Z^2}
#    {\mathbf U}_{nm}(z) e^{-i(\alpha_n x + \beta_m y)}
# with :math:`\alpha_n=\alpha_0 + p_n`, :math:`\beta_m=\beta_0 + q_m`,
# :math:`p_n=2\pi n/d_x` and :math:`q_m=2\pi m/d_y`.
#
# The coefficients of the decomposition can be expressed as:
#
# .. math::
#
#    {\mathbf U}_{nm}(z) = \frac{1}{d_x d_y}\int_{-d_x/2}^{d_x/2}\int_{-d_y/2}^{d_y/2}
#    {\mathbf E}^d(x,y,z) e^{-i(\alpha_n x + \beta_m y)}\mathrm d x  \mathrm d y
#
# Note that we solve for the periodic part of the total field
# :math:`{\mathbf E}_\#^d = {\mathbf E}^d e^{-i(\alpha_n x + \beta_m y)}`.
#
# In the substrate (-), we have :math:`{\mathbf U}_{nm}(z) = {\mathbf V}^{-}_{nm} e^{-i\gamma^{-}_{nm}}`
# and in the superstrate (+), :math:`{\mathbf U}_{nm}(z) = {\mathbf V}^{+}_{nm} e^{i\gamma^{+}_{nm}}`
#
# The total diffracted field is


##  ---------- incident wave ----------
lambda0 = 500
theta0 = 0 * pi / 180
phi0 = 0 * pi / 180
psi0 = 0 * pi / 180

##  ---------- geometry ----------
grooove_thickness = 50
hole_radius = 500 / 2
period = (1000, 1000)
thicknesses = OrderedDict(
    {
        "pml_bottom": lambda0,
        "substrate": lambda0 * 0.25,
        "groove": grooove_thickness,
        "superstrate": lambda0 * 0.25,
        "pml_top": lambda0,
    }
)

##  ---------- mesh ----------
# parmesh = 7

try:
    parmesh = int(sys.argv[1])
except:
    parmesh = 4

N_d_order = 2
degree = 2

parmesh_hole = parmesh * 1.1
parmesh_pml = parmesh * 0.7

mesh_params = dict(
    {
        "pml_bottom": parmesh_pml,
        "substrate": parmesh,
        "groove": parmesh,
        "hole": parmesh_hole,
        "superstrate": parmesh,
        "pml_top": parmesh_pml,
    }
)

##  ---------- materials ----------
eps_groove = (1.75 - 1.5j) ** 2
eps_substrate = 1.5 ** 2

epsilon = dict(
    {
        "substrate": eps_substrate,
        "groove": eps_groove,
        "hole": 1,
        "superstrate": 1,
    }
)
mu = {d: 1 for d in epsilon.keys()}

index = dict()
for e, m in zip(epsilon.items(), mu.items()):
    index[e[0]] = np.mean(
        (np.array(e[1]).real.max() * np.array(m[1]).real.max()) ** 0.5
    ).real
index["pml_top"] = index["superstrate"]
index["pml_bottom"] = index["substrate"]

##  ---------- build geometry ----------

mpi_print("-----------------------------")
mpi_print(">> Building geometry and mesh")
mpi_print("-----------------------------")


model = Layered3D(period, thicknesses, kill=False)

groove = model.layers["groove"]
substrate = model.layers["substrate"]
superstrate = model.layers["superstrate"]
z0 = model.z_position["groove"]

hole = model.add_cylinder(
    0,
    0,
    z0,
    0,
    0,
    grooove_thickness,
    hole_radius,
)

superstrate, substrate, hole, groove = model.fragment(
    [superstrate, substrate, groove], hole
)
# hole, groove = model.fragment(hole, groove)
model.add_physical(groove, "groove")
model.add_physical(hole, "hole")
model.add_physical(substrate, "substrate")
model.add_physical(superstrate, "superstrate")

mesh_sizes = {d: lambda0 / (index[d] * mesh_params[d]) for d in epsilon.keys()}
model.set_mesh_size(mesh_sizes)
mesh_object = model.build()

##  ---------- grating ----------

g = Grating3D(
    model,
    epsilon,
    mu,
    lambda0=lambda0,
    theta0=theta0,
    phi0=phi0,
    psi0=psi0,
    degree=degree,
)

g.mat_degree = 2

g.weak_form()

mpi_print("-------------")
mpi_print(">> Assembling")
mpi_print("-------------")

g.assemble()

mpi_print("----------------------")
mpi_print(">> Computing solution")
mpi_print("----------------------")

g.solve()

list_time()
g.N_d_order = N_d_order

mpi_print("-------------------------------------")
mpi_print(">> Computing diffraction efficiencies")
mpi_print("-------------------------------------")

effs = g.diffraction_efficiencies(orders=True, subdomain_absorption=True)

list_time()
print("diffraction efficiencies")
print("------------------------")
pprint(effs)
print("R00", effs["R"][N_d_order, N_d_order])
print("Σ R = ", np.sum(effs["R"]))
print("Σ T = ", np.sum(effs["T"]))
Q = sum(effs["Q"]["electric"].values()) + sum(effs["Q"]["magnetic"].values())
print("Q   = ", Q)
print("B   = ", effs["B"])
W0 = dolfin.FunctionSpace(g.mesh, "CG", 1)
# W0 = dolfin.FunctionSpace(g.mesh, "DG", 0)
fplot = g.E[0].real + g.Estack_coeff[0].real
# fplot = abs(g.Eper)
dolfin.File("test.pvd") << project(fplot, W0)
dolfin.File("markers.pvd") << g.markers
