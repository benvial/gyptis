#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

from gyptis.bc import BiPeriodic2D
from gyptis.formulation import Maxwell2DBands
from gyptis.geometry import *
from gyptis.helpers import _translation_matrix
from gyptis.materials import *
from gyptis.simulation import Simulation


class Lattice2D(Geometry):
    def __init__(
        self,
        vectors=((1, 0), (0, 1)),
        model_name="Lattice",
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            dim=2,
            **kwargs,
        )
        self.vectors = vectors
        self.vertices = [
            (0, 0),
            (self.vectors[0][0], self.vectors[0][1]),
            (
                self.vectors[0][0] + self.vectors[1][0],
                self.vectors[0][1] + self.vectors[1][1],
            ),
            (self.vectors[1][0], self.vectors[1][1]),
        ]
        p = []
        for v in self.vertices:
            p.append(self.add_point(*v, 0))
        l = []
        for i in range(3):
            l.append(self.add_line(p[i + 1], p[i]))
        l.append(self.add_line(p[3], p[0]))
        cl = self.add_curve_loop(l)
        ps = self.add_plane_surface([cl])
        self.cell = ps
        self.add_physical(self.cell, "cell")

    @property
    def translation(self):
        return _translation_matrix([*self.vectors[0], 0]), _translation_matrix(
            [*self.vectors[1], 0]
        )

    def get_periodic_bnds(self):

        # define lines equations
        def _is_on_line(p, p1, p2):
            x, y = p
            x1, y1 = p1
            x2, y2 = p2
            if x1 == x2:
                return np.allclose(x, x1)
            else:
                return np.allclose(y - y1, (y2 - y1) / (x2 - x1) * (x - x1))

        verts = self.vertices.copy()
        verts.append(self.vertices[0])

        # get all boundaries
        bnds = self.get_entities(1)
        maps = []
        for i in range(4):
            wheres = []
            for b in bnds:
                qb = gmsh.model.getParametrizationBounds(1, b[-1])
                B = []
                for p in qb:
                    val = gmsh.model.getValue(1, b[-1], p)
                    p = val[0:2]
                    belongs = _is_on_line(p, verts[i + 1], verts[i])
                    B.append(belongs)
                alls = np.all(B)
                if alls:
                    wheres.append(b)
            maps.append(wheres)
        s = {}
        s["-1"] = [m[-1] for m in maps[-1]]
        s["+1"] = [m[-1] for m in maps[1]]
        s["-2"] = [m[-1] for m in maps[0]]
        s["+2"] = [m[-1] for m in maps[2]]
        return s

    def build(self, *args, **kwargs):
        periodic_id = self.get_periodic_bnds()
        gmsh.model.mesh.setPeriodic(
            1, periodic_id["+1"], periodic_id["-1"], self.translation[0]
        )
        gmsh.model.mesh.setPeriodic(
            1, periodic_id["+2"], periodic_id["-2"], self.translation[1]
        )
        super().build(*args, **kwargs)


class PhotonicCrystal2D(Simulation):
    def __init__(
        self,
        geometry,
        epsilon,
        mu,
        propagation_vector=(0, 0),
        boundary_conditions={},
        polarization="TM",
        degree=1,
        mat_degree=1,
    ):
        assert isinstance(geometry, Lattice2D)

        self.periodic_bcs = BiPeriodic2D(geometry.vectors)
        function_space = ComplexFunctionSpace(
            geometry.mesh, "CG", degree, constrained_domain=self.periodic_bcs
        )
        epsilon = {k: e + 1e-16j for k, e in epsilon.items()}
        mu = {k: m + 1e-16j for k, m in mu.items()}
        epsilon_coeff = Coefficient(epsilon, geometry, degree=mat_degree)
        mu_coeff = Coefficient(mu, geometry, degree=mat_degree)

        coefficients = epsilon_coeff, mu_coeff
        formulation = Maxwell2DBands(
            geometry,
            coefficients,
            function_space,
            propagation_vector=propagation_vector,
            degree=degree,
            polarization=polarization,
            boundary_conditions=boundary_conditions,
        )

        super().__init__(geometry, formulation)

        self.degree = degree
        self.propagation_vector = propagation_vector

    def eigensolve(self, *args, **kwargs):
        sol = super().eigensolve(*args, **kwargs)
        self.solution["eigenvectors"] = [
            u * self.formulation.phasor for u in sol["eigenvectors"]
        ]
        return self.solution


if __name__ == "__main__":

    # dy = 0.5 * 2 ** 0.5
    # v = (1, 0), (dy, dy)
    # v = (1, 0), (0, 1)
    # R = 0.3
    #
    # lattice = Lattice2D(v)
    #
    # circ = lattice.add_circle(0, 0, 0, R)
    # circ_0, circ1, cell = lattice.fragment(circ, lattice.cell)
    # lattice.remove([(2, circ1)], recursive=True)
    #
    # circ = lattice.add_circle(v[1][0], v[1][1], 0, R)
    # circ1, circ_1, cell = lattice.fragment(circ, cell)
    # lattice.remove([(2, circ1)], recursive=True)
    #
    # circ = lattice.add_circle(v[0][0], v[0][1], 0, R)
    # circ1, circ_2, cell = lattice.fragment(circ, cell)
    # lattice.remove([(2, circ1)], recursive=True)
    #
    # circ = lattice.add_circle(v[0][0] + v[1][0], v[1][1], 0, R)
    # circ1, circ_3, cell = lattice.fragment(circ, cell)
    # lattice.remove([(2, circ1)], recursive=True)
    #
    # lattice.add_physical(cell, "background")
    # lattice.add_physical([circ_0, circ_1, circ_2, circ_3], "inclusion")
    #
    # print(lattice.get_periodic_bnds())
    # # l.set_mesh_size({"background": 0.1, "inclusion": 0.03})
    # lattice.set_size("background", 0.1)
    # lattice.set_size("inclusion", 0.1)
    #
    # lattice.build()

    from gyptis.plot import *

    plt.ion()

    a = 1
    v = (a, 0), (0, a)
    R = 0.2 * a

    lattice = Lattice2D(v)

    circ = lattice.add_circle(a / 2, a / 2, 0, R)
    circ, cell = lattice.fragment(circ, lattice.cell)
    lattice.add_physical(cell, "background")
    lattice.add_physical(circ, "inclusion")

    lattice.set_size("background", 0.05)
    lattice.set_size("inclusion", 0.05)

    lattice.build()
    eps_inclusion = 8.9
    epsilon = dict(background=1, inclusion=eps_inclusion)
    mu = dict(background=1, inclusion=1)

    Nb = 3
    K = np.linspace(0, np.pi / a, Nb)
    bands = np.zeros((3 * Nb - 3, 2))
    bands[:Nb, 0] = K
    bands[Nb : 2 * Nb, 0] = K[-1]
    bands[Nb : 2 * Nb - 1, 1] = K[1:]
    bands[2 * Nb - 1 : 3 * Nb - 3, 0] = bands[2 * Nb - 1 : 3 * Nb - 3, 1] = np.flipud(
        K
    )[1:-1]

    bands_plot = np.zeros(3 * Nb - 2)
    bands_plot[:Nb] = K
    bands_plot[Nb : 2 * Nb - 1] = K[-1] + K[1:]
    bands_plot[2 * Nb - 1 : 3 * Nb - 2] = 2 * K[-1] + 2 ** 0.5 * K[1:]

    n_eig = 6
    BD = {}
    for polarization in ["TE", "TM"]:

        ev_band = []

        for kx, ky in bands:
            phc = PhotonicCrystal2D(
                lattice,
                epsilon,
                mu,
                propagation_vector=(kx, ky),
                polarization=polarization,
                degree=1,
            )
            phc.eigensolve(n_eig=6, wavevector_target=0.1)
            ev_norma = np.array(phc.solution["eigenvalues"]) * a / (2 * np.pi)
            ev_norma = ev_norma[:n_eig].real
            ev_band.append(ev_norma)
        ev_band.append(ev_band[0])
        BD[polarization] = ev_band

    plt.close("all")
    plt.clf()
    plt.figure(figsize=(3.2, 2.5))
    plotTM = plt.plot(bands_plot, BD["TM"], c="#4199b0")
    plotTE = plt.plot(bands_plot, BD["TE"], c="#cf5268")

    plt.annotate("TM modes", (1, 0.05), c="#4199b0")
    plt.annotate("TE modes", (0.33, 0.33), c="#cf5268")

    plt.ylim(0, 0.8)
    plt.xlim(0, bands_plot[-1])
    plt.xticks(
        [0, K[-1], 2 * K[-1], bands_plot[-1]], ["$\Gamma$", "$X$", "$M$", "$\Gamma$"]
    )
    plt.axvline(K[-1], c="k", lw=0.3)
    plt.axvline(2 * K[-1], c="k", lw=0.3)
    plt.ylabel(r"Frequency $\omega a/2\pi c$")
    plt.tight_layout()

    phc = PhotonicCrystal2D(
        lattice,
        epsilon,
        mu,
        propagation_vector=(np.pi / a, 0),
        polarization="TE",
        degree=1,
    )
    phc.eigensolve(n_eig=6, wavevector_target=0.1)
    ev_norma = np.array(phc.solution["eigenvalues"]) * a / (2 * np.pi)
    ev_norma = ev_norma[:n_eig].real

    eig_vects = phc.solution["eigenvectors"]
    for mode, eval in zip(eig_vects, ev_norma):
        if eval.real > 0:
            plot(mode.real, cmap="RdBu_r")
            plt.title(fr"$\omega a/2\pi c = {eval.real:0.3f}+{eval.imag:0.3f}j$")
            H = phc.formulation.get_dual(mode, 1)
            dolfin.plot(H.real, cmap="Greys")
            lattice.plot_subdomains()
            plt.axis("off")
            # plt.xlim(-geom.box_size[0] / 2, geom.box_size[0] / 2)
            # plt.ylim(-geom.box_size[1] / 2, geom.box_size[1] / 2)
