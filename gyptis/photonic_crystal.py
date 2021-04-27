#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

from .bc import BiPeriodic2D
from .formulation import Maxwell2DBands
from .geometry import *
from .helpers import _translation_matrix
from .materials import *
from .simulation import Simulation


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
