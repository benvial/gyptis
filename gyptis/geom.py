#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT
import numpy as np
import pygmsh as pg

#
#
#
# def make_box(box, lc):
#     geom = pg.built_in.Geometry()
#     xbox, ybox = self.lx/2 - -self.lx/2, self.ly/2 - -self.ly/2
#
#     return
#
#     pml_lb = geom.add_rectangle(
#         -self.lx/2 - self.h_pml, -self.lx/2, -self.ly/2 - self.h_pml, -self.ly/2, 0, lpml
#     )
#
#     pml_l = geom.add_rectangle(
#         -self.lx/2 - self.h_pml, -self.lx/2, -self.ly/2, self.ly/2, 0, lpml
#     )
#     pml_lt = geom.add_rectangle(
#         -self.lx/2 - self.h_pml, -self.lx/2, self.ly/2, self.ly/2 + self.h_pml, 0, lpml
#     )
#     pml_t = geom.add_rectangle(
#         -self.lx/2, self.lx/2, self.ly/2, self.ly/2 + self.h_pml, 0, lpml
#     )
#     pml_rt = geom.add_rectangle(
#         self.lx/2, self.lx/2 + self.h_pml, self.ly/2, self.ly/2 + self.h_pml, 0, lpml
#     )
#     pml_r = geom.add_rectangle(
#         self.lx/2, self.lx/2 + self.h_pml, -self.ly/2, self.ly/2, 0, lpml
#     )
#     pml_rb = geom.add_rectangle(
#         self.lx/2, self.lx/2 + self.h_pml, -self.ly/2 - self.h_pml, -self.ly/2, 0, lpml
#     )
#     pml_b = geom.add_rectangle(
#         -self.lx/2, self.lx/2, -self.ly/2 - self.h_pml, -self.ly/2, 0, lpml
#     )
#
#     des = geom.add_rectangle(
#         -self.hx_des / 2, self.hx_des / 2, -self.hy_des / 2, self.hy_des / 2, 0, ldes
#     )
#
#     xtarg = np.linspace(-0.5, 0.5, Ntar) * (self.hx_des - self.delta_x)
#
#     targets = []
#     for xc in xtarg:
#         circle = geom.add_circle(
#             [xc, self.y_target, 0.0], self.r_target, lholes, make_surface=True
#         )
#         targets.append(circle)
#
#     sources = []
#     for xc in xtarg:
#         circle = geom.add_circle(
#             [xc, self.ys, 0.0], self.r_target, lholes, make_surface=True
#         )
#         sources.append(circle)
#
#     holes = [c.line_loop for c in targets]
#     holes += [c.line_loop for c in sources]
#     holes.append(des)
#
#     box = geom.add_rectangle(
#         -self.lx/2, self.lx/2, self.ly/2, -self.ly/2, 0, lhost, holes=holes
#     )
#
#     geom.add_physical(
#         [pml_lb.surface, pml_lt.surface, pml_rt.surface, pml_rb.surface]
#     )  # pml xy
#     geom.add_physical([pml_l.surface, pml_r.surface])  # pml x
#     geom.add_physical([pml_t.surface, pml_b.surface])  # pml y
#     geom.add_physical(box.surface)
#     geom.add_physical(des.surface)
#     [geom.add_physical(c.plane_surface) for c in targets]
#     [geom.add_physical(c.plane_surface) for c in sources]
#
#     # geom.add_raw_code("Coherence;")
#     # geom.add_raw_code("Coherence;")
#     # geom.add_raw_code("Coherence;")
#     # geom.add_raw_code("Mesh.Algorithm=6;")
#
#     code = geom.get_code()  # .replace("'", "")
#
#     with open(self.geom_filename, "w") as f:
#         f.write(code)


def make_box(self, holes=[], pmls=[]):
    geom = pg.built_in.Geometry()
    lx, ly = self.width
    box = geom.add_rectangle(
        -lx / 2, lx / 2, -ly / 2, ly / 2, 0, self.lcar, holes=holes
    )
    #
    # code = geom.get_code()  # .replace("'", "")
    #
    # with open(self.geom_filename, "w") as f:
    #     f.write(code)

    return geom


def add_pml(self, geom):
    cx, cy = self.corner
    wx, wy = self.width
    geom.add_rectangle(
        cx,
        cx + wx,
        cy,
        cy + wy,
        0,
        self.lcar,
    )
    return geom


def _init_pmls(box):
    # if hasattr(box.pml_width,"__len__"):
    width_pml = box.pml_width

    is_pml_dict = type(width_pml) == dict

    bcx, bcy = [b / 2 for b in box.width]
    box_corners = (-bcx, -bcy), (-bcx, bcy), (bcx, bcy), (bcx, -bcy)
    box_corners = np.array(box_corners)
    shift = (-1, -1), (-1, 0), (0, 0), (0, -1)
    shift = np.array(shift)
    pml_positions = []

    # corner PMLS
    pml_dict = dict()

    pos = ["bottom left", "top left", "top right", "bottom right"]

    if is_pml_dict:
        width_pml_ = [width_pml[k] for k in pos]
    else:
        width_pml_ = [width_pml for k in pos]

    pml_corners = box_corners + shift * np.array(width_pml_)

    for p, c, w in zip(*[pos, pml_corners, width_pml_]):
        pml_dict[p] = {"corner": tuple(c), "width": w, "direction": "xy"}
    pml_positions += pos
    # x PMLs
    pos = ["left", "right"]
    if is_pml_dict:
        width_pml_x = [width_pml[k] for k in pos]
    else:
        width_pml_x = [(width_pml[0], box.width[1]) for k in pos]

    pml_corners = box_corners[[0, 3]] + shift[[1, 2]] * np.array(width_pml_x)

    for p, c, w in zip(*[pos, pml_corners, width_pml_x]):
        pml_dict[p] = {"corner": tuple(c), "width": w, "direction": "x"}

    pml_positions += pos
    # y PMLs
    pos = ["bottom", "top"]
    # width_pml_y = box.width[0], width_pml[1]
    if is_pml_dict:
        width_pml_y = [width_pml[k] for k in pos]
    else:
        width_pml_y = [(box.width[0], width_pml[1]) for k in pos]

    pml_corners = box_corners[[0, 1]] + shift[[3, 2]] * np.array(width_pml_y)

    for p, c, w in zip(*[pos, pml_corners, width_pml_y]):
        pml_dict[p] = {"corner": tuple(c), "width": w, "direction": "y"}

    pml_positions += pos

    return pml_dict, pml_positions


if __name__ == "__main__":

    import importlib
    import os

    import core

    importlib.reload(core)
    import core

    box_width = (6, 9)
    box = core.Box(width=box_width, pml_width=(2, 4))
    geom = make_box(box)
    pmls = []
    for pos in box.pml_positions:
        pml = core.PML(**box.pml_dict[pos])
        geom = add_pml(pml, geom)
        pmls.append(pml)

    code = geom.get_code()  # .replace("'", "")
    geom_filename = "test.geo"
    mesh_filename = "test.msh"

    with open(geom_filename, "w") as f:
        f.write(code)

    os.system(f"gmsh -2 {geom_filename}")
    os.system(f"gmsh -m {geom_filename} {mesh_filename}")

    #### ------------------------------
    box_width = (6, 9)
    bx, by = box_width
    htop, hbot, hright, hleft = 2, 5, 0.4, 7

    width_pml = {
        "top": (bx, htop),
        "bottom": (bx, hbot),
        "left": (hleft, by),
        "right": (hright, by),
        "bottom left": (hleft, hbot),
        "top left": (hleft, htop),
        "top right": (hright, htop),
        "bottom right": (hright, hbot),
    }

    box = core.Box(width=box_width, pml_width=width_pml)
    geom = make_box(box)
    pmls = []
    for pos in box.pml_positions[1:4]:
        pml = core.PML(**box.pml_dict[pos])
        geom = add_pml(pml, geom)
        pmls.append(pml)

    code = geom.get_code()  # .replace("'", "")
    geom_filename = "test.geo"
    mesh_filename = "test.msh"

    with open(geom_filename, "w") as f:
        f.write(code)

    os.system(f"gmsh -2 {geom_filename}")
    os.system(f"gmsh -m {geom_filename} {mesh_filename}")
