#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.1.2
# License: MIT
# See the documentation at gyptis.gitlab.io
import re

fn = "examples/index.rst"

with open(fn, "r") as file:
    lines = file.readlines()
#


title = []
tooltip = []
img = []
#
first_tag = " .. figure:: "
second_tag = ".png"
reg = f"(?<={first_tag}).*?(?={second_tag})"
for line in lines:
    #

    if line.startswith('    <div class="sphx-glr-thumbcontainer" tooltip="'):
        # print(line)
        s = line.split('"')
        tooltip.append(s[-2])

    if line.startswith(" .. figure:: "):
        # print(line)
        s = line.replace(" .. figure:: ", "")
        s = s.replace("\n", "")
        s = line.split("/")
        img.append(s[-1])
        # print(s)

    if line.startswith("     :alt: "):
        s = line.replace("     :alt: ", "")
        s = s.replace("\n", "")
        title.append(s)
        # print(s)


def car_item_active(src, title, tooltip):
    return f"""<div class="carousel-item active">
          <img class="d-block w-100" src="_images/{src}" alt="{title}">
            <div class="carousel-caption d-none d-md-block">
            <h5>{title}</h5>
            <p>{tooltip}</p>
          </div>
        </div>"""


def car_item(src, title, tooltip):
    return f"""<div class="carousel-item">
          <img class="d-block w-100" src="_images/{src}" alt="{title}">
            <div class="carousel-caption d-none d-md-block">
            <h5>{title}</h5>
            <p>{tooltip}</p>
          </div>
        </div>"""


inner = []
ind = []
for i, (im, tit, tool) in enumerate(zip(img, title, tooltip)):
    if i == 0:
        s = car_item_active(im, tit, tool)
        ind.append(
            f'<li data-target="#carouselExampleIndicators" data-slide-to="{i}" class="active"></li>'
        )
    else:
        s = car_item(im, tit, tool)
        ind.append(
            f'<li data-target="#carouselExampleIndicators" data-slide-to="{i}"></li>'
        )
    inner.append(s)
inner = "\n".join(inner)
ind = "\n".join(ind)

car = f"""<div id="carouselExampleIndicators" class="carousel slide" data-ride="carousel">
  <ol class="carousel-indicators">
  {ind}
  </ol>
  <div class="carousel-inner">
  {inner}
  </div>
  <a class="carousel-control-prev" href="#carouselExampleIndicators" role="button" data-slide="prev">
    <span class="carousel-control-prev-icon" aria-hidden="true"></span>
    <span class="sr-only">Previous</span>
  </a>
  <a class="carousel-control-next" href="#carouselExampleIndicators" role="button" data-slide="next">
    <span class="carousel-control-next-icon" aria-hidden="true"></span>
    <span class="sr-only">Next</span>
  </a>
</div>"""


# with open("test.html", "w") as file:
#     file.write(car)
# #


# import os
# os.system("cp _build/html/index.html _build/html/bidon.html")

fn = "_build/html/index.html"

with open(fn, "r") as file:
    lines = file.readlines()


nl = []
for line in lines:
    line = line.replace("__CAROUSSEL_PACEHOLDER__", car)
    nl.append(line)
with open(fn, "w") as file:
    file.write("".join(nl))
