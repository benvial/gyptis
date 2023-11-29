#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.0.2
# License: MIT
# See the documentation at gyptis.gitlab.io


import os

import gyptis

header = f"""#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: {gyptis.__version__}
# License: MIT
# See the documentation at gyptis.gitlab.io
"""


def rep_header(python_file, header):
    with open(python_file, "r") as f:
        lines = f.readlines()
    i = 0
    current_header = []
    for line in lines:
        if line.startswith("#"):
            current_header.append(line)
            i += 1
        else:
            break

    new_header = header.splitlines()
    new_header = [h + "\n" for h in new_header]
    if new_header != current_header:
        print(f"updating header in {python_file}")
        new_header = "".join(new_header)
        data = new_header + "".join(lines[i:])
        with open(python_file, "w") as f:
            f.write(data)


def update(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                python_file = os.path.abspath(os.path.join(root, file))
                rep_header(python_file, header)


for directory in ["../"]:
    update(directory)
