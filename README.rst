

.. |release_badge| image:: https://img.shields.io/endpoint?url=https://gitlab.com/gyptis/gyptis/-/jobs/artifacts/master/raw/logobadge.json?job=badge
  :target: https://gitlab.com/gyptis/gyptis/-/releases
  :alt: Release

.. |GL_CI| image:: https://img.shields.io/gitlab/pipeline/gyptis/gyptis/master?logo=gitlab&labelColor=grey&style=for-the-badge
  :target: https://gitlab.com/gyptis/gyptis/commits/master
  :alt: pipeline status

.. |conda| image:: https://img.shields.io/conda/vn/conda-forge/gyptis?logo=conda-forge&color=CD5C5C&logoColor=white&style=for-the-badge   
  :target: https://anaconda.org/conda-forge/gyptis
  :alt: Conda (channel only)

.. |conda_dl| image:: https://img.shields.io/conda/dn/conda-forge/gyptis?logo=conda-forge&logoColor=white&style=for-the-badge
  :alt: Conda

.. |conda_platform| image:: https://img.shields.io/conda/pn/conda-forge/gyptis?logo=conda-forge&logoColor=white&style=for-the-badge
  :alt: Conda


.. |pip| image:: https://img.shields.io/pypi/v/gyptis?color=blue&logo=pypi&logoColor=e9d672&style=for-the-badge
  :target: https://pypi.org/project/gyptis/
  :alt: PyPI
  
.. |pip_dl| image:: https://img.shields.io/pypi/dm/gyptis?logo=pypi&logoColor=e9d672&style=for-the-badge   
  :alt: PyPI - Downloads
   
.. |pip_status| image:: https://img.shields.io/pypi/status/gyptis?logo=pypi&logoColor=e9d672&style=for-the-badge   
  :alt: PyPI - Status

.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg?logo=python&logoColor=e9d672&style=for-the-badge
  :alt: Code style: black
 
.. |coverage| image:: https://img.shields.io/gitlab/coverage/gyptis/gyptis/master?logo=python&logoColor=e9d672&style=for-the-badge
  :target: https://gitlab.com/gyptis/gyptis/commits/master
  :alt: coverage report 

.. |maintainability| image:: https://img.shields.io/codeclimate/maintainability/benvial/gyptis?logo=code-climate&style=for-the-badge   
  :target: https://codeclimate.com/github/benvial/gyptis
  :alt: Code Climate maintainability

.. |zenodo| image:: https://img.shields.io/badge/DOI-10.5281/zenodo.4938573-5fadad?logo=google-scholar&logoColor=ffffff&style=for-the-badge
  :target: https://doi.org/10.5281/zenodo.4938573
 
.. |licence| image:: https://img.shields.io/badge/license-MIT-blue?color=bb798f&logo=open-access&logoColor=white&style=for-the-badge
  :target: https://gitlab.com/gyptis/gyptis/-/blob/master/LICENCE.txt
  :alt: license
 
+----------------------+----------------------+----------------------+
| Release              |            |release_badge|                  |
+----------------------+----------------------+----------------------+
| Deployment           | |pip|                |        |conda|       |
+----------------------+----------------------+----------------------+
| Build Status         |            |GL_CI|                          |
+----------------------+----------------------+----------------------+
| Metrics              | |coverage|           |   |maintainability|  |
+----------------------+----------------------+----------------------+
| Activity             |     |pip_dl|         |      |conda_dl|      |
+----------------------+----------------------+----------------------+
| Citation             |           |zenodo|                          |
+----------------------+----------------------+----------------------+
| License              |           |licence|                         |
+----------------------+----------------------+----------------------+
| Formatter            |           |black|                           |
+----------------------+----------------------+----------------------+





.. inclusion-marker-badges

GYPTIS
======

Computational Photonics in Python
---------------------------------

https://gyptis.gitlab.io

.. image:: https://gitlab.com/gyptis/gyptis/-/raw/master/docs/_assets/landing.png
   :align: center
   :alt: landing


Gyptis is a package to solve Maxwell's equations with the finite element method. 
It includes predefined models and setup commonly used in Photonics.



Installation
------------

.. inclusion-marker-install-start

Conda
~~~~~

The easiest way is using `conda <https://www.anaconda.com/>`_. 
First, add conda-forge to your channels with:

.. code-block:: bash
    
    conda config --add channels conda-forge
    conda config --set channel_priority strict

Once the conda-forge channel has been enabled, gyptis can be installed with:

.. code-block:: bash
  
  conda install gyptis


Alternatively, we provide an `environment.yml <https://gitlab.com/gyptis/gyptis/-/blob/master/environment.yml>`_ 
file with all the dependencies for the master branch. First create the environment:

.. code-block:: bash

  conda env create -f environment.yml

and then activate it with 

.. code-block:: bash

  conda activate gyptis
  

See the `github repository <https://github.com/conda-forge/gyptis-feedstock/>`_ 
where development happens for conda-forge.
  

Pipy
~~~~

The package is `available on pipy <https://pypi.org/project/gyptis/>`_. 
To install, use:


.. code-block:: bash

  pip install gyptis
  

.. note::
  This does not install FeniCS, which should be built separately 
  (see `instructions <https://fenicsproject.org/download/>`_) 


Docker
~~~~~~

Prebuilt container images are available at `DockerHub <https://hub.docker.com/r/gyptis/gyptis>`_

A quick way of testing is to get the installation script

.. code-block:: bash

  curl -s https://gyptis.gitlab.io/get | bash
  

You will then be able to run the container with 

.. code-block:: bash

  gyptis run


Alternatively, you can pull the docker image

.. code-block:: bash

  docker pull gyptis/gyptis:latest
  
  
To run the image, use

.. code-block:: bash

  docker run -it gyptis/gyptis:latest
  
  

From source
~~~~~~~~~~~~

.. code-block:: bash

  git clone https://gitlab.com/gyptis/gyptis.git
  cd gyptis && pip install -e .
  
  
  
.. inclusion-marker-install-end


Documentation and examples
--------------------------

See the `documentation website <https://gyptis.gitlab.io>`_. 
A good starting point is to look at `examples of application <https://gyptis.gitlab.io/examples/index.html>`_ 
for typical problems encountered in photonics.


Contributing
------------

Pull requests are welcome. For major changes, please open an issue first 
to discuss what you would like to change.

Please make sure to update tests as appropriate.


License
-------

MIT, see `LICENSE.txt <https://gitlab.com/gyptis/gyptis/-/blob/master/LICENSE.txt>`_.
