

.. image:: https://img.shields.io/gitlab/pipeline/gyptis/gyptis/master?style=for-the-badge
   :target: https://gitlab.com/gyptis/gyptis/commits/master
   :alt: pipeline status

.. image:: https://img.shields.io/gitlab/coverage/gyptis/gyptis/master?logo=s&logoColor=white&style=for-the-badge
  :target: https://gitlab.com/gyptis/gyptis/commits/master
  :alt: coverage report
  
.. image:: https://img.shields.io/pypi/v/gyptis?color=blue&logo=python&logoColor=yellow&style=for-the-badge   
  :target: https://pypi.org/project/gyptis/
  :alt: PyPI

.. image:: https://img.shields.io/github/license/mashape/apistatus.svg?style=for-the-badge
   :alt: Licence: MIT

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg?style=for-the-badge
   :alt: Code style: black

GYPTIS
======

Computational photonics in Python
---------------------------------

https://gyptis.gitlab.io

.. image:: https://gitlab.com/gyptis/gyptis/-/raw/master/docs/_assets/landing.png
   :align: center
   :alt: landing


Gyptis is a package to solve Maxwell's equations with the finite element method. 
It is in early stage and currently being actively developed, so features might 
come and go.


Installation
------------


Conda
~~~~~

The easiest way is using `conda <https://www.anaconda.com/>`_. 
We provide an `environment.yml <https://gitlab.com/gyptis/gyptis/-/blob/master/environment.yml>`_ 
file with all the dependencies. First create the environment:

.. code-block:: bash

  conda env create -f environment.yml

and then activate it with 

.. code-block:: bash

  conda activate gyptis
  
A `conda-forge <https://github.com/conda-forge/staged-recipes/pull/14424>`_ package 
is being developed and should be available soon.


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

On quick way of testing is getting the installation script

.. code-block:: bash

  curl -s https://gyptis.gitlab.io/get | bash


Or you can pull the docker image

.. code-block:: bash

  docker pull gyptis/gyptis:latest
  
  
To run the image, use

.. code-block:: bash

  docker run -it gyptis/gyptis:latest
  



Contributing
------------

Pull requests are welcome. For major changes, please open an issue first 
to discuss what you would like to change.

Please make sure to update tests as appropriate.


License
-------

MIT, see `LICENSE.txt <https://gitlab.com/gyptis/gyptis/-/blob/master/LICENSE.txt>`_.
