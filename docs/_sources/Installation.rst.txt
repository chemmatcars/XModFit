.. _Installation:

Installation
============
Follow the following instructions for installation:

1) Install Anaconda python (Python 3.8 and higher) for your operating system from `Anaconda website <https://www.anaconda.com/products/individual>`_.

2) If `GIT <https://git-scm.com/book/en/v2/Getting-Started-Installing-Git>`_ is not installed aleady in the system. Install GIT using conda in the terminal::

    conda install git

2) In the terminal run the following to install all the dependency packages::

    pip install --upgrade PyQt5 sqlalchemy scipy six matplotlib pandas lmfit pylint periodictable corner emcee tabulate python-docx numba numba-scipy statsmodels sympy
    pip install pyqtgraph==0.12.1

3) After GIT installation go to the folder (for example: /home/mrinal/Download) you wish to download XModFit run the following in the terminal::

        git clone https://github.com/chemmatcars/XModFit

   The method will create **XModFit** folder with all updated packages in installation folder (i.e. /home/mrinal/Download). The advantage of this method is that it is easy to upgrade the package later on. In order to upgrade, go to the folder named **XModFit** and run the following command::

            git pull

6) Go into the folder XModFit and run the command to run **XModFit**::

            python xmodfit.py


