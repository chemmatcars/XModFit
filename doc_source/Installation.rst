.. _Installation:

Installation
============
Follow the following instructions for installation:

1) Install Anaconda python (Python 3.8 and higher) for your operating system from `Anaconda website <https://www.anaconda.com/products/individual>`_.

2) (**Optional**) If you would like create an anaconda environment 'cmcars' to install XModFit run the following in the terminal (MacOS/linux) or Anaconda terminal (Windows)::

    conda create -n cmcars python=3.9

3) (**Optional**) If you have created or using an anaconda environment the switch to the existing environment (cmcars) by running the following command::

    conda activate cmcars

5) In the terminal run the following to install all the dependency packages::

    conda install git
    pip install --upgrade PyQt5 sqlalchemy scipy six matplotlib pandas lmfit pylint periodictable corner emcee tabulate python-docx numba numba-scipy statsmodels sympy
    pip install pyqtgraph==0.12.1

6) In order to install in a particluar folder (for example: /home/mrinal/Download) run the following in the terminal::

        git clone https://github.com/chemmatcars/XModFit

   The method will create **XModFit** folder with all updated packages in installation folder (i.e. /home/mrinal/Download). The advantage of this method is that it is easy to upgrade the package later on. In order to upgrade, go to the folder named **XModFit** and run the following command::

            git pull

7) Change to the anaconda environment you used for installing XModFit first (if you are not already in that environment) by running::

     conda activate cmcars

8) Go into the folder XModFit and run the command to run **XModFit**::

            python xmodfit.py


