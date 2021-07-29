.. _Installation:

Installation
============
Follow the following instructions for installation:

1) Install Anaconda python (Python 3.8 and higher) for your operating system from `Anaconda website <https://www.anaconda.com/products/individual>`_

2) Open a terminal (Anaconda prompt for Win 10) and create a virtual environment by running this commands::

    conda create --name xmodfit python=3.8

3) Once the virtual environment is installed, activate the environment by running this in the same terminal::

    conda activate xmodfit

4) In the terminal run the following to install all the dependency packages::

    pip install --upgrade PyQt5 pyqtgraph sqlalchemy scipy six matplotlib pandas lmfit pylint periodictable corner emcee tabulate python-docx numba numba-scipy statsmodels sympy

5) The installation can be done in two different ways:

    a) Easier and preferred way is through `GIT <https://git-scm.com/book/en/v2/Getting-Started-Installing-Git>`_. If git is not already installed in your computer you can install it
through Anaconda by running this command in the terminal (Anaconda prompt for Win 10)::

        conda install git

After GIT installation go to the folder (for example: /home/mrinal/Download) you wish to download XModFit run the following in the terminal::

        git clone https://github.com/nayanbera/XModFit


The method will create **XModFit** folder with all updated packages in installation folder (i.e. /home/mrinal/Download). The advantage of this method is that it is easy to upgrade the package later on. In order to upgrade, go to the folder named **XModFit** and run the following command::

            git pull

    b) Universal way which does not need GIT installation:
	    i) Open a web browser and go to the webpage : https://github.com/nayanbera/XModFit
	    ii) Click the green button named "Clone or download"
	    iii) Download the zip file
   	    iv) Extract the zip file into a folder
   	    v) In the Anaconda terminal go the the extracted folder::

   	            cd /home/mrinal/Download/XModFit-master

6) Run the command to run **XModFit**::

            python xmodfit.py

7) Once the installation is completed and for running the software at any later time you need to open a terminal (Anaconda prompt for Win 10) and activate to the 'xmodfit' virtual environment first before running the software::

    conda activate xmodfit
    python xmodfit.py
