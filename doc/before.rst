Before Using BCIPy
==================

Installing Pip
--------------
Installing *pip* (Python Package Manager) is the first step to using BCIPy. @NICK this is for you to describe :-}
Once Pip is installed, install BCIPy and its dependencies from Pypi


Installing BCIPy
----------------
**BCIPy is not yet available on Pypi, please clone the repo and import locally.**

BCIPy is available on Pypi, so you can install it using pip:

    pip install bcipy

Installing from Source
----------------------
If you want to install from source, you can clone the repository and install it using pip:

    git clone

    pip install -e . -> Doesn't work because of disutils deprecation

This will install BCIPy in editable mode, so you can modify the source code and have the changes take effect without having to reinstall BCIPy.

Checking the Installation
-------------------------
To check that BCIPy is installed correctly, you can run the following command in a Python console:

    >>> import bcipy
    >>> print(bcipy)

If BCIPy is installed correctly, you should see the following output:
    
        <module 'bcipy' from '.../bcipy/__init__.py'>

If you see an error message, please check the installation instructions again.

Checking Requirements
---------------------
BCIPy uses a number of external libraries. To check that these are installed correctly, you can run the following command inside the library:

    >>> python3 -c "import pkg_resources; pkg_resources.require(open('requirements.txt',mode='r'))"

If no errors are reported, then all the requirements are installed correctly.