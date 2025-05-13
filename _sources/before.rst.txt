Before Using MindPype
==================

Installing Pip
--------------
Installing *pip* (Python Package Manager) is the first step to using MindPype. @NICK this is for you to describe :-}
Once Pip is installed, install MindPype and its dependencies from Pypi


Installing MindPype
-------------------
**MindPype is not yet available on Pypi, please clone the repo and import locally.**

MindPype is available on Pypi, so you can install it using pip:

    pip install mindpype

Installing from Source
----------------------
If you want to install from source, you can clone the repository and install it using pip:

    git clone

    pip install -e . -> Doesn't work because of disutils deprecation

This will install MindPype in editable mode, so you can modify the source code and have the changes take effect without having to reinstall MindPype.

Checking the Installation
-------------------------
To check that MindPype is installed correctly, you can run the following command in a Python console:

    >>> import mindpype
    >>> print(mindpype)

If MindPype is installed correctly, you should see the following output:

        <module 'mindpype' from '.../mindpype/__init__.py'>

If you see an error message, please check the installation instructions again.

Checking Requirements
---------------------
MindPype uses a number of external libraries. To check that these are installed correctly, you can run the following command inside the library:

    >>> python3 -c "import pkg_resources; pkg_resources.require(open('requirements.txt',mode='r'))"

If no errors are reported, then all the requirements are installed correctly.