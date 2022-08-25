from setuptools import setup

setup(
    name='bci_py',
    version='0.1.0',
    description='A python library to enhance BCI data processing pipeline design and development',
    author='Nicolas Ivanov and Aaron Lio',
    author_email='aaron.lio@mail.utoronto.ca',
    packages=['bcipy'],
    install_requires=['matplotlib==3.5.2', 
                      'more-itertools==8.2.0',
                      'numpy==1.22.4',
                      'numpydoc==0.9.2',
                      'pyriemann==0.2.7',
                      'scikit-learn==1.1.1',
                      'scipy==1.8.1',
                      'pylsl==1.14.0'],
    classifiers=['Programming Language :: Python :: 3',
                 'Intended Audience :: Science/Research',
                 'Topic :: Scientific/Engineering :: Human Machine Interfaces',
                 ],
)