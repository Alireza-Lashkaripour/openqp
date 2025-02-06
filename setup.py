from skbuild import setup
from setuptools import find_packages

setup(
    name="pyoqp",
    version="0.1.0",
    description="OpenQP test with scikit-build",
    author="Your Name",
    packages=find_packages(),  # Or ["pyoqp", "pyoqp.oqp"] if you prefer
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.10.0",
        "libdlfind>=0.0.3",
        "cffi>=1.16.0",
        "mpi4py>=4.0.0",
    ],
    cmake_args=[
        "-DUSE_LIBINT=OFF",
        "-DCMAKE_C_COMPILER=gcc",
        "-DCMAKE_CXX_COMPILER=g++",
        "-DCMAKE_Fortran_COMPILER=gfortran",
        "-DENABLE_OPENMP=ON",
        "-DLINALG_LIB_INT64=OFF",
    ],
    cmake_install_dir="pyoqp/oqp",  

    entry_points={
        "console_scripts": [
            "openqp_test=pyoqp.oqp.pyoqp:main"
        ]
    },
    python_requires=">=3.7",
)

