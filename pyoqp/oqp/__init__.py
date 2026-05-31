"""OQP instance"""

import os
import platform
from oqp.utils.mpi_utils import MPIManager
MPIManager()
# we must import dftd4 ffi lib before oqp to load library correctly
try:
    import dftd4.interface
except ModuleNotFoundError:
    print('\nPyOQP: dftd4 is not available')

try:
    os.environ["OPENQP_ROOT"]
except KeyError:
    os.environ["OPENQP_ROOT"] = os.path.abspath(os.path.dirname(__file__))
#    exit('\nPyQOP: cannot find environment variable $OPENQP_ROOT\n')

try:
    int(os.environ['OMP_NUM_THREADS'])
except (KeyError, ValueError):
    os.environ['OMP_NUM_THREADS'] = '1'


def _oqp_wrapper(func):
    """Decorator for OQP library functions"""

    def wrapper(molecule, *args):
        return func(molecule.data._data, *args)

    return wrapper


if os.environ.get('OQP_RTLD'):
    RTLD = str(os.environ.get('OQP_RTLD')).lower() in ('true', '1', 't', 'y', 'yes', 'on')
else:
    RTLD = True

# When OQP_BACKEND_OPTIONAL is set, a missing or unloadable native library is not
# fatal at import time. This lets pure-Python components (e.g.
# oqp.library.fci.solve_fci, oqp.utils.input_checker) be imported and unit-tested
# without a compiled liboqp build, which is useful for fast CI. Default behaviour
# is unchanged: the import still fails loudly when the backend cannot be loaded.
_BACKEND_OPTIONAL = str(os.environ.get('OQP_BACKEND_OPTIONAL', '')).lower() in (
    'true', '1', 't', 'y', 'yes', 'on'
)

#: True once the compiled liboqp backend has been loaded successfully.
BACKEND_AVAILABLE = False

try:
    if RTLD:
        from cffi import FFI

        ffi = FFI()
        oqp_root = os.environ["OPENQP_ROOT"]

        if platform.uname()[0] == "Windows":
            suffix = "dll"
        elif platform.uname()[0] == "Linux":
            suffix = "so"
        elif platform.uname()[0] == "Darwin":
            suffix = "dylib"
        else:
            suffix = "so"

        with open(f"{oqp_root}/include/oqp.h", "r", encoding="ascii") as oqp_header:
            defs = oqp_header.read().replace("#include", "//#include")

        ffi.cdef(defs)
        lib = ffi.dlopen(f"{oqp_root}/lib/liboqp.{suffix}", ffi.RTLD_GLOBAL)

    else:
        from _oqp import ffi, lib

    for attr_name in dir(lib):
        attr_value = getattr(lib, attr_name)
        if callable(attr_value):
            if attr_name not in ('oqp_init', 'oqp_clean', 'oqp_set_atoms'):
                globals()[attr_name] = _oqp_wrapper(attr_value)
            else:
                globals()[attr_name] = attr_value

    BACKEND_AVAILABLE = True

except Exception as exc:  # pragma: no cover - only reached without a native build
    if not _BACKEND_OPTIONAL:
        raise
    import warnings

    ffi = None
    lib = None
    warnings.warn(
        "PyOQP native backend could not be loaded "
        f"({type(exc).__name__}: {exc}); OQP_BACKEND_OPTIONAL is set, so the "
        "package will still import. Only pure-Python components are available; "
        "any calculation requiring the compiled library will fail.",
        RuntimeWarning,
    )
