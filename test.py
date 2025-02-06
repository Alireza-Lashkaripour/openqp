import os
import sys

# First, just verify that we can import the package.
try:
    import openqp_test
    print("Successfully imported openqp_test!")
except ImportError as e:
    print(f"Failed to import openqp_test: {e}")
    sys.exit(1)

# Now let's dig into the actual classes provided by your pyoqp/oqp code.
# We specifically want to use the Runner class from pyoqp.py
# (adjust the import path if your layout is different).
try:
    from openqp_test.oqp.pyoqp import Runner
    print("Successfully imported Runner from openqp_test.oqp.pyoqp!")
except ImportError as e:
    print(f"Failed to import Runner: {e}")
    sys.exit(1)

# Create a minimal example "input_dict" that mimics the .inp files you normally use.
# The exact content will depend on your code's config schema; here's a generic example:
example_input = {
    "input": {
        "runtype": "energy",     # or "grad", "nac", etc.
        # Potentially add "theory", "charge", "multiplicity", etc. if required
    },
    "basis": {
        "name": "sto-3g",       # a small basis set name you might support
    },
    "molecule": {
        # Provide a minimal geometry. For instance, a hydrogen molecule:
        "xyz": """2
H2 minimal test
H  0.0  0.0  0.0
H  0.7  0.0  0.0
"""
    },
    # You may need other config sections, e.g. "guess", "tests", etc.
    "tests": {
        "do_test": False
    },
    "guess": {
        "type": "none",
        "file": ""
    }
}

# Instantiate the Runner with our in-memory input
runner = Runner(
    project="test_project",
    input_dict=example_input,
    log="test_openqp_test.log",  # or any log filename
    silent=1,                    # 0 for verbose, 1 for silent
    usempi=False                 # Disable MPI in a simple local test
)

# Attempt to run the calculation
try:
    runner.run()
    # If run completes without exceptions, gather results
    results = runner.results()
    print("Calculation ran successfully! Results:")
    print(results)
except Exception as e:
    print("An error occurred during run():", e)
    sys.exit(1)

# If you get here, it means your openqp_test Python package code ran end-to-end.
import pyoqp
