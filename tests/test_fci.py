import importlib
import sys

import numpy as np
import pytest


def _use_real_oqp_package():
    # Several lightweight unit tests stub oqp.* modules in sys.modules. The FCI
    # tests exercise the real installed package, so they must not inherit those
    # stubs when the whole suite runs in filename order.
    for name in list(sys.modules):
        if name == "oqp" or name.startswith("oqp."):
            sys.modules.pop(name, None)
    importlib.invalidate_caches()


def _pyscf_reference(atom, charge=0, spin=0, basis="sto-3g"):
    pyscf = pytest.importorskip("pyscf")
    from pyscf import ao2mo, fci, gto, scf

    mol = gto.Mole()
    mol.atom = atom
    mol.unit = "Angstrom"
    mol.basis = basis
    mol.charge = charge
    mol.spin = spin
    mol.verbose = 0
    mol.build(cart=True)

    mf = scf.RHF(mol)
    mf.conv_tol = 1.0e-12
    mf.kernel()
    assert mf.converged

    coeff = mf.mo_coeff
    h1e = coeff.T @ mf.get_hcore() @ coeff
    eri = ao2mo.restore(1, ao2mo.kernel(mol, coeff), coeff.shape[1])
    energy_ref, _ = fci.FCI(mol, coeff).kernel()
    return h1e, eri, mol.nelec, mol.energy_nuc(), energy_ref


@pytest.mark.parametrize(
    ("atom", "charge", "expected"),
    [
        # Reference totals generated from live PySCF 2.13.0 output.
        ("H 0 0 0; H 0 0 0.740", 0, -1.137283834489),
        ("He 0 0 0; H 0 0 1.4632", 1, -2.826674836464),
    ],
)
def test_dense_fci_solver_matches_live_pyscf_reference(atom, charge, expected):
    _use_real_oqp_package()
    from oqp.library.fci import solve_fci

    h1e, eri, nelec, ecore, energy_ref = _pyscf_reference(atom, charge=charge)
    assert energy_ref == pytest.approx(expected, abs=5.0e-11)

    energies, _ = solve_fci(
        h1e,
        eri,
        nelec,
        ecore=ecore,
        nroot=1,
        max_det=1000,
    )

    assert energies[0] == pytest.approx(energy_ref, abs=1.0e-10)


def test_input_checker_accepts_fci_energy():
    _use_real_oqp_package()
    from oqp.utils.input_checker import check_input_values

    report = check_input_values(
        {
            "input": {
                "system": "\nH 0 0 0\nH 0 0 0.740",
                "basis": "sto-3g",
                "method": "fci",
                "runtype": "energy",
            },
            "guess": {"type": "hcore"},
            "scf": {"type": "rhf", "multiplicity": 1},
            "fci": {"nroot": 1, "max_det": 1000},
            "properties": {"scf_prop": []},
        },
        raise_error=False,
        emit=False,
    )

    assert report.ok, report.to_text()


@pytest.mark.parametrize(
    ("name", "system", "atom", "charge", "abs_tol"),
    [
        (
            "h2_fci",
            "\nH 0 0 0\nH 0 0 0.740",
            "H 0 0 0; H 0 0 0.740",
            "0",
            1.0e-8,
        ),
        (
            "h2o_fci",
            "\nO 0.000000 0.000000 0.000000"
            "\nH 0.000000 -0.757000 0.587000"
            "\nH 0.000000 0.757000 0.587000",
            "O 0.000000 0.000000 0.000000;"
            " H 0.000000 -0.757000 0.587000;"
            " H 0.000000 0.757000 0.587000",
            "0",
            5.0e-8,
        ),
    ],
)
def test_openqp_fci_matches_live_pyscf_reference(tmp_path, name, system, atom, charge, abs_tol):
    pytest.importorskip("pyscf")
    _use_real_oqp_package()
    from oqp.pyoqp import Runner

    _, _, _, _, energy_ref = _pyscf_reference(atom, charge=int(charge))

    runner = Runner(
        project=name,
        input_file=None,
        log=str(tmp_path / f"{name}.log"),
        input_dict={
            "input": {
                "system": system,
                "charge": charge,
                "basis": "sto-3g",
                "method": "fci",
                "runtype": "energy",
            },
            "guess": {"type": "hcore"},
            "scf": {
                "type": "rhf",
                "multiplicity": "1",
                "maxit": "60",
                "forced_attempt": "3",
                "save_molden": "False",
            },
            "properties": {"scf_prop": ""},
            "fci": {"nroot": "1", "max_det": "1000"},
            "tests": {"exception": "True"},
        },
        silent=1,
        usempi=False,
    )

    runner.run(test_mod=True)

    assert np.asarray(runner.mol.energies)[0] == pytest.approx(energy_ref, abs=abs_tol)
