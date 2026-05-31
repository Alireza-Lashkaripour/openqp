import importlib
import os
import sys

import numpy as np
import pytest

# Allow the pure-Python FCI components (solve_fci and the input checker) to be
# imported and tested without a compiled liboqp build. Tests that genuinely need
# the native backend skip themselves when it is unavailable.
os.environ.setdefault("OQP_BACKEND_OPTIONAL", "1")


def _use_real_oqp_package():
    # Several lightweight unit tests stub oqp.* modules in sys.modules. The FCI
    # tests exercise the real installed package, so they must not inherit those
    # stubs when the whole suite runs in filename order.
    for name in list(sys.modules):
        if name == "oqp" or name.startswith("oqp."):
            sys.modules.pop(name, None)
    importlib.invalidate_caches()


def _backend_available() -> bool:
    """True when the compiled liboqp backend loaded (required for Runner tests)."""
    _use_real_oqp_package()
    import oqp

    return bool(getattr(oqp, "BACKEND_AVAILABLE", False))


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
    if not _backend_available():
        pytest.skip("native OQP backend not built; build liboqp to run end-to-end FCI tests")
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


def test_active_space_with_dropped_virtuals_matches_casci():
    """Frozen core + active orbitals smaller than the remaining MO space (so at
    least one virtual is dropped) must reproduce a PySCF CASCI energy."""
    pytest.importorskip("pyscf")
    from pyscf import ao2mo, gto, mcscf, scf

    _use_real_oqp_package()
    from oqp.library.fci import FCISettings, _active_space, solve_fci

    mol = gto.Mole()
    mol.atom = "O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587"
    mol.unit = "Angstrom"
    mol.basis = "sto-3g"
    mol.verbose = 0
    mol.build(cart=True)

    mf = scf.RHF(mol)
    mf.conv_tol = 1.0e-12
    mf.kernel()
    assert mf.converged

    coeff = mf.mo_coeff
    h1e = coeff.T @ mf.get_hcore() @ coeff
    eri = ao2mo.restore(1, ao2mo.kernel(mol, coeff), coeff.shape[1])

    # CAS(2e,2o) on H2O/STO-3G: 4 frozen-core orbitals, 2 active, >=1 dropped virtual.
    energy_cas = mcscf.CASCI(mf, 2, 2).kernel()[0]

    settings = FCISettings(active_electrons=2, active_orbitals=2)
    h_act, eri_act, nelec_act, ecore_act, meta = _active_space(
        h1e, eri, tuple(mol.nelec), float(mol.energy_nuc()), settings
    )

    assert meta["frozen_core"] == 4
    assert meta["active_orbitals"] == 2
    assert meta["norb"] - meta["frozen_core"] - meta["active_orbitals"] >= 1
    assert meta["determinants"] == 4

    energies, _ = solve_fci(h_act, eri_act, nelec_act, ecore=ecore_act, nroot=1, max_det=1000)
    assert energies[0] == pytest.approx(energy_cas, abs=1.0e-9)


def test_solver_rejects_oversized_determinant_space():
    _use_real_oqp_package()
    from oqp.library.fci import solve_fci

    norb = 8  # (4 alpha, 4 beta) -> C(8,4)^2 = 4900 determinants
    h1e = np.zeros((norb, norb))
    eri = np.zeros((norb, norb, norb, norb))
    with pytest.raises(ValueError, match="max_det"):
        solve_fci(h1e, eri, (4, 4), max_det=100)


def test_solver_rejects_oversized_dense_hamiltonian():
    _use_real_oqp_package()
    from oqp.library.fci import solve_fci

    norb = 8  # 4900 determinants -> dense H ~ 192 MiB; cap the budget at 1 MiB.
    h1e = np.zeros((norb, norb))
    eri = np.zeros((norb, norb, norb, norb))
    with pytest.raises(ValueError, match="max_memory"):
        solve_fci(h1e, eri, (4, 4), max_det=100000, max_memory=1)


@pytest.mark.parametrize(
    ("override", "bad_path"),
    [
        ({"input": {"runtype": "grad"}}, "input.runtype"),
        ({"scf": {"type": "uhf", "multiplicity": 1}}, "scf.type"),
        ({"input": {"functional": "b3lyp"}}, "input.functional"),
        ({"fci": {"nroot": 0}}, "fci.nroot"),
        ({"fci": {"max_det": 0}}, "fci.max_det"),
        ({"fci": {"max_memory": 0}}, "fci.max_memory"),
        ({"fci": {"frozen_core": -1}}, "fci.active_space"),
        ({"fci": {"integral_backend": "pyscf"}}, "fci.integral_backend"),
    ],
)
def test_input_checker_rejects_invalid_fci(override, bad_path):
    _use_real_oqp_package()
    from oqp.utils.input_checker import check_input_values

    config = {
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
    }
    for section, values in override.items():
        config.setdefault(section, {}).update(values)

    report = check_input_values(config, raise_error=False, emit=False)

    assert not report.ok, "expected the invalid FCI input to be rejected"
    assert any(d.path == bad_path for d in report.diagnostics), report.to_text()
