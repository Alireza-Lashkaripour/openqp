# FCI examples (`method=fci`)

Small **dense** Full Configuration Interaction energy calculations on top of a
closed-shell RHF reference, using OpenQP-native molecular orbitals and AO
two-electron integrals.

> **MVP scope.** This is a correctness baseline and architecture seed, *not* a
> production-scale FCI engine. It builds an explicit determinant-basis
> Hamiltonian and diagonalizes it densely, so cost and memory grow
> exponentially with the active space. Use it for small systems only.

## How to run

```bash
openqp --nompi examples/FCI/H2_RHF-FCI_ENERGY.inp
```

A minimal input:

```ini
[input]
runtype=energy
method=fci
basis=sto-3g
system=
   H   0.0   0.0   0.0
   H   0.0   0.0   0.74

[scf]
type=rhf
multiplicity=1

[fci]
nroot=1
max_det=1000
```

## `[fci]` options

| key                | default  | meaning |
|--------------------|----------|---------|
| `nroot`            | `1`      | number of CI roots (states) to return |
| `active_electrons` | `0`      | electrons in the active space (`0` = all) |
| `active_orbitals`  | `0`      | spatial orbitals in the active space (`0` = all after the frozen core) |
| `frozen_core`      | `0`      | doubly-occupied orbitals frozen into the core |
| `max_det`          | `5000`   | hard cap on the determinant count (dense solver) |
| `max_memory`       | `2048`   | memory budget in MiB for the dense Hamiltonian / AO integrals |
| `eig_tol`          | `1.0e-10`| residual tolerance for the dense eigensolver sanity check |
| `integral_backend` | `native` | integral source (only `native` is supported) |
| `integral_cutoff`  | `5.0e-11`| magnitude below which integral contributions are skipped |

An active space is defined as `frozen_core` doubly-occupied core orbitals plus
`active_orbitals` active orbitals; any higher virtual orbitals are dropped
(CASCI-style). Setting `active_electrons` infers `frozen_core` automatically.

## Validation

The dense solver has been checked against PySCF (FCI / CASCI) with **identical
MO integrals**, where it reproduces the reference energy to machine precision:

| system / space                  | PySCF reference (Eh) | dets | |Δ| vs PySCF |
|---------------------------------|----------------------|------|--------------|
| H2 / STO-3G, full FCI           | −1.137283834489      | 4    | ~9e-16 |
| H2O / STO-3G, full FCI          | −75.012647118992     | 441  | ~1e-12 |
| HeH⁺ / STO-3G, full FCI         | −2.826674836464      | —    | ~4e-16 |
| H2O / STO-3G, CAS(8e,6o)        | −75.012569053757     | 225  | ~1e-14 |
| H2O / STO-3G, CAS(2e,2o)        | −74.964311401165     | 4    | ~3e-14 |
| H2, three lowest roots          | −1.1373/−0.5308/−0.1684 | — | ~1e-16 |

End to end (OpenQP's own RHF orbitals and AO integrals), `H2_RHF-FCI_ENERGY.inp`
yields −1.137283835869487 Eh (stored in `H2_RHF-FCI_ENERGY.json`). The small
difference from the PySCF total above reflects SCF/integral thresholds, not the
FCI algorithm. These checks are reproduced by `tests/test_fci.py`.

## Current limitations

1. Closed-shell RHF singlet **energy** only (no gradients, no open shell, no UHF/ROHF).
2. Dense determinant Hamiltonian + dense `numpy.linalg.eigh` (no Davidson yet).
3. Dense `nbf**4` AO ERI storage (no symmetry compression); guarded by `max_memory`.
4. Practical only for very small active spaces (a few thousand determinants).
