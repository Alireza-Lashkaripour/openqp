"""Small dense Full Configuration Interaction energy driver."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb

import numpy as np

import oqp
from oqp.utils.file_utils import dump_log


@dataclass(frozen=True)
class FCISettings:
    nroot: int = 1
    active_electrons: int = 0
    active_orbitals: int = 0
    frozen_core: int = 0
    max_det: int = 50000
    eig_tol: float = 1.0e-10
    integral_backend: str = "native"
    integral_cutoff: float = 5.0e-11


def _annihilate(det: int, orb: int) -> tuple[int, int] | None:
    bit = 1 << orb
    if not det & bit:
        return None
    phase = -1 if (det & (bit - 1)).bit_count() % 2 else 1
    return det ^ bit, phase


def _create(det: int, orb: int) -> tuple[int, int] | None:
    bit = 1 << orb
    if det & bit:
        return None
    phase = -1 if (det & (bit - 1)).bit_count() % 2 else 1
    return det | bit, phase


def _occupied(det: int, norb: int) -> list[int]:
    return [orb for orb in range(norb) if det & (1 << orb)]


def _string_basis(norb: int, nelec: int) -> list[int]:
    return [sum(1 << orb for orb in occ) for occ in combinations(range(norb), nelec)]


def _determinants(norb: int, nelec: tuple[int, int]) -> list[int]:
    alpha = _string_basis(norb, nelec[0])
    beta = _string_basis(norb, nelec[1])
    return [a | (b << norb) for a in alpha for b in beta]


def _as_nelec_pair(nelec: int | tuple[int, int] | list[int]) -> tuple[int, int]:
    if isinstance(nelec, (tuple, list)):
        if len(nelec) != 2:
            raise ValueError("nelec must be an integer or an (nalpha, nbeta) pair")
        return int(nelec[0]), int(nelec[1])
    if int(nelec) % 2:
        raise ValueError("integer nelec implies a closed-shell singlet and must be even")
    return int(nelec) // 2, int(nelec) // 2


def _spin_orbital_integrals(
    h1e: np.ndarray,
    eri: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    norb = h1e.shape[0]
    nspin = 2 * norb
    hspin = np.zeros((nspin, nspin), dtype=float)
    gspin = np.zeros((nspin, nspin, nspin, nspin), dtype=float)

    hspin[:norb, :norb] = h1e
    hspin[norb:, norb:] = h1e

    for p in range(nspin):
        ps = 0 if p < norb else 1
        pp = p % norb
        for q in range(nspin):
            qs = 0 if q < norb else 1
            qq = q % norb
            for r in range(nspin):
                if ps != (0 if r < norb else 1):
                    continue
                rr = r % norb
                for s in range(nspin):
                    if qs == (0 if s < norb else 1):
                        gspin[p, q, r, s] = eri[pp, rr, qq, s % norb]

    return hspin, gspin


def solve_fci(
    h1e: np.ndarray,
    eri: np.ndarray,
    nelec: int | tuple[int, int] | list[int],
    *,
    ecore: float = 0.0,
    nroot: int = 1,
    max_det: int = 50000,
    eig_tol: float = 1.0e-10,
    integral_cutoff: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Diagonalize a dense FCI Hamiltonian in a determinant basis.

    The two-electron tensor uses chemists' notation ``(pq|rs)`` over spatial
    orbitals. This routine is intentionally dense and meant for small active
    spaces; it is not a replacement for an iterative production-scale CI solver.
    """

    h1e = np.asarray(h1e, dtype=float)
    eri = np.asarray(eri, dtype=float)
    if h1e.ndim != 2 or h1e.shape[0] != h1e.shape[1]:
        raise ValueError("h1e must be a square matrix")
    norb = h1e.shape[0]
    if eri.shape != (norb, norb, norb, norb):
        raise ValueError("eri must have shape (norb, norb, norb, norb)")

    nalpha, nbeta = _as_nelec_pair(nelec)
    if min(nalpha, nbeta) < 0 or max(nalpha, nbeta) > norb:
        raise ValueError("electron count is incompatible with the orbital count")

    ndet = comb(norb, nalpha) * comb(norb, nbeta)
    if ndet < 1:
        raise ValueError("FCI determinant space is empty")
    if ndet > max_det:
        raise ValueError(
            f"FCI determinant space has {ndet} determinants, exceeding max_det={max_det}"
        )
    if nroot < 1 or nroot > ndet:
        raise ValueError(f"nroot must be between 1 and the determinant count ({ndet})")

    dets = _determinants(norb, (nalpha, nbeta))
    det_index = {det: idx for idx, det in enumerate(dets)}
    nspin = 2 * norb
    hspin, gspin = _spin_orbital_integrals(h1e, eri)
    hamiltonian = np.zeros((ndet, ndet), dtype=float)

    for col, det in enumerate(dets):
        occ = _occupied(det, nspin)

        for q in occ:
            ann_q = _annihilate(det, q)
            if ann_q is None:
                continue
            det_q, phase_q = ann_q
            for p in range(nspin):
                hval = hspin[p, q]
                if abs(hval) <= integral_cutoff:
                    continue
                cre_p = _create(det_q, p)
                if cre_p is None:
                    continue
                det_pq, phase_p = cre_p
                row = det_index.get(det_pq)
                if row is not None:
                    hamiltonian[row, col] += hval * phase_q * phase_p

        for r in occ:
            ann_r = _annihilate(det, r)
            if ann_r is None:
                continue
            det_r, phase_r = ann_r
            for s in _occupied(det_r, nspin):
                ann_s = _annihilate(det_r, s)
                if ann_s is None:
                    continue
                det_rs, phase_s = ann_s
                for q in range(nspin):
                    cre_q = _create(det_rs, q)
                    if cre_q is None:
                        continue
                    det_qrs, phase_q = cre_q
                    for p in range(nspin):
                        gval = gspin[p, q, r, s]
                        if abs(gval) <= integral_cutoff:
                            continue
                        cre_p = _create(det_qrs, p)
                        if cre_p is None:
                            continue
                        det_pqrs, phase_p = cre_p
                        row = det_index.get(det_pqrs)
                        if row is not None:
                            hamiltonian[row, col] += (
                                0.5 * gval * phase_r * phase_s * phase_q * phase_p
                            )

    hamiltonian = 0.5 * (hamiltonian + hamiltonian.T)
    eigvals, eigvecs = np.linalg.eigh(hamiltonian)
    for root in range(nroot):
        residual = np.linalg.norm(
            hamiltonian @ eigvecs[:, root] - eigvals[root] * eigvecs[:, root]
        )
        if residual > eig_tol:
            raise ValueError(
                f"FCI diagonalization residual {residual:.3e} exceeds eig_tol={eig_tol:.3e}"
            )
    return eigvals[:nroot] + float(ecore), eigvecs[:, :nroot]


def _unpack_lower_triangle(packed: np.ndarray, n: int) -> np.ndarray:
    matrix = np.zeros((n, n), dtype=float)
    idx = 0
    for i in range(n):
        for j in range(i + 1):
            matrix[i, j] = packed[idx]
            matrix[j, i] = packed[idx]
            idx += 1
    return matrix


def _transform_integrals(
    hcore_ao: np.ndarray,
    eri_ao: np.ndarray,
    coeff: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    h1e = coeff.T @ hcore_ao @ coeff
    eri = np.einsum(
        "up,vq,wr,xs,uvwx->pqrs",
        coeff,
        coeff,
        coeff,
        coeff,
        eri_ao,
        optimize=True,
    )
    return h1e, eri


def _settings_from_config(config: dict) -> FCISettings:
    raw = config.get("fci", {})
    return FCISettings(
        nroot=int(raw.get("nroot", 1)),
        active_electrons=int(raw.get("active_electrons", 0)),
        active_orbitals=int(raw.get("active_orbitals", 0)),
        frozen_core=int(raw.get("frozen_core", 0)),
        max_det=int(raw.get("max_det", 50000)),
        eig_tol=float(raw.get("eig_tol", 1.0e-10)),
        integral_backend=str(raw.get("integral_backend", "native")).lower(),
        integral_cutoff=float(raw.get("integral_cutoff", 5.0e-11)),
    )


def _active_space(
    h1e: np.ndarray,
    eri: np.ndarray,
    nelec: tuple[int, int],
    ecore: float,
    settings: FCISettings,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int], float, dict[str, int]]:
    nalpha, nbeta = nelec
    norb = h1e.shape[0]
    frozen_core = settings.frozen_core

    if settings.active_electrons:
        inactive_electrons = nalpha + nbeta - settings.active_electrons
        if inactive_electrons < 0 or inactive_electrons % 2:
            raise ValueError("active_electrons must leave a non-negative closed-shell core")
        inferred_core = inactive_electrons // 2
        if frozen_core not in (0, inferred_core):
            raise ValueError("frozen_core is inconsistent with active_electrons")
        frozen_core = inferred_core

    if frozen_core < 0 or frozen_core > min(nalpha, nbeta):
        raise ValueError("frozen_core is incompatible with the electron count")

    active_nelec = (nalpha - frozen_core, nbeta - frozen_core)
    active_norb = settings.active_orbitals or (norb - frozen_core)
    if active_norb < max(active_nelec):
        raise ValueError("active_orbitals is too small for the active electron count")
    if frozen_core + active_norb > norb:
        raise ValueError("active_orbitals extends beyond the available MO space")

    core = range(frozen_core)
    active = slice(frozen_core, frozen_core + active_norb)
    h_active = h1e[active, active].copy()
    eri_active = eri[active, active, active, active].copy()
    ecore_active = float(ecore)

    if frozen_core:
        for i in core:
            ecore_active += 2.0 * h1e[i, i]
        for i in core:
            for j in core:
                ecore_active += 2.0 * eri[i, i, j, j] - eri[i, j, j, i]
        for p in range(active_norb):
            pp = p + frozen_core
            for q in range(active_norb):
                qq = q + frozen_core
                for i in core:
                    h_active[p, q] += 2.0 * eri[pp, qq, i, i] - eri[pp, i, i, qq]

    metadata = {
        "norb": norb,
        "active_orbitals": active_norb,
        "active_electrons": sum(active_nelec),
        "frozen_core": frozen_core,
        "determinants": comb(active_norb, active_nelec[0]) * comb(active_norb, active_nelec[1]),
    }
    return h_active, eri_active, active_nelec, ecore_active, metadata


class FCI:
    """FCI energy calculation using OpenQP-native RHF orbitals and integrals."""

    def __init__(self, mol):
        self.mol = mol
        self.settings = _settings_from_config(mol.config)

    def energy(self, ref_energy=None):
        if self.settings.integral_backend != "native":
            raise ValueError("Only fci.integral_backend=native is implemented")
        if self.mol.config["input"]["functional"]:
            raise ValueError("FCI requires HF integrals; unset input.functional")
        if self.mol.config["scf"]["type"] != "rhf":
            raise ValueError("FCI MVP supports only closed-shell RHF references")
        if self.mol.data["nelec_A"] != self.mol.data["nelec_B"]:
            raise ValueError("FCI MVP supports only closed-shell singlets")

        h1e, eri, nelec, ecore, metadata = self._native_mo_integrals()
        energies, coeffs = solve_fci(
            h1e,
            eri,
            nelec,
            ecore=ecore,
            nroot=self.settings.nroot,
            max_det=self.settings.max_det,
            eig_tol=self.settings.eig_tol,
            integral_cutoff=self.settings.integral_cutoff,
        )

        self.mol.energies = energies.tolist()
        self.mol.data["OQP::FCI_ENERGIES"] = np.ascontiguousarray(energies, dtype=np.float64)
        self.mol.data["OQP::FCI_CI_VECTORS"] = np.ascontiguousarray(coeffs, dtype=np.float64)
        self.mol.data["OQP::FCI_DET_COUNT"] = np.ascontiguousarray(
            [metadata["determinants"]],
            dtype=np.int64,
        )
        self.mol.mol_energy.energy = float(energies[0])

        dump_log(
            self.mol,
            title="PyOQP: Full Configuration Interaction",
            section="fci",
            info={
                "hf_energy": ref_energy[0] if ref_energy else None,
                "energies": self.mol.energies,
                **metadata,
            },
        )
        return self.mol.energies

    def _native_mo_integrals(self):
        oqp.fci_ao_integrals(self.mol)

        nbf = int(self.mol.data.get_basis()["nbf"])
        hcore = _unpack_lower_triangle(np.asarray(self.mol.data["OQP::Hcore"], dtype=float), nbf)
        coeff = np.asarray(self.mol.data["OQP::VEC_MO_A"], dtype=float).reshape((nbf, nbf)).T
        eri_ao = np.asarray(self.mol.data["OQP::AO_ERI"], dtype=float).reshape(
            (nbf, nbf, nbf, nbf),
            order="F",
        )

        h1e, eri = _transform_integrals(hcore, eri_ao, coeff)
        nelec = (int(self.mol.data["nelec_A"]), int(self.mol.data["nelec_B"]))
        ecore = float(self.mol.mol_energy.nenergy)
        return _active_space(h1e, eri, nelec, ecore, self.settings)
