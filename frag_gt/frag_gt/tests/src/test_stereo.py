from __future__ import annotations

from rdkit import Chem

from frag_gt.src.stereo import (
    enumerate_unspecified_stereocenters,
    mol_contains_unspecified_stereo,
)

m_specified = Chem.MolFromSmiles("c1ccccc1[C@@](Cl)(CC)(Br)")
m_unspecified_stereo = Chem.MolFromSmiles("c1ccccc1C(Cl)(CC)(Br)")
m_nostereo = Chem.MolFromSmiles("c1ccccc1")
m_unspecified_bond = Chem.MolFromSmiles("BrC=CC1OC(C2)(F)C2(Cl)C1")


def test_mol_contains_unspecified_stereo() -> None:
    assert mol_contains_unspecified_stereo(m_unspecified_stereo)
    assert mol_contains_unspecified_stereo(m_unspecified_bond)
    assert not mol_contains_unspecified_stereo(m_specified)
    assert not mol_contains_unspecified_stereo(m_nostereo)


def test_enumerate_unspecified_stereocenters() -> None:
    isomers = enumerate_unspecified_stereocenters(m_unspecified_bond)
    assert len(isomers) == 8  # max isomers

    isomers = enumerate_unspecified_stereocenters(m_specified)
    assert len(isomers) == 1
