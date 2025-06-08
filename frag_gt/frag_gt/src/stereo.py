from __future__ import annotations

from rdkit import Chem
from rdkit.Chem.EnumerateStereoisomers import (
    EnumerateStereoisomers,
    StereoEnumerationOptions,
)
from rdkit.Chem.rdchem import StereoSpecified

STEREO_OPTIONS = StereoEnumerationOptions(tryEmbedding=True, unique=True, maxIsomers=8, rand=None)

# TODO: explore fragment on chiral https://sourceforge.net/p/rdkit/mailman/message/35420297/


def mol_contains_unspecified_stereo(m: Chem.Mol) -> bool:
    try:
        si = Chem.FindPotentialStereo(m)
    except ValueError as e:
        print(e)
        print(Chem.MolToSmiles(m))
        return False
    return bool(any(element.specified == StereoSpecified.Unspecified for element in si))


def enumerate_unspecified_stereocenters(m: Chem.Mol) -> list[Chem.Mol]:
    if mol_contains_unspecified_stereo(m):
        isomers = list(EnumerateStereoisomers(m, options=STEREO_OPTIONS))
    else:
        isomers = [m]
    return isomers
