from __future__ import annotations

import argparse
import datetime
import logging
import os
import pickle
from time import time

import numpy as np
from guacamol.utils.helpers import setup_default_logger
from rdkit import Chem

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def chembl_problematic_case(key: str) -> bool:
    """When generating the statistics on ChEMBL, some KeyError exceptions were generated if not checking for
    this special case.
    """
    allowed_key_beginnings = {"[#6R", "[#7R"}

    tokens = key.split("]")

    return "=" in key and tokens[0] not in allowed_key_beginnings


def read_file(file_name: str) -> list[str]:
    """Args:
        file_name: Text file with one SMILES per line

    Returns: a list of SMILES strings

    """
    with open(file_name) as f:
        return [s.strip() for s in f]


def get_counts(
    smarts_list: list[str], smiles_list: list[str], ring: bool = False
) -> tuple[int, dict[str, int]]:
    """Args:
        smarts_list: list of SMARTS of intrest
        smiles_list: a list of SMILES strings
        ring: determines whether or not the matches are uniquified

    Returns:
        tot: sum of SMARTS counts
        probs2: an dictionary of {SMART: counts}

    """
    probs = {}

    for smarts in smarts_list:
        probs[smarts] = 0

    # number_of_molecules = 0
    # tot = 0
    for smiles in smiles_list:
        # print smiles
        # number_of_molecules += 1
        mol = Chem.MolFromSmiles(smiles)
        Chem.Kekulize(mol)
        for smarts in smarts_list:
            matches = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts), uniquify=ring)
            num_bonds = len(matches)  # not the number of bonds, but the number of matches
            probs[smarts] += num_bonds
            # tot += num_bonds

    tot = 0
    probs2 = {}
    for key, value in probs.items():
        if value > 0:
            # print key, probs[key]
            tot += value
            probs2[key] = value

    return tot, probs2


def clean_counts(probs: dict[str, float]):
    """Removes counts for certain SMARTS
    SMARTS are pairs of atoms
    Used only to prepare input for get_rxn_smarts

    Args:
        probs: Dict of {SMARTS: probability}
               probability is actually a count of occurrences
               SMARTS are permutations of pairs of atoms (joined by bonds) where one of these atoms is not in a ring
               {'[#6]-[#6;!R]': 448), '[#6]-[#7;!R]': 173, ...]}

    Returns:
        tot: sum of SMARTS counts
        probs2: an dict of {SMART: counts}

    """
    #            Triple bondedN, Carbonyl, Fl, Cl, Br, I
    exceptions = ["[#7]#", "[#8]=", "[#9]", "[#17]", "[#35]", "[#53]"]
    probs2 = {}

    # for key in probs:
    #   skip = False
    #   for exception in exceptions:
    #     if exception in key:
    #       tokens = re.split('\[|\]|;',key)
    #       alt_key = '['+tokens[3]+']'+tokens[2]+'['+tokens[1]+';!R]'
    #       probs[alt_key] += probs[key]

    for key, value in probs.items():
        skip = False
        for exception in exceptions:
            if exception in key:
                skip = True
        if not skip:
            probs2[key] = value

    tot = sum(probs2.values())

    return tot, probs2


def get_probs(probs: dict[str, float], tot, ignore_problematic: bool = False):
    """From counts to probabilities

    Args:
        probs: Dict of {SMARTS: un-normalised probability}
        ignore_problematic: for cases not supported by get_rxn_smarts_make_rings and get_rxn_smarts_rings, the corresponding smarts must also be ignored here.

    Returns: Dict of {SMARTS: normalised probability}

    """
    p = []

    # When ignoring some smarts, we must adapt the total count in order for the probabilities to be normalized to 1.0
    ignored_count = 0

    for key, value in probs.items():
        if ignore_problematic and chembl_problematic_case(key):
            logger.warning(f"Ignoring key {key} in get_probs to be consistent with other functions")
            ignored_count += value
            continue

        p.append(float(value))

    adapted_tot = tot - ignored_count
    return [prob / adapted_tot for prob in p]


def get_rxn_smarts_make_rings(probs: dict[str, float]) -> list[str]:
    """Generate reaction smarts to form a three-membered ring from two atoms that are not in a ring already
    SMARTS for 3 atom sequences in rings are transformed from XYZ to give ring forming reaction smarts XZ>>X1YZ1
    Transformation CC >> C1CC1 will be performed by [#6;!R:1]=,-[#6;!R:2]>>[*:1]1-[#6R][*:2]1

    Args:
        probs: Dict of {SMARTS: probability}
               probability is actually a count of occurrences (not used)
               SMARTS are permutations of 3 specific atoms in a ring
               {'[#6R]-[#6R]-[#6R]': 296, '[#6R]=[#6R]-[#6R]': 1237, ...}

    Returns: list of reaction SMARTS

    """
    X = {"[#6R": "X4", "[#7R": "X3"}
    rxn_smarts: list[str] = []
    for key in probs:
        if chembl_problematic_case(key):
            logger.warning(f"Ignoring unsupported key {key} in get_rxn_smarts_make_rings")
            continue

        tokens = key.split("]")  # ['[#6R', '-[#6R', '-[#6R', '']

        smarts = ""
        if "=" in key:
            smarts += (
                tokens[0][:-1] + X[tokens[0]] + ";!R:1]"
            )  # [:-1] slice strips trailing R from smarts
        else:
            smarts += tokens[0][:-1] + ";!R:1]=,"  # [:-1] slice strips trailing R from smarts

        smarts += tokens[2][:-1] + ";!R:2]>>"
        smarts += "[*:1]1" + tokens[1] + "][*:2]1"

        # print(rxn_smarts)
        # ['[#6;!R:1]=,-[#6;!R:2]>>[*:1]1-[#6R][*:2]1']
        # ['[#6X4;!R:1]-[#6;!R:2]>>[*:1]1=[#6R][*:2]1'] (if '=' in key)
        rxn_smarts.append(smarts)

    return rxn_smarts


def get_rxn_smarts_rings(probs: dict[str, float]) -> list[str]:
    """Generate reaction smarts to insert one atom in a ring (will not touch 6 or 7-membered rings)
    SMARTS matching 3 atom sequences in rings are transformed from XYZ to give reaction SMARTS XZ>>XYZ
    Transformation C1CCC1 >> C1CCCC1 will be performed by [#6R;!r6;!r7;!R2:1]-[#6R;!r6;!r7:2]>>[*:1]-[#6R][*:2]

    Args:
        probs: Dict of {SMARTS: probability}
               probability is actually a count of occurrences (not used)
               SMARTS are permutations of 3 specific atoms in a ring
               {'[#6R]-[#6R]-[#6R]': 296, '[#6R]=[#6R]-[#6R]': 1237, ...}

    Returns: list of reaction SMARTS

    """
    X = {
        "[#6R": "X4",
        "[#7R": "X3",
    }  # carbons should have four bonds / nitrogens should have three bonds

    rxn_smarts: list[str] = []
    for key in probs:
        if chembl_problematic_case(key):
            logger.warning(f"Ignoring unsupported key {key} in get_rxn_smarts_rings")
            continue

        tokens = key.split("]")  # ['[#6R', '-[#6R', '-[#6R', '']

        smarts = ""
        if "=" in key:
            # This seems to be intended for aromatic ring systems
            # [#6RX4;!r6;!r7;!R2:1] C in a ring where ring size != 6 or 7, with 4 total connections. Not in two rings
            smarts += tokens[0] + X[tokens[0]] + ";!r6;!r7;!R2:1]"
        else:
            # This seems to be intended for aliphatic ring systems
            # [#6R;!r6;!r7;!R2:1] C in a ring of size not 6 and not 7. Not in 2 rings
            smarts += tokens[0] + ";!r6;!r7;!R2:1]"

        smarts += tokens[2] + ";!r6;!r7:2]>>"
        smarts += "[*:1]" + tokens[1] + "][*:2]"

        # print(smarts)
        # [#6R;!r6;!r7;!R2:1]-[#6R;!r6;!r7:2]>>[*:1]-[#6R][*:2]
        # [#6RX4;!r6;!r7;!R2:1]-[#6R;!r6;!r7:2]>>[*:1]=[#6R][*:2] (if '=' in key)
        rxn_smarts.append(smarts)

    return rxn_smarts


def get_rxn_smarts(probs: dict[str, float]) -> list[str]:
    """Generate reaction smarts to add acyclic atoms

    Args:
        probs: Dict of {SMARTS: probability}
               probability is actually a count of occurrences (not used)
               SMARTS are permutations of pairs of atoms (joined by bonds) where one of these atoms is not in a ring
               Unlike other functions, here input has been "cleaned" to remove certain functional groups
               {'[#6]-[#6;!R]': 448, '[#6]-[#7;!R]': 173, ...}

    Returns: list of reaction SMARTS

    """
    rxn_smarts: list[str] = []
    for key in probs:  # key <-> smarts
        # smarts = ''
        tokens = key.split("]")  # ['[#6', '-[#7;!R', '']
        smarts = tokens[0]
        if "-" in key and "#16" not in smarts:  # check for sulfur
            smarts += ";!H0:1]>>[*:1]"  # make sure root atom has one or more hydrogens before adding single bond
        if "=" in key and "#16" not in smarts:  # check for sulfur
            smarts += ";!H1;!H0:1]>>[*:1]"  # make sure root atom has two or more hydrogens before adding double bond
        if "]#[" in key:
            smarts += ";H3:1]>>[*:1]"  # 3 hydrogens are required on root atom in order to introduce a triple bond
        if "#16" in smarts:  # key <-> smarts
            smarts += ":1]>>[*:1]"  # if sulfur, do nothing

        # e.g. [#6;!H0:1]>>[*:1]-[#6;!R] add carbon atom to root carbon if root carbon has one or more hydrogens
        smarts += tokens[-2] + "]"
        rxn_smarts.append(smarts)

    return rxn_smarts


def get_mean_size(smiles_list: list[str]) -> tuple[float, float]:
    """Calculates number of atoms `mean` and `std`
    given a SMILES list

    Args:
        smiles_list: list of SMILES

    Returns: mean, std

    """
    size: list[int] = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        num_atoms = mol.GetNumAtoms()
        size.append(num_atoms)

    return np.mean(size), np.std(size)


def count_macro_cycles(
    smiles_list: list[str], smarts_list: list[str], tot, probs: dict[str, float]
):
    """Args:
        smiles_list: list of SMILES
        smarts_list: list of SMARTS
        tot: counter of ... TODO: why is this passed?
        probs: Dict of {SMARTS: counts}

    Returns:

    """
    # probs = {}
    for smarts in smarts_list:
        probs[smarts] = 0

    for smiles in smiles_list:
        for smarts in smarts_list:
            mol = Chem.MolFromSmiles(smiles)
            Chem.Kekulize(mol)
            matches = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts), uniquify=True)
            if len(matches) > 0:
                probs[smarts] += 1
                tot += 1

    return tot, probs


class StatsCalculator:
    """Contains code of the original main function for a more convenient calculation of statistics."""

    #          Use of '#6' notation conflates aromatic and aliphatic atoms
    #          ['B',  'C',  'N',  'O',  'F',  'Si',  'P',   'S',   'Cl',  'Se',  'Br',  'I']
    elements = (
        "#5",
        "#6",
        "#7",
        "#8",
        "#9",
        "#14",
        "#15",
        "#16",
        "#17",
        "#34",
        "#35",
        "#53",
    )
    bonds = ("-", "=", "#")

    def __init__(self, smiles_file: str) -> None:
        self.smiles_list = read_file(smiles_file)
        self.tot, self.probs = self.smarts_element_and_rings_probs()

    def size_statistics(self) -> tuple[float, float]:
        size_mean, size_stdv = get_mean_size(self.smiles_list)
        return size_mean.item(), size_stdv.item()

    def atom_in_ring_probs(self) -> tuple[int, dict[str, int]]:
        # SMARTS probabilities (atom in ring)
        smarts = [
            "[*]",  # all
            "[R]",  # atom in ring
            "[!R]",  # atom not in ring
            "[R2]",
        ]  # atom in 2 rings

        return get_counts(smarts, self.smiles_list)

    def smarts_ring_probs(self) -> tuple[int, dict[str, int]]:
        # SMARTS probabilities (rings)
        smarts = [
            "[R]~[R]~[R]",  # Any 3 ring atoms connected by any two bonds
            "[R]-[R]-[R]",  # Any 3 ring atoms connected by two single bonds
            "[R]=[R]-[R]",
        ]  # Any 3 ring atoms connected by one single and one double bond

        return get_counts(smarts, self.smiles_list, ring=True)

    def smarts_element_and_element_in_ring_probs(self) -> tuple[int, dict[str, int]]:
        # SMARTS probabilities (elements + elements in ring)
        # smarts = []
        # for element in self.elements:
        #     smarts.append('[' + element + ']')  # elemental abundance

        smarts = [
            "[" + element + "R]" for element in self.elements
        ]  # elemental abundance wihin a ring

        return get_counts(smarts, self.smiles_list)

    def smarts_element_and_rings_probs(self) -> tuple[int, dict[str, int]]:
        tot_Ratoms, probs_Ratoms = self.smarts_element_and_element_in_ring_probs()

        # TODO: rewrite
        R_elements = list(probs_Ratoms)

        # Generate smarts for all permutations of 3 atoms in a ring (limited to atoms in 'elements') e.g. [#6R]=[#7R]-[#6R]
        smarts = []
        for i, e1 in enumerate(R_elements):
            for e2 in R_elements:
                for j, e3 in enumerate(R_elements):
                    if (
                        j >= i
                    ):  # makes sure identical reversed smarts patterns aren't generated (C-N-O and O-N-C)
                        sm_s = e1 + "-" + e2 + "-" + e3
                        if sm_s not in smarts:
                            smarts.append(sm_s)
                    sm_d = e1 + "=" + e2 + "-" + e3
                    if sm_d not in smarts:
                        smarts.append(sm_d)

        return get_counts(smarts, self.smiles_list, ring=True)

    def rxn_smarts_rings(self) -> list[str]:
        return get_rxn_smarts_rings(self.probs)

    def rxn_smarts_make_rings(self) -> list[str]:
        # Generate reaction smarts to grow rings (not 6 or 7-membered) by inserting one atom
        return get_rxn_smarts_make_rings(self.probs)

    def ring_probs(self):
        return get_probs(self.probs, self.tot, ignore_problematic=True)

    def smarts_pair_probs(self) -> tuple[int, dict[str, int]]:
        smarts = []
        for bond in self.bonds:
            for element1 in self.elements:
                smarts.extend(
                    "[" + element1 + "]" + bond + "[" + element2 + ";!R]"
                    for element2 in self.elements
                )

        tot, probs = get_counts(smarts, self.smiles_list)
        return clean_counts(probs)

    def pair_probs(self):
        tot, probs = self.smarts_pair_probs()
        # Normalise probs (which actually contains counts) to give probabilities
        return get_probs(probs, tot)

    def rxn_smarts(self) -> list[str]:
        # Generate reaction smarts to add atoms to a root atom with a specified bond type
        tot, probs = self.smarts_pair_probs()
        return get_rxn_smarts(probs)

    def smarts_macrocycles_probs(self) -> tuple[int, dict[str, int]]:
        # count aliphatic and aromatic rings of size 3-6
        smarts_list = [
            "[*]1-[*]-[*]-1",
            "[*]1-[*]=[*]-1",
            "[*]1-[*]-[*]-[*]-1",
            "[*]1=[*]-[*]-[*]-1",
            "[*]1=[*]-[*]=[*]-1",
            "[*]1-[*]-[*]-[*]-[*]-1",
            "[*]1=[*]-[*]-[*]-[*]-1",
            "[*]1=[*]-[*]=[*]-[*]-1",
            "[*]1-[*]-[*]-[*]-[*]-[*]-1",
            "[*]1=[*]-[*]-[*]-[*]-[*]-1",
            "[*]1=[*]-[*]=[*]-[*]-[*]-1",
            "[*]1=[*]-[*]-[*]=[*]-[*]-1",
            "[*]1=[*]-[*]=[*]-[*]=[*]-1",
        ]

        # count occurence of macrocycles of size 7-12
        smarts_macro = [
            "[r;!r3;!r4;!r5;!r6;!r8;!r9;!r10;!r11;!r12]",
            "[r;!r3;!r4;!r5;!r6;!r7;!r9;!r10;!r11;!r12]",
            "[r;!r3;!r4;!r5;!r6;!r7;!r8;!r10;!r11;!r12]",
            "[r;!r3;!r4;!r5;!r6;!r7;!r8;!r9;!r11;!r12]",
            "[r;!r3;!r4;!r5;!r6;!r7;!r8;!r9;!r10;!r12]",
            "[r;!r3;!r4;!r5;!r6;!r7;!r8;!r9;!r10;!r11]",
        ]

        tot, probs = get_counts(smarts_list, self.smiles_list, ring=True)

        return count_macro_cycles(self.smiles_list, smarts_macro, tot, probs)

    def number_rings(self) -> int:
        tot, probs = self.smarts_macrocycles_probs()
        num_rings = 0
        for key in probs:
            print(key, probs[key])
            num_rings += probs[key]
        return num_rings


def main() -> None:
    setup_default_logger()

    parser = argparse.ArgumentParser(
        description="Generate pickle files for the statistics of a training set for MCTS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--smiles_file",
        default="data/guacamol_v1_all.smiles",
        help="Full path to SMILES file from which to generate the distribution statistics",
    )
    parser.add_argument("--output_dir", default=None, help="Output directory for the pickle files")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.dirname(os.path.realpath(__file__))

    logger.info("Generating probabilities for MCTS...")

    t0 = time()

    stats = StatsCalculator(args.smiles_file)

    size_stats = stats.size_statistics()
    rxn_smarts_rings = stats.rxn_smarts_rings()
    rxn_smarts_make_rings = stats.rxn_smarts_make_rings()
    p_rings = stats.ring_probs()

    pickle.dump(size_stats, open(os.path.join(args.output_dir, "size_stats.p"), "wb"))
    pickle.dump(p_rings, open(os.path.join(args.output_dir, "p_ring.p"), "wb"))
    pickle.dump(rxn_smarts_rings, open(os.path.join(args.output_dir, "rs_ring.p"), "wb"))
    pickle.dump(
        rxn_smarts_make_rings,
        open(os.path.join(args.output_dir, "rs_make_ring.p"), "wb"),
    )

    p = stats.pair_probs()
    rxn_smarts = stats.rxn_smarts()

    pickle.dump(p, open(os.path.join(args.output_dir, "p1.p"), "wb"))
    pickle.dump(rxn_smarts, open(os.path.join(args.output_dir, "r_s1.p"), "wb"))

    print(f"Total time: {datetime.timedelta(seconds=int(time() - t0))!s} secs")


if __name__ == "__main__":
    main()
