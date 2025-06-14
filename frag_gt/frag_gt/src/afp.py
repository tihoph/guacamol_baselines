from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np
from rdkit import Chem
from rdkit.Chem.rdmolops import RDKFingerprint

from frag_gt.src.gene_type_utils import get_attachment_type_idx_pairs

if TYPE_CHECKING:
    from rdkit.DataStructs.cDataStructs import ExplicitBitVect

DUMMY_ATOM = Chem.MolFromSmarts("[#0]")


def create_alignment_fp(frag: Chem.Mol, fp_length: int = 256) -> dict[int, ExplicitBitVect]:
    """Create alignment fingerprint (afp) for rdkit mol object containing wildcard attachment points ([*])
    Assumes no two attachment points have the same idx
    """
    alignment_fp = {}
    for match in frag.GetSubstructMatches(DUMMY_ATOM):
        fp = RDKFingerprint(frag, fromAtoms=match, fpSize=fp_length)
        dummy_idx = int(frag.GetAtomWithIdx(match[0]).GetProp("attachment_idx"))
        alignment_fp[dummy_idx] = fp
    return alignment_fp


def compare_alignment_fps(
    afp1: dict[int, ExplicitBitVect],
    afp2: dict[int, ExplicitBitVect],
    mismatch_idx: int = -1,
) -> tuple[dict[int, int], float]:
    """Get alignment between fragments. Match [*] in afp1 to [*] in afp2 based on similar structural environments.
    The atomic property "attachment_idx" is used to match attachment points (idx must be unique in frag)
    mismatch_idx should be a negative integer value, this will be decremented for each unmatched [*].

    Also returns the alignment score

    Usage:
    >>> m1 = renumber_frag_attachment_idxs(Chem.MolFromSmiles("[2*]c1cc(N[1*])c2c(n1)C([3*])CNC([4*])C2"))
    >>> afp1 = create_alignment_fp(m1)
    >>> m2 = renumber_frag_attachment_idxs(Chem.MolFromSmiles("[1*]c1cc(N[3*])c2c(n1)C([2*])CCCC2"))
    >>> afp2 = create_alignment_fp(m2)
    >>> compare_alignment_fps(afp1, afp2)
    """
    # make frag with the least number of connection points the reference
    # this makes it easier to add 'NONE' tags to those not aligned
    # essentially this is "Loop over the shorter one to match the larger"
    if len(afp1) <= len(afp2):
        afp1_ref = True
        afp_ref, afp_target = afp1, afp2
    else:
        afp1_ref = False
        afp_target, afp_ref = afp1, afp2

    # create distance matrix (based on order of afps not keys, need to rematch to afps later)
    dist = np.zeros(shape=(len(afp_ref), len(afp_target)))
    for idx1, rooted_fp1 in enumerate(afp_ref.values()):
        for idx2, rooted_fp2 in enumerate(afp_target.values()):
            dist[idx1][idx2] = Chem.DataStructs.TanimotoSimilarity(rooted_fp1, rooted_fp2)

    # choose picking strategy that allows the maximum score
    best_alignment = None
    best_score = -1
    ref_idxs = list(range(dist.shape[0]))
    target_idxs = list(range(dist.shape[1]))

    # create all permutations of picking strategies (brute force approach).
    # this is an assignment problem, faster solutions exist e.g. hungarian algorithm
    ref_idx_picking_order_list = itertools.permutations(ref_idxs)

    # loop over picking strategies and choose min scoring alignment
    for idx_order in ref_idx_picking_order_list:
        # list of available target idxs with which to match ref idxs
        available_target_idxs = list(target_idxs)

        alignment = {}
        score = 0
        for idx in idx_order:
            # sort target_idxs by most similar to the ref_idx
            bar_sorted = dist[idx].argsort()

            # if most similar target atom is already matched, go to the next most similar
            # keep track of the score and if best, keep alignment that caused that score
            bar_idx = None
            i = -1
            while bar_idx is None:
                if bar_sorted[i] in available_target_idxs:
                    bar_idx = bar_sorted[i]
                    available_target_idxs.remove(bar_idx)
                    alignment[bar_idx] = idx
                    score += dist[idx][bar_idx]
                else:
                    i += -1

        # if there are any unmatched idxs/attachment points in the target, add to alignment
        if available_target_idxs:
            for target_idx in available_target_idxs:
                alignment[target_idx] = "NONE"

        if score > best_score:
            best_alignment = alignment
            best_score = score

    assert best_alignment is not None, "best alignment was not identified, somethings wrong"

    # match alignment idxs of ref and target back to original keys of afps
    final_alignment = {}
    afp1_idxs, afp2_idxs = list(afp1.keys()), list(afp2.keys())
    if afp1_ref:
        for ref_idx, target_idx in best_alignment.items():
            if not isinstance(target_idx, str):
                final_alignment.update({afp1_idxs[target_idx]: afp2_idxs[ref_idx]})
            else:
                final_alignment.update({mismatch_idx: afp2_idxs[ref_idx]})
                mismatch_idx -= 1
    else:
        for ref_idx, target_idx in best_alignment.items():
            if not isinstance(target_idx, str):
                final_alignment.update({afp1_idxs[ref_idx]: afp2_idxs[target_idx]})
            else:
                final_alignment.update({afp1_idxs[ref_idx]: mismatch_idx})
                mismatch_idx -= 1

    return final_alignment, best_score


def trivial_matching_for_unique_isotope_pairs(
    frag_type_idx_pairs: set[tuple[int, int]],
    ref_type_idx_pairs: set[tuple[int, int]],
) -> tuple[dict[int, int], int]:
    ref_types = [x for x, _ in ref_type_idx_pairs]
    frag_types = [x for x, _ in frag_type_idx_pairs]

    # if there is 1 occurence of a gene type in both ref and frag, map attachment idxs to each other
    alignment = {}
    mismatch_idx = -1
    for t, frag_i in frag_type_idx_pairs:
        if frag_types.count(t) == 1:
            if ref_types.count(t) == 1:
                for pair in ref_type_idx_pairs:
                    if pair[0] == t:
                        alignment[frag_i] = pair[1]
                        break

            elif ref_types.count(t) == 0:
                alignment[frag_i] = mismatch_idx
                mismatch_idx -= 1

    return alignment, mismatch_idx


def renumber_frag_attachment_idxs(
    mol: Chem.Mol,
    idxmap: dict[int, int] | None = None,
) -> Chem.Mol:
    """Given a frag with attachment points (RDKit mol object), add "attachment_idx" property to attachment atoms
    If idxmap is provided, get existing "attachment_idx" and set a new mapped idx
    >>>renumber_frag_attachment_idxs(Chem.MolFromSmiles("c1c([7*])cccc1[4*]"))
    """
    idx = 0
    for m in mol.GetSubstructMatches(DUMMY_ATOM):
        a = mol.GetAtomWithIdx(m[0])
        if idxmap is None:
            a.SetProp("attachment_idx", str(idx))
            idx += 1
        else:
            a.SetProp("attachment_idx", str(idxmap[int(a.GetProp("attachment_idx"))]))
    return mol


def create_alignment_map(frag: Chem.Mol, reference_frag: Chem.Mol) -> dict[int, int]:
    """Return a dictionary mapping `frag` attachment idx -> `reference_frag` attachment idx"""
    # TODO: check symmetrical gene types

    # first check for a trivial alignment (unique isotopes in both ref and frag)
    ref_type_idx_pairs = get_attachment_type_idx_pairs(reference_frag)
    frag_type_idx_pairs = get_attachment_type_idx_pairs(frag)

    trivial_alignment, mismatch_idx = trivial_matching_for_unique_isotope_pairs(
        frag_type_idx_pairs,
        ref_type_idx_pairs,
    )

    if len(trivial_alignment) != max(len(ref_type_idx_pairs), len(frag_type_idx_pairs)):
        afp1 = create_alignment_fp(frag)
        afp2 = create_alignment_fp(reference_frag)

        for k, v in trivial_alignment.items():
            if k >= 0:
                del afp1[k]
            if v >= 0:
                del afp2[v]

        afp_alignment, _ = compare_alignment_fps(afp1, afp2, mismatch_idx=mismatch_idx)

        alignment = {**trivial_alignment, **afp_alignment}
    else:
        alignment = trivial_alignment

    return alignment


def match_fragment_attachment_points(
    frag: Chem.Mol,
    reference_frag: Chem.Mol,
) -> Chem.Mol:
    """Given a reference fragment with attachment points.
    Find the most similar attachment points for each [*] in a query molecule

    Try all alignments of frag([*:A], [*:B], [*:C]) to ref([*:X], [*:Y], [*:Z])
    e.g. (AX, BY, CZ), (AY, BX, CZ), etc...
    Choose alignment which produces an alignment fingerprint (afp) similarity most similar to original molecule
    """
    # Add "attachment_idx" prop to new frag atoms
    frag = renumber_frag_attachment_idxs(frag)

    # align
    alignment = create_alignment_map(frag, reference_frag)

    # Renumber mutant attachment idxs according to alignment
    return renumber_frag_attachment_idxs(frag, idxmap=alignment)


def calculate_alignment_similarity_scores(
    query_frag: Chem.Mol,
    frag_smiles: list[str],
) -> np.ndarray:
    """Score the similarity of attachment points to a query frag
    e.g. if a frag has 3 attachment points [*], the maximum score is 3 (1.0+1.0+1.0)

    note: the intuition behind this fn fails when the number of attachments is not equal between query and ref

    Args:
        query_frag: rdkit mol containing attachments
        frag_smiles: frag smiles with attachments

    Returns:
        numpy array of scores where higher scores are more similar attachment fingerprints

    """
    scores = np.zeros(len(frag_smiles))
    query_frag = renumber_frag_attachment_idxs(Chem.Mol(query_frag))
    afp1 = create_alignment_fp(query_frag)
    assert len(afp1), "query must have attachments"
    for n, s in enumerate(frag_smiles):
        m = renumber_frag_attachment_idxs(Chem.MolFromSmiles(s))
        afp2 = create_alignment_fp(m)
        _, score = compare_alignment_fps(afp1, afp2)
        scores[n] = score
    return scores
