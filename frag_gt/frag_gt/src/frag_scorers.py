from __future__ import annotations

import logging
from random import shuffle

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from frag_gt.src.afp import calculate_alignment_similarity_scores

logger = logging.getLogger(__name__)


class FragScorer:
    """class to sample and score from fragment list

    Method by which to score fragments
    - "afps" uses the quality of alignment by afps to score fragments (slower than the others)
    - "counts" uses a prior based on the prevalence of fragments in the corpus used to generate the
       fragment database (can also be used to supply any external score)
    - "random" ignores the scoring aspect of this function and returns nan as the scores
    - "ecfp4" ranks candidate replacement fragments from the fragstore according to similarity to the query
    """

    def __init__(self, scorer: str = "random", sort: bool = True) -> None:
        self.scorer = scorer
        self.sort = sort
        logger.info(f"fragment sampler initialised: scorer={scorer} sort={sort}")

    def score(
        self,
        gene_frag_list: list[tuple[str, int]],
        query_frag: Chem.Mol = None,
    ) -> list[tuple[str, float]]:
        """Args:
            gene_frag_list: [("[2*]Cc1cc(O)cc(O[4*])c1", 2), ("[2*]CC(=N[4*])C(C)(C)C", 8), ("[2*]CC(N[4*])C(C)C", 1)]
            query_frag: (optional) mol to guide scoring (not used by "counts" or "random")

        Returns:
            list of (smiles, score) tuples

        """
        # unzip list of tuples retrieved from fragstore
        # this will include any precalculated or saved properties stored with each fragment
        # (e.g. count of how many times fragment occurred in corpus)
        smiles, counts = zip(*gene_frag_list)

        if self.scorer == "counts":
            # score frags using count in corpus
            scores = counts
        elif self.scorer == "ecfp4":
            # score frags using ecfp4 similarity to query frag
            scores = ecfp_fragment_scorer(query_frag, list(smiles))
        elif self.scorer == "afps":
            # score frags using the alignment score
            scores = afp_fragment_scorer(query_frag, list(smiles))
        elif self.scorer == "random":
            # random frag scorer
            scores = np.full(len(smiles), np.nan)
            smiles = list(smiles)
            shuffle(smiles)
        else:
            raise ValueError(f"requested scorer not recognised: {self.scorer}")

        if self.sort:
            # return smiles according to decreasing score (deterministically)
            sorted_tuples = sorted(zip(smiles, scores), key=lambda t: t[1], reverse=True)
            smiles = [s for s, sc in sorted_tuples]
            scores = [sc for s, sc in sorted_tuples]

        # zip back into same format as input
        return list(zip(smiles, scores))


def afp_fragment_scorer(query_mol: Chem.Mol, smiles_list: list[str]) -> list[float]:
    assert query_mol is not None, (
        "Must specify `query_frag` argument if using the afp scorer to sample"
    )
    try:
        scores = calculate_alignment_similarity_scores(query_mol, list(smiles_list))
    except AssertionError as e:
        if str(e) == "query must have attachments":
            # if query has no attachment points, score with fingerprints
            scores = ecfp_fragment_scorer(query_mol, smiles_list)
        else:
            raise
    return list(scores)


def ecfp_fragment_scorer(
    query_mol: Chem.Mol,
    smiles_list: list[str],
    nbits: int = 256,
) -> list[float]:
    scores = np.zeros(len(smiles_list))
    query_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, 2, nBits=nbits)
    for n, s in enumerate(smiles_list):
        m = Chem.MolFromSmiles(s)
        if m is None:
            score = np.nan
        else:
            score = DataStructs.TanimotoSimilarity(
                query_fp,
                AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=nbits),
            )
        scores[n] = score
    return list(scores)
