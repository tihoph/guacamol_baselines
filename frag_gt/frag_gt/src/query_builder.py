from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from rdkit import Chem

from frag_gt.src.frag_scorers import FragScorer

if TYPE_CHECKING:
    from frag_gt.src.fragstore import FragStoreBase

logger = logging.getLogger(__name__)


class FragQueryBuilder:
    """This class is used to communicate with a fragment store
    Candidate replacement fragments are identified by their gene type

    if desired a random sample of returned fragments is taken
    and fragments are scored by either:
     - "counts": frequency of frag in fragstore
     - "ecfp4": tanimoto similarity to ref_frag
     - "afps": (dev) afp similarity to ref_frag
     - "random": null scorer

    Also provides the prob of setting n_choices to 1 (via `single_frag_prob`),
    this mimics random selection when used with tournament selection (skip the tournament)
    """

    def __init__(
        self,
        frag_store: FragStoreBase,
        scorer: str = "random",
        sort_by_score: bool = False,
        single_frag_prob: float = 0.0,
        sample_with_replacement: bool = False,
    ) -> None:
        self.frag_sampler = FragScorer(scorer=scorer, sort=sort_by_score)
        self._sort_by_score = sort_by_score
        self.db = frag_store
        self.db.load()
        self.single_frag_prob = single_frag_prob
        self.sample_with_replacement = sample_with_replacement

    def query_frags(
        self,
        gene_type: str,
        ref_frag: Chem.Mol | None = None,
        x_choices: float = -1,
    ) -> tuple[list[str], list[float]]:
        """Function to retrieve fragments to replace a given reference fragment.

        Args:
           gene_type: gene type to query fragstore with (e.g. "5#5")
           ref_frag:  (optional) query mol to guide mol-dependent sampling methods (not used by "counts" or "random")
           x_choices: (optional) number of random frags to return from fragstore,
                    default -1 returns all available frags for a gene_type
                    float between 0 and 1 returns a proportion of the frags available for that gene_type (variable)
                    int value will return that number of frags (up to max available for gene type)

        Returns:
           A list of SMILES strings for the generated molecules
           A list of scores for those molecules (if sort_by_score=False, these are unsorted)

        """
        if gene_type == "":
            logger.debug(f"empty gene_type: {Chem.MolToSmiles(ref_frag)} Skipping mutation")
            return [], []

        logger.debug(f"Finding genes with gene_type: {gene_type}")

        # get pool of haplotype fragment replacements
        # this returns either an empty list (if gene_type not in fragstore), or a nested iterable i.e. ([],)
        gene_type_frags = list(self.db.get_records("gene_types", {"gene_type": gene_type}))
        logger.debug(
            f"Possible genes (fragments) with gene_type {gene_type}: {len(gene_type_frags)}",
        )

        if len(gene_type_frags) == 0:
            return [], []
        if len(gene_type_frags) > 1:
            raise RuntimeError(
                f"More than one gene_type record in FragStore for {gene_type}, something is corrupted.",
            )

        # unzip results to a list of tuples of (smiles, score)
        genes = []  # type: list[tuple[str, int]]
        for record in gene_type_frags[0]["haplotypes"].values():
            # if there are properties saved in the fragstore, we can filter on those here
            # e.g. mw range around `ref_frag` (if mw in fragstore)
            genes.extend([(s, atts["count"]) for s, atts in record["gene_frags"].items()])

        # determine how many genes to sample
        if np.random.uniform(0, 1) <= self.single_frag_prob:
            # return a single frag, regardless of `x_choices` value (see docstring above)
            n_choices = 1
        elif isinstance(x_choices, float):
            # if float, n_choices is a proportion of available frags
            n_choices = max(int(len(genes) * x_choices), 1)
        elif x_choices == -1 or x_choices > len(genes):
            # if -1 or n_choices is larger than available frags, return all
            n_choices = len(genes)
        else:
            # if x_choices is int, return that number of frags from fragstore
            n_choices = x_choices

        # random sample of genes with replacement for tournment selection
        idx_lst = np.random.choice(len(genes), n_choices, replace=self.sample_with_replacement)
        genes = [genes[i] for i in idx_lst]

        # score gene fragments
        scored_genes = self.frag_sampler.score(genes, query_frag=ref_frag)

        smiles, scores = zip(*scored_genes)
        return list(smiles), list(scores)
