from __future__ import annotations

from rdkit import Chem

from frag_gt.src.frag_scorers import FragScorer

FRAG_COUNT_TUPLES = [
    ("[2*]Cc1cc(O)cc(O[4*])c1", 2),
    ("[2*]CC(=N[4*])C(C)(C)C", 8),
    ("[2*]CC(N[4*])C(C)C", 1),
]


def test_frag_scorer() -> None:
    # Given
    fragment_scorer = FragScorer(scorer="random", sort=False)

    # When
    scored_frags = fragment_scorer.score(gene_frag_list=FRAG_COUNT_TUPLES)

    # Then
    assert len(scored_frags) == len(FRAG_COUNT_TUPLES)


def test_frag_scorer_counts_and_sort() -> None:
    # Given
    fragment_scorer = FragScorer(scorer="counts", sort=True)
    query_frag = Chem.MolFromSmiles("[2*]CC(N[4*])C(C)C")

    # When
    scored_frags = fragment_scorer.score(gene_frag_list=FRAG_COUNT_TUPLES, query_frag=query_frag)

    # Then
    frag_smiles, scores = zip(*scored_frags)
    _, original_counts = zip(*FRAG_COUNT_TUPLES)
    assert scores == tuple(sorted(original_counts, reverse=True))


def test_frag_scorer_afp() -> None:
    # Given
    fragment_scorer = FragScorer(scorer="afps", sort=True)
    query_frag = Chem.MolFromSmiles("[2*]CC(N[4*])C(C)C")

    # When
    scored_frags = fragment_scorer.score(
        gene_frag_list=FRAG_COUNT_TUPLES[:1],
        query_frag=query_frag,
    )

    # Then
    assert len(scored_frags) == 1


def test_frag_scorer_ecfp4() -> None:
    # Given
    fragment_scorer = FragScorer(scorer="ecfp4", sort=True)
    query_frag = Chem.MolFromSmiles("[2*]CC(N[4*])C(C)C")

    # When
    scored_frags = fragment_scorer.score(gene_frag_list=FRAG_COUNT_TUPLES, query_frag=query_frag)

    # Then
    assert len(scored_frags) == 3
