from __future__ import annotations

from frag_gt.src.scorers import MolecularWeightScorer


def test_molecular_weight_scorer() -> None:
    # Given
    smiles = "c1ccccc1"

    # When
    scoring_function = MolecularWeightScorer()
    list_score = scoring_function.score_list([smiles])[0]
    single_score = scoring_function.score(smiles)

    # Then
    assert list_score == single_score
    assert single_score == 78.11399999999999
