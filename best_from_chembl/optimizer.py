from __future__ import annotations

from typing import TYPE_CHECKING

from guacamol.goal_directed_generator import GoalDirectedGenerator
from guacamol.utils.parallelize import parallelize

if TYPE_CHECKING:
    from guacamol.scoring_function import ScoringFunction

    from .chembl_file_reader import ChemblFileReader


class BestFromChemblOptimizer(GoalDirectedGenerator):
    """Goal-directed molecule generator that will simply look for the most adequate molecules present in a file."""

    def __init__(self, smiles_reader: ChemblFileReader) -> None:
        # get a list of all the smiles
        self.smiles = list(smiles_reader)

    def top_k(self, smiles: list[str], scoring_function: ScoringFunction, k: int) -> list[str]:
        scores = parallelize(
            scoring_function.score, [(s,) for s in smiles], desc="Scoring", verbose=1
        )
        scored_smiles = list(zip(scores, smiles))
        scored_smiles = sorted(scored_smiles, key=lambda x: x[0], reverse=True)
        return [smile for score, smile in scored_smiles][:k]

    def generate_optimized_molecules(
        self,
        scoring_function: ScoringFunction,
        number_molecules: int,
        starting_population: list[str] | None = None,
    ) -> list[str]:
        """Will iterate through the reference set of SMILES strings and select the best molecules."""
        return self.top_k(self.smiles, scoring_function, number_molecules)
