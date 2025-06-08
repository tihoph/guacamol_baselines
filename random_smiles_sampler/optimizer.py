from __future__ import annotations

from typing import TYPE_CHECKING

from guacamol.goal_directed_generator import GoalDirectedGenerator

if TYPE_CHECKING:
    from guacamol.scoring_function import ScoringFunction

    from .generator import RandomSmilesSampler


class RandomSamplingOptimizer(GoalDirectedGenerator):
    """Mock optimizer that will return molecules drawn from a random sampler"""

    def __init__(self, sampler: RandomSmilesSampler) -> None:
        self.sampler = sampler

    def generate_optimized_molecules(
        self,
        scoring_function: ScoringFunction,
        number_molecules: int,
        starting_population: list[str] | None = None,
    ) -> list[str]:
        return self.sampler.generate(number_samples=number_molecules)
