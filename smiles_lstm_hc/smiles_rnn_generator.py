from __future__ import annotations

from typing import TYPE_CHECKING

from guacamol.distribution_matching_generator import DistributionMatchingGenerator

from .rnn_sampler import SmilesRnnSampler

if TYPE_CHECKING:
    from .rnn_model import SmilesRnn


class SmilesRnnGenerator(DistributionMatchingGenerator):
    """Wraps SmilesRnn in a class satisfying the DistributionMatchingGenerator interface."""

    def __init__(self, model: SmilesRnn, device: str) -> None:
        self.model = model
        self.device = device

    def generate(self, number_samples: int) -> list[str]:
        sampler = SmilesRnnSampler(device=self.device)
        return sampler.sample(model=self.model, num_to_sample=number_samples)
