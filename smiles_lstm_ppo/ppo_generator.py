from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from guacamol.scoring_function import ScoringFunction

from smiles_lstm_hc.rnn_sampler import SmilesRnnSampler
from smiles_lstm_ppo.ppo_trainer import OptResult, PPOTrainer
from smiles_lstm_ppo.rnn_model import SmilesRnnActorCritic

if TYPE_CHECKING:
    import torch
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class PPOMoleculeGenerator:
    def __init__(
        self,
        model: SmilesRnnActorCritic,
        max_seq_length: int,
        device: str | torch.device,
    ) -> None:
        self.model = model
        self.max_seq_length = max_seq_length
        self.device = device
        self.sampler = SmilesRnnSampler(device=device, batch_size=512)

    def optimise(
        self,
        objective: ScoringFunction,
        start_population: list,
        **kwargs,
    ) -> list[OptResult]:
        if start_population:
            logger.warning("PPO algorithm does not support (yet) a starting population")
        num_epochs = kwargs["num_epochs"]
        episode_size = kwargs["optimize_episode_size"]
        batch_size = kwargs["optimize_batch_size"]
        entropy_weight = kwargs["entropy_weight"]
        kl_div_weight = kwargs["kl_div_weight"]
        clip_param = kwargs["clip_param"]

        trainer = PPOTrainer(
            self.model,
            objective,
            device=self.device,
            max_seq_length=self.max_seq_length,
            batch_size=batch_size,
            num_epochs=num_epochs,
            clip_param=clip_param,
            episode_size=episode_size,
            entropy_weight=entropy_weight,
            kl_div_weight=kl_div_weight,
        )
        trainer.train()

        return sorted(trainer.smiles_history, reverse=True)

    def sample(self, num_mols) -> list[str]:
        return self.sampler.sample(
            self.model.smiles_rnn,
            num_to_sample=num_mols,
            max_seq_len=self.max_seq_length,
        )
