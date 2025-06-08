from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from ..smiles_lstm_hc.rnn_model import SmilesRnn


class SmilesRnnActorCritic(nn.Module):
    def __init__(self, smiles_rnn: SmilesRnn) -> None:
        """Creates an Actor-Critic model from a Smiles RNN Language model

        Args:
            smiles_rnn: a SmilesRnn object

        """
        super().__init__()

        self.smiles_rnn = smiles_rnn

        self.critic_decoder = nn.Linear(self.smiles_rnn.hidden_size, 1)

        self.init_weights()

    def init_weights(self) -> None:
        # critic_decoder
        nn.init.xavier_uniform_(self.critic_decoder.weight)
        nn.init.constant_(self.critic_decoder.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        embeds = self.smiles_rnn.encoder(x)
        output, hidden = self.smiles_rnn.rnn(embeds, hidden)
        actor_output = self.smiles_rnn.decoder(output)
        critic_output = self.critic_decoder(output)
        return actor_output, critic_output, hidden
