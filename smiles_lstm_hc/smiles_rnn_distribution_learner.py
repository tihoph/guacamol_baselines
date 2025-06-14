from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from .rnn_model import SmilesRnn
from .rnn_trainer import SmilesRnnTrainer
from .rnn_utils import get_tensor_dataset, load_smiles_from_list, set_random_seed
from .smiles_char_dict import SmilesCharDictionary

if TYPE_CHECKING:
    from guacamol.distribution_matching_generator import DistributionMatchingGenerator

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class SmilesRnnDistributionLearner:
    def __init__(
        self,
        output_dir: str,
        n_epochs: int = 10,
        hidden_size: int = 512,
        n_layers: int = 3,
        max_len: int = 100,
        batch_size: int = 64,
        rnn_dropout: float = 0.2,
        lr: float = 1e-3,
        valid_every: int = 100,
    ) -> None:
        self.n_epochs = n_epochs
        self.output_dir = output_dir
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.max_len = max_len
        self.batch_size = batch_size
        self.rnn_dropout = rnn_dropout
        self.lr = lr
        self.valid_every = valid_every
        self.print_every = 10
        self.seed = 42

    def train(
        self,
        training_set: list[str],
        validation_set: list[str],
    ) -> DistributionMatchingGenerator:
        # GPU if available
        cuda_available = torch.cuda.is_available()
        device_str = "cuda" if cuda_available else "cpu"
        device = torch.device(device_str)
        logger.info(f"CUDA enabled:\t{cuda_available}")

        set_random_seed(self.seed, device)

        # load data
        train_seqs, _ = load_smiles_from_list(training_set, self.max_len)
        valid_seqs, _ = load_smiles_from_list(validation_set, self.max_len)

        train_set = get_tensor_dataset(train_seqs)
        test_set = get_tensor_dataset(valid_seqs)

        sd = SmilesCharDictionary()
        n_characters = sd.get_char_num()

        # build network
        smiles_model = SmilesRnn(
            input_size=n_characters,
            hidden_size=self.hidden_size,
            output_size=n_characters,
            n_layers=self.n_layers,
            rnn_dropout=self.rnn_dropout,
        )

        # wire network for training
        optimizer = torch.optim.Adam(smiles_model.parameters(), lr=self.lr)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=sd.pad_idx)

        trainer = SmilesRnnTrainer(
            model=smiles_model,
            criteria=[criterion],
            optimizer=optimizer,
            device=device,
            log_dir=self.output_dir,
        )

        trainer.fit(
            train_set,
            test_set,
            batch_size=self.batch_size,
            print_every=self.print_every,
            valid_every=self.valid_every,
            n_epochs=self.n_epochs,
        )
