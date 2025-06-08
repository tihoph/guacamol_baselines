# Adapted from https://github.com/molecularsets/moses/blob/master/scripts/aae/train.py
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from guacamol.utils.helpers import setup_default_logger
from moses.aae import AAE, AAETrainer
from moses.aae import get_parser as aae_parser
from moses.script_utils import add_train_args, set_seed
from moses.utils import CharVocab

from moses_baselines.common import read_smiles

if TYPE_CHECKING:
    import argparse


def get_parser() -> argparse.ArgumentParser:
    return add_train_args(aae_parser())


def main(config) -> None:
    setup_default_logger()

    set_seed(config.seed)

    train = read_smiles(config.train_load)

    vocab = CharVocab.from_data(train)
    torch.save(config, config.config_save)
    torch.save(vocab, config.vocab_save)

    device = torch.device(config.device)

    model = AAE(vocab, config)
    model = model.to(device)

    trainer = AAETrainer(config)
    trainer.fit(model, train)

    model.to("cpu")
    torch.save(model.state_dict(), config.model_save)


if __name__ == "__main__":
    parser = get_parser()
    config = parser.parse_known_args()[0]
    main(config)
