# Adapted from https://github.com/molecularsets/moses/blob/master/scripts/organ/train.py
from __future__ import annotations

from multiprocessing import Pool
from typing import TYPE_CHECKING

import torch
from moses.organ import ORGAN, ORGANTrainer
from moses.organ.config import get_parser as organ_parser
from moses.organ.metrics_reward import MetricsReward
from moses.script_utils import add_train_args, set_seed
from moses.utils import CharVocab
from rdkit import RDLogger

from moses_baselines.common import read_smiles

if TYPE_CHECKING:
    import argparse

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


def get_parser() -> argparse.ArgumentParser:
    parser = add_train_args(organ_parser())

    parser.add_argument(
        "--n_ref_subsample",
        type=int,
        default=500,
        help="Number of reference molecules (sampling from training data)",
    )
    parser.add_argument(
        "--addition_rewards",
        nargs="+",
        type=str,
        choices=MetricsReward.supported_metrics,
        default=[],
        help="Adding of addition rewards",
    )

    return parser


def main(config) -> None:
    set_seed(config.seed)

    train = read_smiles(config.train_load)
    vocab = CharVocab.from_data(train)
    device = torch.device(config.device)

    with Pool(config.n_jobs) as pool:
        reward_func = MetricsReward(
            train,
            config.n_ref_subsample,
            config.rollouts,
            pool,
            config.addition_rewards,
        )
        model = ORGAN(vocab, config, reward_func)
        model = model.to(device)

        trainer = ORGANTrainer(config)
        trainer.fit(model, train)

    torch.save(model.state_dict(), config.model_save)
    torch.save(config, config.config_save)
    torch.save(vocab, config.vocab_save)


if __name__ == "__main__":
    parser = get_parser()
    config = parser.parse_known_args()[0]
    main(config)
