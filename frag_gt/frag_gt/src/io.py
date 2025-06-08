from __future__ import annotations

import logging

import joblib
from joblib import Parallel, delayed
from rdkit import Chem
from tqdm import tqdm

logger = logging.getLogger(__name__)


def load_smiles_from_file(smi_file: str) -> list[str]:
    with open(smi_file) as f:
        return [s.strip() for _, s in enumerate(f)]


def _smi2mol(smi: str) -> Chem.Mol | None:
    return Chem.MolFromSmiles(smi)


def valid_mols_from_smiles(smiles_list: list[str], n_jobs: int = -1) -> list[Chem.Mol]:
    if n_jobs < 0:
        n_jobs = joblib.cpu_count()
        logger.info(f"found {n_jobs} cpus available")

    if n_jobs == 1:
        valid_mols = []
        for s in tqdm(smiles_list):
            m = _smi2mol(s)
            if m is not None:
                valid_mols.append(m)
    else:
        with Parallel(n_jobs=n_jobs) as pool:
            parsed_mols = pool(delayed(_smi2mol)(s) for s in smiles_list)
            valid_mols = [m for m in parsed_mols if m is not None]
    logger.info(
        f"parsed {len(valid_mols)} valid mols from a possible {len(smiles_list)} smiles using rdkit",
    )
    return valid_mols
