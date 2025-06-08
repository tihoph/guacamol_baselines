from __future__ import annotations

import json
import os
import time
from datetime import timedelta
from typing import TYPE_CHECKING

import numpy as np
import torch
from guacamol.utils.data import remove_duplicates
from torch import nn
from torch.utils.data import TensorDataset

from .rnn_model import SmilesRnn
from .smiles_char_dict import SmilesCharDictionary

if TYPE_CHECKING:
    from numpy.typing import NDArray


def get_tensor_dataset(numpy_array: NDArray[np.int32]) -> TensorDataset:
    """Gets a numpy array of indices, convert it into a Torch tensor,
    divided it into inputs and targets and wrap it
    into a TensorDataset

    Args:
        numpy_array: to be converted

    Returns: a TensorDataset

    """
    tensor = torch.from_numpy(numpy_array).long()

    inp = tensor[:, :-1]
    target = tensor[:, 1:]

    return TensorDataset(inp, target)


def get_tensor_dataset_on_device(
    numpy_array: NDArray[np.int32], device: str | torch.device
) -> TensorDataset:
    """Get tensor dataset and send it to a device
    Args:
        numpy_array: to be converted
        device: cuda | cpu

    Returns:
        a TensorDataset on the required device

    """
    dataset = get_tensor_dataset(numpy_array)
    dataset.tensors = tuple(t.to(device) for t in dataset.tensors)
    return dataset


def load_model(
    model_class,
    model_definition,
    model_weights,
    device: str | torch.device,
    copy_to_cpu: bool = True,
):
    """Args:
        model_class: what class of model?
        model_definition: path to model json
        model_weights: path to model weights
        device: cuda or cpu
        copy_to_cpu: bool

    Returns: an RNN model

    """
    json_in = open(model_definition).read()
    raw_dict = json.loads(json_in)
    model = model_class(**raw_dict)

    def map_location(storage, loc):
        return storage if copy_to_cpu else None

    model.load_state_dict(torch.load(model_weights, map_location))
    return model.to(device)


def load_rnn_model(
    model_definition, model_weights, device: str | torch.device, copy_to_cpu: bool = True
):
    return load_model(SmilesRnn, model_definition, model_weights, device, copy_to_cpu)


def save_model(model: nn.Module, base_dir: str, base_name: str) -> None:
    model_params = os.path.join(base_dir, base_name + ".pt")
    torch.save(model.state_dict(), model_params)

    model_config = os.path.join(base_dir, base_name + ".json")
    with open(model_config, "w") as mc:
        mc.write(json.dumps(model.config))


def load_smiles_from_file(
    smiles_path: str, rm_invalid: bool = True, rm_duplicates: bool = True, max_len: int = 100
) -> tuple[NDArray[np.int32], list[bool]]:
    """Given a list of SMILES strings, provides a zero padded NumPy array
    with their index representation. Sequences longer than `max_len` are
    discarded. The final array will have dimension (all_valid_smiles, max_len+2)
    as a beginning and end of sequence tokens are added to each string.

    Args:
        smiles_path: a text file with one SMILES string per line
        max_len: dimension 1 of returned array, sequences will be padded

    Returns:
        sequences:list a numpy array of SMILES character indices
        valid_mask: list of len(smiles_list) - a boolean mask vector indicating if each index maps to a valid smiles

    """
    smiles_list = open(smiles_path).readlines()
    return load_smiles_from_list(
        smiles_list,
        rm_invalid=rm_invalid,
        rm_duplicates=rm_duplicates,
        max_len=max_len,
    )


def load_smiles_from_list(
    smiles_list: list[str], rm_invalid: bool = True, rm_duplicates: bool = True, max_len: int = 100
) -> tuple[NDArray[np.int32], list[bool]]:
    """Given a list of SMILES strings, provides a zero padded NumPy array
    with their index representation. Sequences longer than `max_len` are
    discarded. The final array will have dimension (all_valid_smiles, max_len+2)
    as a beginning and end of sequence tokens are added to each string.

    Args:
        smiles_list: a list of SMILES strings
        rm_invalid: bool if True remove invalid smiles from final output. Note that if True the length of the output
          does not
          equal the size of the input  `smiles_list`. Default True
        rm_duplicates: bool if True return remove duplicates from final output. Note that if True the length of the
          output does not equal the size of the input  `smiles_list`. Default True
        max_len: dimension 1 of returned array, sequences will be padded

    Returns:
        sequences:list a numpy array of SMILES character indices
        valid_mask: list of len(smiles_list) - a boolean mask vector indicating if each index maps to a valid smiles

    """
    sd = SmilesCharDictionary()

    # filter valid smiles strings
    valid_smiles = []
    valid_mask = [False] * len(smiles_list)
    for i, s in enumerate(smiles_list):
        s = s.strip()
        if sd.allowed(s) and len(s) <= max_len:
            valid_smiles.append(s)
            valid_mask[i] = True
        elif not rm_invalid:
            valid_smiles.append("C")  # default placeholder

    unique_smiles = remove_duplicates(valid_smiles) if rm_duplicates else valid_smiles

    # max len + two chars for start token 'Q' and stop token '\n'
    max_seq_len = max_len + 2

    # allocate the zero matrix to be filled
    sequences = np.zeros((len(unique_smiles), max_seq_len), dtype=np.int32)

    for i, mol in enumerate(unique_smiles):
        enc_smi = sd.BEGIN + sd.encode(mol) + sd.END
        for c in range(len(enc_smi)):
            sequences[i, c] = sd.char_idx[enc_smi[c]]

    return sequences, valid_mask


def rnn_start_token_vector(batch_size: int, device: str | torch.device = "cpu") -> torch.Tensor:
    """Returns a vector of start tokens for SmilesRnn.
    This vector can be used to start sampling a batch of SMILES strings.

    Args:
        batch_size: how many SMILES will be generated at the same time in SmilesRnn
        device: cpu | cuda

    Returns:
        a tensor (batch_size x 1) containing the start token

    """
    sd = SmilesCharDictionary()
    return torch.LongTensor(batch_size, 1).fill_(sd.begin_idx).to(device)


def time_since(start_time: int) -> str:
    seconds = int(time.time() - start_time)
    return str(timedelta(seconds=seconds))


def set_random_seed(seed: int, device: str | torch.device) -> None:
    """Set the random seed for Numpy and PyTorch operations
    Args:
        seed: seed for the random number generators
        device: "cpu" or "cuda"
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)
