import hydra
import torch
import numpy as np
from omegaconf import DictConfig
from typing import List
from .tuples import SessionBatch


def flatten(dictionary, level=[]):
    """Flattens a dictionary by placing '.' between levels.
    This function flattens a hierarchical dictionary by placing '.'
    between keys at various levels to create a single key for each
    value. It is used internally for converting the configuration
    dictionary to more convenient formats. Implementation was
    inspired by `this StackOverflow post
    <https://stackoverflow.com/questions/6037503/python-unflatten-dict>`_.
    Parameters
    ----------
    dictionary : dict
        The hierarchical dictionary to be flattened.
    level : str, optional
        The string to append to the beginning of this dictionary,
        enabling recursive calls. By default, an empty string.
    Returns
    -------
    dict
        The flattened dictionary.
    """

    tmp_dict = {}
    for key, val in dictionary.items():
        if type(val) == dict:
            tmp_dict.update(flatten(val, level + [key]))
        else:
            tmp_dict[".".join(level + [key])] = val
    return tmp_dict


def transpose_lists(output: List[list]):
    """Transposes the ordering of a list of lists."""
    return list(map(list, zip(*output)))


def send_batch_to_device(batch, device):
    """Recursively searches the batch for tensors and sends them to the device"""

    def send_to_device(obj):
        obj_type = type(obj)
        if obj_type == torch.Tensor:
            return obj.to(device)
        elif obj_type == dict:
            return {k: send_to_device(v) for k, v in obj.items()}
        elif obj_type == list:
            return [send_to_device(o) for o in obj]
        elif obj_type == SessionBatch:
            return SessionBatch(*[send_to_device(o) for o in obj])
        else:
            raise NotImplementedError(
                f"`send_batch_to_device` has not been implemented for {str(obj_type)}."
            )

    return send_to_device(batch)

def get_paths():
    hydra.initialize(config_path="../configs")
    cfg = hydra.compose(config_name="paths")
    return cfg

def get_insert_func(sizes):
    data_ends = np.cumsum(sizes)
    data_starts = np.insert(data_ends, 0, 0)[:-1]
    
    def insert_tensor(tensor, data, index):
        start, end = data_starts[index], data_ends[index]
        tensor[..., start:end] = data

    def exclude_tensor(tensor, index):
        start, end = data_starts[index], data_ends[index]
        indices_to_include = torch.tensor(list(range(start)) + list(range(end, data_ends[-1]))).to(torch.int64)
        sliced_tensor = torch.index_select(tensor, dim=-1, index=indices_to_include.to(tensor.device))
        return sliced_tensor

    return insert_tensor, exclude_tensor

class HParams:
    def __init__(self, hparams):
        for key, value in hparams.items(): setattr(self, key, value)
        
    def add(self, key, value): setattr(self, key, value)
        
# def get_insert_func(sizes):
#     data_ends = np.cumsum(sizes)
#     data_starts = np.insert(data_ends, 0, 0)[:-1]
    
#     def insert_tensor(tensor, data, index, axis=0):
#         axis_starts = [0, 0, 0]
#         axis_ends = [None, None, None]
#         axis_starts[axis] = data_starts[index]
#         axis_ends[axis] = data_ends[index]
#         tensor[axis_starts[0]:axis_ends[0], axis_starts[1]:axis_ends[1], axis_starts[2]:axis_ends[2]] = data

#     return insert_tensor
