import torch
import torch.nn.functional as F
from torch import nn

from .initializers import init_linear_
from .recurrent import ClippedGRUCell

class Communicator(nn.Module): # TODO
    def __init__(self, hparams: dict):
        super().__init__()