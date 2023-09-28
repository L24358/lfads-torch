import hydra
from omegaconf import OmegaConf
from hydra.utils import call, instantiate

OmegaConf.register_new_resolver("eval", eval)

config_path = "."
config_name = "toy"

with hydra.initialize(
        config_path=config_path,
        version_base="1.1",
    ):
        config = hydra.compose(config_name=config_name)
        dic = instantiate(config.model)

import pdb; pdb.set_trace()