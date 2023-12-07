import logging
import os
import shutil
import warnings
import hydra
import pytorch_lightning as pl
import torch
from glob import glob
from pathlib import Path
from hydra.utils import call, instantiate
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict
from ray import tune
from .utils import flatten

CONFIG_PATH = "/root/capsule/configs/"

OmegaConf.register_new_resolver("relpath", lambda p: Path(__file__).parent / ".." / p)
OmegaConf.register_new_resolver("eval", eval)

def run_model(
    overrides: dict = {},
    checkpoint_dir: str = None,
    config_path: str = "../configs/single.yaml",
    do_train: bool = True,
    do_posterior_sample: bool = True,
):
    """Adds overrides to the default config, instantiates all PyTorch Lightning
    objects from config, and runs the training pipeline.
    """

    # Compose the train config with properly formatted overrides
    config_path = Path(config_path)
    overrides = [f"{k}={v}" for k, v in flatten(overrides).items()]
    with hydra.initialize(
        config_path=config_path.parent,
        job_name="run_model",
        version_base="1.1",
    ):
        config = hydra.compose(config_name=config_path.name, overrides=overrides)
        
    # Get the config filenames
    metadata = {}
    @hydra.main(config_path=CONFIG_PATH, config_name=config_path.name)
    def get_metadata(config):
        hydra_cfg = HydraConfig.get()
        metadata.update( OmegaConf.to_container(hydra_cfg.runtime.choices) )
    get_metadata()
    
    # Make local "config" directory and copy relevant files
    os.makedirs("./configs")
    for folder in metadata:
        if "hydra" not in folder:
            source_path = os.path.join(CONFIG_PATH, folder, metadata[folder]+".yaml")
            destination_path = os.path.join(os.path.join(".", "configs", folder))
            os.makedirs(destination_path)
            shutil.copy(source_path, destination_path)
    shutil.copy(os.path.join(CONFIG_PATH, config_path.name), "./configs")

    # Avoid flooding the console with output during multi-model runs
    if config.ignore_warnings:
        logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
        warnings.filterwarnings("ignore")

    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed") is not None:
        pl.seed_everything(config.seed, workers=True)

    # Instantiate `LightningDataModule` and `LightningModule`
    datamodule = instantiate(config.datamodule, _convert_="all")
    model = instantiate(config.model)

    # If `checkpoint_dir` is passed, find the most recent checkpoint in the directory
    if checkpoint_dir:
        ckpt_pattern = os.path.join(checkpoint_dir, "*.ckpt")
        ckpt_path = max(glob(ckpt_pattern), key=os.path.getctime)

    if do_train:
        # If both ray.tune and wandb are being used, ensure that loggers use same name
        if "single" not in str(config_path) and "wandb_logger" in config.logger:
            with open_dict(config):
                config.logger.wandb_logger.name = tune.get_trial_name()
                config.logger.wandb_logger.id = tune.get_trial_name()
        # Instantiate the pytorch_lightning `Trainer` and its callbacks and loggers
        trainer = instantiate(
            config.trainer,
            callbacks=[instantiate(c) for c in config.callbacks.values()],
            logger=[instantiate(lg) for lg in config.logger.values()],
            gpus=int(torch.cuda.is_available()),
        )
        # Temporary workaround for PTL step-resuming bug
        if checkpoint_dir:
            ckpt = torch.load(ckpt_path)
            trainer.fit_loop.epoch_loop._batches_that_stepped = ckpt["global_step"]
        # Train the model
        trainer.fit(
            model=model,
            datamodule=datamodule,
            ckpt_path=ckpt_path if checkpoint_dir else None,
        )
        # Restore the best checkpoint if necessary - otherwise, use last checkpoint
        if config.posterior_sampling.use_best_ckpt:
            ckpt_path = trainer.checkpoint_callback.best_model_path
            model.load_state_dict(torch.load(ckpt_path)["state_dict"])
    else:
        if checkpoint_dir:
            # If not training, restore model from the checkpoint
            ckpt = torch.load(ckpt_path)
            model.load_state_dict(ckpt["state_dict"])
            return model, datamodule, ckpt

    # Run the posterior sampling function
    if do_posterior_sample:
        if torch.cuda.is_available():
            model = model.to("cuda")
        call(config.posterior_sampling.fn, model=model, datamodule=datamodule)

    return None