"""
Analyze how training difficulty might relate to neuron firing statistics.
"""

import os
import shutil
from datetime import datetime
from pathlib import Path
import pytorch_lightning as pl

from lfads_torch.run_model import run_model

# ---------- OPTIONS -----------
PROJECT_STR = "mrsingle_2310142236"
RUN_DIR = Path("/root/capsule/results") / PROJECT_STR
OVERWRITE = False
# ------------------------------

# Switch to the `RUN_DIR` and train the model
os.chdir(RUN_DIR)
model, datamodule = run_model(
    overrides = {},
    checkpoint_dir = RUN_DIR / "lightning_checkpoints",
    config_path = "../../capsule/configs/mrlfads_single.yaml",
    do_train = False,
    do_posterior_sample= False,
)
trainer = pl.Trainer(accelerator='gpu', gpus=1)  # Use 1 GPU

model.eval()
trainer.validate(model, datamodule=datamodule)

for area_name in model.areas:
    avg_rates = model.current_batch[area_name].mean(dim=(0,1))
    

import pdb; pdb.set_trace()
# model.save_var["SMA5"].inputs
# model.current_batch