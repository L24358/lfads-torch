"""
Test to see if ZIPoisson is indeed getting non-zero zero_prob.
"""


import os
import numpy as np
import shutil
from datetime import datetime
from pathlib import Path
import pytorch_lightning as pl
from scipy.ndimage import gaussian_filter1d

from lfads_torch.run_model import run_model

# ---------- OPTIONS -----------
PROJECT_STR = "mrsingle_2310152003"
RUN_DIR = Path("/root/capsule/results") / PROJECT_STR
OVERWRITE = False
s = 0 ## TODO: using the first session only
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

val_dataloader = datamodule.get_val_dataloader()
for batch_idx, batch in enumerate(val_dataloader):
    model.validation_step(batch, batch_idx)

for area_name, area in model.areas.items():
    print(area.recon[s].zero_prob)
