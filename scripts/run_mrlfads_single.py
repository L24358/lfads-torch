import os
import shutil
from datetime import datetime
from pathlib import Path
from lfads_torch.run_model import run_model

# ---------- OPTIONS -----------
PROJECT_STR = "mrlfads_single_" + datetime.now().strftime("%y%m%d%H%M")
RUN_DIR = Path("/root/capsule/results") / PROJECT_STR
OVERWRITE = True
# ------------------------------

# Overwrite the directory
if RUN_DIR.exists() and OVERWRITE:
    shutil.rmtree(RUN_DIR)
RUN_DIR.mkdir(parents=True)

# Copy this script
shutil.copyfile(__file__, RUN_DIR / Path(__file__).name)

# Train the model
os.chdir(RUN_DIR)
run_model(
    config_path = "../configs/mrlfads_single.yaml",
    do_posterior_sample = False,
)
