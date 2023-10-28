"""
Analyze how training difficulty might relate to neuron firing statistics.
"""

import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import shutil
from datetime import datetime
from pathlib import Path
import pytorch_lightning as pl
from scipy.ndimage import gaussian_filter1d

from lfads_torch.run_model import run_model

# ---------- OPTIONS -----------
PROJECT_STR = "mrsingle_2310142236"
RUN_DIR = Path("/root/capsule/data") / PROJECT_STR
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

ic_enc_seq_len = model.hparams.ic_enc_seq_len
def batch_smoothing_func(x):
    smoothing_func = lambda x: gaussian_filter1d(x.astype(float), sigma=10)
    return np.apply_along_axis(smoothing_func, axis=1, arr=x)
def batch_corrcoef(x, y):
    N = x.shape[2]
    x_reshape = x.reshape(-1, N) # shape = (B*T, N)
    y_reshape = y.reshape(-1, N)
    return [np.corrcoef(x_reshape[:, i], y_reshape[:, i])[0][1] for i in range(N)]

nlls_area = []
rates_area = []
area_names = []
palette = sns.color_palette("Set2", n_colors=len(model.areas))
for area_name, area in model.areas.items():
    true_data = model.current_batch[s].recon_data[area_name][:, ic_enc_seq_len:]
    pred_rates = model.save_var[area_name].outputs
    avg_rates = true_data.cpu().detach().numpy().mean(axis=(0,1))
    nlls = area.recon[s].compute_loss_main(true_data, pred_rates).cpu().detach().numpy()
    nlls = nlls.mean(axis=(0,1))
    
    rates_area += list(avg_rates)
    nlls_area += list(nlls)
    area_names += [area_name] * len(nlls)

sns.scatterplot(x=rates_area, y=nlls_area, hue=area_names, palette=palette, s=50)
plt.xlabel("average neuron rates")
plt.ylabel("nll")
plt.savefig("/root/capsule/results/training_difficulty-2.png")
