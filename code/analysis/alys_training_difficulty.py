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

corrs_area = []
rates_area = []
area_names = []
palette = sns.color_palette("Set2", n_colors=len(model.areas))
for ia, area_name in enumerate(model.areas):
    true_data = model.current_batch[s].recon_data[area_name][:, ic_enc_seq_len:].cpu().detach().numpy()
    pred_rates = model.save_var[area_name].outputs.cpu().detach().numpy()
    avg_rates = true_data.mean(axis=(0,1))
    smoothed_rates = batch_smoothing_func(true_data)
    corrs = batch_corrcoef(smoothed_rates, pred_rates)
    
    rates_area += list(avg_rates)
    corrs_area += corrs
    area_names += [area_name] * len(corrs)

corr = np.corrcoef(rates_area, corrs_area)[0][1]
sns.scatterplot(x=rates_area, y=corrs_area, hue=area_names, palette=palette, s=50)
plt.xlabel("average neuron rates")
plt.ylabel("corr(pred, smooth)")
plt.title(f"Correlation: {round(corr, 2)}")
plt.savefig("/root/capsule/results/training_difficulty.png")
