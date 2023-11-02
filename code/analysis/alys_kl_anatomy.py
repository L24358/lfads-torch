"""
Analyze what the anatomy looks like as informed by the KL terms.
"""

import os
import torch
import networkx as nx
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
PROJECT_STR = "mrsingle_2310302342"
RUN_DIR = Path("/root/capsule/data") / PROJECT_STR
OVERWRITE = False
s = 0 ## TODO: using the first session only
# ------------------------------

# Switch to the `RUN_DIR` and train the model
os.chdir(RUN_DIR)
model, datamodule, ckpt = run_model(
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

num_areas = len(model.areas)
kl_weight_dict = {}
hps = model.hparams
area_names = list(model.areas.keys())
co_factor = 1e02
com_factor = 1e04

for area_name, area in model.areas.items():
    co_mean, co_std = torch.split(model.save_var[area_name].co_params, area.hparams.co_dim, dim=2)
    com_mean, com_std = torch.split(model.save_var[area_name].com_params, area.hparams.com_dim * (num_areas-1), dim=2)
    
    com_kl = np.array(area.com_prior.kl_divergence_by_component(com_mean, com_std, area.hparams.com_dim)) * hps.kl_com_scale * com_factor
    co_kl = area.co_prior(co_mean, co_std).item() * hps.kl_co_scale * co_factor

    other_areas = area_names.copy()
    other_areas.remove(area_name)
    for io, other_area in enumerate(other_areas):
        kl_weight_dict[(other_area, area_name)] = com_kl[io]
    kl_weight_dict[("Other", area_name)] = co_kl
    
color_choice = ["black", "red"]
G = nx.DiGraph()
for edge, weight in kl_weight_dict.items():
    G.add_edge(edge[0], edge[1], weight=weight)
pos = nx.circular_layout(G)
edges = G.edges()
weights = np.array([kl_weight_dict[edge] for edge in edges])
colors = np.array([color_choice[int("Other" in edge)] for edge in edges])

nx.draw(G, pos, with_labels=True, node_size=800, node_color='skyblue', font_weight='bold', connectionstyle='arc3, rad = 0.1', width=weights, edge_color=colors)
labels = {(i, j): w['weight'] for (i, j, w) in G.edges(data=True)}
plt.savefig("/root/capsule/results/temp.png")

# factor for kl