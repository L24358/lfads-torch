import io
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from collections import defaultdict
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from .utils import send_batch_to_device, common_label, common_col_title

plt.switch_backend("Agg")
SAVE_DIR = "/root/capsule/results/graphs"
    
    
class OnInitEndCalls(pl.Callback):
    """
    Callbacks that are for on_init_end, but have to be called on_valid_epoch_start to access trainer and pl_module.
    """
    def __init__(self,
                 priority: int = 1):
        self.priority = priority
        self.ran = False
        
    def on_validation_epoch_start(self, trainer, pl_module):
        if self.ran: return
    
        # Common operations
        dataloader = trainer.datamodule.val_dataloader()
        batches = next(iter(dataloader))
        batches = [batch[0] for batch in batches.values()]
        
        # Run functions here
        os.makedirs(SAVE_DIR, exist_ok=True)
        get_maximum_activity_units(trainer, pl_module, batches) 
        import pdb; pdb.set_trace()
        get_conditions(trainer, pl_module, batch, info_strings)
        proctor_preview_plot(trainer, pl_module)
        
        self.ran = True # run only once

class OnEpochEndCalls(pl.Callback):
    """
    Callbacks that are for on_train_epoch_end or on_valid_epoch_end.
    """
    def __init__(self,
                 callbacks: list,
                 in_train: str,
                 priority: int = 1):
        self.priority = priority
        self.callbacks = callbacks
        self.in_train = in_train
        assert len(in_train) == len(callbacks)
        
    def on_train_epoch_end(self, trainer, pl_module):
        # Common operations
        s = 0 # TODO: using the first session only
        batch = pl_module.current_batch[s]
        save_var = pl_module.save_var
        kwargs = {"batch": batch, "save_var": save_var, "log_metrics": self.callbacks[0].metrics} ## Log needs to be the first callback
        
        for i, callback in enumerate(self.callbacks):
            if int(self.in_train[i]):
                new_kwargs = callback.run(trainer, pl_module, **kwargs)
                if not isinstance(new_kwargs, type(None)): kwargs.update(new_kwargs)
            
    def on_validation_epoch_end(self,
                                trainer,
                                pl_module):
        # Common operations
        s = 0 # TODO: using the first session only
        batch = pl_module.current_batch[s]
        save_var = pl_module.save_var
        kwargs = {"batch": batch, "save_var": save_var, "log_metrics": self.callbacks[0].metrics}
        
        for i, callback in enumerate(self.callbacks):
            if not int(self.in_train[i]):
                new_kwargs = callback.run(trainer, pl_module, **kwargs)
                if not isinstance(new_kwargs, type(None)): kwargs.update(new_kwargs)

    
# ===== Classes that are on_validation_epoch_end ===== #

class CalcCorrCoef:
    def __init__(self,
                log_every_n_epochs: int = 1):
        self.log_every_n_epochs = log_every_n_epochs
        self.smoothing_func = lambda x: gaussian_filter1d(x.astype(float), sigma=10)
        
    def run(self, trainer, pl_module, **kwargs):
        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return
        
        # Get variables
        s = 0 ##
        batch, save_var = kwargs["batch"], kwargs["save_var"]
        ic_enc_seq_len = pl_module.hparams.ic_enc_seq_len
        
        for area_name, area in pl_module.areas.items():
            recon_data = batch.recon_data[area_name].detach().cpu().numpy()[:, ic_enc_seq_len:]
            recon_data = recon_data.reshape(-1, area.hparams.encod_data_dim).T # shape = (neurons, batch * time)
            infer_data = torch.exp(save_var[area_name].outputs.detach().cpu()).numpy()
            infer_data = infer_data.reshape(-1, area.hparams.encod_data_dim).T
            
            corrs = []
            for an in area.recon[s].active_neurons:
                smoothed_data = self.smoothing_func(recon_data[an])
                corr = np.corrcoef(infer_data[an], smoothed_data)[0][1]
                corrs.append(corr)
            area.recon[s].log_corrcoef(torch.Tensor(corrs))
            
class FreezeZIPoisson:
    def __init__(self,
                log_every_n_epochs: int = 10):
        self.log_every_n_epochs = log_every_n_epochs
        
    def run(self, trainer, pl_module, **kwargs):
        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return
        
        s = 0 ##
        for area_name, area in pl_module.areas.items():
            area.recon[s].mask_neurons()

class Log:
    def __init__(self,
                 tags: list = []):
        self.metrics = defaultdict(list)
        self.tags = tags
    
    def run(self, trainer, pl_module, **kwargs):
        new_metrics = trainer.logged_metrics
        self.update_dict(self.metrics, new_metrics)
        
        log_dir = trainer.loggers[0].log_dir # Tensorboard logger has to be 1st logger
        event_acc = EventAccumulator(log_dir)
        event_acc.Reload()
        for tag in self.tags:
            if tag in event_acc.Tags()["scalars"]:
                scalar_events = event_acc.Scalars(tag)
                values = [event.value for event in scalar_events]
                self.metrics[tag] = values
            else:
                self.metrics[tag] = []
        return {"log_metrics": self.metrics}
        
    @staticmethod
    def update_dict(old_dict, new_dict):
        for key, value in new_dict.items():
            old_dict[key].append(value.item())
    
class InferredRatesPlot:
    """
    Plots inferred rates with smoothed spiking data.
    """
    def __init__(self, n_samples=3, n_batches=4, log_every_n_epochs=10):
        self.n_samples = n_samples
        self.n_batches = n_batches
        self.log_every_n_epochs = log_every_n_epochs
        self.smoothing_func = lambda x: gaussian_filter1d(x.astype(float), sigma=10)

    def run(self, trainer, pl_module, **kwargs):
        # Check for conditions to not run
        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return

        # Get units
        batch, save_var = kwargs["batch"], kwargs["save_var"]
        units = pl_module.maximum_activity_units(self.n_samples)
        ic_enc_seq_len = pl_module.hparams.ic_enc_seq_len
        
        # Create subplots
        n_rows, n_cols = len(pl_module.area_names) * self.n_samples, self.n_batches
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            sharex=True,
            sharey="row",
            figsize=(3 * n_cols, 2 * n_rows),
        )
        common_label(fig, "time step", "rates")
        common_col_title(fig, [f"Batch {i}" for i in range(n_cols)], (n_rows, n_cols))
        
        # Iterate through areas and take n_sample neurons
        s = 0 ## TODO, New
        count = 0
        for area_name, area in pl_module.areas.items():
            recon_data = batch.recon_data[area_name].detach().cpu().numpy()[:, ic_enc_seq_len:]
            import pdb; pdb.set_trace()
            infer_data = torch.exp(save_var[area_name].outputs.detach().cpu()).numpy()
            
            if area.recon[s].name == "zipoisson":
                non_zero_prob = 1 - area.recon[s].zero_prob.detach().cpu().numpy()
            elif area.recon[s].name == "poisson":
                non_zero_prob = np.ones(infer_data.shape[-1])

            for jn in units[area_name]:
                
                for ib in range(self.n_batches):
                    axes[count][ib].plot(recon_data[ib, :, jn], "gray", alpha=0.5)
                    axes[count][ib].plot(infer_data[ib, :, jn] * non_zero_prob[jn], "b")
                    axes[count][ib].plot(self.smoothing_func(recon_data[ib, :, jn]), "k--")
                    
                axes[count][0].set_ylabel(f"area {area_name}, neuron #{jn}")
                count += 1

        plt.tight_layout()
        plt.savefig(f"{SAVE_DIR}/inferred_rates_plot_epoch{trainer.current_epoch}.png")
        plt.close("all")
        return {}

class PSTHPlot:
    """
    Plot PSTH for all areas.
    """
    def __init__(self, n_samples=3, log_every_n_epochs=10):
        self.n_samples = n_samples
        self.log_every_n_epochs = log_every_n_epochs
        self.smoothing_func = lambda x: gaussian_filter1d(x.astype(float), sigma=10)
        
    def run(self, trainer, pl_module, **kwargs):
        # Check for conditions to not run
        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return
        
        # Get data and outputs
        batch, save_var = kwargs["batch"], kwargs["save_var"]
        units = pl_module.maximum_activity_units(self.n_samples)
        categories, cond_indices = pl_module.conditions
        ic_enc_seq_len = pl_module.hparams.ic_enc_seq_len
            
        # Create subplots
        n_rows, n_cols = len(pl_module.area_names) * self.n_samples, len(categories)
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            sharex=True,
            sharey="row",
            figsize=(3 * n_cols, 2 * n_rows),
        )

        # For each condition (category):
        s = 0 ## TODO, New
        for ic, ax_col in enumerate(axes.T):
            count = 0
            included_batches = cond_indices[ic]

            # Iterate through areas and take n_sample neurons
            for area_name, area in pl_module.areas.items():
                recon_data = batch.recon_data[area_name].detach().cpu().numpy()[:, ic_enc_seq_len:]
                infer_data = torch.exp(save_var[area_name].outputs.detach().cpu()).numpy() # TODO: exp
                
                if area.recon[s].name == "zipoisson":
                    non_zero_prob = 1 - area.recon[s].zero_prob.detach().cpu().numpy()
                elif area.recon[s].name == "poisson":
                    non_zero_prob = np.ones(infer_data.shape[-1])

                for jn in units[area_name]:
                    x_mean = self.smoothing_func(recon_data[included_batches, :, jn].mean(axis=0)) # shape = (T,)
                    r_mean = infer_data[included_batches, :, jn].mean(axis=0) # shape = (T,)
                    x_std = self.smoothing_func(recon_data[included_batches, :, jn].std(axis=0)) # shape = (T,)
                    r_std = infer_data[included_batches, :, jn].std(axis=0) # shape = (T,)
                    
                    r_mean *= non_zero_prob[jn]
                    r_std *= abs(non_zero_prob[jn])
                    ax_col[count].plot(r_mean, "b")
                    ax_col[count].plot(x_mean, "k")
                    ax_col[count].plot(range(len(r_mean)), r_mean, "b")
                    ax_col[count].plot(range(len(x_mean)), x_mean, "k--")
                    ax_col[count].fill_between(range(len(r_mean)), r_mean - r_std, r_mean + r_std,
                                               color="lightblue", alpha=0.5)
                    ax_col[count].fill_between(range(len(x_mean)), x_mean - x_std, x_mean + x_std,
                                               color="gray", alpha=0.5)
                    ax_col[count].set_ylabel(f"{area_name}, neuron #{jn}")
                    ax_col[count].set_title(categories[ic].replace("_", ", "))
                    count += 1

        plt.tight_layout()
        plt.savefig(f"{SAVE_DIR}/psth_plot_epoch{trainer.current_epoch}.png")
        plt.close("all")
        return {}

class ProctorSummaryPlot:
    def __init__(self, log_every_n_epochs=10):
        self.log_every_n_epochs = log_every_n_epochs
        self.count = 0
        self.corrs = {}
        
    def run(self, trainer, pl_module, **kwargs):
        # Check for conditions to not run
        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return
        if self.count < 2:
            self.count += 1
            for area_name in pl_module.areas: self.corrs[area_name] = []
            return
        
        # Access hyperparameters
        hps = pl_module.hparams
        epochs = np.arange(0, trainer.max_epochs)
        log_metrics = kwargs["log_metrics"]
        batch, save_var = kwargs["batch"], kwargs["save_var"]
        seq_len = hps.recon_seq_len - hps.ic_enc_seq_len
    
        # Create subplots
        n_rows, n_cols = 4, 2
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            sharex=False,
            sharey=False,
            figsize=(3 * n_cols, 2 * n_rows),
        )
        common_label(fig, "epochs", "")
        
        # Plot lowest possible learning rate
        axes[0][0].plot(log_metrics["lr-AdamW"][1:], "k")
        axes[0][0].set_title("Learning Rate History")
        axes[0][0].set_ylabel("learning rate")
        axes[0][0].set_xlabel("steps")
        
        # Plot KL divergence ramp history
        axes[0][1].plot(log_metrics["valid/kl/ramp/u"][1:], "k", label="u")
        axes[0][1].plot(log_metrics["valid/kl/ramp/m"][1:], "b--", label="m")
        axes[0][1].set_ylabel("KL divergence")
        axes[0][1].set_title("KL Coefficient History")
        axes[0][1].legend()
        
        for ia, area_name in enumerate(pl_module.areas):
            
            # Compute correlation
            true_data = batch.recon_data[area_name][:, hps.ic_enc_seq_len:].cpu().detach().numpy()
            pred_rates = save_var[area_name].outputs.cpu().detach().numpy()
            avg_rates = true_data.mean(axis=(0,1))
            smoothed_rates = batch_smoothing_func(true_data)
            corrs = batch_corrcoef(smoothed_rates, pred_rates)
            self.corrs[area_name].append(np.mean(corrs))
            
            axes[1][0].plot(log_metrics[f"{area_name}/recon"][1:], label=area_name)
            axes[1][1].plot(self.corrs[area_name][1:], label=area_name)
            axes[2][0].plot(log_metrics[f"{area_name}/kl/co"][1:], label=area_name)
            axes[2][1].plot(log_metrics[f"{area_name}/kl/com"][1:], label=area_name)
            axes[3][0].plot(log_metrics[f"{area_name}/l2"][1:], label=area_name)
            axes[3][1].plot(log_metrics[f"{area_name}/r2"][1:], label=area_name)
        axes[1][0].set_title("Reconstruction Loss")
        axes[1][1].set_title("Correlation")
        axes[2][0].set_title("KL Divergence Loss (u)")
        axes[2][1].set_title("KL Divergence Loss (m)")
        axes[3][0].set_title("L2 Regularization Loss")
        axes[3][1].set_title("Pseudo R-Square")
        axes[1][0].legend()
        axes[1][1].legend()
        axes[2][0].legend()
        axes[2][1].legend()
        axes[3][0].legend()
        axes[3][1].legend()
        axes[1][0].set_ylabel("loss") 
        axes[2][0].set_ylabel("loss") 
        axes[3][0].set_ylabel("loss") 
        
        plt.tight_layout()
        plt.savefig(f"{SAVE_DIR}/proctor_summary_plot_epoch{trainer.current_epoch}.png")
        plt.close("all")
        return {}
    
def batch_smoothing_func(x):
    smoothing_func = lambda x: gaussian_filter1d(x.astype(float), sigma=10)
    return np.apply_along_axis(smoothing_func, axis=1, arr=x)

def batch_corrcoef(x, y):
    N = x.shape[2]
    x_reshape = x.reshape(-1, N) # shape = (B*T, N)
    y_reshape = y.reshape(-1, N)
    return [np.corrcoef(x_reshape[:, i], y_reshape[:, i])[0][1] for i in range(N)]

class CommunicationPSTHPlot:
    """
    Plot Inferred Input and Communication PSTH plots for all areas.
    """
    def __init__(self, log_every_n_epochs=10):
        self.log_every_n_epochs = log_every_n_epochs
        self.count = 0
        
    def run(self, trainer, pl_module, **kwargs):
        # Check for conditions to not run
        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return
        if self.count < 2:
            self.count += 1
            return
        
        # Get data and outputs
        batch, save_var = kwargs["batch"], kwargs["save_var"]
        categories, cond_indices = pl_module.conditions
        log_metrics = kwargs["log_metrics"]
        cmap = sns.color_palette("viridis", as_cmap=True)
            
        # Create subplots
        n_rows, n_cols = len(pl_module.area_names) * 4, len(categories)
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            sharex=True,
            sharey=False,
            figsize=(3 * n_cols, 2 * n_rows),
        )
        common_col_title(fig, categories, (n_rows, n_cols))

        # For each condition (category):
        for ic, ax_col in enumerate(axes.T):
            count = 0
            included_batches = cond_indices[ic]

            # Iterate through areas and take n_sample neurons
            for ia, (area_name, area) in enumerate(pl_module.areas.items()):
                hps = area.hparams
                num_other_areas_name = list(pl_module.areas.keys())
                num_other_areas_name.pop(ia)
                inputs = save_var[area_name].inputs.detach().cpu()
                ci_enc_dim, com_dim, co_dim = hps.ci_enc_dim, hps.com_dim, hps.co_dim
                _, com, co = torch.split(inputs, [ci_enc_dim, com_dim * hps.num_other_areas, co_dim], dim=2)
                
                # Get colors
                # colors = [cmap(x) for x in np.linspace(0, 1, com_dim * hps.num_other_areas)]
                colors = plt.cm.rainbow(np.linspace(0, 1, hps.num_other_areas))

                # Plot co
                for ico in range(co_dim):
                    ax_col[count].plot(co[included_batches, :, ico].mean(axis=0))
                ax_col[count].set_ylabel(f"{area_name}, u")
                count += 1
                
                # Plot kl (co)
                co_mean, co_std = torch.split(save_var[area_name].co_params, [hps.co_dim, hps.co_dim], dim=2)
                co_kl = area.co_prior.kl_divergence_by_component(co_mean[included_batches], co_std[included_batches], 1, tpe="seq")
                for jco in range(co_dim):
                    ax_col[count].plot(co_kl[jco].cpu().detach().numpy())
                ax_col[count].set_ylabel(f"{area_name}, kl (u)")
                count += 1
                
                # Plot com
                count_com = 0
                for icom in range(hps.num_other_areas):
                    for ii in range(com_dim):
                        perturbation = np.random.uniform(-0.25, 0.25, size=3)
                        perturbed_color = np.clip(colors[icom][:3] + perturbation, 0.0, 1.0)
                        sub_color = (*perturbed_color, (ii + 1) / (com_dim + 1)) # colors[icom] is the group color
                        if (ii == 0) and (ic == 0):
                            ax_col[count].plot(com[included_batches, :, count_com].mean(axis=0), color=sub_color, label=f"{num_other_areas_name[icom]}")
                        else:
                            ax_col[count].plot(com[included_batches, :, count_com].mean(axis=0), color=sub_color)
                        count_com += 1
                ax_col[count].set_ylabel(f"{area_name}, m")
                count += 1
                
                # Plot kl (com)
                com_mean, com_std = torch.split(save_var[area_name].com_params, [hps.com_dim * hps.num_other_areas, hps.com_dim * hps.num_other_areas], dim=2)
                com_kl = area.com_prior.kl_divergence_by_component(com_mean[included_batches], com_std[included_batches], 1, tpe="seq")
                count_kl = 0
                for jcom in range(hps.num_other_areas):
                    for jj in range(com_dim):
                        perturbation = np.random.uniform(-0.25, 0.25, size=3)
                        perturbed_color = np.clip(colors[jcom][:3] + perturbation, 0.0, 1.0)
                        sub_color = (*perturbed_color, (jj + 1) / (com_dim + 1)) # colors[jcom] is the group color
                        if (jj == 0) and (ic == 0):
                            ax_col[count].plot(com_kl[count_kl].cpu().detach().numpy(), color=sub_color, label=f"{num_other_areas_name[icom]}")
                        else:
                            ax_col[count].plot(com_kl[count_kl].cpu().detach().numpy(), color=sub_color)
                        count_kl += 1
                # for jcom in range(com_dim * hps.num_other_areas):
                #     ax_col[count].plot(com_kl[jcom].cpu().detach().numpy(), color=colors[jcom])
                ax_col[count].set_ylabel(f"{area_name}, kl (m)")
                count += 1

        plt.tight_layout()
        plt.savefig(f"{SAVE_DIR}/communication_plot_epoch{trainer.current_epoch}.png")
        plt.close("all")
        return {}
        
class ICPCAPlot:
    def __init__(self, log_every_n_epochs=10):
        self.log_every_n_epochs = log_every_n_epochs
        
    def run(self, trainer, pl_module, **kwargs):
        # Check for conditions to not run
        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return
        
        # Get data
        batch, save_var = kwargs["batch"], kwargs["save_var"]
        categories, cond_indices = pl_module.conditions
        
        colors = np.zeros(sum(len(sublist) for sublist in cond_indices))
        for ic, cond_group in enumerate(cond_indices): colors[np.array(cond_group)] = ic
        
        # Create subplots
        n_rows, n_cols = len(pl_module.area_names), 1
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            sharex=True,
            sharey=False,
            figsize=(4 * n_cols, 2 * n_rows),
        )
        common_label(fig, "PCA1", "PCA2")
        
        for ia, (area_name, area) in enumerate(pl_module.areas.items()):
            hps = area.hparams
            ics = save_var[area_name].states[:, 0, -hps.fac_dim:] # shape = (batch, fac_dim)
            
            pca = PCA(n_components=2)
            ics_pca = pca.fit_transform(ics.cpu().detach().numpy())
            sc = axes[ia].scatter(*ics_pca.T, c=colors)
            
            cbar = fig.colorbar(sc, ax=axes[ia])
            cbar.set_ticks(np.arange(len(categories)))
            cbar.set_ticklabels(categories)
            
        plt.tight_layout()
        plt.savefig(f"{SAVE_DIR}/icpca_plot_epoch{trainer.current_epoch}.png")
        plt.close("all")
        return {}
        
# ===== Functions that are on_init_end ===== #
        
def get_maximum_activity_units(trainer, pl_module, batches):
    session_units = []
    for s in range(len(batches)):
        units = {}
        batch = batches[s]
        for area_name in pl_module.area_names:
            arr = batch.recon_data[area_name].detach().cpu().numpy() # shape = (B, T, N)
            arr = arr.reshape(-1, arr.shape[-1]) # shape = (B*T, N)
            indices = np.flip(np.argsort(arr.mean(0))) # according to mean across batch, time
            units[area_name] = indices
        session_units.append(units)
    pl_module.maximum_activity_units = lambda s, n_samples: {k: v[:n_samples] for k, v in session_units[s].items()}
    
def get_conditions(trainer, pl_module, batch, info_strings):
    categories, inverse_indices = np.unique(info_strings, return_inverse=True)
    unique_indices = [np.where(inverse_indices == i)[0] for i in range(len(categories))]
    pl_module.conditions = (categories, unique_indices)
        
def proctor_preview_plot(trainer, pl_module):
    # Access hyperparameters
    hps = pl_module.hparams
    epochs = np.arange(0, trainer.max_epochs)

    # Create subplots
    n_rows, n_cols = 2, 1
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        sharex=True,
        sharey="row",
        figsize=(3 * n_cols, 2 * n_rows),
    )
    common_label(fig, "epochs", "")

    # Plot lowest possible learning rate
    if hps.lr_scheduler:
        geom = lambda epoch: hps.lr_init * np.power(hps.lr_decay, epoch // hps.lr_patience)
        axes[0].plot(epochs, geom(epochs), "k")
        axes[0].set_title(f"Lowest lr: {round(geom(trainer.max_epochs) ,6)}")
    else:
        axes[0].hlines(y=lr_init, xmin=0, xmax=epochs[-1], color='k')
        axes[0].set_title(f"Lowest lr: {hps.lr_init}")
    axes[0].set_ylabel("learning rate")

    # Plot KL divergence history
    kl_ramp_u = pl_module.compute_ramp_inner(torch.from_numpy(epochs), hps.kl_start_epoch_u, hps.kl_increase_epoch_u) * hps.kl_co_scale
    kl_ramp_m = pl_module.compute_ramp_inner(torch.from_numpy(epochs), hps.kl_start_epoch_m, hps.kl_increase_epoch_m) * hps.kl_com_scale
    axes[1].plot(kl_ramp_u, "k", label="u")
    axes[1].plot(kl_ramp_m, "b--", label="m")
    axes[1].set_ylabel("KL divergence")
    axes[1].set_title("KL Divergence History")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/proctor_preview.png")
    
# ===== Original plotting functions by Andrew ===== #

class RasterPlot(pl.Callback):
    """Plots validation spiking data side-by-side with
    inferred inputs and rates and logs to tensorboard.
    """

    def __init__(self, split="valid", n_samples=3, log_every_n_epochs=100):
        """Initializes the callback.
        Parameters
        ----------
        n_samples : int, optional
            The number of samples to plot, by default 3
        log_every_n_epochs : int, optional
            The frequency with which to plot and log, by default 100
        """
        assert split in ["train", "valid"]
        self.split = split
        self.n_samples = n_samples
        self.log_every_n_epochs = log_every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        """Logs plots at the end of the validation epoch.
        Parameters
        ----------
        trainer : pytorch_lightning.Trainer
            The trainer currently handling the model.
        pl_module : pytorch_lightning.LightningModule
            The model currently being trained.
        """
        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return
        # Check for any image loggers
        if not has_image_loggers(trainer.loggers):
            return
        # Get data samples from the dataloaders
        if self.split == "valid":
            dataloader = trainer.datamodule.val_dataloader()
        else:
            dataloader = trainer.datamodule.train_dataloader(shuffle=False)
        batch = next(iter(dataloader))
        # Determine which sessions are in the batch
        sessions = sorted(batch.keys())
        # Move data to the right device
        batch = send_batch_to_device(batch, pl_module.device)
        # Compute model output
        output = pl_module.predict_step(
            batch=batch,
            batch_ix=None,
            sample_posteriors=True,
        )
        # Discard the extra data - only the SessionBatches are relevant here
        batch = {s: b[0] for s, b in batch.items()}
        # Log a few example outputs for each session
        for s in sessions:
            # Convert everything to numpy
            encod_data = batch[s].encod_data.detach().cpu().numpy()
            recon_data = batch[s].recon_data.detach().cpu().numpy()
            truth = batch[s].truth.detach().cpu().numpy()
            means = output[s].output_params.detach().cpu().numpy()
            inputs = output[s].gen_inputs.detach().cpu().numpy()
            # Compute data sizes
            _, steps_encod, neur_encod = encod_data.shape
            _, steps_recon, neur_recon = recon_data.shape
            # Decide on how to plot panels
            if np.all(np.isnan(truth)):
                plot_arrays = [recon_data, means, inputs]
                height_ratios = [3, 3, 1]
            else:
                plot_arrays = [recon_data, truth, means, inputs]
                height_ratios = [3, 3, 3, 1]
            # Create subplots
            fig, axes = plt.subplots(
                len(plot_arrays),
                self.n_samples,
                sharex=True,
                sharey="row",
                figsize=(3 * self.n_samples, 10),
                gridspec_kw={"height_ratios": height_ratios},
            )
            for i, ax_col in enumerate(axes.T):
                for j, (ax, array) in enumerate(zip(ax_col, plot_arrays)):
                    if j < len(plot_arrays) - 1:
                        ax.imshow(array[i].T, interpolation="none", aspect="auto")
                        ax.vlines(steps_encod, 0, neur_recon, color="orange")
                        ax.hlines(neur_encod, 0, steps_recon, color="orange")
                        ax.set_xlim(0, steps_recon)
                        ax.set_ylim(0, neur_recon)
                    else:
                        ax.plot(array[i])
            plt.tight_layout()
            # Log the figure
            log_figure(
                trainer.loggers,
                f"{self.split}/raster_plot/sess{s}",
                fig,
                trainer.global_step,
            )

class TrajectoryPlot(pl.Callback):
    """Plots the top-3 PC's of the latent trajectory for
    all samples in the validation set and logs to tensorboard.
    """

    def __init__(self, log_every_n_epochs=100):
        """Initializes the callback.

        Parameters
        ----------
        log_every_n_epochs : int, optional
            The frequency with which to plot and log, by default 100
        """
        self.log_every_n_epochs = log_every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        """Logs plots at the end of the validation epoch.

        Parameters
        ----------
        trainer : pytorch_lightning.Trainer
            The trainer currently handling the model.
        pl_module : pytorch_lightning.LightningModule
            The model currently being trained.
        """
        # Skip evaluation for most epochs to save time
        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return
        # Check for any image loggers
        if not has_image_loggers(trainer.loggers):
            return
        # Get only the validation dataloaders
        pred_dls = trainer.datamodule.predict_dataloader()
        dataloaders = {s: dls["valid"] for s, dls in pred_dls.items()}
        # Compute outputs and plot for one session at a time
        for s, dataloader in dataloaders.items():
            latents = []
            for batch in dataloader:
                # Move data to the right device
                batch = send_batch_to_device({s: batch}, pl_module.device)
                # Perform the forward pass through the model
                output = pl_module.predict_step(batch, None, sample_posteriors=False)[s]
                latents.append(output.factors)
            latents = torch.cat(latents).detach().cpu().numpy()
            # Reduce dimensionality if necessary
            n_samp, n_step, n_lats = latents.shape
            if n_lats > 3:
                latents_flat = latents.reshape(-1, n_lats)
                pca = PCA(n_components=3)
                latents = pca.fit_transform(latents_flat)
                latents = latents.reshape(n_samp, n_step, 3)
                explained_variance = np.sum(pca.explained_variance_ratio_)
            else:
                explained_variance = 1.0
            # Create figure and plot trajectories
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection="3d")
            for traj in latents:
                ax.plot(*traj.T, alpha=0.2, linewidth=0.5)
            ax.scatter(*latents[:, 0, :].T, alpha=0.1, s=10, c="g")
            ax.scatter(*latents[:, -1, :].T, alpha=0.1, s=10, c="r")
            ax.set_title(f"explained variance: {explained_variance:.2f}")
            plt.tight_layout()
            # Log the figure
            log_figure(
                trainer.loggers,
                f"trajectory_plot/sess{s}",
                fig,
                trainer.global_step,
            )


class TestEval(pl.Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        test_batch = send_batch_to_device(
            trainer.datamodule.test_data[0][0], pl_module.device
        )
        _, esl, edd = test_batch.encod_data.shape
        test_output = pl_module(test_batch, output_means=False)[0]
        test_recon = pl_module.recon[0].compute_loss(
            test_batch.encod_data,
            test_output.output_params[:, :esl, :edd],
        )
        pl_module.log("test/recon", test_recon)

def has_image_loggers(loggers):
    """Checks whether any image loggers are available.

    Parameters
    ----------
    loggers : obj or list[obj]
        An object or list of loggers to search.
    """
    logger_list = loggers if isinstance(loggers, list) else [loggers]
    for logger in logger_list:
        if isinstance(logger, pl.loggers.TensorBoardLogger):
            return True
        elif isinstance(logger, pl.loggers.WandbLogger):
            return True
    return False


def log_figure(loggers, name, fig, step):
    """Logs a figure image to all available image loggers.

    Parameters
    ----------
    loggers : obj or list[obj]
        An object or list of loggers
    name : str
        The name to use for the logged figure
    fig : matplotlib.figure.Figure
        The figure to log
    step : int
        The step to associate with the logged figure
    """
    # Save figure image to in-memory buffer
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format="png")
    image = Image.open(img_buf)
    # Distribute image to all image loggers
    logger_list = loggers if isinstance(loggers, list) else [loggers]
    for logger in logger_list:
        if isinstance(logger, pl.loggers.TensorBoardLogger):
            logger.experiment.add_figure(name, fig, step)
        elif isinstance(logger, pl.loggers.WandbLogger):
            logger.log_image(name, [image], step)
    img_buf.close()