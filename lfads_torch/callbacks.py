import io

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d

from .utils import send_batch_to_device

plt.switch_backend("Agg")


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
    

class OnInitEndCalls(pl.Callback):
    """
    Callbacks that are for on_init_end, but have to be called on_valid_epoch_start to access trainer and pl_module.
    """
    def __init__(self, priority=1):
        self.priority = priority
        self.ran = False
        
    def on_validation_epoch_start(self, trainer, pl_module):
        if self.ran: return
    
        # Common operations
        dataloader = trainer.datamodule.val_dataloader()
        s = 0 # TODO: only using the first session
        batch, info_strings = next(iter(dataloader))[s] # only one single batch
        
        # Run functions here
        get_maximum_activity_units(trainer, pl_module, batch) 
        get_conditions(trainer, pl_module, batch, info_strings)
        proctor_preview_plot(trainer, pl_module)
        
        self.ran = True
        
        
class OnValidationEndCalls(pl.Callback):
    """
    Callbacks that are for on_valid_epoch_end.
    """
    def __init__(self, callbacks, priority=1):
        self.priority = priority
        self.callbacks = callbacks
        
    def on_validation_epoch_end(self, trainer, pl_module):
        # Common operations
        s = 0 # TODO: using the first session only, should change to concatenate all sessions in batch
        batch = pl_module.current_batch[s]
        save_var = pl_module.save_var
        
        for callback in self.callbacks:
            callback.run(trainer, pl_module, batch, save_var)
    
# ===== Classes that are on_validation_epoch_end ===== #
    
class InferredRatesPlot:
    """
    Plots inferred rates with smoothed spiking data.
    """
    def __init__(self, n_samples=3, n_batches=4, log_every_n_epochs=10):
        self.n_samples = n_samples
        self.n_batches = n_batches
        self.log_every_n_epochs = log_every_n_epochs
        self.smoothing_func = lambda x: gaussian_filter1d(x.astype(float), sigma=10)

    def run(self, trainer, pl_module, batch, save_var):
        # Check for conditions to not run
        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return
        if not has_image_loggers(trainer.loggers):
            return

        # Get units
        units = pl_module.maximum_activity_units(self.n_samples)
        
        # Create subplots
        n_rows, n_cols = len(pl_module.area_names) * self.n_samples, self.n_batches
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            sharex=True,
            sharey="row",
            figsize=(3 * n_cols, 2 * n_rows),
        )

        # For the first ``n_batches`` batches:
        for ib, ax_col in enumerate(axes.T):
            count = 0

            # Iterate through areas and take n_sample neurons
            for area_name in pl_module.area_names:
                recon_data = batch.recon_data[area_name].detach().cpu().numpy()
                infer_data = save_var[area_name].outputs.detach().cpu().numpy()
                
                for jn in units[area_name]:
                    ax_col[count].plot(infer_data[ib, :, jn], "b")
                    ax_col[count].plot(self.smoothing_func(recon_data[ib, :, jn]), "k--")
                    count += 1

        plt.tight_layout()
        log_figure(
                trainer.loggers,
                f"{pl_module.current_split}/inferred_rates_plot",
                fig,
                trainer.global_step,
            )

class PSTHPlot:
    """
    Plot PSTH for all areas.
    """
    def __init__(self, n_samples=3, log_every_n_epochs=10):
        self.n_samples = n_samples
        self.log_every_n_epochs = log_every_n_epochs
        self.smoothing_func = lambda x: gaussian_filter1d(x.astype(float), sigma=10)
        
    def run(self, trainer, pl_module, batch, save_var):
        # Check for conditions to not run
        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return
        if not has_image_loggers(trainer.loggers):
            return
        
        # Get data and outputs
        units = pl_module.maximum_activity_units(self.n_samples)
        categories, cond_indices = pl_module.conditions
            
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
        for ic, ax_col in enumerate(axes.T):
            count = 0
            included_batches = cond_indices[ic]

            # Iterate through areas and take n_sample neurons
            for area_name in pl_module.area_names:
                recon_data = batch.recon_data[area_name].detach().cpu().numpy()
                infer_data = save_var[area_name].outputs.detach().cpu().numpy()

                for jn in units[area_name]:
                    ax_col[count].plot(infer_data[included_batches, :, jn].mean(axis=0), "b")
                    ax_col[count].plot(self.smoothing_func(recon_data[included_batches, :, jn].mean(axis=0)), "k")
                    
                    x_mean = self.smoothing_func(recon_data[included_batches, :, jn].mean(axis=0)) # shape = (T,)
                    r_mean = infer_data[included_batches, :, jn].mean(axis=0) # shape = (T,)
                    x_std = self.smoothing_func(recon_data[included_batches, :, jn].std(axis=0)) # shape = (T,)
                    r_std = infer_data[included_batches, :, jn].std(axis=0) # shape = (T,)
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
        log_figure(
                trainer.loggers,
                f"{pl_module.current_split}/psth_plot",
                fig,
                trainer.global_step,
            )

class ProctorSummaryPlot:
    def __init__(self, log_every_n_epochs=10):
        self.log_every_n_epochs = log_every_n_epochs
        
    def run(self, trainer, pl_module, batch, save_var):
        # Check for conditions to not run
        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return
        
        import pdb; pdb.set_trace()
        # Access hyperparameters
        hps = pl_module.hparams
        epochs = np.arange(0, trainer.max_epochs)
    
        # Create subplots
        n_rows, n_cols = 2, 4
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            sharex=True,
            sharey=False,
            figsize=(3 * n_cols, 2 * n_rows),
        )
    
        # Plot lowest possible learning rate
        if hps.lr_scheduler:
            geom = lambda epoch: hps.lr_init * np.power(hps.lr_decay, epoch // hps.lr_patience)
            axes[0].plot(epochs, geom(epochs), "k")
            axes[0].set_title(f"Lowest lr: {round(geom(trainer.max_epochs) ,6)}")
        else:
            axes[0].hline(y=lr_init, xmin=0, xmax=epochs[-1], color='k')
        axes[0].set_title("Learning Rate History")
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("learning rate")
            
        # Plot KL divergence history
        kl_ramp_u = pl_module.compute_ramp_inner(torch.from_numpy(epochs), hps.kl_start_epoch_u, hps.kl_increase_epoch_u)
        kl_ramp_m = pl_module.compute_ramp_inner(torch.from_numpy(epochs), hps.kl_start_epoch_m, hps.kl_increase_epoch_m)
        axes[1].plot(kl_ramp_u, "k", label="u")
        axes[1].plot(kl_ramp_m, "b--", label="m")
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("KL divergence")
        axes[1].set_title("KL Divergence History")
        axes[1].legend()
        
        axes[2].plot([], label="train recon")
        axes[2].plot([], label="val recon")
        axes[2].set_xlabel("epoch")
        axes[2].set_ylabel("loss")
        axes[2].set_title("Reconstruction Loss History")
        
        axes[3].plot([], label="train kl (u)")
        axes[3].plot([], label="val kl (u)")
        axes[3].plot([], label="train kl (m)")
        axes[3].plot([], label="val kl (m)")
        axes[3].plot([], label="train l2")
        axes[3].plot([], label="val l2")
        axes[3].set_xlabel("epoch")
        axes[3].set_ylabel("loss")
        axes[3].set_title("Regularization Loss History")
        
        for area_name in pl_module.area_names:
            axes[4].plot([], label=area_name)
        axes[4].set_xlabel("steps")
        axes[4].set_ylabel("pseudo r-squared")
        axes[4].set_title("Pseudo R-Squared History")
        axes[4].legend()
        
        plt.tight_layout()
        log_figure(
                trainer.loggers,
                f"proctor_summary/epoch{trainer.current_epoch}",
                fig,
                trainer.global_step,
            )
        
# ===== Functions that are on_init_end ===== #
        
def get_maximum_activity_units(trainer, pl_module, batch):
    units = {}
    for area_name in pl_module.area_names:
        arr = batch.recon_data[area_name].detach().cpu().numpy() # shape = (B, T, N)
        arr = arr.reshape(-1, arr.shape[-1]) # shape = (B*T, N)
        indices = np.flip(np.argsort(arr.mean(0))) # according to mean across batch, time
        units[area_name] = indices
    pl_module.maximum_activity_units = lambda n_samples: {k: v[:n_samples] for k, v in units.items()}
    
    
def get_conditions(trainer, pl_module, batch, info_strings):
    categories, inverse_indices = np.unique(info_strings, return_inverse=True)
    unique_indices = [np.where(inverse_indices == i)[0] for i in range(len(categories))]
    pl_module.conditions = (categories, unique_indices)
        
def proctor_preview_plot(trainer, pl_module):
    # Only compute this once
    if hasattr(pl_module, "conditions"): return

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

    # Plot lowest possible learning rate
    if hps.lr_scheduler:
        geom = lambda epoch: hps.lr_init * np.power(hps.lr_decay, epoch // hps.lr_patience)
        axes[0].plot(epochs, geom(epochs), "k")
        axes[0].set_title(f"Lowest lr: {round(geom(trainer.max_epochs) ,6)}")
    else:
        axes[0].hline(y=lr_init, xmin=0, xmax=epochs[-1], color='k')
        axes[0].set_title(f"Lowest lr: {hps.lr_init}")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("learning rate")

    # Plot KL divergence history
    kl_ramp_u = pl_module.compute_ramp_inner(torch.from_numpy(epochs), hps.kl_start_epoch_u, hps.kl_increase_epoch_u)
    kl_ramp_m = pl_module.compute_ramp_inner(torch.from_numpy(epochs), hps.kl_start_epoch_m, hps.kl_increase_epoch_m)
    axes[1].plot(kl_ramp_u, "k", label="u")
    axes[1].plot(kl_ramp_m, "b--", label="m")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("KL divergence")
    axes[1].set_title("KL Divergence History")
    axes[1].legend()

    plt.tight_layout()
    log_figure(
            trainer.loggers,
            f"proctor_preview",
            fig,
            trainer.global_step,
        )
    
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
