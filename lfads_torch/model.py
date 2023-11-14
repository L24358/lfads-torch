import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from typing import Dict, List

from .metrics import ExpSmoothedMetric, r2_score, regional_bits_per_spike
from .modules import augmentations
from .modules.decoder import Decoder, SRDecoder
from .modules.encoder import Encoder, SREncoder
from .modules.communicator import Communicator, AreaCommunicator
from .modules.icsampler import ICSampler
from .modules.l2 import compute_l2_penalty, sr_compute_l2_penalty
from .modules.priors import Null
from .tuples import SessionBatch, SessionOutput, SaveVariables
from .utils import transpose_lists, get_insert_func, HParams

class MRLFADS(pl.LightningModule):
    """
    Multi-Regional LFADS.
    """
    def __init__(
        self,
        areas_info: dict,
        num_other_areas: int,
        encod_seq_len: int,
        recon_seq_len: int,
        ic_enc_seq_len: int,
        train_aug_stack: augmentations.AugmentationStack,
        infer_aug_stack: augmentations.AugmentationStack,
        lr_scheduler: bool,
        lr_init: float,
        lr_stop: float,
        lr_decay: float,
        lr_patience: int,
        lr_adam_beta1: float,
        lr_adam_beta2: float,
        lr_adam_epsilon: float,
        weight_decay: float,
        l2_start_epoch: int,
        l2_increase_epoch: int,
        l2_ic_enc_scale: float,
        l2_ci_enc_scale: float,
        l2_gen_scale: float,
        l2_con_scale: float,
        kl_start_epoch_u: int,
        kl_increase_epoch_u: int,
        kl_start_epoch_m: int,
        kl_increase_epoch_m: int,
        kl_ic_scale: float,
        kl_co_scale: float,
        kl_com_scale: float,
        variational: bool,
        loss_scale: float,
        recon_reduce_mean: bool,
        dropout_rate: float,
        cell_clip: float,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore = ["areas_info"]
        )
        self.valid_recon_smth = ExpSmoothedMetric(coef=0.3)

        # Build all the areas (SR-LFADS)
        self.area_names = list(areas_info.keys())
        assert len(self.area_names) > 1 # must have at least 2 areas
        self._build_areas(areas_info)

    def forward(
        self,
        batch: dict,
        sample_posteriors: bool = False,
    ):
        # Calculate total batch_size
        sessions = sorted(batch.keys())
        batch_sizes = [batch[s].encod_data[self.area_names[0]].size(0) for s in sessions]
        batch_size = sum(batch_sizes)
        self._build_save_var(batch_sizes)
        
        # Run encode
        factor_state_dict = {}
        for ia, (area_name, area) in enumerate(self.areas.items()):

            # readin --> encoder --> icsampler (controller, generator and factors)
            encod_data = torch.cat([area.readin[s](batch[s].encod_data[area_name].float()) for s in sessions])
            ic_mean, ic_std, ci = area.encoder(encod_data.float())
            con_init, gen_init, factor_init = area.icsampler(ic_mean, ic_std, sample_posteriors=sample_posteriors)
            factor_init_split = torch.split(factor_init, batch_sizes, dim=0)
            
            # Save the results
            state = torch.cat([torch.tile(con_init, (batch_size, 1)), gen_init, factor_init], dim=1)
            self.save_var[area_name].states[:,0,:] = state # this causes state to have +1 length
            self.save_var[area_name].inputs[..., :area.hparams.ci_enc_dim] = ci
            self.save_var[area_name].ic_params = torch.cat([ic_mean, ic_std], dim=1)
            factor_state_dict[area_name] = factor_init
            
        # Run decode
        outputs = []
        for t in range(self.hparams.recon_seq_len - self.hparams.ic_enc_seq_len):
            
            # Initialize new factor_cat tensor dict to store factors from each area
            factor_state_dict_new = {}
            
            for ia, (area_name, area) in enumerate(self.areas.items()):
                
                # communicator
                com_samp, com_params = area.communicator(factor_state_dict, sample_posteriors=sample_posteriors)
                self.save_var[area_name].inputs[:, t, area.hparams.ci_enc_dim:-area.hparams.co_dim] = com_samp
                self.save_var[area_name].com_params[:,t,:] = com_params
                
                # external input
                ext_input = torch.cat([batch[s].ext_input[area_name] for s in sessions])
                
                # decoder
                states = self.save_var[area_name].states[:,t,:].clone()
                inputs = self.save_var[area_name].inputs[:,t,:]
                inputs = torch.cat([inputs, ext_input[:,t,:]], dim=1)
                new_state, co_params, con_samp = area.decoder(inputs, states, sample_posteriors=sample_posteriors)   
                self.save_var[area_name].states[:,t+1,:] = new_state
                self.save_var[area_name].co_params[:,t,:] = co_params
                self.save_var[area_name].inputs[:,t,-area.hparams.co_dim:] = con_samp
                
                # readout
                factor_state = new_state[..., -area.hparams.fac_dim:]
                factor_state_dict_new[area_name] = factor_state
                factor_state_split = torch.split(factor_state, batch_sizes)
                
                # rates
                for s in sessions:
                    rates = area.readout[s](factor_state_split[s])
                    self.outputs[area_name][s][:,t,:] = rates
                
            # Reset
            factor_state_dict = factor_state_dict_new
                
        return self.outputs
                
    def _shared_step(self, batch, batch_idx, split):
        hps = self.hparams
        self.current_split = split
        num_areas = len(self.areas)
        
        # Process Augmentations
        sessions = sorted(batch.keys())
        aug_stack = hps.train_aug_stack if split == "train" else self.hparams.infer_aug_stack
        batch = {s: b[0] for s, b in batch.items()} # ignore info, only data is relevant
        batch = {s: aug_stack.process_batch(batch[s]) for s in sessions}
        batch_sizes = [batch[s].encod_data[self.area_names[0]].size(0) for s in sessions]
        batch_size = sum(batch_sizes)
        self.current_batch = batch
        
        # Forward pass
        self.forward(
            batch,
            sample_posteriors = hps.variational and split == "train",
        )
        
        # Compute ramping coefficients
        l2_ramp = self._compute_ramp(hps.l2_start_epoch, hps.l2_increase_epoch)
        kl_ramp_u = self._compute_ramp(hps.kl_start_epoch_u, hps.kl_increase_epoch_u)
        kl_ramp_m = self._compute_ramp(hps.kl_start_epoch_m, hps.kl_increase_epoch_m)
        
        # Calculate all losses
        mr_loss, mr_recon, mr_l2, mr_kl_u, mr_kl_m, mr_r2 = 0, 0, 0, 0, 0, 0
        recon_start = hps.ic_enc_seq_len
        for area_name, area in self.areas.items():
            
            # Compute + process reconstruction loss
            rates_split = self.outputs[area_name]
            recon_all = [area.recon.compute_loss_main(
                batch[s].recon_data[area_name][:,recon_start:],
                rates_split[s])
            for s in sessions]
            recon_all = [aug_stack.process_losses(
                recon_all[s],
                (area_name, batch[s].recon_data[area_name][:,recon_start:]),
                self.log,
                split)
            for s in sessions]
            
            if not hps.recon_reduce_mean: recon_all = [torch.sum(ra, dim=(1, 2)) for ra in recon_all] # uses sum, not mean (except batch dim)
            sess_recon = [ra.mean() for ra in recon_all]
            recon = torch.mean(torch.stack(sess_recon))
            mr_recon += recon
            
            # Compute r-squared using built-in functions
            r2_all = [area.recon.compute_pseudo_r2(
                batch[s].recon_data[area_name][:,recon_start:],
                rates_split[s])
            for s in sessions]
            r2 = np.mean(r2_all)
            
            # Compute L2 loss
            l2 = sr_compute_l2_penalty(area, hps)
            mr_l2 += l2
            
            # Compute KL loss
            ic_mean, ic_std = torch.split(self.save_var[area_name].ic_params, area.hparams.ic_dim, dim=1)
            co_mean, co_std = torch.split(self.save_var[area_name].co_params, area.hparams.co_dim, dim=2)
            com_mean, com_std = torch.split(self.save_var[area_name].com_params, area.hparams.com_dim * (num_areas-1), dim=2)
            ic_kl = area.ic_prior(ic_mean, ic_std) * hps.kl_ic_scale
            co_kl = area.co_prior(co_mean, co_std) * hps.kl_co_scale
            com_kl = area.com_prior(com_mean, com_std) * hps.kl_com_scale
            mr_kl_u += (ic_kl + co_kl)
            mr_kl_m += com_kl
            
            # Compute the final loss
            sr_loss = hps.loss_scale * (recon + l2_ramp * l2 + kl_ramp_u * (ic_kl + co_kl) + kl_ramp_m * com_kl)
            mr_loss += sr_loss
            mr_r2 += r2
            
            # Log area-speific information when on validation
            if split == "valid":
                area_metrics = {
                    f"{area_name}/recon": recon,
                    f"{area_name}/l2": l2,
                    f"{area_name}/kl/ic": ic_kl,
                    f"{area_name}/kl/co": co_kl,
                    f"{area_name}/kl/com": com_kl,
                    f"{area_name}/r2": r2,
                }
                self.log_dict(
                    area_metrics,
                    on_step=False,
                    on_epoch=True,
                    batch_size=sum(batch_sizes),
                )
            
        # Log scalar metrics
        metrics = {
            f"{split}/loss": mr_loss / num_areas,
            f"{split}/recon": mr_recon / num_areas,
            f"{split}/l2": mr_l2 / num_areas,
            f"{split}/kl/u": mr_kl_u / num_areas,
            f"{split}/kl/m": mr_kl_m / num_areas,
            f"{split}/r2": mr_r2 / num_areas,
            
            f"{split}/l2/ramp": l2_ramp,
            f"{split}/kl/ramp/u": kl_ramp_u,
            f"{split}/kl/ramp/m": kl_ramp_m,
        }
        if split == "valid":
            # Update the smoothed reconstruction loss
            self.valid_recon_smth.update(recon, batch_size)
            # Add validation-only metrics
            metrics.update(
                {
                    "valid/recon_smth": self.valid_recon_smth,
                    "hp_metric": recon,
                    "cur_epoch": float(self.current_epoch),
                }
            )
        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            batch_size=sum(batch_sizes),
        )
        
        return mr_loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")
        
    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "valid")
        
    def predict_step(self, batch, batch_ix, sample_posteriors=False):
        return self._shared_step(batch, batch_idx, "valid") ## TODO
        # sessions = sorted(batch.keys())
        # batch = {s: self.hparams.infer_aug_stack.process_batch(batch[s]) for s in sessions}
        # self.current_batch = batch
        # # Reset to clear any saved masks
        # self.hparams.infer_aug_stack.reset()
        # # Perform the forward pass
        # return self.forward(
        #     batch=batch,
        #     sample_posteriors=self.hparams.variational and sample_posteriors,
        # )
        
    def on_validation_epoch_end(self):
        # Log hyperparameters that may change during PBT
        self.log_dict(
            {
                "hp/lr_init": self.hparams.lr_init,
                "hp/dropout_rate": self.hparams.dropout_rate,
                "hp/l2_ic_enc_scale": self.hparams.l2_ic_enc_scale,
                "hp/l2_ci_enc_scale": self.hparams.l2_ci_enc_scale,
                "hp/l2_gen_scale": self.hparams.l2_gen_scale,
                "hp/l2_con_scale": self.hparams.l2_con_scale,
                "hp/kl_co_scale": self.hparams.kl_co_scale,
                "hp/kl_ic_scale": self.hparams.kl_ic_scale,
                "hp/weight_decay": self.hparams.weight_decay,
            }
        )
        # Log CD rate if CD is being used
        for aug in self.hparams.train_aug_stack.batch_transforms:
            if hasattr(aug, "cd_rate"):
                self.log("hp/cd_rate", aug.cd_rate)
    
    def configure_optimizers(self):
        hps = self.hparams
        # Create an optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=hps.lr_init,
            betas=(hps.lr_adam_beta1, hps.lr_adam_beta2),
            eps=hps.lr_adam_epsilon,
            weight_decay=hps.weight_decay,
        )
        if hps.lr_scheduler:
            # Create a scheduler to reduce the learning rate over time
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                mode="min",
                factor=hps.lr_decay,
                patience=hps.lr_patience,
                threshold=0.0,
                min_lr=hps.lr_stop,
                verbose=True,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "valid/recon_smth",
            }
        else:
            return optimizer

    def _build_areas(self, areas_info):
        # MR hyperparameters to copy into SR hyperparameters
        hps_to_copy = ["encod_seq_len", "recon_seq_len", "ic_enc_seq_len", "variational",
                      "dropout_rate", "cell_clip", "num_other_areas"]
        mr_hps_dict = {key: self.hparams[key] for key in hps_to_copy}
        
        # Get total factor dimension for communication
        total_fac_dim_dict = {area_name: area_kwargs["fac_dim"] for area_name, area_kwargs in areas_info.items()}
        mr_hps_dict.update({"total_fac_dim_dict": total_fac_dim_dict,
                            "area_names": self.area_names})
        
        # Build all SR-LFADS instances
        self.areas = nn.ModuleDict()
        for area_name, area_kwargs in areas_info.items():
            area_kwargs.update(mr_hps_dict)
            self.areas[area_name] = SRLFADS(area_name, **area_kwargs)
            
    def _build_save_var(self, batch_sizes):
        self.save_var = {}
        self.outputs = {}
        fac_dims = []
        batch_size = sum(batch_sizes)
        target_len = self.hparams.recon_seq_len - self.hparams.ic_enc_seq_len
        num_other_areas = len(self.area_names) - 1 # number of other areas
        for area_name in self.area_names:
            hps = self.areas[area_name].hparams
            self.save_var[area_name] = SaveVariables(
                # states has 1 extra time in the beginning, will be removed in the end
                states = torch.zeros(batch_size, target_len+1, hps.con_dim + hps.gen_dim + hps.fac_dim).to(self.device),
                inputs = torch.zeros(batch_size, target_len, hps.ci_enc_dim + hps.com_dim * num_other_areas + hps.co_dim).to(self.device),
                ic_params = torch.zeros(batch_size, 2 * hps.ic_dim).to(self.device),
                co_params = torch.zeros(batch_size, target_len, 2 * hps.co_dim).to(self.device),
                com_params = torch.zeros(batch_size, target_len, 2 * hps.com_dim * num_other_areas).to(self.device),   
            )
            fac_dims.append(hps.fac_dim)

            self.outputs[area_name] = []
            for i_sess in range(len(hps.num_neurons)):
                self.outputs[area_name].append(
                torch.zeros(batch_sizes[i_sess], target_len, hps.num_neurons[i_sess]).to(self.device)
                )

        self.insert_factor, self.exclude_factor = get_insert_func(fac_dims)
        
    def _compute_ramp(self, start, increase):
        return self.compute_ramp_inner(self.current_epoch, start, increase)
    
    @staticmethod
    def compute_ramp_inner(epoch, start, increase):
        # Compute a coefficient that ramps from 0 to 1 over `increase` epochs
        ramp = (epoch + 1 - start) / (increase + 1)
        return torch.clamp(torch.tensor(ramp), 0, 1)


class SRLFADS(nn.Module):
    def __init__(
        self,
        area_name,
        reconstruction: nn.ModuleList,
        reconstruction_null: nn.ModuleList,
        co_prior: nn.Module,
        ic_prior: nn.Module,
        com_prior: nn.Module,
        readin: nn.ModuleList,
        readout: nn.ModuleList,
        **kwargs,
    ):
        super().__init__()
        
        hparam_keys = ["total_fac_dim", "encod_data_dim", "encod_seq_len", "recon_seq_len", "ext_input_dim", "ic_enc_seq_len",
                       "ic_enc_dim", "ci_enc_dim", "ci_lag", "con_dim", "co_dim", "ic_dim", "gen_dim", "fac_dim",
                       "com_dim", "dropout_rate", "ic_post_var_min", "m_post_var_min", "cell_clip", "num_neurons"]
        hparam_dict = {key: None for key in hparam_keys}
        hparam_dict.update(kwargs)
        hparam_dict["num_neurons"] = list(hparam_dict["num_neurons"].values())
        self.hparams = HParams(hparam_dict)
        self.hparams.add("co_prior", co_prior)
        self.name = area_name
        
        # Make sure the nn.ModuleList arguments are all the same length
        assert len(readin) == len(readout)
        # Make sure that non-variational models use null priors
        if not self.hparams.variational:
            assert isinstance(ic_prior, Null) and isinstance(co_prior, Null)

        # Set up model components
        self.use_con = all([self.hparams.ci_enc_dim > 0, self.hparams.con_dim > 0, self.hparams.co_dim > 0])
        self.readin = readin
        self.encoder = SREncoder(self.hparams)
        self.decoder = SRDecoder(self.hparams)
        self.icsampler = ICSampler(self.hparams, ic_prior)
        self.communicator = AreaCommunicator(self.hparams, com_prior, area_name)
        self.readout = readout
        self.recon = reconstruction
        self.recon_null = reconstruction_null
        self.ic_prior = ic_prior
        self.co_prior = co_prior
        self.com_prior = com_prior
    
    def forward(self): raise NotImplementedError
    
                
# ===== Original LFADS =====#


class LFADS(pl.LightningModule):
    def __init__(
        self,
        encod_data_dim: int,
        encod_seq_len: int,
        recon_seq_len: int,
        ext_input_dim: int,
        ic_enc_seq_len: int,
        ic_enc_dim: int,
        ci_enc_dim: int,
        ci_lag: int,
        con_dim: int,
        co_dim: int,
        ic_dim: int,
        gen_dim: int,
        fac_dim: int,
        dropout_rate: float,
        reconstruction: nn.Module,
        variational: bool,
        co_prior: nn.Module,
        ic_prior: nn.Module,
        ic_post_var_min: float,
        cell_clip: float,
        train_aug_stack: augmentations.AugmentationStack,
        infer_aug_stack: augmentations.AugmentationStack,
        readin: nn.ModuleList,
        readout: nn.ModuleList,
        loss_scale: float,
        recon_reduce_mean: bool,
        lr_scheduler: bool,
        lr_init: float,
        lr_stop: float,
        lr_decay: float,
        lr_patience: int,
        lr_adam_beta1: float,
        lr_adam_beta2: float,
        lr_adam_epsilon: float,
        weight_decay: float,
        l2_start_epoch: int,
        l2_increase_epoch: int,
        l2_ic_enc_scale: float,
        l2_ci_enc_scale: float,
        l2_gen_scale: float,
        l2_con_scale: float,
        kl_start_epoch: int,
        kl_increase_epoch: int,
        kl_ic_scale: float,
        kl_co_scale: float,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=["ic_prior", "co_prior", "reconstruction", "readin", "readout"],
        )
        # Store `co_prior` on `hparams` so it can be accessed in decoder
        self.hparams.co_prior = co_prior
        # Make sure the nn.ModuleList arguments are all the same length
        assert len(readin) == len(readout) == len(reconstruction)
        # Make sure that non-variational models use null priors
        if not variational:
            assert isinstance(ic_prior, Null) and isinstance(co_prior, Null)

        # Store the readin network
        self.readin = readin
        # Decide whether to use the controller
        self.use_con = all([ci_enc_dim > 0, con_dim > 0, co_dim > 0])
        # Create the encoder and decoder
        self.encoder = Encoder(hparams=self.hparams)
        self.decoder = Decoder(hparams=self.hparams)
        # Store the readout network
        self.readout = readout
        # Create object to manage reconstruction
        self.recon = reconstruction
        # Store the trainable priors
        self.ic_prior = ic_prior
        self.co_prior = co_prior
        # Create metric for exponentially-smoothed `valid/recon`
        self.valid_recon_smth = ExpSmoothedMetric(coef=0.3)
        # Store the data augmentation stacks
        self.train_aug_stack = train_aug_stack
        self.infer_aug_stack = infer_aug_stack

    def forward(
        self,
        batch,
        sample_posteriors: bool = False,
        output_means: bool = True,
    ):        
        # Allow SessionBatch input
        if type(batch) == SessionBatch and len(self.readin) == 1:
            batch = {0: batch}
        # Determine which sessions are in the batch
        sessions = sorted(batch.keys())
        # Keep track of batch sizes so we can split back up
        batch_sizes = [len(batch[s].encod_data) for s in sessions]
        # Pass the data through the readin networks
        encod_data = torch.cat([self.readin[s](batch[s].encod_data) for s in sessions])
        # Collect the external inputs
        ext_input = torch.cat([batch[s].ext_input for s in sessions])
        # Pass the data through the encoders
        ic_mean, ic_std, ci = self.encoder(encod_data)
        # Create the posterior distribution over initial conditions
        ic_post = self.ic_prior.make_posterior(ic_mean, ic_std)
        # Choose to take a sample or to pass the mean
        ic_samp = ic_post.rsample() if sample_posteriors else ic_mean
        
        # Unroll the decoder to estimate latent states
        (
            gen_init,
            gen_states,
            con_states,
            co_means,
            co_stds,
            gen_inputs,
            factors,
        ) = self.decoder(ic_samp, ci, ext_input, sample_posteriors=sample_posteriors)
        # Convert the factors representation into output distribution parameters
        factors = torch.split(factors, batch_sizes)
        output_params = [self.readout[s](f) for s, f in zip(sessions, factors)]
        # Separate parameters of the output distribution
        output_params = [
            self.recon[s].reshape_output_params(op)
            for s, op in zip(sessions, output_params)
        ]
        # Convert the output parameters to means if requested
        if output_means:
            output_params = [
                self.recon[s].compute_means(op)
                for s, op in zip(sessions, output_params)
            ]
        # Separate model outputs by session
        output = transpose_lists(
            [
                output_params,
                factors,
                torch.split(ic_mean, batch_sizes),
                torch.split(ic_std, batch_sizes),
                torch.split(co_means, batch_sizes),
                torch.split(co_stds, batch_sizes),
                torch.split(gen_states, batch_sizes),
                torch.split(gen_init, batch_sizes),
                torch.split(gen_inputs, batch_sizes),
                torch.split(con_states, batch_sizes),
            ]
        )
        # Return the parameter estimates and all intermediate activations
        return {s: SessionOutput(*o) for s, o in zip(sessions, output)}

    def configure_optimizers(self):
        hps = self.hparams
        # Create an optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=hps.lr_init,
            betas=(hps.lr_adam_beta1, hps.lr_adam_beta2),
            eps=hps.lr_adam_epsilon,
            weight_decay=hps.weight_decay,
        )
        if hps.lr_scheduler:
            # Create a scheduler to reduce the learning rate over time
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                mode="min",
                factor=hps.lr_decay,
                patience=hps.lr_patience,
                threshold=0.0,
                min_lr=hps.lr_stop,
                verbose=True,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "valid/recon_smth",
            }
        else:
            return optimizer

    def _compute_ramp(self, start, increase):
        # Compute a coefficient that ramps from 0 to 1 over `increase` epochs
        ramp = (self.current_epoch + 1 - start) / (increase + 1)
        return torch.clamp(torch.tensor(ramp), 0, 1)

    def on_before_optimizer_step(self, optimizer, optimizer_idx):
        hps = self.hparams
        # Gradually ramp weight decay alongside the l2 parameters
        l2_ramp = self._compute_ramp(hps.l2_start_epoch, hps.l2_increase_epoch)
        optimizer.param_groups[0]["weight_decay"] = l2_ramp * hps.weight_decay

    def _shared_step(self, batch, batch_idx, split):
        hps = self.hparams
        # Check that the split argument is valid
        assert split in ["train", "valid"]
        # Determine which sessions are in the batch
        sessions = sorted(batch.keys())
        # Discard the extra data - only the SessionBatches are relevant here
        batch = {s: b[0] for s, b in batch.items()}
        # Process the batch for each session (in order so aug stack can keep track)
        aug_stack = self.train_aug_stack if split == "train" else self.infer_aug_stack
        batch = {s: aug_stack.process_batch(batch[s]) for s in sessions}
        # Perform the forward pass
        output = self.forward(
            batch, sample_posteriors=hps.variational, output_means=False
        )
        # Compute the reconstruction loss
        recon_all = [
            self.recon[s].compute_loss(batch[s].recon_data, output[s].output_params)
            for s in sessions
        ]
        # Apply losses processing
        recon_all = [
            aug_stack.process_losses(ra, batch[s], self.log, split)
            for ra, s in zip(recon_all, sessions)
        ]
        # Compute bits per spike
        sess_bps, sess_co_bps, sess_fp_bps = transpose_lists(
            [
                regional_bits_per_spike(
                    output[s].output_params[..., 0],
                    batch[s].recon_data,
                    hps.encod_data_dim,
                    hps.encod_seq_len,
                )
                for s in sessions
            ]
        )
        bps = torch.mean(torch.stack(sess_bps))
        co_bps = torch.mean(torch.stack(sess_co_bps))
        fp_bps = torch.mean(torch.stack(sess_fp_bps))
        # Aggregate the heldout cost for logging
        if not hps.recon_reduce_mean:
            recon_all = [torch.sum(ra, dim=(1, 2)) for ra in recon_all]
        # Compute reconstruction loss for each session
        sess_recon = [ra.mean() for ra in recon_all]
        recon = torch.mean(torch.stack(sess_recon))
        # Compute the L2 penalty on recurrent weights
        l2 = compute_l2_penalty(self, self.hparams)
        # Collect posterior parameters for fast KL calculation
        ic_mean = torch.cat([output[s].ic_mean for s in sessions])
        ic_std = torch.cat([output[s].ic_std for s in sessions])
        co_means = torch.cat([output[s].co_means for s in sessions])
        co_stds = torch.cat([output[s].co_stds for s in sessions])
        # Compute the KL penalty on posteriors
        ic_kl = self.ic_prior(ic_mean, ic_std) * self.hparams.kl_ic_scale
        co_kl = self.co_prior(co_means, co_stds) * self.hparams.kl_co_scale
        # Compute ramping coefficients
        l2_ramp = self._compute_ramp(hps.l2_start_epoch, hps.l2_increase_epoch)
        kl_ramp = self._compute_ramp(hps.kl_start_epoch, hps.kl_increase_epoch)
        # Compute the final loss
        loss = hps.loss_scale * (recon + l2_ramp * l2 + kl_ramp * (ic_kl + co_kl))
        # Compute the reconstruction accuracy, if applicable
        if batch[0].truth.numel() > 0:
            output_means = [
                self.recon[s].compute_means(output[s].output_params) for s in sessions
            ]
            r2 = torch.mean(
                torch.stack(
                    [
                        r2_score(om, batch[s].truth)
                        for om, s in zip(output_means, sessions)
                    ]
                )
            )
        else:
            r2 = float("nan")
        # Compute batch sizes for logging
        batch_sizes = [len(batch[s].encod_data) for s in sessions]
        # Log per-session metrics
        for s, recon_value, batch_size in zip(sessions, sess_recon, batch_sizes):
            self.log(
                name=f"{split}/recon/sess{s}",
                value=recon_value,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )
        # Collect metrics for logging
        metrics = {
            f"{split}/loss": loss,
            f"{split}/recon": recon,
            f"{split}/bps": max(bps, -1.0),
            f"{split}/co_bps": max(co_bps, -1.0),
            f"{split}/fp_bps": max(fp_bps, -1.0),
            f"{split}/r2": r2,
            f"{split}/wt_l2": l2,
            f"{split}/wt_l2/ramp": l2_ramp,
            f"{split}/wt_kl": ic_kl + co_kl,
            f"{split}/wt_kl/ic": ic_kl,
            f"{split}/wt_kl/co": co_kl,
            f"{split}/wt_kl/ramp": kl_ramp,
        }
        if split == "valid":
            # Update the smoothed reconstruction loss
            self.valid_recon_smth.update(recon, batch_size)
            # Add validation-only metrics
            metrics.update(
                {
                    "valid/recon_smth": self.valid_recon_smth,
                    "hp_metric": recon,
                    "cur_epoch": float(self.current_epoch),
                }
            )
        # Log overall metrics
        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            batch_size=sum(batch_sizes),
        )

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "valid")

    def predict_step(self, batch, batch_ix, sample_posteriors=True):
        # Discard the extra data - only the SessionBatches are relevant here
        batch = {s: b[0] for s, b in batch.items()}
        # Process the batch for each session
        batch = {s: self.infer_aug_stack.process_batch(b) for s, b in batch.items()}
        # Reset to clear any saved masks
        self.infer_aug_stack.reset()
        # Perform the forward pass
        return self.forward(
            batch=batch,
            sample_posteriors=self.hparams.variational and sample_posteriors,
            output_means=True,
        )

    def on_validation_epoch_end(self):
        # Log hyperparameters that may change during PBT
        self.log_dict(
            {
                "hp/lr_init": self.hparams.lr_init,
                "hp/dropout_rate": self.hparams.dropout_rate,
                "hp/l2_ic_enc_scale": self.hparams.l2_ic_enc_scale,
                "hp/l2_ci_enc_scale": self.hparams.l2_ci_enc_scale,
                "hp/l2_gen_scale": self.hparams.l2_gen_scale,
                "hp/l2_con_scale": self.hparams.l2_con_scale,
                "hp/kl_co_scale": self.hparams.kl_co_scale,
                "hp/kl_ic_scale": self.hparams.kl_ic_scale,
                "hp/weight_decay": self.hparams.weight_decay,
            }
        )
        # Log CD rate if CD is being used
        for aug in self.train_aug_stack.batch_transforms:
            if hasattr(aug, "cd_rate"):
                self.log("hp/cd_rate", aug.cd_rate)