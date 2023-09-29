import torch
import torch.nn as nn
from .initializers import init_linear_
from .decoder import KernelNormalizedLinear

class Communicator(nn.Module):
    def __init__(self, hparams, com_prior):
        super().__init__()
        self.hparams = hps = hparams
        self.com_prior = com_prior
        self.f_to_m = nn.Linear(hps.total_fac_dim - hps.fac_dim, 2 * hps.com_dim)
        
    def forward(
        self,
        factor_state,
        sample_posteriors: bool = True,
    ):
        m_params = self.f_to_m(factor_state)
        m_mean, m_logvar = torch.split(m_params, self.hparams.com_dim, dim=1)
        m_std = torch.sqrt(torch.exp(m_logvar) + self.hparams.m_post_var_min)
        m_post = self.com_prior.make_posterior(m_mean, m_std)
        m_samp = m_post.rsample() if sample_posteriors else m_mean
        return m_samp, torch.cat([m_mean, m_std], dim=1)
        