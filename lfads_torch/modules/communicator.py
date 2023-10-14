import torch
import torch.nn as nn
from copy import deepcopy
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
        
class AreaCommunicator(nn.Module):
    def __init__(self, hparams, com_prior, self_name):
        super().__init__()
        self.name = self_name
        self.hparams = hparams
        self.other_area_names = deepcopy(hparams.area_names)
        self.other_area_names.remove(self.name)
        self.com_dim_total = hparams.com_dim * len(self.other_area_names)
        self._build_areas(com_prior)
        
    def forward(
        self,
        factor_state,
        sample_posteriors: bool = True,
    ):
        hps = self.hparams
        batch_size = factor_state[self.name].size(0)
        com_samp = torch.zeros(batch_size, self.com_dim_total)
        com_params = torch.zeros(batch_size, self.com_dim_total * 2)
        
        base = hps.com_dim * len(self.other_area_names)
        for ia, area_name in enumerate(self.other_area_names):
            m_params = self.areas_linear[area_name](factor_state[area_name])
            m_mean, m_logvar = torch.split(m_params, hps.com_dim, dim=1)
            m_std = torch.sqrt(torch.exp(m_logvar) + hps.m_post_var_min)
            m_post = self.areas_prior[area_name].make_posterior(m_mean, m_std)
            m_samp = m_post.rsample() if sample_posteriors else m_mean
            
            com_samp[:, hps.com_dim * ia: hps.com_dim * (ia+1)] = m_samp
            com_params[:, hps.com_dim * ia: hps.com_dim * (ia+1)] = m_mean
            com_params[:, base + hps.com_dim * ia: base + hps.com_dim * (ia+1)] = m_std
        return com_samp, com_params
    
    def _build_areas(self,
                     com_prior):
        hps = self.hparams
        self.areas_linear = nn.ModuleDict()
        self.areas_prior = nn.ModuleDict()
        for ia, area_name in enumerate(self.other_area_names):
            self.areas_linear[area_name] = nn.Linear(hps.total_fac_dim_dict[area_name], 2 * hps.com_dim)
            self.areas_prior[area_name] = deepcopy(com_prior)