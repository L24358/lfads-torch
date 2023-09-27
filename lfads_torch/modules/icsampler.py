import torch
import torch.nn as nn

class ICSampler(nn.Module):
    def __init__(self, hparams, ic_prior):
        super().__init__()
        self.hparams = hps = hparams
        self.ic_prior = ic_prior
        self.dropout = nn.Dropout(hps.dropout_rate)
        self.ic_to_g0 = nn.Linear(hps.ic_dim, hps.gen_dim)
        init_linear_(self.ic_to_g0)
        self.fac_linear = KernelNormalizedLinear(hps.gen_dim, hps.fac_dim, bias=False)
        self.con_h0 = nn.Parameter(torch.zeros((1, hps.con_dim), requires_grad=True))
        
    def forward(
        self,
        ic_mean,
        ic_std,
    ):
        # Create the posterior distribution over initial conditions
        ic_post = self.ic_prior.make_posterior(ic_mean, ic_std)
        # Choose to take a sample or to pass the mean
        ic_samp = ic_post.rsample() if sample_posteriors else ic_mean
        # Calculate initial generator state and pass it to the RNN with dropout rate
        gen_init = self.ic_to_g0(ic_samp)
        gen_init_drop = self.dropout(gen_init)
        factor_init = self.fac_linear(gen_init_drop)
        dec_rnn_input = torch.cat([ci, ext_input], dim=2)
        return self.con_h0, gen_init_drop, factor_init
        