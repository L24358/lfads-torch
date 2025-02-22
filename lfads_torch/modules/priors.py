import torch
from torch import nn
from torch.distributions import Independent, Normal, StudentT, kl_divergence
from torch.distributions import MultivariateNormal as FullRankMultivariateNormal
from torch.distributions.transforms import AffineTransform


class Null(nn.Module):
    def make_posterior(self, *args):
        return None

    def forward(self, *args):
        return 0


class MultivariateNormal(nn.Module):
    def __init__(
        self,
        mean: float,
        variance: float,
        shape: int,
    ):
        super().__init__()
        # Create distribution parameter tensors
        means = torch.ones(shape) * mean
        logvars = torch.log(torch.ones(shape) * variance)
        self.mean = nn.Parameter(means, requires_grad=False)
        self.logvar = nn.Parameter(logvars, requires_grad=True)

    def make_posterior(self, post_mean, post_std):
        post_mean = torch.nan_to_num(post_mean, posinf=1e6, neginf=-1e6)
        post_std = torch.nan_to_num(post_std, posinf=1e6, neginf=-1e6, nan=1e-10)
        return Independent(Normal(post_mean, post_std), 1)

    def forward(self, post_mean, post_std):
        # kl_batch = self.kl_divergence_by_component(post_mean, post_std)
        # Create the posterior distribution
        posterior = self.make_posterior(post_mean, post_std)
        # Create the prior and posterior
        prior_std = torch.exp(0.5 * self.logvar)
        prior_mean = torch.nan_to_num(self.mean, posinf=1e6, neginf=-1e6)
        prior_std = torch.nan_to_num(prior_std, posinf=1e6, neginf=-1e6, nan=1e-10)
        prior = Independent(Normal(prior_mean, prior_std), 1)
        # Compute KL analytically
        kl_batch = kl_divergence(posterior, prior)
        return torch.mean(kl_batch)
    
    def kl_divergence_by_component(self, post_mean, post_std, com_dim, tpe="mean"):
        post_mean = torch.split(post_mean, com_dim, dim=2)
        post_std = torch.split(post_std, com_dim, dim=2)
        prior_std = torch.split(torch.exp(0.5 * self.logvar), com_dim)
        prior_mean = torch.split(self.mean, com_dim)
        
        kls = []
        for i in range(len(post_mean)):
            posterior = self.make_posterior(post_mean[i], post_std[i])
            prior = self.make_posterior(prior_mean[i], prior_std[i])
            
            if tpe == "mean":
                kls.append(kl_divergence(posterior, prior).mean().item())
            elif tpe == "seq":
                kls.append(kl_divergence(posterior, prior).mean(dim=0))
        return kls

class AutoregressiveMultivariateNormal(nn.Module):
    def __init__(
        self,
        tau: float,
        nvar: float,
        shape: int,
    ):
        super().__init__()
        # Create the distribution parameters
        logtaus = torch.log(torch.ones(shape) * tau)
        lognvars = torch.log(torch.ones(shape) * nvar)
        self.logtaus = nn.Parameter(logtaus, requires_grad=True)
        self.lognvars = nn.Parameter(lognvars, requires_grad=True)

    def make_posterior(self, post_mean, post_std):
        return Independent(Normal(post_mean, post_std), 2)

    def log_prob(self, sample):
        # Compute alpha and process variance
        alphas = torch.exp(-1.0 / torch.exp(self.logtaus))
        logpvars = self.lognvars - torch.log(1 - alphas**2)
        # Create autocorrelative transformation
        transform = AffineTransform(loc=0, scale=alphas)
        # Align previous samples and compute means and stddevs
        prev_samp = torch.roll(sample, shifts=1, dims=1)
        means = transform(prev_samp)
        stddevs = torch.ones_like(means) * torch.exp(0.5 * self.lognvars)
        # Correct the first time point
        means[:, 0] = 0.0
        stddevs[:, 0] = torch.exp(0.5 * logpvars)
        # Create the prior and compute the log-probability
        prior = Independent(Normal(means, stddevs), 2)
        return prior.log_prob(sample)

    def forward(self, post_mean, post_std):
        posterior = self.make_posterior(post_mean, post_std)
        sample = posterior.rsample()
        log_q = posterior.log_prob(sample)
        log_p = self.log_prob(sample)
        kl_batch = log_q - log_p
        return torch.mean(kl_batch)


class MultivariateStudentT(nn.Module):
    def __init__(
        self,
        loc: float,
        scale: float,
        df: int,
        shape: int,
    ):
        super().__init__()
        # Create the distribution parameters
        loc = torch.ones(shape) * scale
        self.loc = nn.Parameter(loc, requires_grad=True)
        logscale = torch.log(torch.ones(shape) * scale)
        self.logscale = nn.Parameter(logscale, requires_grad=True)
        self.df = df

    def make_posterior(self, post_loc, post_scale):
        # TODO: Should probably be inferring degrees of freedom along with loc and scale
        return Independent(StudentT(self.df, post_loc, post_scale), 1)

    def forward(self, post_loc, post_scale):
        # Create the posterior distribution
        posterior = self.make_posterior(post_loc, post_scale)
        # Create the prior distribution
        prior_scale = torch.exp(self.logscale)
        prior = Independent(StudentT(self.df, self.loc, prior_scale), 1)
        # Approximate KL divergence
        sample = posterior.rsample()
        log_q = posterior.log_prob(sample)
        log_p = prior.log_prob(sample)
        kl_batch = log_q - log_p
        return torch.mean(kl_batch)

class FRMultivariateNormal(nn.Module):
    def __init__(self,):
        super().__init__()

    def make_dist(self, mean, std):
        return FullRankMultivariateNormal(mean, std)

    def forward(self, pre_mean, pre_std, post_mean, post_std):
        prior = self.make_dist(pre_mean, pre_std)
        posterior = self.make_dist(post_mean, post_std)
        kl_batch = kl_divergence(posterior, prior)
        return torch.mean(kl_batch)