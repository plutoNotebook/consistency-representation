import torch
import ml_collections

from torch import nn
from copy import deepcopy
from consistency_models.consistency_models import *
from consistency_models import ConsistencySamplingAndEditing, ImprovedConsistencyTraining, pseudo_huber_loss, improved_timesteps_schedule
from consistency_models.utils import update_ema_model_
from ncsn.ncsnpp import NCSNpp, NCSNppEncoder

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class GenerativeRepresentation(nn.Module):
    def __init__(
        self,
        model_config: ml_collections.ConfigDict,
        total_step: int,
        ema_decay: float,
        rep_dim: int = 256,
        emb_dim: int = 1024,
        batch_size: int = 64,
        num_classes: int = 64,
        reg_weight: float = 1.0,
        lambd: float = 0.0051,
        is_stochastic: bool = False
    ):
        super().__init__()
        self.model_config = model_config
        self.total_step = total_step
        self.ema_decay = ema_decay
        self.rep_dim = rep_dim
        self.emb_dim = emb_dim
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.reg_weight = reg_weight
        self.lambd = lambd
        self.is_stochastic = is_stochastic

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.generator = NCSNpp(self.model_config)
        self.generator_ema = deepcopy(self.generator)
        self.generator_ema.load_state_dict(self.generator.state_dict())
        for param in self.generator_ema.parameters():
            param.requires_grad = False
        self.generator_ema.eval()

        self.encoder = NCSNppEncoder(self.model_config)

        self.projector = nn.Sequential(
            nn.Linear(self.rep_dim, self.emb_dim, bias=False),
            nn.BatchNorm1d(self.emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.emb_dim, self.emb_dim)
        )
        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(self.emb_dim, affine=False)
        # linear head for classification
        self.head = nn.Linear(self.rep_dim, self.num_classes)

         # Initialize the training module 
        self.improved_consistency_training = ImprovedConsistencyTraining(
        sigma_min=0.002,
        sigma_max=80.0,
        rho=7.0,
        sigma_data=0.5,
        initial_timesteps=10,
        final_timesteps=1280,
        lognormal_mean=-1.1,
        lognormal_std=2.0,
        )
    
        self.consistency_sampling_and_editing = ConsistencySamplingAndEditing(
            sigma_min=0.002,
            sigma_data=0.5,
        )

    def forward(self, x, step): 
            
        # Forward Pass
        recon_output = self.improved_consistency_training(self.generator, x, step, self.total_step)

        x1, x2, num_timesteps, sigmas, weights = recon_output.predicted, recon_output.target, recon_output.num_timesteps, recon_output.sigmas, recon_output.loss_weights
        # Loss Computation
        recon_loss = (pseudo_huber_loss(x1, x2) * weights).mean()

        noise = torch.randn_like(x).to(self.device)
        aug_timestep = lognormal_timestep_distribution(
            x.shape[0], sigmas, mean=-1.1, std=2.0
        )
        aug_sigma = sigmas[aug_timestep]
        noisy_aug = x + pad_dims_like(aug_sigma, x) * noise
        x_aug = model_forward_wrapper(self.generator, noisy_aug, aug_sigma, 0.5, 0.002)
        aug1, aug2 = x1, x_aug
        
        if self.is_stochastic:
            mu1, logvar1 = self.encoder(aug1)
            mu2, logvar2 = self.encoder(aug2)

            z1 = self.reparameterization(mu1, logvar1)
            z2 = self.reparameterization(mu2, logvar2)
            
            kl_div1 = -0.5 * torch.sum(1 + logvar1 - mu1.pow(2) - logvar1.exp())
            kl_div2 = -0.5 * torch.sum(1 + logvar2 - mu2.pow(2) - logvar2.exp())
            kl_div = (kl_div1 + kl_div2) / 2

        else:
            z1 = self.encoder(aug1) # representation
            z2 = self.encoder(aug2)
        
        y1 = self.projector(z1) # embedding
        y2 = self.projector(z2) 

        # empirical cross-correlation matrix
        c = self.bn(y1).T @ self.bn(y2)

        # sum the cross-correlation matrix between all gpus
        c.div_(self.batch_size)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        rep_loss = on_diag + self.lambd * off_diag

        if self.is_stochastic:
            rep_loss = rep_loss + self.reg_weight * kl_div
        
        return recon_loss, rep_loss
    
    @torch.no_grad()
    def get_sample(self, sigmas, num_sample, ema=False):
        size = self.model_config.data.image_size
        noise = torch.randn((num_sample, 3, size, size), device=self.device)
        if ema:
            model = self.generator_ema
        else:
            model = self.generator
        sample = self.consistency_sampling_and_editing(model, noise, sigmas=sigmas, clip_denoised=True, verbose=True)

        return sample

    @torch.no_grad()
    def get_augmentation_sample(self, x, step, num_samples):

        model = self.generator
        batch = x[:num_samples]
        num_timesteps = improved_timesteps_schedule(step, self.total_step, 10, 1280)
        sigmas = karras_schedule(num_timesteps, 0.002, 80.0, 7.0, self.device)
        noise = torch.randn_like(batch).to(self.device)
        timesteps = lognormal_timestep_distribution(num_samples, sigmas=sigmas, mean=-1.1, std=2.0)
        aug_timestep = lognormal_timestep_distribution(num_samples, sigmas, mean=-1.1, std=2.0)
        sigma = sigmas[timesteps]
        aug_sigma = sigmas[aug_timestep]
        noisy_x = batch + pad_dims_like(sigma, batch) * noise
        noisy_x_aug = batch + pad_dims_like(aug_sigma, batch) * noise
        x_recon = model_forward_wrapper(model, noisy_x, sigma,  0.5, 0.002)
        x_aug_recon = model_forward_wrapper(model, noisy_x_aug, aug_sigma,  0.5, 0.002)
        x_recon = x_recon.clamp(min=-1.0, max=1.0)
        x_aug_recon = x_aug_recon.clamp(min=-1.0, max=1.0)
        return x_recon, x_aug_recon
        
    
    @torch.no_grad()
    def get_representation(self, x):
        representation = self.encoder(x)

        return representation
    
    def reparameterization(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std
    
    def update_ema(self):
        update_ema_model_(self.generator_ema, self.generator, self.ema_decay)

    def get_current_num_timestep(self, step):
        return improved_timesteps_schedule(step, self.total_step)
        
