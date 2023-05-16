##########################
# Autor: Junyeob Baek
# email: wnsdlqjtm@gmail.com
##########################

import torch
from torch import nn
from torch.nn import functional as F
from typing import Dict, Any
from aae.distribution import diagonal_gaussian, gaussian_mixture


class MLP(nn.Module):
    "A simple MLP module for general purpose"

    def __init__(self, in_dim: int, out_dim: int, binary_mode: bool = False) -> None:
        super().__init__()
        self.binary_mode = binary_mode

        self.in_net = nn.Linear(in_dim, 512)
        self.hidden = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.1),
        )
        self.out_net = nn.Linear(512, out_dim)

    def forward(self, X: Any) -> Any:
        X = self.in_net(X)
        X = self.hidden(X)
        out = self.out_net(X)
        return out if not self.binary_mode else F.sigmoid(out)


class AAE(nn.Module):
    """Adversarial Autoencoder Model"""

    def __init__(
        self,
        in_dim: int,
        latent_dim: int,
        recon_lr: float = 1e-3,
        reg_lr: float = 1e-5,
        dist_type: str = "gaussian",
        device=torch.device("cuda"),
    ) -> None:
        super(AAE, self).__init__()
        self.device = device
        self._latent_dim = latent_dim

        # models
        self._encoder = MLP(in_dim, latent_dim)
        self._decoder = MLP(latent_dim, in_dim)
        self._discriminator = MLP(latent_dim, 1, binary_mode=True)

        # type of prior distribution
        self.dist_type = dist_type

        # optimizers
        self._recon_optim = torch.optim.Adam(
            nn.ModuleList([self._encoder, self._decoder]).parameters(), lr=recon_lr
        )
        self._gen_optim = torch.optim.Adam(self._encoder.parameters(), lr=reg_lr)
        self._disc_optim = torch.optim.Adam(self._discriminator.parameters(), lr=reg_lr)


    def eval_mode(self):
        self._encoder.eval()
        self._decoder.eval()
        self._discriminator.eval()
        
    def encode(self, x: Any) -> Any:
        return self._encoder(x)

    def decode(self, z: Any) -> Any:
        return self._decoder(z)

    def discriminate(self, z: Any) -> Any:
        return self._discriminator(z)

    def get_latent_dim(self):
        return self._latent_dim

    def forward(self) -> None:
        """don't use this function"""
        return Exception("this function is not implemented")

    def train(self, x: Any) -> Dict[str, float]:
        self._encoder.train()
        self._decoder.train()
        self._discriminator.train()

        B, D = x.size()

        ###############################
        # 1. reconstruction phase
        ###############################
        z = self.encode(x)
        x_hat = self.decode(z)

        recon_loss = F.mse_loss(x_hat, x)

        # update encoder and decoder to minimize reconstruction loss
        self._recon_optim.zero_grad()
        recon_loss.mean().backward()
        self._recon_optim.step()

        ###############################
        # 2. regularation phase
        ###############################
        # First, update discriminator
        self._encoder.eval()

        z_fake = self.encode(x)

        z_real = (
            diagonal_gaussian(z_fake.size(0), z_fake.size(1))
            if self.dist_type == "gaussian"
            else gaussian_mixture(
                z_fake.size(0), z_fake.size(1), 10, x_var=0.5, y_var=0.1
            )
        )  # 10 mode of gaussian mixture
        z_real = torch.Tensor(z_real).to(self.device)

        d_real = self.discriminate(z_real)
        d_fake = self.discriminate(z_fake)

        disc_loss = -(torch.log(d_real) + torch.log(1 - d_fake))

        # backpropagate to minimize discriminative loss
        self._disc_optim.zero_grad()
        disc_loss.mean().backward()
        self._disc_optim.step()

        # Second, update generator
        self._encoder.train()

        z_fake = self.encode(x)
        d_fake = self.discriminate(z_fake)

        gen_loss = -torch.log(d_fake)

        # backpropagate to confuse discriminaive network
        self._gen_optim.zero_grad()
        gen_loss.mean().backward()
        self._gen_optim.step()

        return {
            "recon_loss": recon_loss.detach(),
            "disc_loss": disc_loss.detach(),
            "gen_loss": gen_loss.detach(),
        }

    def eval(self, x):
        self._encoder.eval()
        self._decoder.eval()
        self._discriminator.eval()

        # eval autoencoder
        z_fake = self.encode(x)
        x_hat = self.decode(z_fake)

        # eval adversarial network
        z_real = (
            diagonal_gaussian(z_fake.size(0), z_fake.size(1))
            if self.dist_type == "gaussian"
            else gaussian_mixture(
                z_fake.size(0), z_fake.size(1), 10, x_var=0.5, y_var=0.1
            )
        )  # 10 mode of gaussian mixture
        z_real = torch.Tensor(z_real).to(self.device)

        d_real = self.discriminate(z_real)
        d_fake = self.discriminate(z_fake)

        recon_loss = F.mse_loss(x_hat, x)
        disc_loss = -(torch.log(d_real) + torch.log(1 - d_fake))
        gen_loss = -torch.log(d_fake)

        return {
            "recon_loss": recon_loss.detach(),
            "disc_loss": disc_loss.detach(),
            "gen_loss": gen_loss.detach(),
        }, x_hat


class VAE(nn.Module):
    """Variational Auto Encoder"""

    def __init__(
        self,
        in_dim: int,
        latent_dim: int,
        recon_lr: float = 1e-3,
        kl_weight: float = 1e-3,
        dist_type: str = "gaussian",
        device=torch.device("cuda"),
    ) -> None:
        super(VAE, self).__init__()
        self.device = device
        self._latent_dim = latent_dim

        self.kl_weight = kl_weight

        # type of prior distribution
        self.dist_type = dist_type
        assert (
            dist_type == "gaussian"
        ), "it's not implemented yet other than gaussian prior"

        # models
        self._encoder = MLP(in_dim, 128)
        self._decoder = MLP(latent_dim, in_dim)

        self.mu_vec = nn.Linear(128, latent_dim)
        self.std_vec = nn.Linear(128, latent_dim)

        # optimizers
        self._optim = torch.optim.Adam(
            nn.ModuleList([self._encoder, self._decoder]).parameters(), lr=recon_lr
        )

    def eval_mode(self):
        self._encoder.eval()
        self._decoder.eval()

    def encode(self, x: Any) -> Any:
        return self._encoder(x)

    def decode(self, z: Any) -> Any:
        return self._decoder(z)

    def get_latent_dim(self):
        return self._latent_dim

    def reparametize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        noise = torch.randn_like(std).to(self.device)

        z = mu + noise * std
        return z

    def forward(self) -> None:
        """don't use this function"""
        return Exception("this function is not implemented")

    def train(self, x: Any) -> Dict[str, float]:
        self._encoder.train()
        self._decoder.train()

        z_ = self.encode(x)

        mu = self.mu_vec(z_)
        logvar = self.std_vec(z_)
        z = self.reparametize(mu, logvar)

        x_hat = self.decode(z)

        recon_loss = F.mse_loss(x_hat, x)
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1), dim=0
        )

        loss = recon_loss + self.kl_weight * kld_loss

        self._optim.zero_grad()
        loss.mean().backward()
        self._optim.step()

        return {
            "recon_loss": recon_loss.detach(),
            "kl_loss": kld_loss.detach(),
        }

    def eval(self, x):
        self._encoder.eval()
        self._decoder.eval()

        # eval autoencoder
        z_ = self.encode(x)

        mu = self.mu_vec(z_)
        logvar = self.std_vec(z_)
        z = self.reparametize(mu, logvar)

        x_hat = self.decode(z)

        recon_loss = F.mse_loss(x_hat, x)
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1), dim=0
        )

        loss = recon_loss + self.kl_weight * kld_loss

        return {
            "recon_loss": recon_loss.detach(),
            "kl_loss": kld_loss.detach(),
        }, x_hat


class GAN:
    pass
