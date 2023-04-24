##########################
# Autor: Junyeob Baek
# email: wnsdlqjtm@gmail.com
##########################

import torch
from torch import nn
from torch.nn import functional as F
from typing import Dict, Any


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
        device=torch.device("cuda"),
    ) -> None:
        super(AAE, self).__init__()
        self.device = device

        # models
        self._encoder = MLP(in_dim, latent_dim)
        self._decoder = MLP(latent_dim, in_dim)
        self._discriminator = MLP(latent_dim, 1, binary_mode=True)

        # optimizers
        self._recon_optim = torch.optim.Adam(
            nn.ModuleList([self._encoder, self._decoder]).parameters(), lr=recon_lr
        )
        self._gen_optim = torch.optim.Adam(self._encoder.parameters(), lr=reg_lr)
        self._disc_optim = torch.optim.Adam(self._discriminator.parameters(), lr=reg_lr)

    def encode(self, x: Any) -> Any:
        return self._encoder(x)

    def decode(self, z: Any) -> Any:
        return self._decoder(z)

    def discriminate(self, z: Any) -> Any:
        return self._discriminator(z)

    def forward(self) -> None:
        """don't use this function"""
        return Exception("this function is not implemented")

    def train(self, x: Any) -> Dict[str, float]:
        self._encoder.train()
        self._decoder.train()
        self._discriminator.train()

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
        z_real = torch.randn_like(z_fake)

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
        z_real = torch.randn_like(z_fake)

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


class VAE:
    pass


class GAN:
    pass
