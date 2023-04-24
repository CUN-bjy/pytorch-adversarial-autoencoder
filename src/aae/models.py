##########################
# Autor: Junyeob Baek
# email: wnsdlqjtm@gmail.com
##########################

import torch
from torch import nn
from torch.nn import functional as F
from typing import Dict, Any, Tuple


class MLP(nn.Module):
    "A simple MLP module for general purpose"
    def __init__(self, in_dim: int, out_dim: int)-> None:
        super().__init__()
        self.in_net = nn.Linear(in_dim, 512)
        self.hidden = nn.Sequential(
            nn.Linear(512, 1024), nn.ReLU(), nn.Dropout(p=0.1), 
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(p=0.1),
        )
        self.out_net = nn.Linear(512, out_dim)

    def forward(self, X: Any)-> Any:
        X = self.in_net(X)
        X = self.hidden(X)
        out = self.out_net(X)
        return out


class Discriminator(nn.Module):
    """Discriminator Model"""
    def __init__(self):
        super(Discriminator, self).__init__()
        
    def forward(self, x):
        return x

class AAE(nn.Module):
    """Adversarial Autoencoder Model"""
    def __init__(self, in_dim: int, latent_dim:int, device=torch.device("cuda")) -> None:
        super(AAE, self).__init__()
        self.device = device
        
        self._encoder = MLP(in_dim, latent_dim)
        self._decoder = MLP(latent_dim, in_dim)
        
    def encode(self, x: Any) -> Any:
        return self._encoder(x)
    
    def decode(self, z: Any) -> Any:
        return self._decoder(z)

    def forward(self, x: Any) -> Tuple[float, Tuple[float]]:
        z = self.encode(x)
        x_hat = self.decode(z)
        
        losses = self.loss_function(x_hat, x)
        
        return losses["loss"], x_hat, (losses["recon_loss"],)

    def loss_function(self, *args, **kwargs) -> Dict:
        recons = args[0]
        input = args[1]
        
        recon_loss = F.mse_loss(recons, input)
        
        loss = recon_loss
        
        return {
            "loss": loss,
            "recon_loss": recon_loss.detach(),
        }

class VAE():
    pass

class GAN():
    pass