##########################
# Autor: Junyeob Baek
# email: wnsdlqjtm@gmail.com
##########################

from aae.models import AAE, VAE
from aae.distribution import interpolate
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch

from torch.utils.tensorboard import SummaryWriter

import easydict
import os, io
import time
from tqdm import tqdm
from datetime import datetime

log_dir = os.path.join("runs/", datetime.today().isoformat())
writer = SummaryWriter(log_dir)


def generate_manifold(args, model, writer):
    # change to eval mode
    model.eval_mode()
    with torch.no_grad():
        plot_size = 20
        random_code = interpolate(plot_size=plot_size, interpolate_range=[-3, 3, -3, 3])
        random_code = (
            torch.Tensor(random_code).to(model.device).view(plot_size, plot_size, -1)
        )

        manifold = torch.zeros(
            (
                args.num_channels,
                args.input_size[0] * plot_size,
                args.input_size[1] * plot_size,
            )
        )
        for i in range(plot_size):
            for j in range(plot_size):
                z_sample = random_code[i][j]
                x_gen = model.decode(z_sample)
                x_gen = (
                    x_gen.view(args.num_channels, args.input_size[0], -1)
                    .unsqueeze(0)
                    .cpu()
                )
                manifold[
                    :,
                    i * args.input_size[0] : (i + 1) * args.input_size[0],
                    j * args.input_size[1] : (j + 1) * args.input_size[1],
                ] = x_gen
        manifold = manifold.view(
            1,
            args.num_channels,
            plot_size * args.input_size[0],
            plot_size * args.input_size[1],
        )
        writer.add_images(f"generated_manifold", manifold, 0)


def visualize_latent_space(args, model, test_loader, writer):
    # change to eval mode
    model.eval_mode()
    with torch.no_grad():
        test_iterator = tqdm(
            enumerate(test_loader), total=len(test_loader), desc="testing"
        )
        latents = torch.Tensor([]).to(args.device)
        for i, batch_data in test_iterator:
            images, labels = batch_data
            images = images.view(images.size(0), -1).to(args.device)
            labels = labels.to(args.device)

            z_set = model.encode(images)
            to_plot = torch.cat((z_set, labels.unsqueeze(-1)), axis=-1)
            latents = torch.cat((latents, to_plot), axis=0)
        plt.scatter(
            latents[:, 0].cpu(),
            latents[:, 1].cpu(),
            c=latents[:, 2].cpu(),
            alpha=0.7,
            s=0.7,
        )
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)

        writer.add_figure(f"latent_space", plt.gcf())


if __name__ == "__main__":
    args = easydict.EasyDict(
        {
            "device": torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu"),
            "model": "vae",  # aae or vae
            "input_size": (28, 28),  # (28,28) for MNIST, (218, 178) for CelebA
            "num_channels": 1,  # 1 for MNIST, 3 for CelebA
            "latent_size": 2,  # 128 for MNIST, 512 for CelebA
            "checkpoint_path": "/home/junyeob/pytorch-adversarial-autoencoder/runs/230516VAE-MNIST-2d-gaussian/aae_checkpoint.pt",
        }
    )

    # MNIST Dataset
    test_set = dsets.MNIST(
        root="./data", train=True, transform=transforms.ToTensor(), download=True
    )

    # Data Loader
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=8, shuffle=False
    )

    # Declare model
    in_dim = args.num_channels * args.input_size[0] * args.input_size[1]
    if args.model == "aae":
        model = AAE(
            in_dim, args.latent_size, device=args.device
        )
    else:
        model = VAE(
            in_dim, args.latent_size, device=args.device
        )
    model.to(args.device)

    # Load Checkpoints
    checkpoint = torch.load(args.checkpoint_path, map_location=args.device)
    model.load_state_dict(checkpoint["model"])

    # generate manifold by sampling uniformly in the latent space Z
    generate_manifold(args, model, writer)

    # visualize latent space
    visualize_latent_space(args, model, test_loader, writer)

    time.sleep(1)
