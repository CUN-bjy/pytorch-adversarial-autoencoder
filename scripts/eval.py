##########################
# Autor: Junyeob Baek
# email: wnsdlqjtm@gmail.com
##########################

from aae.models import AAE
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch

from torch.utils.tensorboard import SummaryWriter

import easydict
import numpy as np
from tqdm import tqdm

writer = SummaryWriter()


## visualization
def imshow(data, title="MNIST"):
    plt.imshow(data)
    plt.title(title, fontsize=30)
    plt.savefig(f"{title}.png")
    plt.cla()


if __name__ == "__main__":
    args = easydict.EasyDict(
        {
            "device": torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu"),
            "input_size": 784,
            "latent_size": 128,
        }
    )

    # MNIST Dataset
    test_set = dsets.MNIST(
        root="./data", train=False, transform=transforms.ToTensor(), download=True
    )

    # Data Loader
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=args.batch_size, shuffle=False
    )

    # Declare AAE model
    model = AAE(args.input_size, args.latent_size, device=args.device)
    model.to(args.device)
