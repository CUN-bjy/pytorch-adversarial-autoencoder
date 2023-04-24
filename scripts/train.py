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


def train_model(args, model, train_loader, test_loader, debug=False):
    # interaction setup
    epochs = tqdm(range(args.max_iter // len(train_loader) + 1))

    # training the model
    count = 0
    for epoch in epochs:
        #### TRAIN PHASE
        train_iterator = tqdm(
            enumerate(train_loader), total=len(train_loader), desc="training"
        )

        if count > args.max_iter:
            return
        count += 1

        train_losses = []
        recon_losses, disc_losses, gen_losses = [], [], []
        for i, batch_data in train_iterator:
            images, labels = batch_data
            images = images.view(images.size(0), -1).to(args.device)

            if debug:
                images = images[0].unsqueeze(0)

            llog = model.train(images)

            recon_loss, disc_loss, gen_loss = (
                float(llog["recon_loss"].mean()),
                float(llog["disc_loss"].mean()),
                float(llog["gen_loss"].mean()),
            )
            train_loss = recon_loss + disc_loss + gen_loss
            train_iterator.set_postfix({"train_loss": train_loss})
            train_losses.append(train_loss)
            recon_losses.append(recon_loss)
            disc_losses.append(disc_loss)
            gen_losses.append(gen_loss)
            if debug:
                break
        writer.add_scalar("train/loss", sum(train_losses) / len(train_losses), epoch)
        writer.add_scalar(
            "train/recon_loss", sum(recon_losses) / len(recon_losses), epoch
        )
        writer.add_scalar("train/disc_loss", sum(disc_losses) / len(disc_losses), epoch)
        writer.add_scalar("train/gen_loss", sum(gen_losses) / len(gen_losses), epoch)

        ### EVAL PHASE
        test_iterator = tqdm(
            enumerate(test_loader), total=len(test_loader), desc="testing"
        )
        eval_losses = []
        recon_losses, disc_losses, gen_losses = [], [], []
        with torch.no_grad():
            for i, batch_data in test_iterator:
                images, labels = batch_data
                images = images.view(images.size(0), -1).to(args.device)

                if debug:
                    images = images[0].unsqueeze(0)

                llog, recon_x = model.eval(images)

                recon_loss, disc_loss, gen_loss = (
                    float(llog["recon_loss"].mean()),
                    float(llog["disc_loss"].mean()),
                    float(llog["gen_loss"].mean()),
                )
                eval_loss = recon_loss + disc_loss + gen_loss
                test_iterator.set_postfix({"eval_loss": eval_loss})
                eval_losses.append(eval_loss)
                recon_losses.append(recon_loss)
                disc_losses.append(disc_loss)
                gen_losses.append(gen_loss)

                if i == 0:
                    nhw_orig = images.view(
                        images.size(0), int(np.sqrt(images.size(-1))), -1
                    )[0]
                    nhw_recon = recon_x.view(
                        images.size(0), int(np.sqrt(images.size(-1))), -1
                    )[0]
                    imshow(nhw_orig.cpu(), f"orig{epoch}")
                    imshow(nhw_recon.cpu(), f"recon{epoch}")
                    # writer.add_images(f"original{i}", nchw_orig, epoch)
                    # writer.add_images(f"reconstructed{i}", nchw_recon, epoch)
                    if debug:
                        break

        print(f"Evaluation Score: [{sum(eval_losses)/len(eval_losses)}]")
        writer.add_scalar("eval/loss", sum(eval_losses) / len(eval_losses), epoch)
        writer.add_scalar(
            "eval/recon_loss", sum(recon_losses) / len(recon_losses), epoch
        )
        writer.add_scalar("eval/disc_loss", sum(disc_losses) / len(disc_losses), epoch)
        writer.add_scalar("eval/gen_loss", sum(gen_losses) / len(gen_losses), epoch)
        if debug:
            break


if __name__ == "__main__":
    args = easydict.EasyDict(
        {
            "batch_size": 512,
            "device": torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu"),
            "input_size": 784,
            "latent_size": 128,
            "learning_rate": 0.001,
            "max_iter": 1000,
            "debug": False,
        }
    )

    # MNIST Dataset
    train_set = dsets.MNIST(
        root="./data", train=True, transform=transforms.ToTensor(), download=True
    )

    test_set = dsets.MNIST(
        root="./data", train=False, transform=transforms.ToTensor(), download=True
    )

    # Data Loader
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=args.batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=args.batch_size, shuffle=False
    )

    # Declare AAE model
    model = AAE(args.input_size, args.latent_size, device=args.device)
    model.to(args.device)

    # Train
    train_model(args, model, train_loader, test_loader, debug=args.debug)
