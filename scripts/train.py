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
from tqdm import tqdm
import os
from datetime import datetime

log_dir = os.path.join("runs/", datetime.today().isoformat())
writer = SummaryWriter(log_dir)


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
            train_iterator.set_postfix({"train_loss": recon_loss})
            recon_losses.append(recon_loss)
            disc_losses.append(disc_loss)
            gen_losses.append(gen_loss)
            if debug:
                break
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
                test_iterator.set_postfix({"eval_loss": recon_loss})
                recon_losses.append(recon_loss)
                disc_losses.append(disc_loss)
                gen_losses.append(gen_loss)

                if i == 0:
                    # nhw_orig = images.view(
                    #     images.size(0), int(np.sqrt(images.size(-1))), -1
                    # )[0]
                    # nhw_recon = recon_x.view(
                    #     images.size(0), int(np.sqrt(images.size(-1))), -1
                    # )[0]
                    # imshow(nhw_orig.cpu(), f"orig{epoch}")
                    # imshow(nhw_recon.cpu(), f"recon{epoch}")

                    nhw_orig = images.view(
                        images.size(0), args.num_channels, args.input_size[0], -1
                    )[0].unsqueeze(0)
                    nhw_recon = recon_x.view(
                        images.size(0), args.num_channels, args.input_size[0], -1
                    )[0].unsqueeze(0)
                    writer.add_images(f"original{i}", nhw_orig.cpu(), epoch)
                    writer.add_images(f"reconstructed{i}", nhw_recon.cpu(), epoch)
                    if debug:
                        break

        print(f"Evaluation Score: [{sum(recon_losses) / len(recon_losses)}]")
        writer.add_scalar(
            "eval/recon_loss", sum(recon_losses) / len(recon_losses), epoch
        )
        writer.add_scalar("eval/disc_loss", sum(disc_losses) / len(disc_losses), epoch)
        writer.add_scalar("eval/gen_loss", sum(gen_losses) / len(gen_losses), epoch)

        checkpoint = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
        }
        torch.save(checkpoint, os.path.join(log_dir, "aae_checkpoint.pt"))
        if debug:
            break


if __name__ == "__main__":
    args = easydict.EasyDict(
        {
            "batch_size": 512,
            "device": torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu"),
            "input_size": (28, 28), # (28,28) for MNIST, (218, 178) for CelebA
            "num_channels": 1, # 1 for MNIST, 3 for CelebA
            "latent_size": 2, # 2 for MNIST, 128 for CelebA
            "learning_rate": 0.001,
            "dataset": "mnist",
            "max_iter": 2000,
            "debug": False,
        }
    )

    # Dataset
    if args.dataset == "mnist":
        train_set = dsets.MNIST(
            root="./data", train=True, transform=transforms.ToTensor(), download=True
        )

        test_set = dsets.MNIST(
            root="./data", train=False, transform=transforms.ToTensor(), download=True
        )
    else:
        train_set = dsets.CelebA(
            root="./data", split="train", transform=transforms.ToTensor(), download=True
        )

        test_set = dsets.CelebA(
            root="./data", split="valid", transform=transforms.ToTensor(), download=True
        )

    # Data Loader
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=args.batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=args.batch_size, shuffle=False
    )

    # Declare AAE model
    in_dim = args.num_channels * args.input_size[0] * args.input_size[1]
    model = AAE(in_dim, args.latent_size, device=args.device)
    model.to(args.device)

    # Train
    train_model(args, model, train_loader, test_loader, debug=args.debug)
