##########################
# Autor: Junyeob Baek
# email: wnsdlqjtm@gmail.com
##########################

from aae.models import AAE
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch

import easydict
import numpy as np
from tqdm import tqdm


## visualization
def imshow(data, title="MNIST"):
    plt.imshow(data)
    plt.title(title, fontsize=30)
    plt.savefig(f"{title}.png")
    plt.cla()

def train(args, model, train_loader, test_loader, debug=False):
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # interaction setup
    epochs = tqdm(range(args.max_iter // len(train_loader) + 1))
    
    # training
    count = 0
    for epoch in epochs:
        #### TRAIN PHASE
        model.train()
        optimizer.zero_grad()
        train_iterator = tqdm(enumerate(train_loader), total=len(train_loader), desc="training")
        
        for i, batch_data in train_iterator:
            if count > args.max_iter:
                return model
            count+=1
            
            images, labels = batch_data
            images = images.view(images.size(0), -1).to(args.device)
            
            if debug: images = images[0].unsqueeze(0)
            
            mloss, _, llog = model(images)
            
            # backward and optimize
            optimizer.zero_grad()
            mloss.mean().backward()
            optimizer.step()
            
            train_iterator.set_postfix({"train_loss": float(mloss.mean())})
            if debug: break

        ### EVAL PHASE
        model.eval()
        eval_loss = 0
        test_iterator = tqdm(enumerate(test_loader), total=len(test_loader), desc="testing")
        with torch.no_grad():
            for i, batch_data in test_iterator:
                
                images, labels = batch_data
                images = images.view(images.size(0), -1).to(args.device)
            
                if debug: images = images[0].unsqueeze(0)
                
                mloss, recon_x, llog = model(images)
                
                eval_loss += mloss.mean().item()
                
                test_iterator.set_postfix({"eval_loss": float(mloss.mean())})
                
                if i == 0:
                    nhw_orig = images.view(images.size(0), int(np.sqrt(images.size(-1))), -1)[0]
                    nhw_recon = recon_x.view(images.size(0), int(np.sqrt(images.size(-1))), -1)[0]
                    imshow(nhw_orig.cpu(), f"orig{epoch}")
                    imshow(nhw_recon.cpu(), f"recon{epoch}")
                    # writer.add_images(f"original{i}", nchw_orig, epoch)
                    # writer.add_images(f"reconstructed{i}", nchw_recon, epoch)
                    if debug: break
                
        eval_loss = eval_loss / len(test_loader)
        print(f"Evaluation Score: [{eval_loss}]")
    

if __name__=="__main__":

    args = easydict.EasyDict(
        {
            "batch_size": 512,
            "device":  torch.device("cuda")
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
    train_set = dsets.MNIST(root='./data', 
                        train=True, 
                        transform=transforms.ToTensor(),  
                        download=True)
    
    test_set = dsets.MNIST(root='./data', 
                        train=False, 
                        transform=transforms.ToTensor(),  
                        download=True)

    # Data Loader
    train_loader = torch.utils.data.DataLoader(dataset=train_set, 
                                            batch_size=args.batch_size, 
                                            shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, 
                                            batch_size=args.batch_size, 
                                            shuffle=False)

    # Declare AAE model
    model = AAE(args.input_size, args.latent_size, device=args.device)
    model.to(args.device)

    
    # Train
    train(args, model, train_loader, test_loader, debug=args.debug)