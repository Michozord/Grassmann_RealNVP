__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2019/11/03 21:50:25"

import numpy as np
import torch
torch.set_default_dtype(torch.float64)
import torch.optim as optim
import pickle
import math
import os
from sys import exit
import matplotlib.pyplot as plt
from sklearn import datasets
from RealNVP_2D import *

## Set seed for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

## Detect and select device: CUDA > torch_directml > CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA device")
else:
    try:
        import torch_directml
        device = torch_directml.device()
        print("Using DirectML device")
    except ImportError:
        device = torch.device("cpu")
        print("Using CPU device")

## dimenstion of hidden units used in scale and translation transformation
hidden_dim = 128

## construct the RealNVP_2D object
realNVP = RealNVP_2D(n=2, r=1, num_layers=8, hidden_dim=hidden_dim)
# realNVP = realNVP.to(device)
device = next(realNVP.parameters()).device

optimizer = optim.Adam(realNVP.parameters(), lr = 1e-4)
num_steps = 30_000

## Create checkpoint directory
checkpoint_dir = "./checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

## the following loop learns the RealNVP_2D model by data
## in each loop, data is dynamically sampled from the scipy moon dataset
for idx_step in range(num_steps):
    ## sample data from the scipy moon dataset
    X, label = datasets.make_moons(n_samples = 512, noise = 0.05)
    X = torch.Tensor(X).to(device = device)

    ## transform data X to latent space Z
    z, logdet = realNVP.inverse(X)

    ## calculate the negative loglikelihood of X
    loss = torch.log(z.new_tensor([2*math.pi])) + torch.mean(torch.sum(0.5*z**2, -1) - logdet)
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(realNVP.parameters(), max_norm=1.0)

    optimizer.step()

    if (idx_step + 1) % 100 == 0:
        print(f"idx_steps: {idx_step:}, loss: {loss.item():.5f}")
    
    ## Save model and generate pictures every 1000 steps
    if (idx_step + 1) % 1000 == 0:
        step = idx_step + 1
        
        ## Save model
        model_path = os.path.join(checkpoint_dir, f"realNVP_step_{step}.pt")
        torch.save(realNVP.state_dict(), model_path)
        print(f"Model saved at step {step}: {model_path}")
        
        ## Generate and save picture 1: Z transformed from X
        X_test, label_test = datasets.make_moons(n_samples=1000, noise=0.05)
        X_test = torch.Tensor(X_test).to(device=device)
        z_test, _ = realNVP.inverse(X_test)
        z_test = z_test.cpu().detach().numpy()
        X_test = X_test.cpu().detach().numpy()
        
        fig = plt.figure(figsize=(12.8, 4.8))
        fig.clf()
        plt.subplot(1, 2, 1)
        plt.plot(X_test[label_test==0,0], X_test[label_test==0,1], ".")
        plt.plot(X_test[label_test==1,0], X_test[label_test==1,1], ".")
        plt.title("X sampled from Moon dataset")
        plt.xlabel(r"$x_1$")
        plt.ylabel(r"$x_2$")
        
        plt.subplot(1, 2, 2)
        plt.plot(z_test[label_test==0,0], z_test[label_test==0,1], ".")
        plt.plot(z_test[label_test==1,0], z_test[label_test==1,1], ".")
        plt.title("Z transformed from X")
        plt.xlabel(r"$z_1$")
        plt.ylabel(r"$z_2$")
        
        pic1_path = os.path.join(checkpoint_dir, f"moon_z_transformed_from_x_step_{step}.png")
        plt.savefig(pic1_path)
        plt.close()
        print(f"Picture 1 saved: {pic1_path}")
        
        ## Generate and save picture 2: X transformed from Z
        z_sample = torch.normal(0, 1, size=(1000, 2)).to(device=device)
        X_sample, _ = realNVP(z_sample)
        X_sample = X_sample.cpu().detach().numpy()
        z_sample = z_sample.cpu().detach().numpy()
        
        fig = plt.figure(figsize=(12.8, 4.8))
        fig.clf()
        plt.subplot(1, 2, 1)
        plt.plot(z_sample[:,0], z_sample[:,1], ".")
        plt.title("Z sampled from normal distribution")
        plt.xlabel(r"$z_1$")
        plt.ylabel(r"$z_2$")
        
        plt.subplot(1, 2, 2)
        plt.plot(X_sample[:,0], X_sample[:,1], ".")
        plt.title("X transformed from Z")
        plt.xlabel(r"$x_1$")
        plt.ylabel(r"$x_2$")
        
        pic2_path = os.path.join(checkpoint_dir, f"moon_x_transformed_from_z_step_{step}.png")
        plt.savefig(pic2_path)
        plt.close()
        print(f"Picture 2 saved: {pic2_path}")

print(f"Training completed! Models and pictures saved in {checkpoint_dir}")

