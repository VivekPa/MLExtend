import torch
import torch.distributions
from torch.distributions import Normal, MultivariateNormal
from torch.utils.data import DataLoader, TensorDataset

import sklearn
from sklearn.datasets import make_moons
from sklearn.preprocessing import scale

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import copy
import sys, os, argparse, datetime, time
from statsmodels.stats.correlation_tools import cov_nearest

import torch.nn.functional as F
from torch import nn

sns.set()

parser = argparse.ArgumentParser()

parser.add_argument('-num_samples', type=int, default=200)
parser.add_argument('-num_inducing_points', type=int, default=10)
parser.add_argument('-x_noise_std', type=float, default=0.01)
parser.add_argument('-y_noise_std', type=float, default=0.5)
parser.add_argument('-zoom', type=int, default=10)

parser.add_argument('-lr_kernel', type=float, default=0.01)
parser.add_argument('-lr_ip', type=float, default=0.1)

parser.add_argument('-num_epochs', type=int, default=200)

params = parser.parse_args()

def generate_data():
    x = np.linspace(-1, 1, params.num_samples)
    x_noise = np.random.normal(0., params.x_noise_std, size=x.shape)
    y_noise = np.random.normal(0., params.y_noise_std, size=x.shape)

    y = x + np.exp(x + x_noise) + np.sin(2*np.pi*x + x_noise) + y_noise

    x, y = x.reshape(-1, 1), y.reshape(-1, 1)

    if True:
        plt.scatter(x, y)
        plt.xlabel('X')
        plt.ylabel("Y")
        plt.show()

    return x, y

class GP_IP(nn.Module):
    def __init__(self, xs, ys, num_ip=params.num_inducing_points, dim=1) -> None:
        super().__init__()

        self.x = xs
        self.y = ys

        self.num_ip = num_ip

        inducing_x = torch.linspace(xs.min().item(), xs.max().item(), self.num_ip).reshape(-1, 1)
        self.inducing_x_hat = nn.Parameter(inducing_x + torch.randn_like(inducing_x).clamp(-0.1, 0.1))
        self.inducing_y_hat = nn.Parameter(torch.FloatTensor(self.num_ip, dim).uniform_(-0.5, 0.5))

        self.ls = nn.Parameter(torch.scalar_tensor(0.2))
        self.noise = nn.Parameter(torch.scalar_tensor(0.5))

    def kernel_matrix(self, xi, xj):
        pdist = (xi - xj.T)**2
        return torch.exp(-pdist*0.5/(self.ls + 0.01))

    def forward(self, Xs):
        self.K_XsX = self.kernel_matrix(Xs, self.inducing_x_hat)
        self.K_XX = self.kernel_matrix(self.inducing_x_hat, self.inducing_x_hat)
        self.K_XsXs = self.kernel_matrix(Xs, Xs)

        self.K_XX_inv = torch.inverse(self.K_XX + 1e-6*torch.eye(self.K_XX.shape[0]))

        mu = self.K_XsX @ self.K_XX_inv @ self.inducing_y_hat
        sigma = self.K_XsXs - self.K_XsX @ self.K_XX_inv @ self.K_XsX.T + self.noise*torch.eye(self.K_XsXs.shape[0])

        return mu, torch.diag(sigma)[:, None]

    def loss(self, Xs, y):
        self.ls.data = self.ls.data.clamp(1e-6, 3)
        self.noise.data = self.noise.data.clamp(1e-6, 3)

        self.K_XsX = self.kernel_matrix(Xs, self.inducing_x_hat)
        self.K_XX = self.kernel_matrix(self.inducing_x_hat, self.inducing_x_hat)
        self.K_XsXs = self.kernel_matrix(Xs, Xs)

        self.K_XX_inv = torch.inverse(self.K_XX + 1e-6*torch.eye(self.K_XX.shape[0]))

        Q_XX = self.K_XsXs - self.K_XsX @ self.K_XX_inv @ self.K_XsX.T

        mu = self.K_XsX @ self.K_XX_inv @ self.inducing_y_hat
        sigma = Q_XX + self.noise**2*torch.eye(Q_XX.shape[0])

        spy_sigma = cov_nearest(sigma.detach().numpy(), method='nearest')
        cov_sigma = torch.FloatTensor(spy_sigma)

        p_y = MultivariateNormal(mu.squeeze(), covariance_matrix=cov_sigma)

        log_py = p_y.log_prob(y.squeeze()) - 1/(2*self.noise**2)*torch.trace(Q_XX)

        return -log_py

    def plot(self, title, save=False):
        x = torch.linspace(self.x.mean() - (self.x.max() - self.x.min())*1.5*0.5, self.x.mean() + (self.x.max() - self.x.min())*1.5*0.5, 200).reshape(-1, 1)

        with torch.no_grad():
            mu, sigma = self.forward(x)

        x = x.numpy().squeeze()
        mu = mu.numpy().squeeze()
        sigma = sigma.numpy().squeeze()

        plt.title(title)
        plt.scatter(self.inducing_x_hat.detach().numpy(), self.inducing_y_hat.detach().numpy(), label='Inducing Points')
        plt.scatter(self.x.detach().numpy(), self.y.detach().numpy())
        plt.fill_between(x, mu-2*sigma, mu+2*sigma, alpha=0.1, label='95% CI')
        plt.plot(x, mu, label='Mean Function')

        plt.xlim(self.x.mean() - (self.x.max() - self.x.min())*1.5*0.5, self.x.mean() + (self.x.max() - self.x.min())*1.5*0.5)
        plt.ylim(-3, 3)

        plt.legend()
        if save:
            plt.savefig(f"figures/{title}.jpg")

        plt.show()


if __name__ == "__main__":
    X, y = generate_data()
    X, y = torch.FloatTensor(scale(X)), torch.FloatTensor(scale(y))

    gp = GP_IP(xs=X, ys=y)
    gp.plot("Initial")

    opt = torch.optim.Adam([
        {"params": [gp.ls, gp.noise], "lr":params.lr_kernel},
        {"params": [gp.inducing_x_hat, gp.inducing_y_hat], "lr":params.lr_ip}
    ])

    train_data = DataLoader(TensorDataset(X, y), batch_size=params.num_samples, shuffle=True, num_workers=1)
    
    for epoch in range(params.num_epochs):
        for _, (data, label) in enumerate(train_data):
            opt.zero_grad()

            loglike = gp.loss(data, label)

            loglike.backward()
            opt.step()

            if epoch%(params.num_epochs//10) == 0:
                # print(f"Epoch: {epoch} \t Log-Likelihood = {loglike:.2f} \t GP Lengthscale = {gp.ls:.2f} \t GP Noise = {gp.noise:.2f}")
                # gp.plot(title=f"Epoch: {epoch} \t Log-Likelihood = {-loglike:.2f} \t GP Lengthscale = {gp.ls:.2f} \t GP Noise = {gp.noise:.2f}")
                gp.plot(f'Inducing Points (Epoch {int(epoch)})', save=True)
                # pass

        # gp.plot(title="Inducing Points")
        print(f"Epoch: {epoch} \t Log-Likelihood = {-loglike.item():.2f} \t GP Lengthscale = {gp.ls.item():.2f} \t GP Noise = {gp.noise.item():.2f}")

