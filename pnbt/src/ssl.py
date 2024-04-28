# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:09:08 2019

Barlow Twins for PointNet

@author: tadahaya
"""
import torch
import torch.nn as nn

def flatten(t):
    return t.reshape(t.shape[0], -1)


def off_diagonal(x):
    """ return a flattened view of the off-diagonal elements of a square matrix """
    n, m = x.shape
    assert n==m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwins(nn.Module):
    """
    single GPU version based on https://github.com/MaxLikesMath/Barlow-Twins-Pytorch/tree/main

    """
    def __init__(self, backbone, config):
        """
        Parameters
        ----------
        backbone: Model

        
        Config keys
        -----------
        latent_id: name or index of the layer to be fed to the projection

        projection_sizes: size of the hidden layers in the projection

        lambd: tradeoff function

        scale_factor: factor to scale loss by

        """
        super().__init__()
        # config
        default_config = {
            "lambd": 5e-3,
            "scale_factor": 1.0,
            "projection_sizes": [1024, 1024, 1024, 1024]
        }
        self.config = {**default_config, **config}
        # for readability
        self.backbone = backbone
        self.lambd = self.config["lambd"]
        self.scale_factor = self.config["scale_factor"]
        self.projection_sizes = self.config["projection_sizes"]
        # projector
        layers = []
        for i in range(len(self.projection_sizes) - 2):
            layers.append(nn.Linear(
                self.projection_sizes[i], self.projection_sizes[i + 1], bias=False)
            ) # BatchNorm入れるのでbias=False
            layers.append(nn.BatchNorm1d(self.projection_sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(
            self.projection_sizes[-2], self.projection_sizes[-1], bias=False)
        ) # BatchNorm入れるのでbias=False
        self.projector = nn.Sequential(*layers)
        # normalization layer for z1 and z2
        self.bn = nn.BatchNorm1d(self.projection_sizes[-1], affine=False)


    def forward(self, y1, y2): # 2つのtensorをinput
        z1, _ = self.backbone(y1) # (global features, critical indices)を返す
        z2, _ = self.backbone(y2)
        z1 = self.projector(z1)
        z2 = self.projector(z2)
        # empirical cross-correlation matrix
        c = torch.mm(self.bn(z1).T, self.bn(z2))
        c.div_(z1.shape[0])
        # scaling
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = self.scale_factor * (on_diag + self.lambd * off_diag)
        return loss
    

    @torch.no_grad()
    def get_latent(self, x):
        z, critidx = self.backbone(x)
        return self.projector(z).detach(), critidx


# ToDo 全くを手を付けてないのでテストする
class LinearHead(nn.Module):
    def __init__(self, backbone, num_classes:int):
        super().__init__()
        self.backbone = backbone
        self.fc = nn.Linear(backbone.fc.in_features, num_classes)


    def forward(self, x):
        out = self.backbone(x)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out