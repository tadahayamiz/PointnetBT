# -*- coding: utf-8 -*-
"""
Created on Mon Aug 9 15:00:00 2021

Pointnet for backbone model

inspired from the following:
https://github.com/itberrios/3D/tree/main/point_net
https://medium.com/@itberrios6/point-net-from-scratch-78935690e496

@author: tadahaya
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

def make_model(config):
    return PointNetBackbone(config)


class SharedMLPBlock(nn.Module):
    """ Shared MLP block with batch normalization using conv1d """
    def __init__(self, in_channels, out_channels):
        """
        Parameters
        ----------
        in_channels:int
            number of input channels

        out_channels:int
            number of output channels

        """
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.bn(F.relu(self.conv(x)))
        return x


class NonlinearBlock(nn.Module):
    """ Linear block with batch normalization """
    def __init__(self, input_dim, output_dim):
        """
        Parameters
        ----------
        in_channels:int
            number of input channels

        out_channels:int
            number of output channels

        """
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False) # batch norm already has bias
        self.bn = nn.BatchNorm1d(output_dim)


    def forward(self, x):
        x = self.bn(F.relu(self.linear(x)))
        return x


class Tnet(nn.Module):
    """ T-Net learns a Transformation matrix with a specified dimension """
    def __init__(self, config):
        """
        Config keys
        -----------
        input_dim:int
            input dimension

        num_points:int
            number of points in each point cloud data

        dropout_ratio:float
            dropout ratio

        """
        super().__init__()
        # config
        default_config = {
            "input_dim": 3,
            "dropout_ratio": 0.1
        }
        self.config = {**default_config, **config}
        # for readability
        self.dim = self.config["input_dim"] # input dim
        self.dropout_ratio = self.config["dropout_ratio"]
        self.num_points = self.config["num_points"]
        # model
        self.smlp = nn.Sequential(
            SharedMLPBlock(self.dim, 64),
            SharedMLPBlock(64, 128),
            SharedMLPBlock(128, 512)
        )
        self.max_pool = nn.MaxPool1d(kernel_size=self.num_points)
        self.nonlinear = nn.Sequential(
            NonlinearBlock(512, 256),
            NonlinearBlock(256, 128)
        )
        self.dropout = nn.Dropout(p=self.dropout_ratio)
        self.fc = nn.Linear(128, self.dim**2)

        # note: original implementation
        # self.smlp = nn.Sequential(
        #     SharedMLPBlock(self.dim, 64),
        #     SharedMLPBlock(64, 128),
        #     SharedMLPBlock(128, 1024)
        # )
        # self.max_pool = nn.MaxPool1d(kernel_size=self.num_points)
        # self.nonlinear = nn.Sequential(
        #     NonlinearBlock(1024, 512),
        #     NonlinearBlock(512, 256)
        # )
        # self.dropout = nn.Dropout(p=self.dropout_ratio)
        # self.fc = nn.Linear(256, self.dim**2)


    def forward(self, x):
        bs = x.shape[0] # get batch size
        # pass through shared MLP layers (conv1d)
        x = self.smlp(x)
        # max pool over num points
        x = self.max_pool(x).view(bs, -1)
        # pass through MLP
        x = self.nonlinear(x)
        x = self.dropout(x)
        x = self.fc(x)
        # initialize identity matrix to keep the original point cloud shape
        iden = torch.eye(self.dim, requires_grad=True).repeat(bs, 1, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x.view(-1, self.dim, self.dim) + iden
        return x


class PointNetBackbone(nn.Module):
    def __init__(self, config):
        """
        Config keys
        -----------
        input_dim:int
            input dimension

        num_points:int
            number of points in each point cloud data

        dim_global_feats:int
            dimension of global features

        local_feats:bool
            whether to include local features or not

        dropout_ratio:float
            dropout ratio

        """
        super().__init__()
        # config
        default_config = {
            "input_dim": 3,
            "dim_global_feats": 1024,
            "local_feats": False,
            "dropout_ratio": 0.1
        }
        self.config = {**default_config, **config}
        # for readability
        self.input_dim = self.config["input_dim"]
        self.dim_global_feats = self.config["dim_global_feats"]
        self.local_feats = self.config["local_feats"]
        self.dropout_ratio = self.config["dropout_ratio"]
        self.num_points = self.config["num_points"]
        # Spatial Transformer Networks (T-nets)
        config2 = self.config.copy()
        config2["input_dim"] = 64
        self.tnet1 = Tnet(self.config)
        self.tnet2 = Tnet(config2)
        # shared MLP 1
        self.smlp1 = nn.Sequential(
            SharedMLPBlock(self.input_dim, 64),
            SharedMLPBlock(64, 64)
            )
        # shared MLP 2
        self.smlp2 = nn.Sequential(
            SharedMLPBlock(64, 64),
            SharedMLPBlock(64, 128),
            SharedMLPBlock(128, self.dim_global_feats)
            )
        # max pool to get the global features
        self.max_pool = nn.MaxPool1d(
            kernel_size=self.num_points, return_indices=True
            )


    def forward(self, x):
        # get batch size
        bs = x.shape[0]
        # pass through first Tnet to get transform matrix
        A_input = self.tnet1(x)
        # perform first transformation across each point in the batch
        x = torch.bmm(x.transpose(2, 1), A_input).transpose(2, 1)
        # pass through first shared MLP
        x = self.smlp1(x)
        # get feature transform
        A_feat = self.tnet2(x)
        # perform second transformation across each (64 dim) feature in the batch
        x = torch.bmm(x.transpose(2, 1), A_feat).transpose(2, 1)
        # store local point features for segmentation head
        local_features = x.clone()
        # pass through second MLP
        x = self.smlp2(x)
        # get global feature vector and critical indexes
        global_features, critical_indices = self.max_pool(x)
        global_features = global_features.view(bs, -1)
        critical_indices = critical_indices.view(bs, -1)
        if self.local_feats:
            combined_features = torch.cat((local_features,
                                  global_features.unsqueeze(-1).repeat(1, 1, self.num_points)),
                                  dim=1)
            return combined_features, global_features, critical_indices
        else:
            return global_features, critical_indices