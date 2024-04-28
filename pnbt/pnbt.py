# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:09:08 2019

ihvit module

@author: tadahaya
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import yaml
from tqdm.auto import tqdm

from .src.backbone import PointNetBackbone
from .src.ssl import BarlowTwins
from .src.utils import save_experiment, load_experiment
from .src.trainer import Trainer
from .src.data_handler import prep_data


class PointNetBT:
    def __init__(self, config=None, config_path=None):
        # config
        if config is None:
            if config_path is not None:
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)
            else:
                config = dict()
        default_config = {
            # backbone config
            "input_dim": 3,
            "num_points": 256,
            "dim_global_feats": 128, # 1024
            "local_feats": False,
            "dropout_ratio": 0.3,
            # ssl config
            "lambd": 1e-3,
            "projection_sizes": [128, 256, 128], # [1024, 1024, 1024, 1024]
            "scale_factor": 1.0,
            # trainer config
            "exp_name": "experiment",
            "base_dir": None,
            "epochs": 20,
            "batch_size": 64,
            "save_model_every": 10,
            "optimizer": {
                "name": "AdamW",
                "lr": 1e-3,
                "weight_decay": 1e-2,
            }
        }
        self.config = {**default_config, **config}
        # model
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        bb = PointNetBackbone(self.config)
        self.model = BarlowTwins(bb, self.config)
        self.trainer = Trainer(self.model, self.config)


    def prep_data(self, x_train, x_test=None, train=True):
        """
        data preparation
        
        Parameters
        ----------
        x_train: np.array
            training data or all data, (batch_size, num_points, input_dim)
        
        """
        train_loader, test_loader = prep_data(
            x_train, x_test=x_test, num_points=self.config["num_points"],
            batch_size=self.config["batch_size"], train=train
            )
        return train_loader, test_loader


    def fit(self, train_loader, test_loader):
        """ training """
        # training
        train_losses, test_losses, accuracies = self.trainer.train(train_loader, test_loader)
        # save experiment
        save_experiment(
            self.config["exp_name"], self.config["base_dir"], self.config,
            self.model, train_losses, test_losses, accuracies
            )


    def load_model(self, exp_name, base_dir):
        """
        load model
        
        Parameters
        ----------
        exp_name: str
            experiment name
        
        base_dir: str
            base directory path
        
        """
        self.config, cpfile, _, _, _ = load_experiment(
            exp_name, base_dir
            )
        bb = PointNetBackbone(self.config)
        self.model = BarlowTwins(bb, self.config)
        self.model.load_state_dict(torch.load(cpfile))
        self.trainer = Trainer(self.model, self.config)


    def get_latent(self, X, return_idx=False):
        """
        get latent features, return numpy array
        
        Parameters
        ----------
        X: np.array
            input data, (batch_size, num_points, input_dim)
                
        Returns
        -------
        latent: np.array
            latent features
        
        """
        # data loading
        data_loader, _ = prep_data(X, train=False)
        latents = []
        crit_indices = []
        for y1, y2 in tqdm(data_loader):
            y1, y2 = y1.to(self.device), y2.to(self.device)
            z1, ci1 = self.model.get_latent(y1)
            if return_idx:
                latents.append(z1)
                crit_indices.append(ci1)
            else:
                z2, ci2 = self.model.get_latent(y2)
                z = (z1 + z2) / 2
                latents.append(z)
        latents = torch.cat(latents, dim=0).cpu().numpy()
        if return_idx:
            crit_indices = torch.cat(crit_indices, dim=0).numpy()
            return latents, crit_indices
        return latents