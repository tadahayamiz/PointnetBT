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
from typing import Tuple
import yaml

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
        data_loader = prep_data(X, train=False)
        latents = []
        crit_indices = []
        for y1, y2 in data_loader:
            z1, ci1 = self.model.get_latent(y1)
            if return_idx:
                latents.append(z1)
                crit_indices.append(ci1)
            else:
                z2, ci2 = self.model.get_latent(y2)
                z = (z1 + z2) / 2
                latents.append(z)
        latents = torch.cat(latents, dim=0).numpy()
        if return_idx:
            crit_indices = torch.cat(crit_indices, dim=0).numpy()
            return latents, crit_indices
        return latents


# hard coding
from collections import Counter
from itertools import chain
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Data:
    """
    data格納モジュール, 基本的にハード

    """
    def __init__(self, input, colors:list=["green", "red"]):
        # 読み込み
        self.data = None
        self.colors = colors
        self.dim = len(colors)
        assert (self.dim > 0) & (self.dim <= 2)
        if type(input) == str:
            self.data = pd.read_csv(input, index_col=0)
        elif type(input) == type(pd.DataFrame()):
            self.data = input
        else:
            raise ValueError("!! Provide url or dataframe !!")
        # データの中身の把握
        col = list(self.data.columns)
        self.dic_components = dict()
        for c in col:
            tmp = self.data[c].values.flatten().tolist()
            self.dic_components[c] = Counter(tmp)


    def conditioned(self, condition:dict):
        """ 解析対象とするデータを条件付けする """
        for k, v in condition.items():
            try:
                self.data = self.data[self.data[k]==v]
            except KeyError:
                raise KeyError("!! Wrong key in condition: check the keys of condition !!")


    def sampling(
        self, sid:int, n_sampling:int=32, n_points:int=256,
        v_name:str="value", s_name:str="sample_id"
        ):
        """
        指定した検体からn_sampleの回の輝点のサンプリングを行う

        Parameters
        ----------
        sid: int
            Specimen ID, 検体をidentifyする

        v_name: str
            valueカラムの名称
            DPPVIの場合は素直にvalue

        Returns
        -------
        a list of sampled data

        """
        tmp = self.data[self.data[s_name]==sid]
        tmp0 = tmp[tmp["color"]==self.colors[0]]
        tmp1 = tmp[tmp["color"]==self.colors[1]]
        common_well = set(tmp0["well_id"]) & set(tmp1["well_id"])
        tmp0 = tmp0[tmp0["well_id"].isin(common_well)]
        tmp1 = tmp1[tmp1["well_id"].isin(common_well)] # sample_idとcolorを絞るとwell_idのサイズに一致
        x0 = tmp0[v_name].values
        x1 = tmp1[v_name].values
        X = np.array([x0, x1]).T
        idx = list(range(n_points))
        rng = np.random.default_rng()
        res = []
        for i in range(n_sampling):
            tmp_idx = idx.copy()
            rng.shuffle(tmp_idx)
            res.append(X[tmp_idx, :])
        return res


    def prep_data(
            self, n_sampling:int=32, n_points:int=256,
            shuffle:bool=True, v_name:str="value", s_name:str="sample_id"
            ):
        """
        指定したsamplesizeまでサンプリングを行う
        dataをlistで返す
        2dなら[2d-array]

        """
        specimens = list(set(list(self.data[s_name])))
        specimens.sort()
        res = []
        specimen = []
        if self.dim == 2:
            for s in tqdm(specimens):
                tmp = self.sampling(s, n_sampling, n_points, v_name, s_name)
                res.append(tmp)
                specimen.append([s] * n_sampling)
        else:
            raise ValueError("!! check |colors|, which should be 1 or 2 !!")
        res = list(chain.from_iterable(res))
        specimen = list(chain.from_iterable(specimen))
        if shuffle:
            rng = np.random.default_rng()
            idx = list(range(len(res)))
            rng.shuffle(idx)
            res = [res[i] for i in idx]
            specimen = [specimen[i] for i in idx]
        res = np.array(res)
        specimen = np.array(specimen)
        return res, specimen


    def imshow(
            self, sid:int, symbol_size:int=8, symbol_alpha:float=0.4,
            linewidths:float=0.0, pixel:tuple=(64, 64), dpi:int=100,
            ratio:float=0.9, condition:dict=dict(),
            outdir:str="", v_name:str="value", s_name:str="sample_id",
            figsize=(), fontsize:int=16
            ):
        """ 指定したIDの画像を表示する """
        # data
        if len(condition) > 0:
            self.conditioned(condition)
        data = self.sampling(sid, 2, ratio, v_name, s_name)[0]
        # show
        if len(figsize)==0:
            figsize = pixel[0] / dpi, pixel[1] / dpi
        fig = plt.figure(figsize=figsize)
        plt.rcParams["font.size"] = fontsize
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(sid)
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        ax.scatter(
            data[:, 0], data[:, 1],
            color="black", s=symbol_size, alpha=symbol_alpha,
            linewidths=linewidths
            )
        ax.set_xlabel(self.colors[0])
        ax.set_ylabel(self.colors[1])
        if len(outdir) > 0:
            plt.savefig(outdir + f"/scatter_{sid}.png")
        plt.show()