# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:09:08 2019

utils

@author: tadahaya
"""
import json, os, math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import datetime


# save and load functions
def save_experiment(
        experiment_name, base_dir, config, model,
        train_losses, test_losses, accuracies
        ):
    """ save the experiment: config, model, metrics, and progress plot """
    outdir = os.path.join(base_dir, experiment_name)
    os.makedirs(outdir, exist_ok=True)

    # save config
    now = datetime.datetime.now()
    config["timestamp"] = now.strftime('%Y-%m-%d %H:%M:%S')
    configfile = os.path.join(outdir, 'config.json')
    with open(configfile, 'w') as f:
        json.dump(config, f, sort_keys=True, indent=4)
    
    # save metrics
    jsonfile = os.path.join(outdir, 'metrics.json')
    with open(jsonfile, 'w') as f:
        data = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'accuracies': accuracies,
        }
        json.dump(data, f, sort_keys=True, indent=4)

    # plot progress
    plot_progress(
        experiment_name, train_losses, test_losses, config["epochs"], base_dir=base_dir
        )

    # save the model
    save_checkpoint(experiment_name, model, "final", base_dir=base_dir)


def save_checkpoint(experiment_name, model, epoch, base_dir):
    """
    save the model
    
    # Trainerの中で走る
    
    """
    outdir = os.path.join(base_dir, experiment_name)
    os.makedirs(outdir, exist_ok=True)
    cpfile = os.path.join(outdir, f"model_{epoch}.pt")
    torch.save(model.state_dict(), cpfile)


def load_experiments(
        model, experiment_name, base_dir="experiments", checkpoint_name="model_final.pt"
        ):
    outdir = os.path.join(base_dir, experiment_name)
    # load config
    configfile = os.path.join(outdir, "config.json")
    with open(configfile, 'r') as f:
        config = json.load(f)
    # load metrics
    jsonfile = os.path.join(outdir, 'metrics.json')
    with open(jsonfile, 'r') as f:
        data = json.load(f)
    train_losses = data["train_losses"]
    test_losses = data["test_losses"]
    accuracies = data["accuracies"]
    # load model
    model = model(config)
    cpfile = os.path.join(outdir, checkpoint_name)
    model.load_state_dict(torch.load(cpfile)) # checkpointを読み込んでから
    return config, model, train_losses, test_losses, accuracies


# model preparation functions
def make_optimizer(params, name, **kwargs):
    """
    make optimizer
    
    Parameters
    ----------
    params: model.parameters()
    
    name: str
        optimizer name to be used
        - "Adam"
        - "SGD"
        - "RMSprop"
        - "Adadelta"
        - "Adagrad"
        - "AdamW"
        - "SparseAdam"
        - "Adamax"
        - "ASGD"
        - "LBFGS"
        - "Rprop"
    
    """
    return optim.__dict__[name](params, **kwargs)


def make_loss_fn(name, **kwargs):
    """
    make loss function
    
    Parameters
    ----------
    name: str
        loss function name to be used
        - "CrossEntropyLoss"
        - "MSELoss"
        - "NLLLoss"
        - "PoissonNLLLoss"
        - "KLDivLoss"
        - "BCELoss"
        - "BCEWithLogitsLoss"
        - "MarginRankingLoss"
        - "HingeEmbeddingLoss"
        - "MultiLabelMarginLoss"
        - "SmoothL1Loss"
        - "SoftMarginLoss"
        - "MultiLabelSoftMarginLoss"
        - "CosineEmbeddingLoss"
        - "MultiMarginLoss"
        - "TripletMarginLoss"
        - "CTCLoss"
        - "NLLLoss2d"
        - "PoissonNLLLoss"
        - "KLDivLoss"

    """
    return nn.__dict__[name](**kwargs)


# visualization functions
def plot_progress(
        experiment_name:str, train_loss:list, test_loss:list, num_epoch:int,
        base_dir:str="experiments", xlabel="epoch", ylabel="loss"
        ):
    """ plot learning progress """
    outdir = os.path.join(base_dir, experiment_name)
    os.makedirs(outdir, exist_ok=True)
    epochs = list(range(1, num_epoch + 1, 1))
    fig, ax = plt.subplots()
    plt.rcParams['font.size'] = 14
    ax.plot(epochs, train_loss, c='navy', label='train')
    ax.plot(epochs, test_loss, c='darkgoldenrod', label='test')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid()
    ax.legend()
    plt.tight_layout()
    plt.savefig(outdir + f'/progress_{ylabel}.tif', dpi=300, bbox_inches='tight')


def visualize_images(mydataset, indices:list=[], output:str="", nrow:int=3, ncol:int=4):
    """
    visualize the images in the given dataset
    
    """
    # indicesの準備
    assert len(indices) <= len(mydataset), "!! indices should be less than the total number of images !!"
    num_vis = np.min((len(mydataset), nrow * ncol))
    if len(indices) == 0:
        indices = torch.randperm(len(mydataset))[:num_vis]
    else:
        num_vis = len(indices)
    classes = mydataset.classes
    images = [np.asarray(mydataset[i][0]) for i in indices]
    labels = [mydataset[i][1] for i in indices]
    # 描画
    fig = plt.figure()
    for i in range(num_vis):
        ax = fig.add_subplot(nrow, ncol, i+1, xticks=[], yticks=[])
        ax.imshow(images[i])
        ax.set_title(classes[labels[i]])
    plt.tight_layout()
    if len(output) > 0:
        plt.savefig(output, dpi=300, bbox_inches='tight')