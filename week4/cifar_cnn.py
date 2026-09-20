import yaml
import argparse
import torch
from torch import nn
import torch.nn.functional as T
import matplotlib.pyplot as plt
import numpy as np
import wandb
import time
from src.models import MODELS
from src.data import DATASETS
from src.train import main as train

def main(config, project):
    train(config, project)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--device', default=None)  # CLI override
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.device:
        config['device'] = args.device

    wandb.login()
    torch.manual_seed(config['seed'])
    main(config, config['project'])