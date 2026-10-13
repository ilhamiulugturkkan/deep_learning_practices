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

def main(config, project):

    device = config.get("device") or (torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu")
    print(f"Using {device} device")


    train_dataload, val_dataload = DATASETS[config['dataset']](batch_size=config.get('batch_size', 64), shuffle = config.get('shuffle', True))
 
    print(config['model_args'])
    model = MODELS[config['model']](**config['model_args']).to(device)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr = config['lr'])
    t_max = config['epochs']
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)

    start_time = time.perf_counter()

    best_val_loss = float("inf")
    patience_counter = 0


    with wandb.init(project=project, name=config['run_name'], config=config) as run:
        
        for epoch in range(config["epochs"]):
            train_losses = []
            for images, labels in train_dataload:
                images = images.to(device=device)
                labels = labels.to(device=device)
                logits = model(images)

                loss = loss_fn(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                train_losses.append(loss.item())
                optimizer.step()
                

            train_loss = sum(train_losses) / len(train_losses)
            run.log({"train_loss": train_loss})


            scheduler.step()

            
            val_losses = []
            accuracies = []
            for images, labels in val_dataload:
                with torch.no_grad():
                    images = images.to(device=device)
                    labels = labels.to(device=device)

                    logits = model(images)
                    loss = loss_fn(logits, labels)
                    val_losses.append(loss.item())
                    preds = torch.argmax(logits, dim=1)
                    accuracies.append(torch.sum(preds==labels).item() / labels.shape[0] * 100.0)

            val_loss = sum(val_losses) / len(val_losses)
            accuracy = sum(accuracies) / len(accuracies)
            run.log({"val_loss": val_loss, 'accuracy': accuracy})
            print({"val_loss": val_loss, 'accuracy': accuracy})

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                model_name = config["run_name"] + ".pt"
                torch.save(model.state_dict(), model_name)   # checkpoint
            else:
                patience_counter += 1
                if patience_counter >= config["patience"]:
                    print(f"Early stopping at epoch {epoch}")
                    break

    end_time = time.perf_counter()

    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.6f} seconds")


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