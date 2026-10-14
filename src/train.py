import yaml
import argparse
import math
import os
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

    accumulation_step = config.get('accum_step', 1)
    total_steps = config['epochs'] * math.ceil(len(train_dataload) / accumulation_step)
    warmup_steps = max(1, int(total_steps * 0.03))
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)

    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps-warmup_steps)

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, main_scheduler], 
        milestones=[warmup_steps] # SWITCH FROM LIENARLR TO COSINE DECAY
    )

    start_time = time.perf_counter()

    best_val_loss = float("inf")
    patience_counter = 0
    scaler = torch.GradScaler(device, enabled=not config.get('bf16'))

    amp_dtype = torch.bfloat16 if config.get('bf16') else torch.float16

    with wandb.init(project=project, name=config['run_name'], config=config) as run:
        
        for epoch in range(config["epochs"]):
            train_losses = []
            temp_losses =[]
            for i, (images, labels) in enumerate(train_dataload):
                images = images.to(device=device)
                labels = labels.to(device=device)

                with torch.autocast(device_type=device, dtype=amp_dtype):
                    logits = model(images)
                    loss = loss_fn(logits, labels)
                scaler.scale(loss / accumulation_step).backward()  

                temp_losses.append(loss.item())

                if (i+1) % accumulation_step == 0 or (i+1) == len(train_dataload):
                    if config.get('grad_clip', False) == True:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                    scheduler.step()

                    avg_loss = sum(temp_losses) / len(temp_losses)
                    train_losses.append(avg_loss)
                    temp_losses = []

            train_loss = sum(train_losses) / len(train_losses)
            run.log({"train_loss": train_loss})

            
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


            # Checkpoint!!! 
            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "patience_counter": patience_counter,
            }
            torch.save(ckpt, "last.tmp")
            os.replace("last.tmp", "last.pt")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(ckpt, "best.tmp")
                os.replace("best.tmp", "best.pt")
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