import argparse
import glob
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import os
from datetime import datetime


# --------------------------
# Dataset
# --------------------------
class BestPuppiDataset(Dataset):
    def __init__(self, pattern: str):
        files = sorted(glob.glob(pattern))
        random.shuffle(files)
        self.files = files[:500]

        print("Preloading dataset...")
        self.data = []

        for f in self.files:
            chunk = torch.load(f, map_location="cpu")
            self.data.extend(chunk)

        print(f"Loaded {len(self.data)} graphs")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# --------------------------
# Model
# --------------------------
class DeepSetRegressor(nn.Module):
    def __init__(self, in_dim, hidden_dim=256):
        super().__init__()

        self.phi = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        self.rho = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        #self.scale = nn.Parameter(torch.tensor(1.0))
        #self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, data):
        x, batch = data.x, data.batch

        h = self.phi(x)

        # sum pooling
        h_sum = torch.zeros(
            batch.max() + 1,
            h.size(1),
            device=h.device
        )
        h_sum = h_sum.index_add(0, batch, h)

        # mean pooling
        counts = torch.bincount(batch)
        h_mean = h_sum / counts.unsqueeze(1)

        # center particle
        center_mask = data.x[:, -1] > 0.5
        h_center = h[center_mask]

        h_out = torch.cat([h_sum, h_mean, h_center], dim=1)

        return self.rho(h_out).view(-1)


# --------------------------
# Helpers
# --------------------------
def huber_numpy(pred, target, delta=1.0):
    err = pred - target
    abs_err = np.abs(err)
    return np.mean(
        np.where(abs_err < delta, 0.5 * err**2, delta * (abs_err - 0.5 * delta))
    )


def make_splits(n, seed=42):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)

    n_train = int(0.7 * n)
    n_val = int(0.15 * n)

    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]


def make_run_dir():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"runs/run_{ts}"
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


# --------------------------
# Train
# --------------------------
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0

    for data in tqdm(loader, desc="Train", leave=False):
        data = data.to(device)

        pred = F.softplus(model(data)) # enforce positive output
        target = data.y.view(-1)

        # huber loss
        loss_abs = F.huber_loss(pred, target, reduction="none")

        # penalize if pred deviates from target (underprediction & overprediction penalized equally)
        eps = 1e-3
        
        log_ratio = torch.log(pred + eps) - torch.log(target + eps)
        loss_ratio = log_ratio**2
        
        # add up loss terms; alpha controls the penalty term strength 
        nonzero_mask = (target > 0.05).float()
        alpha = 0.5

        loss = loss_abs + alpha * loss_ratio * nonzero_mask  # only apply it to non-zero target case; otherwise things are a bit skewed? boh

        #bias_loss = (pred.mean() - target.mean().detach())**2 # if things shift globally, this terms seems to help? but boh
        #loss += bias_loss
 
        loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# --------------------------
# Eval
# --------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    preds, targets = [], []
    event_idx, center_idx = [], []

    for data in tqdm(loader, desc="Eval", leave=False):
        data = data.to(device)

        pred = F.softplus(model(data))

        preds.append(pred.cpu())
        targets.append(data.y.view(-1).cpu())
        event_idx.append(data.event_idx.view(-1).cpu())
        center_idx.append(data.center_idx.view(-1).cpu())

    return {
        "pred": torch.cat(preds).numpy(),
        "target": torch.cat(targets).numpy(),
        "event_idx": torch.cat(event_idx).numpy(),
        "center_idx": torch.cat(center_idx).numpy(),
    }


# --------------------------
# Main
# --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    run_dir = make_run_dir()
    print("Run dir:", run_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = BestPuppiDataset(args.data)

    train_idx, val_idx, test_idx = make_splits(len(dataset))

    train_loader = DataLoader(
        torch.utils.data.Subset(dataset, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        torch.utils.data.Subset(dataset, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
    )

    test_loader = DataLoader(
        torch.utils.data.Subset(dataset, test_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
    )

    in_dim = dataset[0].x.shape[1]
    model = DeepSetRegressor(in_dim).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_losses, val_mse, val_huber = [], [], []

    print("Start training!")

    for epoch in range(1, args.epochs + 1):

        train_loss = train_epoch(model, train_loader, optimizer, device)

        val_metrics = evaluate(model, val_loader, device)

        pred = val_metrics["pred"]
        target = val_metrics["target"]

        mse = np.mean((pred - target) ** 2)
        hub = huber_numpy(pred, target)

        train_losses.append(train_loss)
        val_mse.append(mse)
        val_huber.append(hub)

        print(
            f"Epoch {epoch:03d} | "
            f"train={train_loss:.5f} | "
            f"val_mse={mse:.5f} | "
            f"val_huber={hub:.5f}"
        )

        test_metrics = evaluate(model, test_loader, device)

        torch.save(model.state_dict(),
                   os.path.join(run_dir, f"model_epoch{epoch}.pt"))

        np.savez(
            os.path.join(run_dir, f"outputs_epoch{epoch}.npz"),
            **test_metrics
        )

        np.savez(
            os.path.join(run_dir, "loss_history.npz"),
            epoch=np.arange(1, epoch + 1),
            train_loss=np.array(train_losses),
            val_loss_mse=np.array(val_mse),
            val_loss_huber=np.array(val_huber),
        )


if __name__ == "__main__":
    main()

