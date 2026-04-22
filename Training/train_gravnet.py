import argparse
import glob
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GravNetConv
from torch_geometric.nn import EdgeConv
from torch_geometric.nn.pool import knn_graph
from tqdm import tqdm
import os
from datetime import datetime



# --------------------------
# Dataset
# --------------------------

class BestPuppiDataset(Dataset):
    def __init__(self, pattern: str):
        self.files = sorted(glob.glob(pattern))[:500]

        if len(self.files) == 0:
            raise RuntimeError(f"No files matched pattern: {pattern}")

        print("Preloading dataset into RAM...")

        self.data = []
        total = 0

        for f in self.files:
            print(f"Loading {f}...")
            chunk = torch.load(f, map_location="cpu")
            self.data.extend(chunk)
            total += len(chunk)
            print(f"  -> {len(chunk)} graphs (total so far: {total})")

        print(f"Finished loading {total} graphs into memory.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# --------------------------
# Model: try both GravNet and EdgeConv
# --------------------------
class EdgeConvRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, k: int = 8):
        super().__init__()
        self.k = k

        self.input_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.conv1 = EdgeConv(
            nn=nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            ),
            aggr="mean",
        )

        self.conv2 = EdgeConv(
            nn=nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            ),
            aggr="mean",
        )

        self.out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data):
        x, batch = data.x, data.batch

        x = self.input_mlp(x)

        # build kNN graph inside each batch item; turn off self-loop
        edge_index = knn_graph(x, k=self.k, batch=batch, loop=False)
        x = self.conv1(x, edge_index)

        edge_index = knn_graph(x, k=self.k, batch=batch, loop=False)
        x = self.conv2(x, edge_index)

        # one prediction per center graph
        center_mask = data.x[:, -1] > 0.5
        x = x[center_mask]

        return F.softplus(self.out(x)).view(-1)

class GravNetRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64):
        super().__init__()

        self.input_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.conv1 = GravNetConv(hidden_dim, hidden_dim, 4, 16, 12)
        self.conv2 = GravNetConv(hidden_dim, hidden_dim, 4, 16, 12)

        self.out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data):
        x, batch = data.x, data.batch

        x = self.input_mlp(x)
        x = F.relu(self.conv1(x, batch))
        x = F.relu(self.conv2(x, batch))

        # last feature is center flag
        center_mask = data.x[:, -1] > 0.5
        x = x[center_mask]

        return F.softplus(self.out(x)).view(-1)


# --------------------------
# Helpers
# --------------------------
def huber_numpy(pred: np.ndarray, target: np.ndarray, delta: float = 1.0) -> float:
    err = pred - target
    abs_err = np.abs(err)
    return np.mean(np.where(abs_err < delta, 0.5 * err**2, delta * (abs_err - 0.5 * delta)))


def make_splits(n: int, seed: int = 42):
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)

    n_train = int(0.7 * n)
    n_val = int(0.15 * n)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    return train_idx, val_idx, test_idx


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for data in tqdm(loader, desc="Train", leave=False):
        data = data.to(device)

        pred = model(data)
        target = data.y.view(-1)

        # log transform because there's this peak at zero that i wanna smooth out
        target_log = torch.log1p(target)

        loss = F.huber_loss(pred, target_log)
        #loss = F.huber_loss(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    preds, targets = [], []
    event_idx, center_idx = [], []

    for data in tqdm(loader, desc="Eval", leave=False):
        data = data.to(device)

        #pred = model(data)
        pred_log = model(data)
        pred = torch.expm1(pred_log).clamp(min=0.0)

        preds.append(pred.cpu())
        targets.append(data.y.view(-1).cpu())
        event_idx.append(data.event_idx.view(-1).cpu())
        center_idx.append(data.center_idx.view(-1).cpu())

    pred = torch.cat(preds).numpy()
    target = torch.cat(targets).numpy()
    event_idx = torch.cat(event_idx).numpy()
    center_idx = torch.cat(center_idx).numpy()

    return {
        "pred": pred,
        "target": target,
        "event_idx": event_idx,
        "center_idx": center_idx,
    }

def make_run_dir(base="runs"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base, f"run_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

# --------------------------
# Main
# --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help='Glob pattern, e.g. "/eos/.../graphs_*.pt"')
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_dir = make_run_dir()
    print(f"Saving outputs to: {run_dir}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = BestPuppiDataset(args.data)

    train_idx, val_idx, test_idx = make_splits(len(dataset), seed=args.seed)

    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset = torch.utils.data.Subset(dataset, val_idx)
    test_subset = torch.utils.data.Subset(dataset, test_idx)

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    print(next(iter(train_loader)))

    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    sample = dataset[0]
    in_dim = sample.x.shape[1]
    print(f"Input feature dimension: {in_dim}")

    #model = EdgeConvRegressor(in_dim=in_dim, hidden_dim = 64, k = 8).to(device)
    model = GravNetRegressor(in_dim=in_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_losses = []
    val_losses_mse = []
    val_losses_huber = []

    print("Start training!")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)

        val_metrics = evaluate(model, val_loader, device)
        pred_val = val_metrics["pred"]
        target_val = val_metrics["target"]

        val_loss_mse = np.mean((pred_val - target_val) ** 2)
        val_loss_huber = huber_numpy(pred_val, target_val, delta=1.0)

        train_losses.append(train_loss)
        val_losses_mse.append(val_loss_mse)
        val_losses_huber.append(val_loss_huber)

        print(
            f"Epoch {epoch:03d}: "
            f"train_huber={train_loss:.6f} | "
            f"val_mse={val_loss_mse:.6f} | "
            f"val_huber={val_loss_huber:.6f}"
        )

        test_metrics = evaluate(model, test_loader, device)

        torch.save(
            model.state_dict(),
            os.path.join(run_dir, f"model_epoch{epoch:03d}.pt")
        )

        np.savez(
            os.path.join(run_dir, f"outputs_epoch{epoch}.npz"),
            pred=test_metrics["pred"],
            target=test_metrics["target"],
            event_idx=test_metrics["event_idx"],
            center_idx=test_metrics["center_idx"],
        )

        np.savez(
            os.path.join(run_dir, "loss_history.npz"),
            epoch=np.arange(1, epoch + 1),
            train_loss=np.array(train_losses, dtype=np.float32),
            val_loss_mse=np.array(val_losses_mse, dtype=np.float32),
            val_loss_huber=np.array(val_losses_huber, dtype=np.float32),
        )

if __name__ == "__main__":
    main()
