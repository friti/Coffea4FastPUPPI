import argparse
import glob
import json
import math
import os
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# Dataset can contain a superset of features, but when we run the training some features can be commented out.
INPUT_FEATURES = [
    "log_pt",
    "slog_px",
    "slog_py",
    "abs_eta",
    "charge",
    "nnvtx",
    # "vz",
    "dxy",
    "z0",
    "puppiWeight",
    # "idProbPu",
    # "idProbEm",
    # "idProbPi",
    "pdg_abs_211",
    "pdg_abs_130",
    "pdg_abs_22",
    "pdg_abs_11",
    "pdg_abs_13",
    "deta",
    "dphi",
    "dr",
    "is_center",
]


class BestPuppiDataset(Dataset):
    def __init__(self, pattern: str, max_files=10):
        files = sorted(glob.glob(pattern))
        random.shuffle(files)
        self.files = files[:max_files] # at the moment each dataset contains 50k neutrals, so 500k in total if 10 files

        print("len(self.files)", len(self.files))
        print("Preloading dataset...")

        self.data = []
        self.metadata = None
        self.feature_names = None

        for f in self.files:
            t0 = time.time()
            payload = torch.load(f, map_location="cpu", weights_only=False)

            if isinstance(payload, dict) and "graphs" in payload:
                graphs = payload["graphs"]
                metadata = payload.get("metadata", {})
                feature_names = metadata.get("feature_names", None)

                if self.metadata is None:
                    self.metadata = metadata
                    self.feature_names = feature_names
                elif feature_names != self.feature_names: # some protection boh
                    raise RuntimeError(
                        f"Feature-name mismatch in {f}\n"
                        f"existing: {self.feature_names}\n"
                        f"new     : {feature_names}"
                    )
            else:
                graphs = payload
                metadata = {}
                feature_names = None

            self.data.extend(graphs)
            print(f"Loaded {f}: {len(graphs)} graphs in {time.time() - t0:.1f} s")

        print(f"Loaded {len(self.data)} graphs from {len(self.files)} files") # check if there's some I/O bottleneck bc sometimes it takes forever

        if self.feature_names is None:
            in_dim = self.data[0].x.shape[1]
            self.feature_names = default_feature_names(in_dim)
            self.metadata = {
                "feature_names": self.feature_names,
                "note": "Inferred feature names from old-format dataset.",
            }

        print("Dataset feature names:")
        for i, name in enumerate(self.feature_names):
            print(f"  {i:2d}: {name}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DeepSetRegressor(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, is_center_idx=-1):
        super().__init__()
        self.is_center_idx = is_center_idx

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

    def forward(self, data):
        x, batch = data.x, data.batch
        h = self.phi(x)

        num_graphs = int(batch.max().item()) + 1

        h_sum = torch.zeros(num_graphs, h.size(1), device=h.device, dtype=h.dtype)
        h_sum = h_sum.index_add(0, batch, h)

        counts = torch.bincount(batch, minlength=num_graphs).to(h.dtype)
        h_mean = h_sum / counts.unsqueeze(1).clamp(min=1.0)

        center_mask = x[:, self.is_center_idx] > 0.5
        num_centers = torch.bincount(batch[center_mask], minlength=num_graphs)

        if not torch.all(num_centers == 1):
            raise RuntimeError(f"Expected exactly one center per graph, got {num_centers}")

        h_center = h[center_mask]
        h_out = torch.cat([h_sum, h_mean, h_center], dim=1)

        return self.rho(h_out).view(-1)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def apply_output_activation(raw, activation="softplus"):
    if activation == "relu":
        return F.relu(raw)
    if activation == "softplus":
        return F.softplus(raw)
    if activation == "identity":
        return raw
    raise ValueError(f"Unknown output activation: {activation}")


def default_feature_names(in_dim):
    # if working with "old dataset" that doesn't contain metadata of feature names in the dataset; just infer boh
    if in_dim == 15:
        return [
            "log_pt", "slog_px", "slog_py", "charge", "nnvtx", "puppiWeight",
            "pdg_abs_211", "pdg_abs_130", "pdg_abs_22", "pdg_abs_11", "pdg_abs_13",
            "deta", "dphi", "dr", "is_center",
        ]

    if in_dim == 16:
        return [
            "log_pt", "slog_px", "slog_py", "abs_eta", "charge", "nnvtx", "puppiWeight",
            "pdg_abs_211", "pdg_abs_130", "pdg_abs_22", "pdg_abs_11", "pdg_abs_13",
            "deta", "dphi", "dr", "is_center",
        ]

    return [f"feature_{i}" for i in range(in_dim)]


def build_feature_indices(dataset_feature_names, input_features):
    feature_to_idx = {name: i for i, name in enumerate(dataset_feature_names)}

    missing = [f for f in input_features if f not in feature_to_idx]
    if missing:
        raise RuntimeError(
            "Requested input features are missing from dataset metadata:\n"
            f"missing = {missing}\n"
            f"available = {dataset_feature_names}"
        )

    return [feature_to_idx[f] for f in input_features]


def select_features(data, feature_idx):
    data.x = data.x[:, feature_idx]
    return data


def global_shift_penalty(pred): # doesn't seem to work :( 
    return pred.mean()


def mse_penalty(pred, target):  # log ratio penalty seems to work much better than this
    return torch.mean((pred - target) ** 2)


# Tried bunch of different log ratio penalty term to see if the "global shift" goes away
def centered_log_ratio_penalty(pred, target, eps=1e-3, zero_thr=0.05):
    pos_mask = target > zero_thr
    loss_ratio = torch.zeros_like(target)

    if pos_mask.sum() == 0:
        return loss_ratio

    log_ratio = torch.log(pred.clamp_min(0.0) + eps) - torch.log(target + eps)
    pos_log_ratio = log_ratio[pos_mask]
    centered = pos_log_ratio - pos_log_ratio.mean()

    loss_ratio[pos_mask] = centered.pow(2)
    return loss_ratio


def under_log_ratio_penalty(pred, target, eps=1e-3, zero_thr=0.05):
    pos_mask = target > zero_thr
    loss_ratio = torch.zeros_like(target)

    if pos_mask.sum() == 0:
        return loss_ratio

    log_ratio = torch.log(pred.clamp_min(0.0) + eps) - torch.log(target + eps)
    loss_ratio[pos_mask] = F.relu(-log_ratio[pos_mask]).pow(2)

    return loss_ratio


def selective_under_log_penalty(pred, target, *, eps=1e-3, target_thr=0.05, ratio_floor=0.5):
    pos_mask = target > target_thr

    if pos_mask.sum() == 0:
        return pred.new_tensor(0.0)

    pred_pos = pred[pos_mask].clamp_min(0.0)
    target_pos = target[pos_mask]

    log_ratio = torch.log(pred_pos + eps) - torch.log(target_pos + eps)
    margin = math.log(ratio_floor)
    under_violation = F.relu(margin - log_ratio)

    return under_violation.pow(2).mean()

# Main loss function
def puppi_loss(
    pred,
    target,
    loss_type="weighted_huber",
    alpha=1.0,                 # this controls the log-ratio penalty term
    mse_penalty_alpha=0.0,     # if using MSE penalty term
    global_shift_alpha=0.0,    # if using "global shift" penalty term in addition to log ratio term 
    eps=1e-3,                  # some tiny value to avoid 0 in the denominator
    zero_thr=0.05,             # a small threshold to decide what to consider zero-target or not
    raw=None,
    pos_weight=9.0,            # a value that controls the weighted huber (flatten out by target distribution)
    pos_weight_threshold=0.05, # what to consider zero / non-zero
):
    if loss_type == "huber":
        loss_abs = F.huber_loss(pred, target, reduction="none")

    elif loss_type == "weighted_huber": # this is the thing that works the best for now
        loss_abs = F.huber_loss(pred, target, reduction="none")

        weight = torch.ones_like(target)
        pos_mask = target > pos_weight_threshold
        weight[pos_mask] = pos_weight

        loss_abs = loss_abs * weight

    elif loss_type == "mse":
        loss_abs = F.mse_loss(pred, target, reduction="none")

    elif loss_type == "mae":
        loss_abs = F.l1_loss(pred, target, reduction="none")

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    loss = loss_abs.mean()

    if mse_penalty_alpha > 0.0: # this didn't seem to work well
        loss = loss + mse_penalty_alpha * torch.mean((pred - target) ** 2)

    # what works best for now : plain symmetric log-ratio MSE on positive targets
    if alpha > 0.0:
        pos_mask = target > zero_thr

        if torch.any(pos_mask):
            pred_pos = pred[pos_mask].clamp_min(eps)
            target_pos = target[pos_mask].clamp_min(eps)

            log_ratio = torch.log(pred_pos) - torch.log(target_pos)
            loss_logratio = torch.mean(log_ratio ** 2)

            loss = loss + alpha * loss_logratio

    # Kept for future messing-around, disabled by default
    if global_shift_alpha > 0.0:
        loss = loss + global_shift_alpha * global_shift_penalty(pred)

    return loss


def compute_diagnostic_metrics(pred, target, eps=1e-3, zero_thr=0.05, pred_thr=0.05):
    pred = np.asarray(pred)
    target = np.asarray(target)

    err = pred - target

    metrics = {}
    metrics["mse"] = float(np.mean(err ** 2))
    metrics["mae"] = float(np.mean(np.abs(err)))

    abs_err = np.abs(err)
    metrics["huber"] = float(np.mean(np.where(abs_err < 1.0, 0.5 * err**2, abs_err - 0.5)))

    zero_mask = target <= zero_thr
    pos_mask = target > zero_thr

    metrics["target_zero_frac"] = float(np.mean(zero_mask))
    metrics["target_pos_frac"] = float(np.mean(pos_mask))
    metrics["pred_mean"] = float(np.mean(pred))
    metrics["target_mean"] = float(np.mean(target))
    metrics["global_bias"] = float(np.mean(pred - target))

    if np.any(zero_mask):
        metrics["zero_mean_pred"] = float(np.mean(pred[zero_mask]))
        metrics["zero_median_pred"] = float(np.median(pred[zero_mask]))
        metrics["zero_nonzero_pred_frac"] = float(np.mean(pred[zero_mask] > pred_thr))
    else:
        metrics["zero_mean_pred"] = np.nan
        metrics["zero_median_pred"] = np.nan
        metrics["zero_nonzero_pred_frac"] = np.nan

    if np.any(pos_mask):
        log_ratio = np.log(pred[pos_mask] + eps) - np.log(target[pos_mask] + eps)
        ratio = (pred[pos_mask] + eps) / (target[pos_mask] + eps)

        metrics["pos_mae"] = float(np.mean(np.abs(pred[pos_mask] - target[pos_mask])))
        metrics["pos_bias"] = float(np.mean(pred[pos_mask] - target[pos_mask]))
        metrics["pos_logratio_mean"] = float(np.mean(log_ratio))
        metrics["pos_logratio_rms"] = float(np.sqrt(np.mean(log_ratio ** 2)))
        metrics["pos_ratio_median"] = float(np.median(ratio))
        metrics["pos_under_zero_frac"] = float(np.mean(pred[pos_mask] <= pred_thr))
    else:
        metrics["pos_mae"] = np.nan
        metrics["pos_bias"] = np.nan
        metrics["pos_logratio_mean"] = np.nan
        metrics["pos_logratio_rms"] = np.nan
        metrics["pos_ratio_median"] = np.nan
        metrics["pos_under_zero_frac"] = np.nan

    return metrics


def extract_center_feature(data, feature_idx, is_center_idx):
    x, batch = data.x, data.batch
    center_mask = x[:, is_center_idx] > 0.5

    num_graphs = int(batch.max().item()) + 1
    num_centers = torch.bincount(batch[center_mask], minlength=num_graphs)

    if not torch.all(num_centers == 1):
        raise RuntimeError(f"Expected exactly one center per graph, got {num_centers}")

    return x[center_mask, feature_idx]


def extract_center_pt(data, feature_name_to_idx):
    log_pt = extract_center_feature(
        data,
        feature_idx=feature_name_to_idx["log_pt"],
        is_center_idx=feature_name_to_idx["is_center"],
    )
    return torch.exp(log_pt)


def extract_center_eta(data, feature_name_to_idx):
    return extract_center_feature(
        data,
        feature_idx=feature_name_to_idx["abs_eta"],
        is_center_idx=feature_name_to_idx["is_center"],
    )


def train_epoch(model, loader, optimizer, device, args, feature_idx):
    model.train()
    total_loss = 0.0

    for data in tqdm(loader, desc="Train", leave=False):
        data = data.to(device)
        data = select_features(data, feature_idx)

        raw = model(data)
        pred = apply_output_activation(raw, args.output_activation)
        target = data.y.view(-1)

        loss = puppi_loss(
            pred,
            target,
            loss_type=args.loss_type,
            alpha=args.penalty_alpha,
            mse_penalty_alpha=args.mse_penalty_alpha,
            global_shift_alpha=args.global_shift_alpha,
            eps=args.penalty_eps,
            zero_thr=args.penalty_zero_thr,
            raw=raw,
            pos_weight=args.pos_weight,
            pos_weight_threshold=args.pos_weight_threshold,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device, args, feature_idx, input_feature_name_to_idx):
    model.eval()

    preds, raws, targets = [], [], []
    event_idx, center_idx, center_pt, center_eta = [], [], [], []
    losses = []

    for data in tqdm(loader, desc="Eval", leave=False):
        data = data.to(device)
        data = select_features(data, feature_idx)

        raw = model(data)
        pred = apply_output_activation(raw, args.output_activation)

        pred_for_metrics = pred.clamp_min(0.0) if args.eval_clip_nonnegative else pred
        target = data.y.view(-1)

        loss = puppi_loss(
            pred,
            target,
            loss_type=args.loss_type,
            alpha=args.penalty_alpha,
            mse_penalty_alpha=args.mse_penalty_alpha,
            global_shift_alpha=args.global_shift_alpha,
            eps=args.penalty_eps,
            zero_thr=args.penalty_zero_thr,
            raw=raw,
            pos_weight=args.pos_weight,
            pos_weight_threshold=args.pos_weight_threshold,
        )

        losses.append(loss.item())

        raws.append(raw.cpu())
        preds.append(pred_for_metrics.cpu())
        targets.append(target.cpu())

        event_idx.append(data.event_idx.view(-1).cpu())
        center_idx.append(data.center_idx.view(-1).cpu())
        center_pt.append(extract_center_pt(data, input_feature_name_to_idx).cpu())

        if "abs_eta" in input_feature_name_to_idx:
            center_eta.append(extract_center_eta(data, input_feature_name_to_idx).cpu())

    out = {
        "loss": float(np.mean(losses)),
        "raw": torch.cat(raws).numpy(),
        "pred": torch.cat(preds).numpy(),
        "target": torch.cat(targets).numpy(),
        "event_idx": torch.cat(event_idx).numpy(),
        "center_idx": torch.cat(center_idx).numpy(),
        "center_pt": torch.cat(center_pt).numpy(),
    }

    if len(center_eta) > 0:
        out["center_eta"] = torch.cat(center_eta).numpy()

    return out


@torch.no_grad()
def permutation_feature_importance(
    model,
    loader,
    device,
    args,
    feature_idx,
    input_feature_names,
    max_batches=20,
): # SHAF feature importance expects fixed number of features (i.e. same number of neutrals in a cone) so just use this instead
    model.eval()

    baseline_losses = []
    cached_batches = []

    for ibatch, data in enumerate(loader):
        if ibatch >= max_batches:
            break

        cached_batches.append(data)

        data = data.to(device)
        data = select_features(data, feature_idx)

        raw = model(data)
        pred = apply_output_activation(raw, args.output_activation)
        target = data.y.view(-1)

        loss = puppi_loss(
            pred,
            target,
            loss_type=args.loss_type,
            alpha=args.penalty_alpha,
            mse_penalty_alpha=args.mse_penalty_alpha,
            global_shift_alpha=args.global_shift_alpha,
            eps=args.penalty_eps,
            zero_thr=args.penalty_zero_thr,
            raw=raw,
            pos_weight=args.pos_weight,
            pos_weight_threshold=args.pos_weight_threshold,
        )

        baseline_losses.append(loss.item())

    baseline_loss = float(np.mean(baseline_losses))

    results = []
    protected_features = {"is_center"}

    for ifeat, name in enumerate(input_feature_names):
        if name in protected_features:
            results.append({
                "feature_index": ifeat,
                "feature": name,
                "baseline_loss": baseline_loss,
                "permuted_loss": np.nan,
                "delta_loss": np.nan,
                "note": "Skipped because permuting this feature breaks graph structure.",
            })
            continue

        perm_losses = []

        for data_cpu in cached_batches:
            data = data_cpu.clone()
            data.x = data.x.clone()
            data = select_features(data, feature_idx)

            perm = torch.randperm(data.x.size(0))
            data.x[:, ifeat] = data.x[perm, ifeat]

            data = data.to(device)

            raw = model(data)
            pred = apply_output_activation(raw, args.output_activation)
            target = data.y.view(-1)

            loss = puppi_loss(
                pred,
                target,
                loss_type=args.loss_type,
                alpha=args.penalty_alpha,
                mse_penalty_alpha=args.mse_penalty_alpha,
                global_shift_alpha=args.global_shift_alpha,
                eps=args.penalty_eps,
                zero_thr=args.penalty_zero_thr,
                raw=raw,
                pos_weight=args.pos_weight,
                pos_weight_threshold=args.pos_weight_threshold,
            )

            perm_losses.append(loss.item())

        perm_loss = float(np.mean(perm_losses))

        results.append({
            "feature_index": ifeat,
            "feature": name,
            "baseline_loss": baseline_loss,
            "permuted_loss": perm_loss,
            "delta_loss": perm_loss - baseline_loss,
        })

    return sorted(results, key=lambda x: x["delta_loss"], reverse=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_files", type=int, default=10)
    parser.add_argument("--hidden_dim", type=int, default=256)

    parser.add_argument("--output_activation", choices=["relu", "softplus", "identity"], default="softplus")
    parser.add_argument("--loss_type", choices=["huber", "mse", "mae", "weighted_huber"], default="weighted_huber")

    parser.add_argument("--pos_weight", type=float, default=9.0)
    parser.add_argument("--pos_weight_threshold", type=float, default=0.05)

    parser.add_argument("--penalty_alpha", type=float, default=1.0)
    parser.add_argument("--penalty_eps", type=float, default=1e-3)
    parser.add_argument("--penalty_zero_thr", type=float, default=0.05)

    parser.add_argument("--negative_penalty_beta", type=float, default=0.0)
    parser.add_argument("--mse_penalty_alpha", type=float, default=0.0)
    parser.add_argument("--global_shift_alpha", type=float, default=0.0)

    parser.add_argument("--feature_importance", action="store_true")
    parser.add_argument("--feature_importance_batches", type=int, default=20)
    parser.add_argument("--eval_clip_nonnegative", action="store_true")

    args = parser.parse_args()

    set_seed(args.seed)

    run_dir = make_run_dir()
    print("Run dir:", run_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = BestPuppiDataset(args.data, max_files=args.max_files)

    dataset_feature_names = dataset.feature_names
    feature_idx = build_feature_indices(dataset_feature_names, INPUT_FEATURES)
    input_feature_name_to_idx = {name: i for i, name in enumerate(INPUT_FEATURES)}

    print("Training input features:")
    for i, name in enumerate(INPUT_FEATURES):
        print(f"  {i:2d}: {name}  dataset_idx={feature_idx[i]}")

    train_idx, val_idx, test_idx = make_splits(len(dataset), seed=args.seed)

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

    model = DeepSetRegressor(
        in_dim=len(INPUT_FEATURES),
        hidden_dim=args.hidden_dim,
        is_center_idx=input_feature_name_to_idx["is_center"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    config = vars(args).copy()
    config["dataset_metadata"] = dataset.metadata
    config["dataset_feature_names"] = dataset_feature_names
    config["input_features"] = INPUT_FEATURES
    config["feature_idx"] = feature_idx
    config["in_dim"] = len(INPUT_FEATURES)
    config["model"] = "DeepSetRegressor_sum_mean_center"
    config["prediction"] = f"{args.output_activation}(raw)"
    config["baseline"] = "softplus + weighted_huber(pos_weight=9) + plain log-ratio penalty(alpha=1)"
    config["train_size"] = len(train_idx)
    config["val_size"] = len(val_idx)
    config["test_size"] = len(test_idx)

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    history = {"epoch": [], "train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_epoch = -1

    print("Start training!")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, args, feature_idx)

        val_out = evaluate(model, val_loader, device, args, feature_idx, input_feature_name_to_idx)
        val_loss = val_out["loss"]

        diag = compute_diagnostic_metrics(
            val_out["pred"],
            val_out["target"],
            eps=args.penalty_eps,
            zero_thr=args.penalty_zero_thr,
            pred_thr=args.pos_weight_threshold,
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        for k, v in diag.items():
            history.setdefault(f"val_{k}", []).append(v)

        print(
            f"Epoch {epoch:03d} | "
            f"train={train_loss:.5f} | "
            f"val={val_loss:.5f} | "
            f"val_mse={diag['mse']:.5f} | "
            f"val_huber={diag['huber']:.5f} | "
            f"zero_mean_pred={diag['zero_mean_pred']:.5f} | "
            f"zero_nonzero_frac={diag['zero_nonzero_pred_frac']:.4f} | "
            f"pos_logR_rms={diag['pos_logratio_rms']:.5f}"
        )

        np.savez(os.path.join(run_dir, f"val_outputs_epoch{epoch}.npz"), **val_out)
        np.savez(os.path.join(run_dir, "loss_history.npz"), **{k: np.array(v) for k, v in history.items()})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch

            torch.save(model.state_dict(), os.path.join(run_dir, "model_best.pt"))
            np.savez(os.path.join(run_dir, "val_outputs_best.npz"), **val_out)

        torch.save(model.state_dict(), os.path.join(run_dir, f"model_epoch{epoch}.pt"))

    print(f"Best epoch: {best_epoch} with val_loss={best_val_loss:.6f}")

    best_path = os.path.join(run_dir, "model_best.pt")
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))

    test_out = evaluate(model, test_loader, device, args, feature_idx, input_feature_name_to_idx)

    test_diag = compute_diagnostic_metrics(
        test_out["pred"],
        test_out["target"],
        eps=args.penalty_eps,
        zero_thr=args.penalty_zero_thr,
        pred_thr=args.pos_weight_threshold,
    )

    np.savez(os.path.join(run_dir, "test_outputs_best.npz"), **test_out)

    with open(os.path.join(run_dir, "test_metrics_best.json"), "w") as f:
        json.dump(test_diag, f, indent=2)

    print("Test metrics from best model:")
    for k, v in test_diag.items():
        print(f"  {k}: {v}")

    if args.feature_importance:
        print("Running permutation feature importance...")

        fi = permutation_feature_importance(
            model,
            val_loader,
            device,
            args,
            feature_idx=feature_idx,
            input_feature_names=INPUT_FEATURES,
            max_batches=args.feature_importance_batches,
        )

        with open(os.path.join(run_dir, "feature_importance.json"), "w") as f:
            json.dump(fi, f, indent=2)

        print("Top feature importances:")
        for item in fi[:15]:
            print(
                f"  {item['feature_index']:2d} {item['feature']:15s} "
                f"delta_loss={item['delta_loss']:.6f}"
            )


if __name__ == "__main__":
    main()
