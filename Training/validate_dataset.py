########################################
# --- PUPPI ML: Dataset validation --- #
########################################

# HowToRun: python3 validate_dataset.py   --input "/eos/cms/store/group/cmst3/group/l1tr/elfontan/PUPPIML/DEEPSets/graphs-dR0p3_removingRequirementAtLeast1PF/graphs_*pt" --output_dir /eos/user/e/elfontan/www/PHASE2L1_Correlator/ML_NeutralRegression/DATASET-Validation-AllData-dR0p3/ --n_examples 20

import argparse
import glob
import os
import sys
import tempfile
import types
import warnings

import numpy as np
import torch
from tqdm import tqdm

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ----------------
# Features
# ----------------

FEATURE_NAMES = [
    "log_pt",
    "slog_px",
    "slog_py",
    "charge",
    "nnvtx",
    "puppi",
    "is_chHad",
    "is_nHad",
    "is_gamma",
    "is_ele",
    "is_mu",
    "deta",
    "dphi",
    "dr",
    "is_seed",
]


def install_torch_geometric_stubs():
    if "torch_geometric.data.data" in sys.modules:
        return

    tg_mod = types.ModuleType("torch_geometric")
    tg_data_pkg = types.ModuleType("torch_geometric.data")
    tg_data_mod = types.ModuleType("torch_geometric.data.data")
    tg_storage_mod = types.ModuleType("torch_geometric.data.storage")

    class GlobalStorage(dict):
        pass

    class Data:
        def __getattr__(self, name):
            store = self.__dict__.get("_store", None)
            if store is None:
                raise AttributeError(name)

            mapping = getattr(store, "_mapping", {})
            if name in mapping:
                return mapping[name]

            raise AttributeError(name)

        def keys(self):
            store = self.__dict__.get("_store", None)
            if store is None:
                return []
            return list(getattr(store, "_mapping", {}).keys())

    class DataEdgeAttr:
        pass

    class DataTensorAttr:
        pass

    tg_data_mod.Data = Data
    tg_data_mod.DataEdgeAttr = DataEdgeAttr
    tg_data_mod.DataTensorAttr = DataTensorAttr
    tg_storage_mod.GlobalStorage = GlobalStorage

    sys.modules["torch_geometric"] = tg_mod
    sys.modules["torch_geometric.data"] = tg_data_pkg
    sys.modules["torch_geometric.data.data"] = tg_data_mod
    sys.modules["torch_geometric.data.storage"] = tg_storage_mod


def load_graphs(path):
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, message="You are using `torch.load`")
            return torch.load(path, map_location="cpu")
    except ModuleNotFoundError as exc:
        if "torch_geometric" not in str(exc):
            raise
        install_torch_geometric_stubs()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, message="You are using `torch.load`")
            return torch.load(path, map_location="cpu")


def get_graph_attr(graph, name):
    if hasattr(graph, name):
        value = getattr(graph, name)
        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
        return value

    store = getattr(graph, "_store", None)
    mapping = getattr(store, "_mapping", {})
    value = mapping.get(name, None)
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return value


def finalize_axis(ax, xlabel, ylabel="Entries", title=None, logy=False):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if logy:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.25)


def hist_or_empty(ax, values, bins, xlabel, title=None, logy=False, hist_range=None):
    if values.size:
        ax.hist(values, bins=bins, range=hist_range, histtype="stepfilled", alpha=0.75)
    finalize_axis(ax, xlabel, title=title, logy=logy)


def make_output_dir(path):
    os.makedirs(path, exist_ok=True)
    return os.path.abspath(path)


def plot_input_variables(stats, outdir):
    fig, axes = plt.subplots(3, 3, figsize=(16, 13))
    axes = axes.flatten()

    hist_or_empty(axes[0], stats["pt_all"], 80, "PF candidate pT [GeV]", logy=True)
    hist_or_empty(axes[1], stats["log_pt_all"], 80, "log(pT)", logy=True)
    hist_or_empty(axes[2], stats["puppi_all"], 80, "PUPPI weight", logy=True, hist_range=(0.0, 1.05))
    hist_or_empty(axes[3], stats["nnvtx_all"], 80, "nnVtxScore", logy=True)
    hist_or_empty(axes[4], stats["charge_all"], np.arange(-1.5, 2.6, 1.0), "Charge")
    hist_or_empty(axes[5], stats["dr_all"], 80, "DeltaR to seed", logy=True, hist_range=(0.0, 0.31))
    hist_or_empty(axes[6], stats["deta_all"], 80, "DeltaEta", logy=True, hist_range=(-0.31, 0.31))
    hist_or_empty(axes[7], stats["dphi_all"], 80, "DeltaPhi", logy=True, hist_range=(-0.31, 0.31))

    particle_labels = ["ch had", "n had", "gamma", "ele", "mu"]
    axes[8].bar(particle_labels, stats["particle_counts"], alpha=0.8)
    finalize_axis(axes[8], "PF candidate type", ylabel="Candidates", title="PDG class composition")

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "input_variables.png"), dpi=150)
    plt.close(fig)


def plot_target_vs_puppi(stats, outdir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    hist_or_empty(
        axes[0],
        stats["seed_puppi"],
        80,
        "Seed PUPPI weight",
        title="Seed PUPPI distribution",
        logy=True,
        hist_range=(0.0, 1.05),
    )

    hist_or_empty(
        axes[1],
        stats["target"],
        100,
        "Target",
        title="Target distribution",
        logy=True,
    )

    residual = stats["target"] - stats["seed_puppi"]
    hist_or_empty(
        axes[2],
        residual,
        100,
        "Target - seed PUPPI",
        title="Residual distribution",
        logy=True,
        hist_range=(-3.0, 3.0),
    )

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "target_vs_puppi.png"), dpi=150)
    plt.close(fig)


def plot_cone_multiplicity(stats, outdir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    hist_or_empty(axes[0], stats["n_nodes"], 60, "PF candidates in cone", title="Total multiplicity", logy=True)
    hist_or_empty(axes[1], stats["n_neutral"], 60, "Neutral PF candidates in cone", title="Neutral multiplicity", logy=True)
    hist_or_empty(axes[2], stats["n_charged"], 60, "Charged PF candidates in cone", title="Charged multiplicity", logy=True)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "cone_multiplicity.png"), dpi=150)
    plt.close(fig)


def plot_pf_cloud_examples(example_graphs, outdir):
    examples_dir = os.path.join(outdir, "Clouds-around-NeutralSeed")
    os.makedirs(examples_dir, exist_ok=True)

    for iexample, example in enumerate(example_graphs):
        x = example["x"]
        target = example["target"]
        seed_puppi = example["seed_puppi"]
        cone_index = example["cone_index"]

        deta = x[:, 11]
        dphi = x[:, 12]
        pt = np.exp(x[:, 0])
        charge = x[:, 3]
        is_seed = x[:, 14] > 0.5
        is_charged = np.abs(charge) > 0.5
        is_neutral = ~is_charged
        title = (
            f"PF cloud around neutral seed\n"
            f"cone={cone_index}"
        )

        fig, ax = plt.subplots(figsize=(9, 7))

        scatter = None
        if np.any(is_neutral & ~is_seed):
            scatter = ax.scatter(
                deta[is_neutral & ~is_seed],
                dphi[is_neutral & ~is_seed],
                c=pt[is_neutral & ~is_seed],
                s=40,
                cmap="viridis",
                marker="o",
                alpha=0.90,
                edgecolors="none",
                label="Neutral PF",
            )
        if np.any(is_charged):
            charged = ax.scatter(
                deta[is_charged],
                dphi[is_charged],
                c=pt[is_charged],
                s=55,
                cmap="viridis",
                marker="^",
                alpha=0.95,
                edgecolors="black",
                linewidths=0.3,
                label="Charged PF",
            )
            if scatter is None:
                scatter = charged
        if np.any(is_seed):
            seed = ax.scatter(
                deta[is_seed],
                dphi[is_seed],
                c=pt[is_seed],
                s=220,
                cmap="viridis",
                marker="*",
                alpha=1.0,
                edgecolors="red",
                linewidths=1.0,
                label="Neutral seed",
            )
            if scatter is None:
                scatter = seed

        ax.axhline(0.0, color="gray", linewidth=1.0, alpha=0.7)
        ax.axvline(0.0, color="gray", linewidth=1.0, alpha=0.7)
        finalize_axis(ax, "Δη", ylabel="Δφ", title=title)
        ax.set_xlim(-0.31, 0.31)
        ax.set_ylim(-0.31, 0.31)
        ax.legend(loc="upper right", fontsize=9)

        if scatter is not None:
            cbar = fig.colorbar(scatter, ax=ax, shrink=0.92, pad=0.02)
            cbar.set_label("PF candidate pT [GeV]")

        fig.tight_layout()
        fig.savefig(os.path.join(examples_dir, f"cone_{iexample:02d}_2d.png"), dpi=160)
        plt.close(fig)

        fig3d = plt.figure(figsize=(9, 7))
        ax3d = fig3d.add_subplot(1, 1, 1, projection="3d")

        scatter3d = None
        if np.any(is_neutral & ~is_seed):
            scatter3d = ax3d.scatter(
                deta[is_neutral & ~is_seed],
                dphi[is_neutral & ~is_seed],
                pt[is_neutral & ~is_seed],
                c=pt[is_neutral & ~is_seed],
                s=40,
                cmap="viridis",
                marker="o",
                alpha=0.90,
                edgecolors="none",
                label="Neutral PF",
            )
        if np.any(is_charged):
            charged3d = ax3d.scatter(
                deta[is_charged],
                dphi[is_charged],
                pt[is_charged],
                c=pt[is_charged],
                s=55,
                cmap="viridis",
                marker="^",
                alpha=0.95,
                edgecolors="black",
                linewidths=0.3,
                label="Charged PF",
            )
            if scatter3d is None:
                scatter3d = charged3d
        if np.any(is_seed):
            seed3d = ax3d.scatter(
                deta[is_seed],
                dphi[is_seed],
                pt[is_seed],
                c=pt[is_seed],
                s=220,
                cmap="viridis",
                marker="*",
                alpha=1.0,
                edgecolors="red",
                linewidths=1.0,
                label="Neutral seed",
            )
            if scatter3d is None:
                scatter3d = seed3d

        ax3d.set_xlabel("Δη")
        ax3d.set_ylabel("Δφ")
        ax3d.set_zlabel("PF pT [GeV]")
        ax3d.set_xlim(-0.31, 0.31)
        ax3d.set_ylim(-0.31, 0.31)
        ax3d.set_title(title, fontsize=11)
        ax3d.legend(loc="upper right", fontsize=9)

        if scatter3d is not None:
            cbar3d = fig3d.colorbar(scatter3d, ax=ax3d, shrink=0.82, pad=0.08)
            cbar3d.set_label("PF candidate pT [GeV]")

        fig3d.subplots_adjust(left=0.02, right=0.90, bottom=0.08, top=0.88)
        fig3d.savefig(os.path.join(examples_dir, f"cone_{iexample:02d}_3d.png"), dpi=160)
        plt.close(fig3d)


def write_summary(stats, outdir, files_used):
    summary_path = os.path.join(outdir, "summary.txt")
    with open(summary_path, "w", encoding="ascii") as handle:
        handle.write(f"Files used: {files_used}\n")
        handle.write(f"Cones processed: {len(stats['target'])}\n")
        handle.write(f"PF candidates processed: {len(stats['pt_all'])}\n")
        if len(stats["target"]):
            handle.write(f"Target mean: {np.mean(stats['target']):.6f}\n")
            handle.write(f"Target std: {np.std(stats['target']):.6f}\n")
        if len(stats["seed_puppi"]):
            handle.write(f"Seed PUPPI mean: {np.mean(stats['seed_puppi']):.6f}\n")
            handle.write(f"Seed PUPPI std: {np.std(stats['seed_puppi']):.6f}\n")


def main():
    parser = argparse.ArgumentParser(description="Validation plots for DEEPSets neutral-regression graph batches")
    parser.add_argument(
        "--input",
        required=True,
        help="Glob pattern for the input .pt graph files",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where the plots and summary are written",
    )
    parser.add_argument("--max_files", type=int, default=None, help="Optional cap on the number of .pt files")
    parser.add_argument("--max_graphs", type=int, default=None, help="Optional cap on the number of cones")
    parser.add_argument("--n_examples", type=int, default=5, help="Number of explicit cone-cloud examples to save")
    args = parser.parse_args()

    files = sorted(glob.glob(args.input))
    if args.max_files is not None:
        files = files[: args.max_files]

    if not files:
        raise FileNotFoundError(f"No .pt files matched pattern: {args.input}")

    outdir = make_output_dir(args.output_dir)

    stats = {
        "pt_all": [],
        "log_pt_all": [],
        "puppi_all": [],
        "nnvtx_all": [],
        "charge_all": [],
        "deta_all": [],
        "dphi_all": [],
        "dr_all": [],
        "target": [],
        "seed_puppi": [],
        "n_nodes": [],
        "n_neutral": [],
        "n_charged": [],
        "particle_counts": np.zeros(5, dtype=np.int64),
    }
    example_graphs = []

    graphs_seen = 0

    for path in tqdm(files, desc="Loading .pt files"):
        graphs = load_graphs(path)
        for graph in graphs:
            x = get_graph_attr(graph, "x")
            y = get_graph_attr(graph, "y")

            if x is None or y is None:
                continue

            x = np.asarray(x, dtype=np.float32)
            y = np.asarray(y, dtype=np.float32).reshape(-1)

            if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES):
                raise ValueError(
                    f"Unexpected feature shape {x.shape} in {path}. "
                    f"Expected (_, {len(FEATURE_NAMES)}) from prepare_dataset_batches.py."
                )

            log_pt = x[:, 0]
            pt = np.exp(log_pt)
            charge = x[:, 3]
            nnvtx = x[:, 4]
            puppi = x[:, 5]
            onehot = x[:, 6:11]
            deta = x[:, 11]
            dphi = x[:, 12]
            dr = x[:, 13]
            is_seed = x[:, 14] > 0.5

            if is_seed.sum() != 1:
                continue

            seed_idx = np.argmax(is_seed)

            stats["pt_all"].append(pt)
            stats["log_pt_all"].append(log_pt)
            stats["puppi_all"].append(puppi)
            stats["nnvtx_all"].append(nnvtx)
            stats["charge_all"].append(charge)
            stats["deta_all"].append(deta)
            stats["dphi_all"].append(dphi)
            stats["dr_all"].append(dr)
            stats["target"].append(float(y[0]))
            stats["seed_puppi"].append(float(puppi[seed_idx]))
            stats["n_nodes"].append(x.shape[0])
            stats["n_charged"].append(int(np.sum(np.abs(charge) > 0.5)))
            stats["n_neutral"].append(int(np.sum(np.abs(charge) <= 0.5)))
            stats["particle_counts"] += np.rint(onehot.sum(axis=0)).astype(np.int64)
            if len(example_graphs) < args.n_examples:
                example_graphs.append(
                    {
                        "x": x.copy(),
                        "target": float(y[0]),
                        "seed_puppi": float(puppi[seed_idx]),
                        "cone_index": graphs_seen,
                    }
                )

            graphs_seen += 1
            if args.max_graphs is not None and graphs_seen >= args.max_graphs:
                break

        if args.max_graphs is not None and graphs_seen >= args.max_graphs:
            break

    for key, value in list(stats.items()):
        if key == "particle_counts":
            continue
        stats[key] = np.concatenate(value) if key.endswith("_all") and value else np.array([], dtype=np.float32)
        if key in {"target", "seed_puppi", "n_nodes", "n_charged", "n_neutral"}:
            stats[key] = np.asarray(value)

    plot_input_variables(stats, outdir)
    plot_target_vs_puppi(stats, outdir)
    plot_cone_multiplicity(stats, outdir)
    plot_pf_cloud_examples(example_graphs, outdir)
    write_summary(stats, outdir, len(files))

    print(f"Saved validation plots in: {outdir}")


if __name__ == "__main__":
    main()
