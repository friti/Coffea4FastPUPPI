import os
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm


# -------------------------------------------------
# Helpers
# -------------------------------------------------

def make_plot_dir(run_dir):

    plot_dir = os.path.join(run_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    return plot_dir


def savefig(plot_dir, name):

    path = os.path.join(plot_dir, name)

    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    print("Saved", path)


def load_outputs(run_dir):

    outputs = np.load(
        os.path.join(run_dir, "test_outputs_best.npz")
    )

    pred = outputs["pred"]
    target = outputs["target"]

    center_pt = outputs["center_pt"]

    center_eta = None

    if "center_eta" in outputs:
        center_eta = outputs["center_eta"]

    return pred, target, center_pt, center_eta


# -------------------------------------------------
# Plotting
# -------------------------------------------------

def plot_loss_history(run_dir, plot_dir):

    hist = np.load(
        os.path.join(run_dir, "loss_history.npz")
    )

    epoch = hist["epoch"]
    train_loss = hist["train_loss"]
    val_loss = hist["val_loss"]

    plt.figure(figsize=(6, 5))

    plt.plot(epoch, train_loss, label="Train")
    plt.plot(epoch, val_loss, label="Validation")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title("Loss vs. epoch")

    plt.legend()

    savefig(plot_dir, "loss_vs_epoch.png")


def plot_residual(
    pred,
    target,
    plot_dir,
    filename,
    title,
    xlim=None,
    mask=None,
    ylabel="Candidates",
    density=False,
    bins=100,
):

    if mask is not None:
        pred = pred[mask]
        target = target[mask]

    residual = pred - target

    plt.figure(figsize=(6, 5))

    plt.hist(
        residual,
        bins=bins,
        histtype="step",
        linewidth=2,
        density=density,
    )

    plt.xlabel("Prediction - Target")
    plt.ylabel(ylabel)

    plt.title(title)

    if xlim is not None:
        plt.xlim(*xlim)

    savefig(plot_dir, filename)


def plot_residual_compare(
    pred1,
    target1,
    pred2,
    target2,
    plot_dir,
    filename,
    title,
    labels,
    xlim=None,
    mask1=None,
    mask2=None,
    density=True,
    bins=100,
):

    if mask1 is not None:
        pred1 = pred1[mask1]
        target1 = target1[mask1]

    if mask2 is not None:
        pred2 = pred2[mask2]
        target2 = target2[mask2]

    residual1 = pred1 - target1
    residual2 = pred2 - target2

    plt.figure(figsize=(6, 5))

    plt.hist(
        residual1,
        bins=bins,
        histtype="step",
        linewidth=2,
        density=density,
        label=labels[0],
    )

    plt.hist(
        residual2,
        bins=bins,
        histtype="step",
        linewidth=2,
        density=density,
        label=labels[1],
    )

    plt.xlabel("Prediction - Target")
    plt.ylabel("A.U." if density else "Candidates")

    plt.title(title)

    if xlim is not None:
        plt.xlim(*xlim)

    plt.legend()

    savefig(plot_dir, filename)


def plot_response_2d(
    pred,
    target,
    plot_dir,
    filename,
    title,
    mask=None,
    xlim=(0, 1.5),
    ylim=(0, 1.5),
):

    if mask is not None:
        pred = pred[mask]
        target = target[mask]

    plt.figure(figsize=(6, 5))

    plt.hist2d(
        target,
        pred,
        bins=[100, 100],
        range=[[xlim[0],xlim[1]],[ylim[0],ylim[1]]],
        norm=LogNorm(),
    )

    plt.xlabel("Target")
    plt.ylabel("Prediction")

    plt.title(title)

    plt.xlim(*xlim)
    plt.ylim(*ylim)

    plt.colorbar(label="Candidates")

    plt.plot(
        [xlim[0], xlim[1]],
        [ylim[0], ylim[1]],
        linestyle="--",
        linewidth=1,
        color="red",
    )

    savefig(plot_dir, filename)

def plot_target_pt(
    target,
    center_pt,
    plot_dir,
    filename,
    title,
    mask=None,
    xlim=(0, 50),
    ylim=(0, 50),
):

    if mask is not None:
        target = target[mask]
        center_pt = center_pt[mask]

    target_pt = center_pt * target

    plt.figure(figsize=(6, 5))

    plt.hist2d(
        center_pt,
        target_pt,
        bins=[100, 100],
        range=[[xlim[0], xlim[1]], [ylim[0], ylim[1]]],
        norm=LogNorm(),
    )

    plt.xlabel("Neutral candidate pT")
    plt.ylabel("Target * pT")

    plt.title(title)

    plt.colorbar(label="Candidates")

    plt.plot(
        [xlim[0], xlim[1]],
        [ylim[0], ylim[1]],
        linestyle="--",
        linewidth=1,
        color="red",
    )

    savefig(plot_dir, filename)


def plot_corrected_pt(
    pred,
    center_pt,
    plot_dir,
    filename,
    title,
    mask=None,
    xlim=(0, 50),
    ylim=(0, 50),
):

    if mask is not None:
        pred = pred[mask]
        center_pt = center_pt[mask]

    corrected_pt = center_pt * pred

    plt.figure(figsize=(6, 5))

    plt.hist2d(
        center_pt,
        corrected_pt,
        range=[[xlim[0],xlim[1]],[ylim[0],ylim[1]]],
        bins=[100, 100],
        norm=LogNorm(),
    )

    plt.xlabel("Neutral candidate pT")
    plt.ylabel("Regressed wgt * pT")

    plt.title(title)

    plt.colorbar(label="Candidates")

    savefig(plot_dir, filename)


def plot_feature_importance(run_dir, plot_dir):

    path = os.path.join(
        run_dir,
        "feature_importance.json",
    )

    if not os.path.exists(path):
        print("No feature importance found")
        return

    with open(path) as f:
        fi = json.load(f)

    rename_map = {
        "pdg_abs_211": "is_PF_charged_hadron",
        "pdg_abs_130": "is_PF_neutral_hadron",
        "pdg_abs_22": "is_PF_photon",
        "pdg_abs_11": "is_PF_electron",
        "pdg_abs_13": "is_PF_muon",
    }

    names = []
    values = []

    for item in fi:

        if np.isnan(item["delta_loss"]):
            continue

        name = item["feature"]

        if name in rename_map:
            name = rename_map[name]

        names.append(name)
        values.append(item["delta_loss"])

    names = names[::-1]
    values = values[::-1]

    plt.figure(figsize=(8, 6))

    plt.barh(names, values)

    plt.xlabel("Permutation importance")
    plt.title("Feature importance")

    savefig(plot_dir, "feature_importance.png")


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run",
        required=True,
    )

    parser.add_argument(
        "--compare_run",
        default=None,
    )

    args = parser.parse_args()

    run_dir = args.run

    plot_dir = make_plot_dir(run_dir)

    pred, target, center_pt, center_eta = load_outputs(run_dir)

    # -------------------------------------------------
    # Loss history
    # -------------------------------------------------

    plot_loss_history(run_dir, plot_dir)

    # -------------------------------------------------
    # Residuals
    # -------------------------------------------------

    plot_residual(
        pred,
        target,
        plot_dir,
        "residual_all.png",
        "Residual",
        xlim=(-2, 2),
        bins=200,
    )

    plot_residual(
        pred,
        target,
        plot_dir,
        "residual_zero_target.png",
        "Residual for neutrals with zero target",
        xlim=(0, 1.5),
        mask=(target <= 0.05),
        bins=100,
    )

    plot_residual(
        pred,
        target,
        plot_dir,
        "residual_nonzero_target.png",
        "Residual for neutrals with non-zero target",
        xlim=(-2, 2),
        mask=(target > 0.05),
        bins=200,
    )

    # -------------------------------------------------
    # Inclusive response
    # -------------------------------------------------

    plot_response_2d(
        pred,
        target,
        plot_dir,
        "response_2d.png",
        "Target vs. regressed",
    )

    # -------------------------------------------------
    # pT bins
    # -------------------------------------------------

    pt_bins = [
        (1, 5, "1 GeV < pT < 5 GeV", "response_pt_1_5.png"),
        (5, 10, "5 GeV < pT < 10 GeV", "response_pt_5_10.png"),
        (10, np.inf, "10 GeV < pT", "response_pt_10_inf.png"),
    ]

    for ptmin, ptmax, title, filename in pt_bins:

        mask = (
            (center_pt >= ptmin)
            & (center_pt < ptmax)
        )

        plot_response_2d(
            pred,
            target,
            plot_dir,
            filename,
            title,
            mask=mask,
        )

    # -------------------------------------------------
    # eta bins
    # -------------------------------------------------

    if center_eta is not None:

        eta_bins = [
            (0, 1.4, "Barrel", "response_eta_barrel.png"),
            (1.5, 3.0, "Endcap", "response_eta_endcap.png"),
            (3.0, np.inf, "HF", "response_eta_hf.png"),
        ]

        for etamin, etamax, title, filename in eta_bins:

            mask = (
                (center_eta >= etamin)
                & (center_eta < etamax)
            )

            plot_response_2d(
                pred,
                target,
                plot_dir,
                filename,
                title,
                mask=mask,
            )

    # -------------------------------------------------
    # Corrected pT
    # -------------------------------------------------

    plot_target_pt(
        target,
        center_pt,
        plot_dir,
        "target_pt.png",
        "Neutral candidate pT vs. target pT",
    )

    plot_corrected_pt(
        pred,
        center_pt,
        plot_dir,
        "corrected_pt.png",
        "Neutral candidate pT vs. regressed pT",
    )

    # -------------------------------------------------
    # Feature importance
    # -------------------------------------------------

    plot_feature_importance(
        run_dir,
        plot_dir,
    )

    # -------------------------------------------------
    # Compare runs
    # -------------------------------------------------

    if args.compare_run is not None:

        pred2, target2, _, _ = load_outputs(
            args.compare_run
        )

        plot_residual_compare(
            pred,
            target,
            pred2,
            target2,
            plot_dir,
            "compare_nonzero_residual.png",
            "Residual for neutrals with non-zero target",
            labels=[
                os.path.basename(run_dir),
                os.path.basename(args.compare_run),
            ],
            xlim=(-1, 1),
            mask1=(target > 0.05),
            mask2=(target2 > 0.05),
            density=True,
            bins=100,
        )


if __name__ == "__main__":
    main()
