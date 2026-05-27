import os
import argparse

import awkward as ak
import numpy as np
import torch
import uproot

from tqdm import tqdm
from torch_geometric.data import Data


# -------------------------------------------------
# Features to skip (i.e. not available in the current set of ntuples)
# -------------------------------------------------

SKIP_FEATURES = [
    "vz",
    "idProbPu",
    "idProbEm",
    "idProbPi",
]


# -------------------------------------------------
# Branch map
# -------------------------------------------------

BRANCH_MAP = {
    "pt": "L1PuppiCands_pt",
    "eta": "L1PuppiCands_eta",
    "phi": "L1PuppiCands_phi",

    "charge": "L1PuppiCands_charge",
    "pdgId": "L1PuppiCands_pdgId",

    "nnvtx": "L1PuppiCands_nnVtxScore",
    "vz": "L1PuppiCands_vz",
    "dxy": "L1PuppiCands_dxy",
    "z0": "L1PuppiCands_z0",

    "puppiWeight": "L1PuppiCands_puppiWeight",

    "idProbPu": "L1PuppiCands_idProbPu",
    "idProbEm": "L1PuppiCands_idProbEm",
    "idProbPi": "L1PuppiCands_idProbPi",

    "genPtSum0p2": "L1PuppiCands_genPtSum0p2",
    "recoDen_seedPlusAll0p2": "L1PuppiCands_recoDen_seedPlusAll0p2",
}


# -------------------------------------------------
# Feature ordering
# -------------------------------------------------

FEATURE_NAMES = [
    "log_pt",
    "slog_px",
    "slog_py",
    "abs_eta",

    "charge",

    "nnvtx",
    "vz",
    "dxy",
    "z0",

    "puppiWeight",

    "idProbPu",
    "idProbEm",
    "idProbPi",

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

FEATURE_NAMES = [
    f for f in FEATURE_NAMES
    if f not in SKIP_FEATURES
]


# -------------------------------------------------
# Branches to load
# -------------------------------------------------

BRANCHES = [
    branch
    for key, branch in BRANCH_MAP.items()
    if key not in SKIP_FEATURES
]


# -------------------------------------------------
# Feature groups
# -------------------------------------------------

TRACK_FEATURES = {
    "nnvtx",
    "vz",
    "dxy",
    "z0",
}

HGCAL_FEATURES = {
    "idProbPu",
    "idProbEm",
    "idProbPi",
}

GEOMETRY_FEATURES = {
    "deta",
    "dphi",
    "dr",
    "is_center",
}


# -------------------------------------------------
# Helpers
# -------------------------------------------------

def pdg_to_onehot_vec(pdgIds):

    pid = np.abs(pdgIds.astype(np.int32))

    return {
        "pdg_abs_211": (pid == 211).astype(np.float32),
        "pdg_abs_130": (pid == 130).astype(np.float32),
        "pdg_abs_22":  (pid == 22).astype(np.float32),
        "pdg_abs_11":  (pid == 11).astype(np.float32),
        "pdg_abs_13":  (pid == 13).astype(np.float32),
    }


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dr", type=float, default=0.3)
    parser.add_argument("--max_nodes", type=int, default=48)
    parser.add_argument("--graphs_per_file", type=int, default=50000)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("\nUsing features:\n")
    for i, f in enumerate(FEATURE_NAMES):
        print(f"{i:2d}: {f}")

    print("\nReading branches:\n")
    for b in BRANCHES:
        print(" ", b)

    # -------------------------------------------------
    # Load ROOT file
    # -------------------------------------------------

    file = uproot.open(args.input)
    tree = file[file.keys()[0]]

    arrays = tree.arrays(BRANCHES, library="ak")

    graphs = []
    file_idx = 0

    n_events = tree.num_entries

    # -------------------------------------------------
    # Event loop
    # -------------------------------------------------

    for iev in tqdm(range(n_events), desc="Processing events"):

        pts = ak.to_numpy(arrays[BRANCH_MAP["pt"]][iev])
        etas = ak.to_numpy(arrays[BRANCH_MAP["eta"]][iev])
        phis = ak.to_numpy(arrays[BRANCH_MAP["phi"]][iev])

        charges = ak.to_numpy(arrays[BRANCH_MAP["charge"]][iev])
        pdgIds = ak.to_numpy(arrays[BRANCH_MAP["pdgId"]][iev])

        genSum = ak.to_numpy(arrays[BRANCH_MAP["genPtSum0p2"]][iev])

        recoDen = ak.to_numpy(arrays[BRANCH_MAP["recoDen_seedPlusAll0p2"]][iev])

        if len(pts) < 1:
            continue

        # -------------------------------------------------
        # Relative geometry
        # -------------------------------------------------

        deta = etas[None, :] - etas[:, None]
        dphi = ((phis[None, :] - phis[:, None] + np.pi) % (2 * np.pi) - np.pi) # take care of boundaries
        dr = np.sqrt(deta**2 + dphi**2)

        # -------------------------------------------------
        # Kinematics
        # -------------------------------------------------

        px = pts * np.cos(phis)
        py = pts * np.sin(phis)

        log_pt = np.log(np.clip(pts, 1e-6, None))

        slog_px = np.sign(px) * np.log(np.abs(px) + 1e-6)
        slog_py = np.sign(py) * np.log(np.abs(py) + 1e-6)

        abs_eta = np.abs(etas)

        hgcal_mask = ((abs_eta >= 1.5) & (abs_eta <= 2.4))

        # -------------------------------------------------
        # Base features
        # -------------------------------------------------

        feature_arrays = {
            "log_pt": log_pt,
            "slog_px": slog_px,
            "slog_py": slog_py,
            "abs_eta": abs_eta,
            "charge": charges.astype(np.float32),
        }

        # -------------------------------------------------
        # Optional feature builders
        # -------------------------------------------------

        optional_feature_builders = {
            "nnvtx": lambda:
                ak.to_numpy(arrays[BRANCH_MAP["nnvtx"]][iev]).astype(np.float32),

            "vz": lambda:
                ak.to_numpy(arrays[BRANCH_MAP["vz"]][iev]).astype(np.float32),

            "dxy": lambda:
                ak.to_numpy(arrays[BRANCH_MAP["dxy"]][iev]).astype(np.float32),

            "z0": lambda:
                ak.to_numpy(arrays[BRANCH_MAP["z0"]][iev]).astype(np.float32),

            "puppiWeight": lambda:
                ak.to_numpy(arrays[BRANCH_MAP["puppiWeight"]][iev]).astype(np.float32),

            "idProbPu": lambda:
                ak.to_numpy(arrays[BRANCH_MAP["idProbPu"]][iev]).astype(np.float32),

            "idProbEm": lambda:
                ak.to_numpy(arrays[BRANCH_MAP["idProbEm"]][iev]).astype(np.float32),

            "idProbPi": lambda:
                ak.to_numpy(arrays[BRANCH_MAP["idProbPi"]][iev]).astype(np.float32),
        }

        # -------------------------------------------------
        # Build optional features (if not present in skip list)
        # -------------------------------------------------

        for feat, builder in optional_feature_builders.items():
            if feat in SKIP_FEATURES:
                continue

            arr = builder()

            if feat in TRACK_FEATURES:
                arr = arr.copy()
                arr[charges == 0] = -1.0

            if feat in HGCAL_FEATURES:
                arr = arr.copy()
                arr[~hgcal_mask] = -1.0

            feature_arrays[feat] = arr

        # -------------------------------------------------
        # PDG one-hot
        # -------------------------------------------------

        feature_arrays.update(pdg_to_onehot_vec(pdgIds))

        # -------------------------------------------------
        # Base feature tensor
        # -------------------------------------------------

        base_feature_names = [
            f for f in FEATURE_NAMES
            if f not in GEOMETRY_FEATURES
        ]

        base_feats = np.stack(
            [feature_arrays[f] for f in base_feature_names],
            axis=1,
        ).astype(np.float32)

        # -------------------------------------------------
        # Center neutral candidates
        # -------------------------------------------------

        centers = np.where((charges == 0) & (pts >= 1.0) & (recoDen > 0))[0]

        # -------------------------------------------------
        # Build graph around each center
        # -------------------------------------------------

        for i in centers:
            target = genSum[i] / recoDen[i]

            idx = np.where(dr[i] < args.dr)[0]

            if len(idx) < 1:
                continue

            idx = idx[np.argsort(dr[i][idx])]
            idx = idx[:args.max_nodes]

            extra_features = []

            if "deta" in FEATURE_NAMES:
                extra_features.append(deta[i][idx][:, None])

            if "dphi" in FEATURE_NAMES:
                extra_features.append(dphi[i][idx][:, None])

            if "dr" in FEATURE_NAMES:
                extra_features.append(dr[i][idx][:, None])

            if "is_center" in FEATURE_NAMES:
                extra_features.append((idx == i).astype(np.float32)[:, None])

            feats = np.concatenate(
                [base_feats[idx]] + extra_features,
                axis=1,
            )

            graph = Data(
                x=torch.tensor(feats, dtype=torch.float32),
                y=torch.tensor([target], dtype=torch.float32),

                event_idx=torch.tensor([iev]),
                center_idx=torch.tensor([i]),
            )

            graphs.append(graph)

            # -------------------------------------------------
            # Flush chunk
            # -------------------------------------------------

            if len(graphs) >= args.graphs_per_file:

                out = os.path.join(
                    args.output_dir,
                    f"graphs_{file_idx}.pt",
                )

                metadata = {
                    "feature_names": FEATURE_NAMES,
                    "skip_features": SKIP_FEATURES,
                    "branch_map": BRANCH_MAP,
                    "dr": args.dr,
                    "max_nodes": args.max_nodes,
                }

                torch.save(
                    {
                        "graphs": graphs,
                        "metadata": metadata,
                    },
                    out,
                )

                print(f"Saved {len(graphs)} graphs -> {out}")

                graphs = []
                file_idx += 1

    # -------------------------------------------------
    # Final flush
    # -------------------------------------------------

    if len(graphs) > 0:

        out = os.path.join(
            args.output_dir,
            f"graphs_{file_idx}.pt",
        )

        metadata = {
            "feature_names": FEATURE_NAMES,
            "skip_features": SKIP_FEATURES,
            "branch_map": BRANCH_MAP,
            "dr": args.dr,
            "max_nodes": args.max_nodes,
        }

        torch.save(
            {
                "graphs": graphs,
                "metadata": metadata,
            },
            out,
        )

        print(f"Saved {len(graphs)} graphs -> {out}")


if __name__ == "__main__":
    main()
