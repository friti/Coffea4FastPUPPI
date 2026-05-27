import uproot
import awkward as ak
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import argparse
import os
import h5py
import json


BRANCHES = [
    "L1PuppiCands_pt",
    "L1PuppiCands_eta",
    "L1PuppiCands_phi",
    "L1PuppiCands_charge",
    "L1PuppiCands_pdgId",
    "L1PuppiCands_nnVtxScore",
    "L1PuppiCands_puppiWeight",
    "L1PuppiCands_genPtSum0p2",
    "L1PuppiCands_recoDen_seedPlusAll0p2",
]


def pdg_to_onehot_vec(pdgIds):
    pid = np.abs(pdgIds.astype(np.int32))
    out = np.zeros((len(pid), 5), dtype=np.float32)
    out[:, 0] = (pid == 211)
    out[:, 1] = (pid == 130)
    out[:, 2] = (pid == 22)
    out[:, 3] = (pid == 11)
    out[:, 4] = (pid == 13)
    return out


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--dr", type=float, default=0.3)
    parser.add_argument("--max_nodes", type=int, default=128)
    parser.add_argument("--graphs_per_file", type=int, default=50000)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    pt_dir = os.path.join(args.output_dir, "pt")
    os.makedirs(pt_dir, exist_ok=True)

    file = uproot.open(args.input)
    tree = file[file.keys()[0]]
    arrays = tree.arrays(BRANCHES, library="ak")

    # HDF5 accumulators
    X, MASK, Y = [], [], []

    puppi_list = []
    gen_list = []
    reco_list = []
    seed_pt_list = []
    seed_eta_list = []
    event_idx_list = []
    center_idx_list = []
    n_raw_list = []
    n_kept_list = []

    feature_names = [
        "log_pt", "slog_px", "slog_py",
        "charge", "nnvtx", "puppi",
        "is_chHad", "is_neuHad", "is_gamma", "is_ele", "is_mu",
        "deta", "dphi", "dr", "is_seed"
    ]

    graphs = []
    file_idx = 0

    EPS = 1e-6

    # event loop
    for iev in tqdm(range(tree.num_entries), desc="Processing events"):

        pts = ak.to_numpy(arrays["L1PuppiCands_pt"][iev])
        etas = ak.to_numpy(arrays["L1PuppiCands_eta"][iev])
        phis = ak.to_numpy(arrays["L1PuppiCands_phi"][iev])
        charges = ak.to_numpy(arrays["L1PuppiCands_charge"][iev])
        pdgIds = ak.to_numpy(arrays["L1PuppiCands_pdgId"][iev])
        nnvtx = ak.to_numpy(arrays["L1PuppiCands_nnVtxScore"][iev])
        puppi = ak.to_numpy(arrays["L1PuppiCands_puppiWeight"][iev])
        genSum = ak.to_numpy(arrays["L1PuppiCands_genPtSum0p2"][iev])
        recoDen = ak.to_numpy(arrays["L1PuppiCands_recoDen_seedPlusAll0p2"][iev])

        if len(pts) < 2:
            continue

        # geometry
        deta = etas[None, :] - etas[:, None]
        dphi = (phis[None, :] - phis[:, None] + np.pi) % (2*np.pi) - np.pi
        dr = np.sqrt(deta**2 + dphi**2)

        # kinematics
        px = pts * np.cos(phis)
        py = pts * np.sin(phis)

        log_pt = np.log(pts + EPS)
        slog_px = np.sign(px) * np.log(np.abs(px) + EPS)
        slog_py = np.sign(py) * np.log(np.abs(py) + EPS)

        base_feats = np.stack([
            log_pt,
            slog_px,
            slog_py,
            charges,
            nnvtx,
            puppi
        ], axis=1)

        base_feats = np.concatenate([base_feats, pdg_to_onehot_vec(pdgIds)], axis=1)

        # select neutral seeds
        centers = np.where((charges == 0) & (pts >= 1.0) & (recoDen > 0))[0]

        for i in centers:

            target = genSum[i] / (recoDen[i] + EPS)

            idx = np.where(dr[i] < args.dr)[0]
            if len(idx) < 2:
                continue

            # sort by distance
            idx = idx[np.argsort(dr[i][idx])]

            n_raw = len(idx)

            # truncate
            idx = idx[:args.max_nodes]
            n_kept = len(idx)

            feats = np.concatenate([
                base_feats[idx],
                deta[i][idx][:, None],
                dphi[i][idx][:, None],
                dr[i][idx][:, None],
                (idx == i).astype(float)[:, None],
            ], axis=1)

            # PT (variable-length for graph + some deepset)
            graphs.append(
                Data(
                    x=torch.tensor(feats, dtype=torch.float32),
                    y=torch.tensor([target], dtype=torch.float32),
                    event_idx=torch.tensor([iev]),
                    center_idx=torch.tensor([i]),
                    seed_pt=torch.tensor([pts[i]]),
                    seed_eta=torch.tensor([etas[i]]),
                    genPtSum=torch.tensor([genSum[i]]),
                    recoPtSum=torch.tensor([recoDen[i]]),
                )
            )

            if len(graphs) >= args.graphs_per_file:
                out = os.path.join(pt_dir, f"graphs_{file_idx}.pt")
                torch.save(graphs, out)
                print(f"Saved {len(graphs)} → {out}")
                graphs = []
                file_idx += 1

            # HDF5 (padded + mask)
            x_pad = np.zeros((args.max_nodes, feats.shape[1]), dtype=np.float32)
            mask = np.zeros(args.max_nodes, dtype=np.float32)

            x_pad[:n_kept] = feats
            mask[:n_kept] = 1.0

            X.append(x_pad)
            MASK.append(mask)
            Y.append(target)

            puppi_list.append(puppi[i])
            gen_list.append(genSum[i])
            reco_list.append(recoDen[i])
            seed_pt_list.append(pts[i])
            seed_eta_list.append(etas[i])
            event_idx_list.append(iev)
            center_idx_list.append(i)
            n_raw_list.append(n_raw)
            n_kept_list.append(n_kept)

    # save pt
    if len(graphs) > 0:
        out = os.path.join(pt_dir, f"graphs_{file_idx}.pt")
        torch.save(graphs, out)
        print(f"Saved {len(graphs)} pt files in: {out}")

    # write HDF5 (padded + mask for fixed size)
    out_h5 = os.path.join(args.output_dir, "dataset.h5")

    with h5py.File(out_h5, "w") as hf:
        hf.create_dataset("x", data=np.array(X))
        hf.create_dataset("mask", data=np.array(MASK))
        hf.create_dataset("target_weight", data=np.array(Y))

        hf.create_dataset("puppiWeight", data=np.array(puppi_list))
        hf.create_dataset("genPtSum", data=np.array(gen_list))
        hf.create_dataset("recoPtSum", data=np.array(reco_list))

        hf.create_dataset("seed_pt", data=np.array(seed_pt_list))
        hf.create_dataset("seed_eta", data=np.array(seed_eta_list))

        hf.create_dataset("event_idx", data=np.array(event_idx_list))
        hf.create_dataset("center_idx", data=np.array(center_idx_list))

        hf.create_dataset("n_pf_raw", data=np.array(n_raw_list))
        hf.create_dataset("n_pf_kept", data=np.array(n_kept_list))

        hf.attrs["model_features"] = json.dumps(feature_names)
        hf.attrs["dr_used"] = args.dr

    print(f"HDF5 saved in: {out_h5}")


if __name__ == "__main__":
    main()
