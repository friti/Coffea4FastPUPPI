import uproot
import awkward as ak
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import argparse
import os


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
    parser.add_argument("--max_nodes", type=int, default=48)
    parser.add_argument("--graphs_per_file", type=int, default=50000)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    file = uproot.open(args.input)
    tree = file[file.keys()[0]]
    arrays = tree.arrays(BRANCHES, library="ak")

    graphs = []
    file_idx = 0

    n_events = tree.num_entries

    for iev in tqdm(range(n_events), desc="Processing events"):

        pts = ak.to_numpy(arrays["L1PuppiCands_pt"][iev])
        etas = ak.to_numpy(arrays["L1PuppiCands_eta"][iev])
        phis = ak.to_numpy(arrays["L1PuppiCands_phi"][iev])
        charges = ak.to_numpy(arrays["L1PuppiCands_charge"][iev])
        pdgIds = ak.to_numpy(arrays["L1PuppiCands_pdgId"][iev])
        nnvtx = ak.to_numpy(arrays["L1PuppiCands_nnVtxScore"][iev])
        puppi = ak.to_numpy(arrays["L1PuppiCands_puppiWeight"][iev])
        genSum = ak.to_numpy(arrays["L1PuppiCands_genPtSum0p2"][iev])
        recoDen = ak.to_numpy(arrays["L1PuppiCands_recoDen_seedPlusAll0p2"][iev])

        n = len(pts)
        if n < 2:
            continue

        # geometry
        deta = etas[None, :] - etas[:, None]
        dphi = (phis[None, :] - phis[:, None] + np.pi) % (2*np.pi) - np.pi
        dr = np.sqrt(deta**2 + dphi**2)

        # kinematics
        px = pts * np.cos(phis)
        py = pts * np.sin(phis)

        log_pt = np.log(pts)
        slog_px = np.sign(px) * np.log(np.abs(px) + (px == 0))
        slog_py = np.sign(py) * np.log(np.abs(py) + (py == 0))

        base_feats = np.stack([
            log_pt,
            slog_px,
            slog_py,
            charges,
            nnvtx,
            puppi
        ], axis=1)

        base_feats = np.concatenate([base_feats, pdg_to_onehot_vec(pdgIds)], axis=1)

        centers = np.where((charges == 0) & (pts >= 1.0) & (recoDen > 0))[0]

        for i in centers:

            target = genSum[i] / recoDen[i]

            idx = np.where(dr[i] < args.dr)[0]
            if len(idx) < 2:
                continue

            idx = idx[np.argsort(dr[i][idx])][:args.max_nodes]

            feats = np.concatenate([
                base_feats[idx],
                deta[i][idx][:, None],
                dphi[i][idx][:, None],
                dr[i][idx][:, None],
                (idx == i).astype(float)[:, None],
            ], axis=1)

            graphs.append(
                Data(
                    x=torch.tensor(feats, dtype=torch.float32),
                    y=torch.tensor([target], dtype=torch.float32),
                    event_idx=torch.tensor([iev]),
                    center_idx=torch.tensor([i]),
                )
            )

            # flush to disk
            if len(graphs) >= args.graphs_per_file:
                out = os.path.join(args.output_dir, f"graphs_{file_idx}.pt")
                torch.save(graphs, out)
                print(f"Saved {len(graphs)} → {out}")
                graphs = []
                file_idx += 1

    # final flush
    if len(graphs) > 0:
        out = os.path.join(args.output_dir, f"graphs_{file_idx}.pt")
        torch.save(graphs, out)
        print(f"Saved {len(graphs)} → {out}")

if __name__ == "__main__":
    main()
