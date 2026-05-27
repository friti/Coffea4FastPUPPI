import glob
import numpy as np
import torch
from tqdm import tqdm


def load_pt_dataset(pattern, max_nodes, limit_files=None, return_metadata=False):
    files = sorted(glob.glob(pattern))
    if limit_files is not None:
        files = files[:limit_files]

    print(f"Found {len(files)} files")

    data_list = []
    for f in tqdm(files, desc="Loading .pt files"):
        data_list.extend(torch.load(f, map_location="cpu"))

    print(f"Total samples: {len(data_list)}")

    X, MASK, Y = [], [], []

    # optional metadata
    meta = {
        "seed_pt": [],
        "seed_eta": [],
        "genPtSum": [],
        "recoPtSum": [],
        "event_idx": [],
        "center_idx": [],
    }

    for d in tqdm(data_list, desc="Converting to dense tensors"):

        x = d.x.numpy()
        n = x.shape[0]

        x_pad = np.zeros((max_nodes, x.shape[1]), dtype=np.float32)
        mask = np.zeros(max_nodes, dtype=np.float32)

        n_keep = min(n, max_nodes)

        x_pad[:n_keep] = x[:n_keep]
        mask[:n_keep] = 1.0

        X.append(x_pad)
        MASK.append(mask)
        Y.append(d.y.item())

        if return_metadata:
            meta["seed_pt"].append(d.seed_pt.item())
            meta["seed_eta"].append(d.seed_eta.item())
            meta["genPtSum"].append(d.genPtSum.item())
            meta["recoPtSum"].append(d.recoPtSum.item())
            meta["event_idx"].append(d.event_idx.item())
            meta["center_idx"].append(d.center_idx.item())

    X = np.array(X)
    MASK = np.array(MASK)
    Y = np.array(Y)

    print("Shapes:")
    print("X:", X.shape)
    print("MASK:", MASK.shape)
    print("Y:", Y.shape)

    if return_metadata:
        for k in meta:
            meta[k] = np.array(meta[k])
        return X, MASK, Y, meta

    return X, MASK, Y

## Now this can be used in the training like the following:

"""
X, mask, y, meta = load_pt_dataset(
    "dataset/pt/graphs_*.pt",
    max_nodes=128,
    return_metadata=True
)

x = X
mask = mask
y = y

"""
