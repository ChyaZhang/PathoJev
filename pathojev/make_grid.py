"""Pool TRIDENT CONCH v1.5 patch features (20x, 512 x 512 px) of one slide onto a 32 x 32 grid.

Patch coordinates are mapped onto a square grid of side max(width, height) / 32 of the tissue bounding box; each
non-empty cell keeps the mean feature of its patches.

  python -m pathojev.make_grid <slide>.h5 <slide>.pt
"""
import sys

import h5py
import numpy as np
import torch


def make_grid(h5_path, grid=32):
    with h5py.File(h5_path, "r") as h:
        x = h["features"][:].astype(np.float32).reshape(-1, h["features"].shape[-1])
        xy = h["coords"][:].astype(np.float64).reshape(-1, 2)
    lo = xy.min(0)
    side = max((xy.max(0) - lo).max(), 1.0) / grid * (1 + 1e-6)
    rc = np.minimum(((xy - lo) / side).astype(np.int64), grid - 1)[:, ::-1]  # (row = y, col = x)
    key = rc[:, 0] * grid + rc[:, 1]
    uniq, inv = np.unique(key, return_inverse=True)
    acc = np.zeros((len(uniq), x.shape[1]), np.float64)
    np.add.at(acc, inv, x)
    acc /= np.bincount(inv)[:, None]
    return {"rc": torch.from_numpy(np.stack([uniq // grid, uniq % grid], 1)).short(), "x": torch.from_numpy(acc).half()}


if __name__ == "__main__":
    torch.save(make_grid(sys.argv[1]), sys.argv[2])
