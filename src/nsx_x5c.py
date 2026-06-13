# SPDX-License-Identifier: MIT
# Copyright (c) 2026 P. Salmond — hwy-spectral-survey (see LICENSE, NOTICE)
"""nsx_x5c.py -- complex-grid landscape survey on the tapered basis.
r(lam) = sqrt(lam_min(At - lam Gt^T - conj(lam) Gt + |lam|^2 I)), dense
eigh-subset per point (cut 1e-12 prep). Conjugate symmetry: Im > 0 only.
Coarse grid Re in [0, 0.35] x Im in [0.05, 0.60], step 0.05 -> 96 points,
checkpoint per row. Any local dip -> refine later.
"""
import os, time
import numpy as np
import scipy.linalg as sla

t0 = time.time()
def log(s): print(f"[{time.time()-t0:7.1f}s] {s}", flush=True)

Zp = np.load('nsx_x5_prep.npz')
Gt = Zp['Gt']; At = Zp['At_12']
n = At.shape[0]
GtT = Gt.T
log(f"loaded prep (n={n})")

def rmin(lam):
    Nl = (At - lam*GtT - np.conj(lam)*Gt
          + (abs(lam)**2)*np.eye(n)).astype(np.complex128)
    w = sla.eigh(Nl, eigvals_only=True, subset_by_index=[0, 0],
                 check_finite=False, overwrite_a=True)
    return np.sqrt(max(w[0], 0.0))

Re = np.arange(0.0, 0.351, 0.05)
Im = np.arange(0.05, 0.601, 0.05)
R = np.full((len(Re), len(Im)), np.nan)
ck = 'nsx_x5_grid_ck.npz'
i0 = 0
if os.path.exists(ck):
    Zc = np.load(ck)
    R[:] = Zc['R']; i0 = int(Zc['i0'])
    log(f"resuming at row {i0}")
for i in range(i0, len(Re)):
    for j in range(len(Im)):
        R[i, j] = rmin(complex(Re[i], Im[j]))
    np.savez(ck, R=R, i0=i+1, Re=Re, Im=Im)
    log(f"Re={Re[i]:+.2f}: " +
        " ".join(f"{R[i, j]:.3e}" for j in range(len(Im))))
np.savez('nsx_x5_grid.npz', R=R, Re=Re, Im=Im)
log("saved nsx_x5_grid.npz; done")
