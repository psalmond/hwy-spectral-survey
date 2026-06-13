# SPDX-License-Identifier: MIT
# Copyright (c) 2026 P. Salmond — hwy-spectral-survey (see LICENSE, NOTICE)
"""nsx_x4b.py -- lean Schur + landscape on the masked, origin-regularized
basis (grams from nsx_x5_grams.npz). Memory-ordered to stay under ~1.8 GB.
Verdict numbers: bottom(Ad~) (instrument noise floor in r^2), r(lam1) valley
vs off-valley, real-axis scan.
"""
import time
import numpy as np
import scipy.linalg as sla
from nsx_basis import _leg_derivs
from nsx_op import hermite_full
from nsx_basis import radial_factors
from ns_part3f import gl_nodes
from ns_part3_spectrum import S
from scipy.special import eval_genlaguerre

LAM = 0.11314203274385946
t0 = time.time()
def log(s): print(f"[{time.time()-t0:7.1f}s] {s}", flush=True)

# ---- rebuild the 1-D pressure-span pieces (cheap)
Nb, Nt = 480, 220
b, wb = gl_nodes(Nb, 0.0, np.pi/2)
t, wt = gl_nodes(Nt, 0.0, np.pi/2)
rho = S*np.tan(b)
wr = rho**2*S/np.cos(b)**2*wb
wa = np.sin(t)*wt
def lag0(nm, ell):
    x = rho/ell; E = np.exp(-0.5*x)
    Sf, S1 = [], []
    for m in range(nm):
        L0 = eval_genlaguerre(m, 0, x)
        L1 = -eval_genlaguerre(m-1, 1, x) if m >= 1 else 0*x
        Sf.append(L0*E); S1.append((L1 - 0.5*L0)*E/ell)
    return np.array(Sf), np.array(S1)
HH = hermite_full(range(60), 20.0, 2.5, rho)
H2 = hermite_full(range(30), 20.0, 1.2, rho)
L3 = lag0(12, 3.0); L8 = lag0(8, 8.0)
Sf = np.vstack([HH[0], H2[0], L3[0], L8[0]])
S1g = np.vstack([HH[1], H2[1], L3[1], L8[1]])
Sor = Sf/rho[None, :]
mu, st = np.cos(t), np.sin(t)
Pq, Pq1 = [], []
for k in [2*i for i in range(72)]:
    P, P1 = _leg_derivs(k+1, mu, 1)[:2]
    Pq.append(P); Pq1.append(-st*P1)
Pq = np.array(Pq); Pq1 = np.array(Pq1)
Mg = np.kron((S1g*wr) @ S1g.T, (Pq*wa) @ Pq.T)
Mg += np.kron((Sor*wr) @ Sor.T, (Pq1*wa) @ Pq1.T)
log("M built")
evM, QM = sla.eigh(Mg, overwrite_a=True, check_finite=False)
del Mg
log("eigh(M) done")
Z = np.load('nsx_x5_grams.npz')
B = Z['B']
Y = QM.T @ B.T
del QM, B
log("Y formed; QM/B freed")
AA, G, H = Z['AA'], Z['G'], Z['H']
del Z
evH, QH = sla.eigh(H, check_finite=False)
kpH = evH > 1e-10*evH[-1]
T = QH[:, kpH]/np.sqrt(evH[kpH])
del QH, H
log(f"H whitened (kept {kpH.sum()})")
Gt = T.T @ G @ T
del G
Ats = {}
for cut in (1e-10, 1e-12):
    kp = evM > cut*evM[-1]
    Ys = Y[kp]/np.sqrt(evM[kp])[:, None]
    AAd = AA - Ys.T @ Ys
    del Ys
    At = T.T @ AAd @ T
    del AAd
    bA = sla.eigh(At, eigvals_only=True, subset_by_index=[0, 2],
                  check_finite=False)
    log(f"cut {cut:.0e}: kept {kp.sum()} grad modes; bottom(Ad~) = {bA}")
    Ats[cut] = At
np.savez('nsx_x5_prep.npz', Gt=Gt, At_10=Ats[1e-10], At_12=Ats[1e-12])
log("saved nsx_x5_prep.npz")

def rmin(At, lam):
    n = At.shape[0]
    Nl = At - lam*(Gt + Gt.T) + lam*lam*np.eye(n)
    w = sla.eigh(Nl, eigvals_only=True, subset_by_index=[0, 0],
                 check_finite=False)
    return np.sqrt(max(w[0], 0.0)), w[0]

for cut in (1e-10, 1e-12):
    r1, raw1 = rmin(Ats[cut], LAM)
    r0, _ = rmin(Ats[cut], 0.0)
    r25, _ = rmin(Ats[cut], 0.25)
    log(f"cut {cut:.0e}: r(lam1)={r1:.4e} (raw {raw1:+.3e})  "
        f"r(0)={r0:.4e}  r(0.25)={r25:.4e}")

At = Ats[1e-12]
lams = np.arange(-0.10, 0.3001, 0.005)
rr = np.array([rmin(At, x)[0] for x in lams])
np.savez('nsx_x5_real.npz', lams=lams, r=rr)
i0 = np.argmin(rr)
log("real-axis scan (cut 1e-12):")
for i in range(0, len(lams), 8):
    print("   " + " ".join(f"{lams[j]:+.3f}:{rr[j]:.3e}"
                           for j in range(i, min(i+8, len(lams)))), flush=True)
fine = LAM + np.arange(-0.012, 0.0121, 0.001)
rf = np.array([rmin(At, x)[0] for x in fine])
j0 = np.argmin(rf)
log(f"coarse min at lam={lams[i0]:+.4f} r={rr[i0]:.4e};  "
    f"fine min at lam={fine[j0]:+.6f} r={rf[j0]:.4e}")
log("done")
