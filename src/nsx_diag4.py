# SPDX-License-Identifier: MIT
# Copyright (c) 2026 P. Salmond — hwy-spectral-survey (see LICENSE, NOTICE)
"""nsx_diag4.py -- decide between (a) stored pressure slot is not a pure
gradient, (b) basis has hidden pathology, (c) my IBP/quadrature wrong.
T1: synthetic scalar q = exp(-(rho-20)^2/8)*cos(2 th); grad q analytic;
    <w_i, grad q> must be ~edge-flux-tiny if identity+quadrature are right.
T2: curl test of stored slot (g1,g2)=(fr[p], ft[p]/rho):
    gradient <=> d(g1)/dth == d(rho g2)/drho ; FD on the tensor grid.
T3: locate giant entries of raw H; print top raw-norm basis indices.
"""
import numpy as np
from nsx_op import hermite_full
from nsx_basis import angular_factors, radial_factors
from ns_part3k import load_family
from ns_part3f import B0, T0, T1, gl_nodes
from ns_part12_gate import Phys
from ns_part3_spectrum import S

NJH, RHO0, SIG = 50, 20.0, 2.5
NJL, ELLL = 12, 3.0
NKP = 40
BHI = np.pi/2 - 0.04
Nb, Nt = 1000, 200
b, wb = gl_nodes(Nb, B0, BHI)
t, wt = gl_nodes(Nt, T0, T1)
rho = S*np.tan(b)
wr = rho**2*S/np.cos(b)**2*wb
wa = np.sin(t)*wt
RH = hermite_full(range(NJH), RHO0, SIG, rho)
RL = radial_factors(range(NJL), ELLL, rho)
R = np.vstack([RH[0], RL['R']]); R1 = np.vstack([RH[1], RL['R1']])
D = 2*R + rho[None, :]*R1
ksP = [2*i for i in range(NKP)]
kk = np.array(ksP, float); fac = (kk+1)*(kk+2)
AP, BP = angular_factors(ksP, t)
aF = AP['f']*fac[:, None]; bb = BP['f']

def pair(g1, g2):
    return (R*wr) @ g1 @ (wa*aF).T + (D*wr) @ (-g2) @ (wa*bb).T

# T1 synthetic
q = np.exp(-(rho[:, None]-20.0)**2/8.0)*np.cos(2*t[None, :])
q1 = (-(rho[:, None]-20.0)/4.0)*q
q2 = np.exp(-(rho[:, None]-20.0)**2/8.0)*(-2*np.sin(2*t[None, :]))/rho[:, None]
dsyn = pair(q1, q2)
# scale reference: same pairing applied to the raw field (q1,q2) magnitude
ref = np.linalg.norm(pair(np.abs(q1), np.abs(q2)))
print(f"T1 synthetic grad: ||<w,grad q>|| = {np.linalg.norm(dsyn):.3e}  "
      f"(|pairing| scale {ref:.3e};  ratio {np.linalg.norm(dsyn)/ref:.3e})")

# T2 curl of stored pressure slot (FD on grid, interior points)
fams, _ = load_family('odd')
P = Phys(fams[-1], b, t, 'cos')
g1 = P.fr['p']; g2 = P.ft['p']/rho[:, None]
db = np.gradient(rho)
dth = np.gradient(t)
d_g1_dth = np.gradient(g1, axis=1)/dth[None, :]
d_rg2_drho = np.gradient(rho[:, None]*g2, axis=0)/db[:, None]
curl = d_g1_dth - d_rg2_drho
sc = max(np.abs(d_g1_dth).max(), np.abs(d_rg2_drho).max())
print(f"T2 curl(stored grad-p slot): max|curl|/scale = "
      f"{np.abs(curl[5:-5,5:-5]).max()/sc:.3e}  (scale {sc:.3e})")
# also: how big is the slot itself
print(f"   max|fr[p]|={np.abs(g1).max():.3e}  max|ft[p]/rho|={np.abs(g2).max():.3e}")

# T3 raw norms
W1 = np.einsum('it,jt->ij', wa*aF, aF)
W2 = np.einsum('it,jt->ij', wa*bb, bb)
nrm2 = (np.einsum('pb,ii,pb->pi', R*wr, W1, R)
        + np.einsum('pb,ii,pb->pi', D*wr, W2, D)) if False else None
diag1 = np.einsum('pb,qb->pq', R*R*wr, np.ones((1, 1))) # placeholder
n2 = (R*R*wr).sum(1)[:, None]*np.diag(W1)[None, :] \
     + (D*D*wr).sum(1)[:, None]*np.diag(W2)[None, :]
idx = np.unravel_index(np.argsort(n2.ravel())[::-1][:6], n2.shape)
print("T3 top raw ||w||^2 (r index, k index):")
for r, k in zip(*idx):
    blk = 'H' if r < NJH else f'L{r-NJH}'
    print(f"   r={r:2d}({blk})  k={ksP[k]:3d}  ||w||^2={n2[r,k]:.3e}  "
          f"radial part ||R||^2={(R[r]**2*wr).sum():.3e} ||D||^2={(D[r]**2*wr).sum():.3e}")
