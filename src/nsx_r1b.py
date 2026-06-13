# SPDX-License-Identifier: MIT
# Copyright (c) 2026 P. Salmond — hwy-spectral-survey (see LICENSE, NOTICE)
"""nsx_r1b.py -- iterate the pressure-deflation scalar span on control B:
  rB = min_q ||L_mom v1 - LAM v1 + grad q|| / ||v1||
(v1 exact, pressure dropped). Target floor ~1e-3. Each trial only changes
the span; L_mom v1 is computed once. Also localizes the post-deflation
residual in rho and theta for the best span, to guide enrichment.
"""
import time
import numpy as np
import nsx_op
from nsx_op import hermite_full
from nsx_basis import _leg_derivs
from ns_part3k import load_family, profile_F, Lcols
from ns_part3f import make_geo, gl_nodes
from ns_part12_gate import Phys
from ns_part3_spectrum import S
from scipy.special import eval_genlaguerre

LAM = 0.11314203274385946
t0 = time.time()
Nb, Nt = 480, 220
b, wb = gl_nodes(Nb, 0.0, np.pi/2)
t, wt = gl_nodes(Nt, 0.0, np.pi/2)
geo = make_geo(b, t)
rho = S*np.tan(b)
wr = rho**2*S/np.cos(b)**2*wb
wa = np.sin(t)*wt
WGT = wr[:, None]*wa[None, :]

UF = profile_F()
PU = Phys(UF, b, t, 'cos')
Uval = {k: PU.f[k] for k in ('u1', 'u2', 'u3')}
Ubf = {k: dict(f=PU.f[k], fr=PU.fr[k], frr=PU.frr[k],
               ft=PU.ft[k], ftt=PU.ftt[k]) for k in ('u1', 'u2', 'u3')}
fams, _ = load_family('odd')
Pv1 = Phys(fams[-1], b, t, 'cos')
Pv1.fr['p'] = np.zeros_like(Pv1.f['u1'])
Pv1.ft['p'] = np.zeros_like(Pv1.f['u1'])
out = Lcols(Pv1, Uval, Ubf, geo, True)
Ng = Nb*Nt
r1 = out[:Ng].reshape(Nb, Nt) - LAM*Pv1.f['u1']
r2 = out[Ng:2*Ng].reshape(Nb, Nt) - LAM*Pv1.f['u2']
r3 = out[2*Ng:].reshape(Nb, Nt) - LAM*Pv1.f['u3']
vn2 = (WGT*(Pv1.f['u1']**2 + Pv1.f['u2']**2 + Pv1.f['u3']**2)).sum()
print(f"[{time.time()-t0:5.1f}s] residual field built; "
      f"raw r = {np.sqrt((WGT*(r1*r1+r2*r2+r3*r3)).sum()/vn2):.3e}", flush=True)

def lag0(nm, ell):
    x = rho/ell; E = np.exp(-0.5*x)
    Sf, S1 = [], []
    for m in range(nm):
        L0 = eval_genlaguerre(m, 0, x)
        L1 = -eval_genlaguerre(m-1, 1, x) if m >= 1 else 0*x
        Sf.append(L0*E); S1.append((L1 - 0.5*L0)*E/ell)
    return np.array(Sf), np.array(S1)

def herm(nm, r0, sg):
    A = hermite_full(range(nm), r0, sg, rho)
    return A[0], A[1]

def trial(tag, blocks, nkq):
    Sf = np.vstack([x[0] for x in blocks]); S1 = np.vstack([x[1] for x in blocks])
    Sor = Sf/rho[None, :]
    ksQ = [2*i for i in range(nkq)]
    mu, st = np.cos(t), np.sin(t)
    Pq, Pq1 = [], []
    for k in ksQ:
        P, P1 = _leg_derivs(k+1, mu, 1)[:2]
        Pq.append(P); Pq1.append(-st*P1)
    Pq = np.array(Pq); Pq1 = np.array(Pq1)
    M = (np.kron((S1*wr) @ S1.T, (Pq*wa) @ Pq.T)
         + np.kron((Sor*wr) @ Sor.T, (Pq1*wa) @ Pq1.T))
    rhs = ((S1*wr) @ r1 @ (wa[:, None]*Pq.T)
           + (Sor*wr) @ r2 @ (wa[:, None]*Pq1.T)).ravel()
    evM, QM = np.linalg.eigh(M)
    y = QM.T @ rhs
    n2 = (WGT*(r1*r1 + r2*r2 + r3*r3)).sum()
    res = {}
    for cut in (1e-10, 1e-12, 1e-13):
        kp = evM > cut*evM[-1]
        res[cut] = np.sqrt(max(n2 - (y[kp]**2/evM[kp]).sum(), 0)/vn2)
    print(f"{tag:34s} Nq={M.shape[0]:5d}  " +
          "  ".join(f"{c:.0e}:{v:.3e}" for c, v in res.items()), flush=True)
    return res, (Sf, S1, Pq, Pq1, QM, evM, y, n2)

# baseline (= nsx_r1 span)
trial("base H40s2.5+L3x12+L8x8, kq48",
      [herm(40, 20.0, 2.5), lag0(12, 3.0), lag0(8, 8.0)], 48)
# more angular
trial("kq 48->72", [herm(40, 20.0, 2.5), lag0(12, 3.0), lag0(8, 8.0)], 72)
# more shell radial
trial("H 40->60", [herm(60, 20.0, 2.5), lag0(12, 3.0), lag0(8, 8.0)], 48)
# both
res, pack = trial("H60 kq72",
                  [herm(60, 20.0, 2.5), lag0(12, 3.0), lag0(8, 8.0)], 72)
# wider Laguerre ladder
trial("H60 kq72 +L16x8", [herm(60, 20.0, 2.5), lag0(12, 3.0),
                          lag0(8, 8.0), lag0(8, 16.0)], 72)
# sub-shell sharper hermite family
res2, pack2 = trial("H60 + H30s1.2 kq72",
                    [herm(60, 20.0, 2.5), herm(30, 20.0, 1.2),
                     lag0(12, 3.0), lag0(8, 8.0)], 72)

# ---- localize the residual for the best span so far
best = pack2 if min(res2.values()) < min(res.values()) else pack
Sf, S1, Pq, Pq1, QM, evM, y, n2 = best
kp = evM > 1e-12*evM[-1]
d = QM[:, kp] @ (y[kp]/evM[kp])
ns, nk = Sf.shape[0], Pq.shape[0]
dq = d.reshape(ns, nk)
g1 = (S1.T @ dq) @ Pq
g2 = ((Sf/rho[None, :]).T @ dq) @ Pq1
q1 = r1 - g1; q2 = r2 - g2; q3 = r3
dens_r = (q1*q1 + q2*q2 + q3*q3)*WGT
pr = dens_r.sum(1); pt = dens_r.sum(0)
ir = np.argsort(pr)[::-1][:6]; it = np.argsort(pt)[::-1][:6]
print("post-deflation residual concentration:", flush=True)
print("  rho   hot: " + " ".join(f"{rho[i]:8.3f}({pr[i]/dens_r.sum():.2f})"
                                 for i in ir), flush=True)
print("  theta hot: " + " ".join(f"{t[i]:6.4f}({pt[i]/dens_r.sum():.2f})"
                                 for i in it), flush=True)
print(f"[{time.time()-t0:5.1f}s] done", flush=True)
