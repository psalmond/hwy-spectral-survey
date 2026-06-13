# SPDX-License-Identifier: MIT
# Copyright (c) 2026 P. Salmond — hwy-spectral-survey (see LICENSE, NOTICE)
"""nsx_x1.py -- X1 gate on the decay-adapted div-free basis.
Assembles H, G (with profile), G0 (U:=0) at gate size on GL(400,200),
whitens H with a CUT-STABILITY SCAN (rev-H protocol), and checks:
  X1 : Ritz spectrum contains lambda1 = +0.11314203274385946 (tol 5e-3),
       stable across cuts;
  X1b: a Ritz value near the exact symmetry eigenvalue -1/2 (d_z U mode;
       representable only via the broad Laguerre block, so tolerance loose);
  X3 : U:=0 control: max Re Ritz(G0) <= -1/4 + eps (numerical-range bound
       Re<v,L0 v> <= -1/4 ||v||^2 holds for ANY subspace -- sharp control);
  X0r: v1 fit error within this exact span (for the record).
Caches G,H,G0 to nsx_x1_grams.npz for reuse (landscape next).
"""
import sys, time
import numpy as np
import nsx_op
from nsx_op import extract_coeffs, composite_radial, assemble, assemble_H
from nsx_basis import angular_factors
from ns_part3k import load_family
from ns_part3f import make_geo, B0, T0, T1, gl_nodes
from ns_part12_gate import Phys
from ns_part3_spectrum import S

LAM = 0.11314203274385946
Nb, Nt = 480, 220
NJH, RHO0, SIG = 50, 20.0, 2.5
NJL, ELLL = 12, 3.0
NKP, NKT = 40, 24

t0 = time.time()
b, wb = gl_nodes(Nb, 0.0, np.pi/2)   # OPEN quarter: kills pressure flux (diag5)
t, wt = gl_nodes(Nt, 0.0, np.pi/2)   # OPEN quarter
geo = make_geo(b, t)
rho = S*np.tan(b)
nsx_op.RHO = rho
wr = rho**2*S/np.cos(b)**2*wb
wa = np.sin(t)*wt

rad, nr = composite_radial(rho, wr, NJH, RHO0, SIG, NJL, ELLL)
ksP = [2*i for i in range(NKP)]; ksT = [2*i+1 for i in range(NKT)]
AP, BP = angular_factors(ksP, t)
_,  BT = angular_factors(ksT, t)
angP = (AP['f'], AP['t'], AP['tt'], BP['f'], BP['t'], BP['tt'])
angT_ = (BT['f'], BT['t'], BT['tt'])
NP, NT = nr*NKP, nr*NKT
N = NP + NT
print(f"[{time.time()-t0:6.1f}s] basis: nr={nr}, N={N}", flush=True)

C  = extract_coeffs(b, t, geo, withU=True)
print(f"[{time.time()-t0:6.1f}s] coefficients (withU) extracted", flush=True)
G  = assemble(C, wr, wa, rad, angP, angT_)
print(f"[{time.time()-t0:6.1f}s] G assembled", flush=True)
C0 = extract_coeffs(b, t, geo, withU=False)
G0 = assemble(C0, wr, wa, rad, angP, angT_)
del C, C0
print(f"[{time.time()-t0:6.1f}s] G0 assembled", flush=True)
H  = assemble_H(wr, wa, rad, angP, angT_)
np.savez('nsx_x1_grams.npz', G=G, G0=G0, H=H, nr=nr, NKP=NKP, NKT=NKT,
         NJH=NJH, RHO0=RHO0, SIG=SIG, NJL=NJL, ELLL=ELLL, Nb=Nb, Nt=Nt)
print(f"[{time.time()-t0:6.1f}s] H assembled; grams cached", flush=True)

# ---- X0r: v1 fit in this exact span
fams, _ = load_family('odd')
P = Phys(fams[-1], b, t, 'cos')
Y = {k: P.f[k] for k in ('u1', 'u2', 'u3')}
R, R1 = rad[0], rad[1]
D = 2*R + rho[None, :]*R1
a = angP[0]; bb = angP[3]; b3 = angT_[0]
y1 = (R*wr) @ Y['u1'] @ (wa*a).T
y2 = (D*wr) @ Y['u2'] @ (wa*bb).T
y3 = (R*wr) @ Y['u3'] @ (wa*b3).T
rhs = np.concatenate([(y1 - y2).ravel(), y3.ravel()])
Etot = ((wr[:, None]*wa[None, :])*(Y['u1']**2+Y['u2']**2+Y['u3']**2)).sum()
evH = np.linalg.eigvalsh(H)
cfit = np.linalg.solve(H + 1e-13*evH[-1]*np.eye(N), rhs)
errfit = np.sqrt(max(Etot - cfit @ rhs, 0)/Etot)
print(f"X0r: v1 fit error in pencil span = {errfit:.3e}", flush=True)

# ---- whiten + eig with cut-stability scan (rev-H protocol)
def ritz(Gm, cut):
    keep = evH > cut*evH[-1]
    Q = np.linalg.eigh(H)[1][:, keep] if False else None
    # (recompute vectors once outside for efficiency)
    return keep

evH_, QH = np.linalg.eigh(H)
results = {}
for cut in (1e-8, 1e-10, 1e-12):
    keep = evH_ > cut*evH_[-1]
    T = QH[:, keep]/np.sqrt(evH_[keep])
    Gt = T.T @ G @ T
    ev = np.linalg.eigvals(Gt)
    G0t = T.T @ G0 @ T
    ev0 = np.linalg.eigvals(G0t)
    i1 = np.argmin(np.abs(ev - LAM))
    ihalf = np.argmin(np.abs(ev - (-0.5)))
    res = dict(nkept=int(keep.sum()),
               lam1=ev[i1], dlam1=abs(ev[i1]-LAM),
               near_half=ev[ihalf],
               maxRe0=ev0.real.max(),
               topRe=np.sort(ev.real)[-6:])
    results[cut] = res
    print(f"cut={cut:.0e}: kept {res['nkept']}/{N}  "
          f"lam1_ritz={res['lam1']:.8f} (|d|={res['dlam1']:.2e})  "
          f"near(-1/2)={res['near_half']:.5f}  "
          f"X3 maxRe(U=0)={res['maxRe0']:+.5f}", flush=True)
    print(f"          top Re Ritz: " +
          " ".join(f"{z:+.5f}" for z in res['topRe']), flush=True)

ok1 = all(results[c]['dlam1'] < 5e-3 for c in results)
okb = any(abs(results[c]['near_half'].real + 0.5) < 5e-2 and
          abs(results[c]['near_half'].imag) < 5e-2 for c in results)
ok3 = all(results[c]['maxRe0'] < -0.25 + 5e-3 for c in results)
print(f"\nX1  (lambda1 to <5e-3, all cuts): {'PASS' if ok1 else 'FAIL'}")
print(f"X1b (Ritz near -1/2)            : {'PASS' if okb else 'WEAK/FAIL'}")
print(f"X3  (U:=0 maxRe <= -1/4+eps)    : {'PASS' if ok3 else 'FAIL'}")
print(f"[{time.time()-t0:6.1f}s] done", flush=True)
