# SPDX-License-Identifier: MIT
# Copyright (c) 2026 P. Salmond — hwy-spectral-survey (see LICENSE, NOTICE)
"""nsx_x5r.py -- pointwise certification of the landscape valley minimizer:
extract argmin vector at lam1 from the prep pencil, rebuild the field on the
grid, ONE direct Lcols, explicit grid deflation -> certified upper bound,
independent of the gram pipeline. Should reproduce r(lam1) ~ 9.3e-3.
"""
import numpy as np, time, scipy.linalg as sla
import nsx_op
from nsx_op import hermite_full
from nsx_basis import angular_factors, radial_factors, _leg_derivs
from ns_part3k import load_family, profile_F, Lcols
from ns_part3f import make_geo, gl_nodes
from ns_part12_gate import Phys
from ns_part3_spectrum import S
from scipy.special import eval_genlaguerre
import nsx_x5 as X5

LAM = 0.11314203274385946
t0 = time.time()
def log(s): print(f"[{time.time()-t0:6.1f}s] {s}", flush=True)

Nb, Nt = 480, 260
b, wb = gl_nodes(Nb, 0.0, np.pi/2)
t, wt = gl_nodes(Nt, 0.0, np.pi/2)
geo = make_geo(b, t)
rho = S*np.tan(b)
nsx_op.RHO = rho
wr = rho**2*S/np.cos(b)**2*wb
wa = np.sin(t)*wt
WGT = wr[:, None]*wa[None, :]
rad, _, _, _ = X5.build_radial(rho, wr, 60, 24, 12)
slim = list(range(36)) + list(range(60, 66)) + list(range(84, 88))
torr = list(range(24)) + list(range(60, 62)) + list(range(84, 88))
radS, radT = X5.sub(rad, slim), X5.sub(rad, torr)
ksP_lo = [2*i for i in range(20)]
ksP_hi = [2*i for i in range(20, 61)]
ksT_ = [2*i+1 for i in range(52)]
APl, BPl = angular_factors(ksP_lo, t)
APh, BPh = angular_factors(ksP_hi, t)
_, BT = angular_factors(ksT_, t)
log("basis rebuilt")

# minimizer at lam1 from prep
Zp = np.load('nsx_x5_prep.npz')
Gt, At = Zp['Gt'], Zp['At_12']
n = At.shape[0]
Nl = At - LAM*(Gt + Gt.T) + LAM*LAM*np.eye(n)
w, V = sla.eigh(Nl, subset_by_index=[0, 0], check_finite=False)
log(f"argmin extracted: r^2 = {w[0]:.6e} -> r = {np.sqrt(w[0]):.4e}")
# map whitened -> coefficients
Z = np.load('nsx_x5_grams.npz')
H = Z['H']; del Z
evH, QH = sla.eigh(H, check_finite=False)
kp = evH > 1e-10*evH[-1]
T = QH[:, kp]/np.sqrt(evH[kp])
c = T @ V[:, 0]
del H, QH, T
log("coefficients mapped")

# build field blockwise
sizes = [96*20, 46*41, 30*52]
off = np.concatenate([[0], np.cumsum(sizes)]).astype(int)
class F: pass
f = F(); f.f, f.fr, f.frr, f.ft, f.ftt = {}, {}, {}, {}, {}
for k in ('u1', 'u2', 'u3'):
    for dd in ('f', 'fr', 'frr', 'ft', 'ftt'):
        getattr(f, dd)[k] = np.zeros((Nb, Nt))
def addpol(rd, AP, BP, cP):
    R, R1, R2, R3 = rd
    D = 2*R + rho[None, :]*R1
    D1 = 3*R1 + rho[None, :]*R2
    D2 = 4*R2 + rho[None, :]*R3
    mix = lambda Rm, Am: (Rm.T @ cP) @ Am
    f.f['u1'] += mix(R, AP['f']); f.fr['u1'] += mix(R1, AP['f'])
    f.frr['u1'] += mix(R2, AP['f'])
    f.ft['u1'] += mix(R, AP['t']); f.ftt['u1'] += mix(R, AP['tt'])
    f.f['u2'] -= mix(D, BP['f']); f.fr['u2'] -= mix(D1, BP['f'])
    f.frr['u2'] -= mix(D2, BP['f'])
    f.ft['u2'] -= mix(D, BP['t']); f.ftt['u2'] -= mix(D, BP['tt'])
addpol(rad, APl, BPl, c[off[0]:off[1]].reshape(96, 20))
addpol(radS, APh, BPh, c[off[1]:off[2]].reshape(46, 41))
R, R1, R2, R3 = radT
cT = c[off[2]:off[3]].reshape(30, 52)
mixT = lambda Rm, Am: (Rm.T @ cT) @ Am
f.f['u3'] += mixT(R, BT['f']); f.fr['u3'] += mixT(R1, BT['f'])
f.frr['u3'] += mixT(R2, BT['f'])
f.ft['u3'] += mixT(R, BT['t']); f.ftt['u3'] += mixT(R, BT['tt'])
f.fr['p'] = np.zeros((Nb, Nt)); f.ft['p'] = np.zeros((Nb, Nt))
vn2 = (WGT*(f.f['u1']**2 + f.f['u2']**2 + f.f['u3']**2)).sum()
log(f"field built; ||v||^2 = {vn2:.6f} (expect ~1)")

UF = profile_F(); PU = Phys(UF, b, t, 'cos')
Uval = {k: PU.f[k] for k in ('u1', 'u2', 'u3')}
Ubf = {k: dict(f=PU.f[k], fr=PU.fr[k], frr=PU.frr[k],
               ft=PU.ft[k], ftt=PU.ftt[k]) for k in ('u1', 'u2', 'u3')}
out = Lcols(f, Uval, Ubf, geo, True)
Ng = Nb*Nt
r1 = out[:Ng].reshape(Nb, Nt) - LAM*f.f['u1']
r2 = out[Ng:2*Ng].reshape(Nb, Nt) - LAM*f.f['u2']
r3 = out[2*Ng:].reshape(Nb, Nt) - LAM*f.f['u3']
log(f"raw pointwise r = {np.sqrt((WGT*(r1*r1+r2*r2+r3*r3)).sum()/vn2):.4e}")

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
Mg = np.kron((S1g*wr) @ S1g.T, (Pq*wa) @ Pq.T) \
   + np.kron((Sor*wr) @ Sor.T, (Pq1*wa) @ Pq1.T)
rhsq = ((S1g*wr) @ r1 @ (wa[:, None]*Pq.T)
        + (Sor*wr) @ r2 @ (wa[:, None]*Pq1.T)).ravel()
n2 = (WGT*(r1*r1 + r2*r2 + r3*r3)).sum()
tr = np.trace(Mg)/len(Mg)
for jit in (1e-10, 1e-12):
    Cf = sla.cho_factor(Mg + jit*tr*np.eye(len(Mg)), lower=True)
    d = sla.cho_solve(Cf, rhsq)
    del Cf
    rr = np.sqrt(max(n2 - d @ rhsq, 0)/vn2)
    print(f"CERTIFIED pointwise deflated r(lam1) jit{jit:.0e}: {rr:.4e}  "
          f"(gram pipeline said 9.314e-03)", flush=True)
log("done")
