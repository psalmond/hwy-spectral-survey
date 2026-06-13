# SPDX-License-Identifier: MIT
# Copyright (c) 2026 P. Salmond — hwy-spectral-survey (see LICENSE, NOTICE)
"""nsx_x0f.py -- measure the target instead of guessing families.
(1) Exact angular spectra of v1 on the open quarter domain:
      u1 in span{P_{k+1}}, u2 in span{sin P'_{k+1}} (k even),
      u3 in span{sin P'_{k+1}} (k odd) -- mutually orthogonal by parity,
      so Parseval is exact (gates printed). Cumulative tails -> required K.
(2) Per-k DIV-FREE-CONSTRAINED radial fits (R drives u1, D=2R+rhoR' drives
    u2 jointly) for three radial families at matched size:
      A: Hermite50(20,2.5) + Lag12(3)   [current, origin-masked]
      B: log-Gaussian shell lattice in ln(rho)  ["fractal" family]
      C: A union B (can the union beat both?)
    Aggregate fit error vs K_max -> the design curve and the verdict.
"""
import numpy as np, time
from nsx_op import hermite_full
from nsx_basis import radial_factors
from ns_part3k import load_family
from ns_part3f import gl_nodes
from ns_part12_gate import Phys
from ns_part3_spectrum import S
from scipy.special import eval_legendre

t0 = time.time()
Nb, Nt = 560, 300
b, wb = gl_nodes(Nb, 0.0, np.pi/2)
t, wt = gl_nodes(Nt, 0.0, np.pi/2)
rho = S*np.tan(b)
wr = rho**2*S/np.cos(b)**2*wb
wa = np.sin(t)*wt
mu = np.cos(t)
fams, _ = load_family('odd')
P = Phys(fams[-1], b, t, 'cos')
U1, U2, U3 = P.f['u1'], P.f['u2'], P.f['u3']
E1t = ((wr[:, None]*wa[None, :])*U1*U1).sum()
E2t = ((wr[:, None]*wa[None, :])*U2*U2).sum()
E3t = ((wr[:, None]*wa[None, :])*U3*U3).sum()
Etot = E1t + E2t + E3t
print(f"[{time.time()-t0:5.1f}s] component energies: "
      f"u1 {E1t/Etot:.3f}  u2 {E2t/Etot:.3f}  u3 {E3t/Etot:.3f}", flush=True)

# ---------------- angular families up to high degree (recurrence)
KMAX = 240                      # max Legendre degree used (n = k+1)
Pn = np.empty((KMAX+1, Nt))     # P_n(mu)
Pn[0] = 1.0; Pn[1] = mu
for n in range(1, KMAX):
    Pn[n+1] = ((2*n+1)*mu*Pn[n] - n*Pn[n-1])/(n+1)
dPn = np.empty_like(Pn)         # P_n'(mu)
dPn[0] = 0.0; dPn[1] = 1.0
for n in range(1, KMAX):
    dPn[n+1] = dPn[n-1] + (2*n+1)*Pn[n]
st = np.sin(t)

ksP = np.arange(0, 200, 2)      # poloidal k (even), n = k+1 odd
ksT = np.arange(1, 200, 2)      # toroidal k (odd),  n = k+1 even
nP1 = 1.0/(2*ksP + 3)                                   # ||P_n||^2 quarter
nP2 = (ksP+1)*(ksP+2)/(2.0*ksP + 3)                     # ||sin P_n'||^2
nT2 = (ksT+1)*(ksT+2)/(2.0*ksT + 3)

A1 = Pn[ksP+1]                  # (K, Nt)
A2 = st[None, :]*dPn[ksP+1]
A3 = st[None, :]*dPn[ksT+1]
C1 = (U1 @ (wa*A1).T)/nP1[None, :]      # (Nb, K) coefficient profiles
C2 = (U2 @ (wa*A2).T)/nP2[None, :]
C3 = (U3 @ (wa*A3).T)/nT2[None, :]
e1 = (wr[:, None]*C1*C1*nP1[None, :]).sum(0)
e2 = (wr[:, None]*C2*C2*nP2[None, :]).sum(0)
e3 = (wr[:, None]*C3*C3*nT2[None, :]).sum(0)
print(f"[{time.time()-t0:5.1f}s] Parseval gates: "
      f"u1 {e1.sum()/E1t:.8f}  u2 {e2.sum()/E2t:.8f}  u3 {e3.sum()/E3t:.8f}",
      flush=True)

tail = (e1[::-1].cumsum()[::-1] + e2[::-1].cumsum()[::-1]
        + e3[::-1].cumsum()[::-1])/Etot     # tail energy from index i on
print("angular tail sqrt(energy fraction) beyond k:", flush=True)
for i in (10, 20, 28, 36, 44, 52, 60, 70, 80, 90, 99):
    print(f"   k>={ksP[i]:3d}: {np.sqrt(tail[i]):.3e}", flush=True)
need = np.where(np.sqrt(tail) < 3e-4)[0]
print(f"   K needed for angular tail < 3e-4: "
      f"{'k=' + str(ksP[need[0]]) if len(need) else '>198 (NOT reached)'}",
      flush=True)

# ---------------- radial families (origin-masked)
def msk(R, R1):
    m0 = rho*rho/(rho*rho + 4.0)
    m1 = 8.0*rho/(rho*rho + 4.0)**2
    return R*m0, R1*m0 + R*m1

RH, RH1 = hermite_full(range(50), 20.0, 2.5, rho)[:2]
rl = radial_factors(range(12), 3.0, rho)
famA = msk(np.vstack([RH, rl['R']]), np.vstack([RH1, rl['R1']]))

x = np.log(rho)
def logshell(centers, sx):
    z = (x[None, :] - centers[:, None])/sx
    R = np.exp(-0.5*z*z)
    R1 = R*(-z/sx)/rho[None, :]          # d/drho = (1/rho) d/dx
    return R, R1
cset = np.arange(np.log(5.0), np.log(55.0), 0.055)       # ~44 shells
famB = msk(*logshell(cset, 0.075))
famC = (np.vstack([famA[0], famB[0]]), np.vstack([famA[1], famB[1]]))
print(f"[{time.time()-t0:5.1f}s] families: A {famA[0].shape[0]}  "
      f"B {famB[0].shape[0]}  C {famC[0].shape[0]}", flush=True)

# ---------------- per-k constrained fits
def fit_curve(fam, tag):
    R, R1 = fam
    D = 2*R + rho[None, :]*R1
    nf = R.shape[0]
    capP = np.zeros(len(ksP)); capT = np.zeros(len(ksT))
    for i, k in enumerate(ksP):
        fac = (k+1.0)*(k+2.0)
        M = (fac*fac*nP1[i])*((R*wr) @ R.T) + nP2[i]*((D*wr) @ D.T)
        r = (fac*nP1[i])*((R*wr) @ C1[:, i]) - nP2[i]*((D*wr) @ C2[:, i])
        ev = np.linalg.eigvalsh(M)
        c = np.linalg.solve(M + 1e-13*ev[-1]*np.eye(nf), r)
        capP[i] = c @ r
    GRR = (R*wr) @ R.T
    evG = np.linalg.eigvalsh(GRR)
    GRRi = GRR + 1e-13*evG[-1]*np.eye(nf)
    for i, k in enumerate(ksT):
        r = (R*wr) @ C3[:, i]
        c = np.linalg.solve(GRRi, r)
        capT[i] = nT2[i]*(c @ r)
    print(f"  {tag}: aggregate fit error vs K_max "
          "(both pol/tor truncated at K):", flush=True)
    for K in (40, 56, 72, 88, 104, 120, 140, 160, 180, 198):
        iP = ksP <= K; iT = ksT <= K
        cap = capP[iP].sum() + capT[iT].sum()
        miss = (Etot - cap)/Etot
        print(f"     K={K:3d}: err={np.sqrt(max(miss, 0)):.4e}", flush=True)

fit_curve(famC, "C union (A + logshell s0.075)")
# finer log lattice
famB2 = msk(*logshell(np.arange(np.log(5.0), np.log(55.0), 0.030), 0.042))
print(f"B2: {famB2[0].shape[0]} shells", flush=True)
fit_curve(famB2, "B2 log-shells fine s0.042")
# dual-sigma Hermite (pressure-span trick)
RH9, RH91 = hermite_full(range(70), 20.0, 2.5, rho)[:2]
RS, RS1 = hermite_full(range(30), 20.0, 1.2, rho)[:2]
famA2 = msk(np.vstack([RH9, RS, rl['R']]), np.vstack([RH91, RS1, rl['R1']]))
fit_curve(famA2, "A2 H70s2.5+H30s1.2+Lag12")
# Chebyshev in beta (HWY native coordinate)
bb_ = 2*b/(np.pi/2) - 1.0
mC = 90
TC = np.cos(np.arange(mC)[:, None]*np.arccos(bb_)[None, :])
dT = np.zeros_like(TC)
for m in range(1, mC):
    sb = np.sqrt(np.maximum(1 - bb_*bb_, 1e-300))
    dT[m] = m*np.sin(m*np.arccos(bb_))/sb
dT *= (2/(np.pi/2))*(np.cos(b)**2/S)[None, :]   # d/drho = cos^2(beta)/S d/dbeta
famD = msk(TC, dT)
fit_curve(famD, "D Chebyshev-in-beta m90")
famE = (np.vstack([famA[0], famD[0]]), np.vstack([famA[1], famD[1]]))
# economy scan: minimal (radial, KP, KT) with fit <= 3e-4
def fit_at(fam, KP, KT):
    R, R1 = fam
    D = 2*R + rho[None, :]*R1
    nf = R.shape[0]
    cap = 0.0
    for i, k in enumerate(ksP):
        if k > KP: break
        fac = (k+1.0)*(k+2.0)
        M = (fac*fac*nP1[i])*((R*wr) @ R.T) + nP2[i]*((D*wr) @ D.T)
        r = (fac*nP1[i])*((R*wr) @ C1[:, i]) - nP2[i]*((D*wr) @ C2[:, i])
        ev = np.linalg.eigvalsh(M)
        c = np.linalg.solve(M + 1e-13*ev[-1]*np.eye(nf), r)
        cap += c @ r
    GRR = (R*wr) @ R.T
    evG = np.linalg.eigvalsh(GRR)
    GRRi = GRR + 1e-13*evG[-1]*np.eye(nf)
    for i, k in enumerate(ksT):
        if k > KT: break
        r = (R*wr) @ C3[:, i]
        cap += nT2[i]*((np.linalg.solve(GRRi, r)) @ r)
    return np.sqrt(max((Etot - cap)/Etot, 0))

def econ(njA, njS, KP, KT):
    RA, RA1 = hermite_full(range(njA), 20.0, 2.5, rho)[:2]
    RS, RS1 = hermite_full(range(njS), 20.0, 1.2, rho)[:2]
    fam = msk(np.vstack([RA, RS, rl['R']]), np.vstack([RA1, RS1, rl['R1']]))
    nfam = fam[0].shape[0]
    NKPi = KP//2 + 1; NKTi = (KT+1)//2
    N = nfam*(NKPi + NKTi) - (nfam - 12)*0  # full tensor (mask later)
    e = fit_at(fam, KP, KT)
    print(f"H{njA}+H{njS}s1.2+L12 (nf={nfam})  KP={KP} KT={KT}  "
          f"N~{nfam*(NKPi+NKTi):5d}  err={e:.4e}", flush=True)

econ(70, 30, 120, 89)
econ(70, 30, 104, 63)
econ(60, 24, 104, 63)
econ(60, 24, 104, 47)
econ(50, 24, 104, 47)
econ(60, 24,  96, 47)
econ(60, 20, 104, 31)
econ(50, 20,  96, 31)
print(f"[{time.time()-t0:5.1f}s] done", flush=True)
