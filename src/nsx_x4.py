# SPDX-License-Identifier: MIT
# Copyright (c) 2026 P. Salmond — hwy-spectral-survey (see LICENSE, NOTICE)
"""nsx_x4.py -- tame-basis rebuild: kill the Laguerre x high-k rough tail.

Measured pathology: jointly-whitened basis contains Laguerre(ell=3) x k~79
modes with ||L w|| ~ 1e3-1e4 (Delta ~ k^2/rho^2 at rho~1). These (a) poison
the gram-Schur P_perp pipeline with float64 cancellation (bottom(Ad~) =
-0.011) and (b) are useless for shell-localized targets.

Fix: whiten the Hermite and Laguerre radial blocks SEPARATELY (identifiable
indices), assemble the full tensor grams, then MASK: keep
   Hermite r  : all k        (shell physics, k^2/rho^2 <= ~16 at rho=20)
   Laguerre r : k-index < 12 (pol) / < 8 (tor)   (broad smooth modes only)

Stages in one run:
  A: G, G0, H assembly (separable, T4-validated machinery) + mask
     -> X1-style gates on the masked pencil (lam1, X3, cut scan);
  B: AA, B grams (T5-validated aa_block/b_block) + mask
     -> Schur prep, bottom(Ad~) noise check, landscape real sweep.
Saves nsx_x4_grams.npz, nsx_x4_real.npz.
"""
import time
import numpy as np
import scipy.linalg as sla
import nsx_op
from nsx_op import extract_coeffs, assemble, assemble_H, hermite_full
from nsx_basis import angular_factors, radial_factors, _leg_derivs
from ns_part3k import load_family
from ns_part3f import make_geo, gl_nodes
from ns_part12_gate import Phys
from ns_part3_spectrum import S
from nsx_x2_aa import aa_block, b_block, slot_tables, pressure_span
from scipy.special import eval_genlaguerre

LAM = 0.11314203274385946
Nb, Nt = 480, 220
NJH, RHO0, SIG = 50, 20.0, 2.5
NJL, ELLL = 12, 3.0
NKP, NKT = 40, 24
KL_POL, KL_TOR = 12, 8        # Laguerre rows keep only these many k's
SPEC = [('H', 60, 20.0, 2.5), ('H', 30, 20.0, 1.2),
        ('L', 12, 3.0), ('L', 8, 8.0)]
NKQ = 72
t0 = time.time()
def log(s): print(f"[{time.time()-t0:7.1f}s] {s}", flush=True)

# ---------------- split-whitened composite radial
def origin_mask(rho, a=2.0):
    den = rho*rho + a*a
    m0 = rho*rho/den
    m1 = 2*a*a*rho/den**2
    m2 = 2*a*a*(a*a - 3*rho*rho)/den**3
    m3 = 24*a*a*rho*(rho*rho - a*a)/den**4
    return m0, m1, m2, m3

def apply_mask(R, R1, R2, R3, rho):
    m0, m1, m2, m3 = origin_mask(rho)
    Rm  = R*m0
    Rm1 = R1*m0 + R*m1
    Rm2 = R2*m0 + 2*R1*m1 + R*m2
    Rm3 = R3*m0 + 3*R2*m1 + 3*R1*m2 + R*m3
    return Rm, Rm1, Rm2, Rm3

def composite_radial_split(rho, wr, cut=1e-12):
    RH, RH1, RH2, RH3 = hermite_full(range(NJH), RHO0, SIG, rho)
    RH, RH1, RH2, RH3 = apply_mask(RH, RH1, RH2, RH3, rho)
    radL = radial_factors(range(NJL), ELLL, rho)
    x = rho/ELLL; E = np.exp(-0.5*x)
    RL3 = []
    for j in range(NJL):
        n = 1.0/(ELLL**1.5*np.sqrt((j+1.0)*(j+2.0)))
        L0 = eval_genlaguerre(j, 2, x)
        L1 = -eval_genlaguerre(j-1, 3, x) if j >= 1 else 0*x
        L2 = eval_genlaguerre(j-2, 4, x) if j >= 2 else 0*x
        L3 = -eval_genlaguerre(j-3, 5, x) if j >= 3 else 0*x
        RL3.append(n*(L3 - 1.5*L2 + 0.75*L1 - 0.125*L0)*E/ELLL**3)
    RL, RL1, RL2, RL3a = apply_mask(radL['R'], radL['R1'], radL['R2'],
                                    np.array(RL3), rho)
    def whiten(Rb, derivs):
        Gr = (Rb*wr) @ Rb.T
        ev, Q = np.linalg.eigh(Gr)
        keep = ev > cut*ev[-1]
        T = (Q[:, keep]/np.sqrt(ev[keep])).T
        return [T @ X for X in derivs], int(keep.sum())
    Hw, nH = whiten(RH, (RH, RH1, RH2, RH3))
    Lw, nL = whiten(RL, (RL, RL1, RL2, RL3a))
    rad = [np.vstack([Hw[i], Lw[i]]) for i in range(4)]
    return rad, nH, nL

# ---------------- grid
b, wb = gl_nodes(Nb, 0.0, np.pi/2)
t, wt = gl_nodes(Nt, 0.0, np.pi/2)
geo = make_geo(b, t)
rho = S*np.tan(b)
nsx_op.RHO = rho
wr = rho**2*S/np.cos(b)**2*wb
wa = np.sin(t)*wt

rad, nH, nL = composite_radial_split(rho, wr)
nr = nH + nL
ksP = [2*i for i in range(NKP)]; ksT = [2*i+1 for i in range(NKT)]
AP, BP = angular_factors(ksP, t)
_,  BT = angular_factors(ksT, t)
angP = (AP['f'], AP['t'], AP['tt'], BP['f'], BP['t'], BP['tt'])
angT_ = (BT['f'], BT['t'], BT['tt'])
NP, NT = nr*NKP, nr*NKT; N = NP + NT
# mask in the flattened [POL (r x kP), TOR (r x kT)] ordering
mP = np.zeros((nr, NKP), bool); mP[:nH] = True; mP[nH:, :KL_POL] = True
mT = np.zeros((nr, NKT), bool); mT[:nH] = True; mT[nH:, :KL_TOR] = True
mask = np.concatenate([mP.ravel(), mT.ravel()])
Nm = int(mask.sum())
log(f"basis: nr={nr} (H{nH}+L{nL}), full N={N}, masked Nm={Nm}")

# ================= stage A: pencil gates =================
C = extract_coeffs(b, t, geo, withU=True)
G = assemble(C, wr, wa, rad, angP, angT_)
C0 = extract_coeffs(b, t, geo, withU=False)
G0 = assemble(C0, wr, wa, rad, angP, angT_)
H = assemble_H(wr, wa, rad, angP, angT_)
G = G[np.ix_(mask, mask)]; G0 = G0[np.ix_(mask, mask)]
H = H[np.ix_(mask, mask)]
log("G, G0, H assembled + masked")

# X0r on masked span
fams, _ = load_family('odd')
Pv1 = Phys(fams[-1], b, t, 'cos')
R, R1 = rad[0], rad[1]
D = 2*R + rho[None, :]*R1
y1 = (R*wr) @ Pv1.f['u1'] @ (wa*AP['f']).T
y2 = (D*wr) @ Pv1.f['u2'] @ (wa*BP['f']).T
y3 = (R*wr) @ Pv1.f['u3'] @ (wa*BT['f']).T
rhs = np.concatenate([(y1 - y2).ravel(), y3.ravel()])[mask]
Etot = ((wr[:, None]*wa[None, :])
        * (Pv1.f['u1']**2 + Pv1.f['u2']**2 + Pv1.f['u3']**2)).sum()
evH, QH = sla.eigh(H, check_finite=False)
cfit = np.linalg.solve(H + 1e-13*evH[-1]*np.eye(Nm), rhs)
log(f"X0r (masked span): v1 fit error = "
    f"{np.sqrt(max(Etot - cfit @ rhs, 0)/Etot):.3e}")

results = {}
for cut in (1e-8, 1e-10, 1e-12):
    keep = evH > cut*evH[-1]
    T = QH[:, keep]/np.sqrt(evH[keep])
    ev = np.linalg.eigvals(T.T @ G @ T)
    ev0 = np.linalg.eigvals(T.T @ G0 @ T)
    i1 = np.argmin(np.abs(ev - LAM))
    res = dict(nk=int(keep.sum()), lam1=ev[i1], d=abs(ev[i1]-LAM),
               m0=ev0.real.max(), top=np.sort(ev.real)[-6:])
    results[cut] = res
    log(f"cut={cut:.0e}: kept {res['nk']}/{Nm}  lam1={res['lam1']:.8f} "
        f"(|d|={res['d']:.2e})  X3 maxRe(U=0)={res['m0']:+.5f}")
    log("   top Re Ritz: " + " ".join(f"{z:+.5f}" for z in res['top']))
ok1 = all(results[c]['d'] < 5e-3 for c in results)
ok3 = all(results[c]['m0'] < -0.25 + 5e-3 for c in results)
log(f"X1 (masked): {'PASS' if ok1 else 'FAIL'};  "
    f"X3: {'PASS' if ok3 else 'FAIL'}")

# ================= stage B: AA, B, Schur, landscape =================
POL, TOR = slot_tables(rad, angP, angT_, rho)
AA = np.zeros((N, N))
AA[:NP, :NP] = aa_block(C, wr, wa, POL, POL)
AA[:NP, NP:] = aa_block(C, wr, wa, POL, TOR)
AA[NP:, :NP] = AA[:NP, NP:].T
AA[NP:, NP:] = aa_block(C, wr, wa, TOR, TOR)
asym = np.linalg.norm(AA - AA.T)/np.linalg.norm(AA)
AA = 0.5*(AA + AA.T)
AA = AA[np.ix_(mask, mask)]
log(f"AA assembled+masked (asym gate {asym:.2e}); "
    f"max diag {AA.diagonal().max():.3e}")
Sf, S1, Pq, Pq1 = pressure_span(rho, t, SPEC, NKQ)
Sor = Sf/rho[None, :]
B = np.vstack([b_block(C, wr, wa, POL, S1, Sor, Pq, Pq1),
               b_block(C, wr, wa, TOR, S1, Sor, Pq, Pq1)])[mask]
del C, C0
log("B assembled+masked")
np.savez('nsx_x4_grams.npz', G=G, G0=G0, H=H, AA=AA, B=B,
         mask=mask, nH=nH, nL=nL, NKP=NKP, NKT=NKT)

Mg = np.kron((S1*wr) @ S1.T, (Pq*wa) @ Pq.T)
Mg += np.kron((Sor*wr) @ Sor.T, (Pq1*wa) @ Pq1.T)
evM, QM = sla.eigh(Mg, overwrite_a=True, check_finite=False)
del Mg
Y = QM.T @ B.T
del QM
log("Schur pieces ready")
T = QH[:, evH > 1e-10*evH[-1]]/np.sqrt(evH[evH > 1e-10*evH[-1]])
out = {}
for cut in (1e-10, 1e-12):
    kp = evM > cut*evM[-1]
    Ys = Y[kp]/np.sqrt(evM[kp])[:, None]
    AAd = AA - Ys.T @ Ys
    At = T.T @ AAd @ T
    bA = sla.eigh(At, eigvals_only=True, subset_by_index=[0, 2],
                  check_finite=False)
    out[cut] = At
    log(f"cut {cut:.0e}: kept {kp.sum()} grad modes; "
        f"bottom(Ad~) = {bA}  <- noise floor in r^2")
Gt = T.T @ G @ T
np.savez('nsx_x4_prep.npz', Gt=Gt, At_10=out[1e-10], At_12=out[1e-12])

def rmin(At, lam):
    n = At.shape[0]
    Nl = At - lam*(Gt + Gt.T) + lam*lam*np.eye(n)
    w = sla.eigh(Nl, eigvals_only=True, subset_by_index=[0, 0],
                 check_finite=False)
    return np.sqrt(max(w[0], 0.0)), w[0]

for cut in (1e-10, 1e-12):
    r1, raw1 = rmin(out[cut], LAM)
    r2, _ = rmin(out[cut], 0.25)
    r3, _ = rmin(out[cut], 0.0)
    log(f"cut {cut:.0e}: r(lam1)={r1:.4e} (raw {raw1:+.3e})  "
        f"r(0)={r3:.4e}  r(0.25)={r2:.4e}")

At = out[1e-12]
lams = np.arange(-0.10, 0.3001, 0.005)
rr = np.array([rmin(At, x)[0] for x in lams])
np.savez('nsx_x4_real.npz', lams=lams, r=rr)
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
