# SPDX-License-Identifier: MIT
# Copyright (c) 2026 P. Salmond — hwy-spectral-survey (see LICENSE, NOTICE)
"""nsx_r1.py -- pointwise residual check (run20-style) of the lambda1 Ritz
vector from the X1 gate on the rebuilt decay-adapted basis.

Definition: r(v,lam) = min_q ||L_mom v - lam v + grad q|| / ||v||  on the
open quarter domain, with the gradient deflated over a rich scalar span
q = S_m(rho) P_{k+1}(mu), k even (odd-sector pressure parity), S_m drawn
from shell-Hermite(rho0=20,sig=2.5) (+) Laguerre(ell=3) (+) Laguerre(ell=8).
Separable normal equations -- no large LSQ matrix is ever materialized.

Controls:
  A: exact v1 with its OWN stored pressure, lam = LAM_KNOWN  (expect ~2e-5,
     the gate-validated eigenpair residual -- calibrates quadrature);
  B: exact v1, pressure zeroed, gradient DEFLATED, lam = LAM_KNOWN
     (calibrates adequacy of the deflation span);
  C: lambda1 Ritz vector, deflated                            (headline);
  D: next-highest-Re Ritz vector (-0.063...), deflated        (context).
"""
import time
import numpy as np
import nsx_op
from nsx_op import composite_radial, hermite_full
from nsx_basis import angular_factors, _leg_derivs
from ns_part3k import load_family, profile_F, Lcols
from ns_part3f import make_geo, gl_nodes
from ns_part12_gate import Phys
from ns_part3_spectrum import S
from scipy.special import eval_genlaguerre

LAM = 0.11314203274385946
t0 = time.time()

# ---------------- grid + basis exactly as cached in nsx_x1_grams.npz
Z = np.load('nsx_x1_grams.npz')
G, H = Z['G'], Z['H']
nr, NKP, NKT = int(Z['nr']), int(Z['NKP']), int(Z['NKT'])
NJH, RHO0, SIG = int(Z['NJH']), float(Z['RHO0']), float(Z['SIG'])
NJL, ELLL = int(Z['NJL']), float(Z['ELLL'])
Nb, Nt = int(Z['Nb']), int(Z['Nt'])

b, wb = gl_nodes(Nb, 0.0, np.pi/2)
t, wt = gl_nodes(Nt, 0.0, np.pi/2)
geo = make_geo(b, t)
rho = S*np.tan(b)
nsx_op.RHO = rho
wr = rho**2*S/np.cos(b)**2*wb
wa = np.sin(t)*wt
WGT = wr[:, None]*wa[None, :]

rad, nr2 = composite_radial(rho, wr, NJH, RHO0, SIG, NJL, ELLL)
assert nr2 == nr, (nr2, nr)
R, R1, R2, R3 = rad
D = 2*R + rho[None, :]*R1
D1 = 3*R1 + rho[None, :]*R2
D2 = 4*R2 + rho[None, :]*R3
ksP = [2*i for i in range(NKP)]; ksT = [2*i+1 for i in range(NKT)]
AP, BP = angular_factors(ksP, t)
_,  BT = angular_factors(ksT, t)
NP, NT = nr*NKP, nr*NKT; N = NP + NT
print(f"[{time.time()-t0:6.1f}s] grid+basis rebuilt, N={N}", flush=True)

# ---------------- profile fields for Lcols
UF = profile_F()
PU = Phys(UF, b, t, 'cos')
Uval = {k: PU.f[k] for k in ('u1', 'u2', 'u3')}
Ubf = {k: dict(f=PU.f[k], fr=PU.fr[k], frr=PU.frr[k],
               ft=PU.ft[k], ftt=PU.ftt[k]) for k in ('u1', 'u2', 'u3')}
print(f"[{time.time()-t0:6.1f}s] profile evaluated", flush=True)

# ---------------- eigen-solve (cut 1e-10) and target Ritz vectors
evH, QH = np.linalg.eigh(H)
keep = evH > 1e-10*evH[-1]
T = QH[:, keep]/np.sqrt(evH[keep])
Gt = T.T @ G @ T
ev, W = np.linalg.eig(Gt)
print(f"[{time.time()-t0:6.1f}s] eig done, kept {keep.sum()}/{N}", flush=True)

def coeff_for(target):
    i = np.argmin(np.abs(ev - target))
    w = W[:, i]
    ph = w[np.argmax(np.abs(w))]
    w = (w*np.conj(ph/abs(ph)))
    im = np.abs(w.imag).max()/np.abs(w.real).max()
    c = T @ w.real
    c = c/np.sqrt(c @ H @ c)
    return ev[i], c, im

# ---------------- field reconstruction from coefficients
class F:
    pass

def build_field(c):
    cP = c[:NP].reshape(nr, NKP)
    cT = c[NP:].reshape(nr, NKT)
    f = F()
    f.f, f.fr, f.frr, f.ft, f.ftt = {}, {}, {}, {}, {}
    def mix(Rm, cm, Am):
        return (Rm.T @ cm) @ Am
    f.f['u1'] = mix(R, cP, AP['f']);  f.fr['u1'] = mix(R1, cP, AP['f'])
    f.frr['u1'] = mix(R2, cP, AP['f'])
    f.ft['u1'] = mix(R, cP, AP['t']); f.ftt['u1'] = mix(R, cP, AP['tt'])
    f.f['u2'] = -mix(D, cP, BP['f']); f.fr['u2'] = -mix(D1, cP, BP['f'])
    f.frr['u2'] = -mix(D2, cP, BP['f'])
    f.ft['u2'] = -mix(D, cP, BP['t']); f.ftt['u2'] = -mix(D, cP, BP['tt'])
    f.f['u3'] = mix(R, cT, BT['f']);  f.fr['u3'] = mix(R1, cT, BT['f'])
    f.frr['u3'] = mix(R2, cT, BT['f'])
    f.ft['u3'] = mix(R, cT, BT['t']); f.ftt['u3'] = mix(R, cT, BT['tt'])
    f.fr['p'] = np.zeros_like(f.f['u1']); f.ft['p'] = np.zeros_like(f.f['u1'])
    return f

def norm2_of(v1, v2, v3):
    return (WGT*(v1*v1 + v2*v2 + v3*v3)).sum()

def apply_L(field):
    out = Lcols(field, Uval, Ubf, geo, True)
    Ng = Nb*Nt
    return (out[:Ng].reshape(Nb, Nt), out[Ng:2*Ng].reshape(Nb, Nt),
            out[2*Ng:].reshape(Nb, Nt))

# ---------------- gradient-deflation span (separable normal equations)
NSH, NL3, NL8, NKQ = 40, 12, 8, 48
SH, SH1, _, _ = hermite_full(range(NSH), RHO0, SIG, rho)
def lag0(nm, ell):
    x = rho/ell; E = np.exp(-0.5*x)
    Sf, S1 = [], []
    for m in range(nm):
        L0 = eval_genlaguerre(m, 0, x)
        L1 = -eval_genlaguerre(m-1, 1, x) if m >= 1 else 0*x
        Sf.append(L0*E); S1.append((L1 - 0.5*L0)*E/ell)
    return np.array(Sf), np.array(S1)
SL3, SL31 = lag0(NL3, 3.0)
SL8, SL81 = lag0(NL8, 8.0)
Sf = np.vstack([SH, SL3, SL8]); S1 = np.vstack([SH1, SL31, SL81])
Sor = Sf/rho[None, :]
ksQ = [2*i for i in range(NKQ)]
Pq, Pq1 = [], []
mu, st = np.cos(t), np.sin(t)
for k in ksQ:
    P, P1 = _leg_derivs(k+1, mu, 1)[:2]
    Pq.append(P); Pq1.append(-st*P1)        # d/dt = -st d/dmu
Pq = np.array(Pq); Pq1 = np.array(Pq1)
A1 = (S1*wr) @ S1.T;  A0 = (Sor*wr) @ Sor.T
P0 = (Pq*wa) @ Pq.T;  P1g = (Pq1*wa) @ Pq1.T
M = np.kron(A1, P0) + np.kron(A0, P1g)
evM, QM = np.linalg.eigh(M)
print(f"[{time.time()-t0:6.1f}s] gradient span: Nq={M.shape[0]}, "
      f"cond head {evM[-1]:.2e} tail {evM[0]:.2e}", flush=True)

def deflated_norm2(r1, r2, r3, cuts=(1e-10, 1e-12)):
    rhs = ((S1*wr) @ r1 @ (wa[:, None]*Pq.T)
           + (Sor*wr) @ r2 @ (wa[:, None]*Pq1.T)).ravel()
    n2 = norm2_of(r1, r2, r3)
    out = {}
    y = QM.T @ rhs
    for cut in cuts:
        kp = evM > cut*evM[-1]
        out[cut] = n2 - (y[kp]**2/evM[kp]).sum()
    return n2, out

def report(tag, r1, r2, r3, vn2, deflate=True):
    n2, dn = (deflated_norm2(r1, r2, r3) if deflate
              else (norm2_of(r1, r2, r3), None))
    raw = np.sqrt(n2/vn2)
    if dn is None:
        print(f"{tag}: r = {raw:.3e}  (no deflation)", flush=True)
    else:
        s = "  ".join(f"cut{c:.0e}: {np.sqrt(max(v,0)/vn2):.3e}"
                      for c, v in dn.items())
        print(f"{tag}: raw {raw:.3e}  deflated {s}", flush=True)

# ================= CONTROL A/B: exact v1 =================
fams, lams = load_family('odd')
Pv1 = Phys(fams[-1], b, t, 'cos')
vn2 = norm2_of(Pv1.f['u1'], Pv1.f['u2'], Pv1.f['u3'])
L1, L2, L3 = apply_L(Pv1)
rA = (L1 - LAM*Pv1.f['u1'], L2 - LAM*Pv1.f['u2'], L3 - LAM*Pv1.f['u3'])
report("A  v1 + own pressure       ", *rA, vn2, deflate=False)

p_fr, p_ft = Pv1.fr['p'].copy(), Pv1.ft['p'].copy()
Pv1.fr['p'] = np.zeros_like(p_fr); Pv1.ft['p'] = np.zeros_like(p_ft)
L1, L2, L3 = apply_L(Pv1)
rB = (L1 - LAM*Pv1.f['u1'], L2 - LAM*Pv1.f['u2'], L3 - LAM*Pv1.f['u3'])
report("B  v1, p dropped, deflated ", *rB, vn2)
Pv1.fr['p'] = p_fr; Pv1.ft['p'] = p_ft
print(f"[{time.time()-t0:6.1f}s] controls done", flush=True)

# ================= C: lambda1 Ritz vector =================
lam1, c1, im1 = coeff_for(LAM)
print(f"lambda1 Ritz = {lam1.real:+.8f}{lam1.imag:+.1e}j  "
      f"(eigvec imag ratio {im1:.1e})", flush=True)
fC = build_field(c1)
vn2C = norm2_of(fC.f['u1'], fC.f['u2'], fC.f['u3'])
print(f"   grid norm^2 = {vn2C:.6f} (coeff-norm 1; quadrature consistency)",
      flush=True)
L1, L2, L3 = apply_L(fC)
rC = (L1 - lam1.real*fC.f['u1'], L2 - lam1.real*fC.f['u2'],
      L3 - lam1.real*fC.f['u3'])
report("C  Ritz(lam1), deflated    ", *rC, vn2C)
np.savez('nsx_r1_out.npz', c1=c1, lam1=lam1.real)

# ================= D: next-highest-Re Ritz =================
re_sorted = np.sort(ev.real)
tgt = re_sorted[-2]
lam2, c2, im2 = coeff_for(tgt)
fD = build_field(c2)
vn2D = norm2_of(fD.f['u1'], fD.f['u2'], fD.f['u3'])
L1, L2, L3 = apply_L(fD)
rD = (L1 - lam2.real*fD.f['u1'], L2 - lam2.real*fD.f['u2'],
      L3 - lam2.real*fD.f['u3'])
print(f"next Ritz = {lam2.real:+.6f}{lam2.imag:+.1e}j "
      f"(imag ratio {im2:.1e})", flush=True)
report("D  Ritz(next), deflated    ", *rD, vn2D)
print(f"[{time.time()-t0:6.1f}s] done", flush=True)
