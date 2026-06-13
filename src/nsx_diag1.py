# SPDX-License-Identifier: MIT
# Copyright (c) 2026 P. Salmond — hwy-spectral-survey (see LICENSE, NOTICE)
"""nsx_diag1.py -- split the X1 failure into (a) non-normal amplification
of representation error, (b) illegitimately dropped pressure term, or
(c) assembly inconsistency, using the exact stored eigenpair (v1, p1):
  g_i = <w_i, L^mom v1>        (validated Lcols, pressure slots zeroed)
  d_i = <w_i, grad p1>         (stored eigen-pressure)
  h_i = <w_i, v1>
  S1: ||g + d - lam1 h|| / ||g||   ~ machinery sanity (must be ~1e-4 or less)
  S2: ||d|| / ||g||                 ~ size of the dropped term
  S3: lam_RQ = c'Gc/c'Hc for the fitted c (from cached grams)
  S4: oblique pencil landscape eta(lam) = min ||(G-lam H)c||/||Hc|| coarse
"""
import numpy as np
import nsx_op
from nsx_op import composite_radial, COMPS, DERIVS
from nsx_basis import angular_factors
from ns_part3k import load_family, profile_F, Lcols
from ns_part3f import make_geo, B0, T0, T1, gl_nodes
from ns_part12_gate import Phys
from ns_part3_spectrum import S

LAM = 0.11314203274385946
Z = np.load('nsx_x1_grams.npz')
G, H = Z['G'], Z['H']
Nb, Nt = int(Z['Nb']), int(Z['Nt'])
b, wb = gl_nodes(Nb, B0, np.pi/2 - 0.04)
t, wt = gl_nodes(Nt, T0, T1)
geo = make_geo(b, t)
rho = S*np.tan(b)
nsx_op.RHO = rho
wr = rho**2*S/np.cos(b)**2*wb
wa = np.sin(t)*wt
rad, nr = composite_radial(rho, wr, int(Z['NJH']), float(Z['RHO0']),
                           float(Z['SIG']), int(Z['NJL']), float(Z['ELLL']))
NKP, NKT = int(Z['NKP']), int(Z['NKT'])
ksP = [2*i for i in range(NKP)]; ksT = [2*i+1 for i in range(NKT)]
AP, BP = angular_factors(ksP, t)
_,  BT = angular_factors(ksT, t)
R, R1 = rad[0], rad[1]
D = 2*R + rho[None, :]*R1
a = AP['f']; bb = BP['f']; b3 = BT['f']

fams, _ = load_family('odd')
P = Phys(fams[-1], b, t, 'cos')
has_p = 'p' in P.f
print(f"stored eigen-pressure available: {has_p}")

# ---- test-side contraction of an arbitrary grid 3-vector field
def testdot(F1, F2, F3):
    y1 = (R*wr) @ F1 @ (wa*a).T
    y2 = (D*wr) @ F2 @ (wa*bb).T
    y3 = (R*wr) @ F3 @ (wa*b3).T
    return np.concatenate([(y1 - y2).ravel(), y3.ravel()])

# ---- g: Lcols on exact v1 with pressure slots zeroed
class _Wrap:
    def __init__(self, P):
        self.f = {k: P.f[k] for k in COMPS}
        self.fr = {k: P.fr[k] for k in COMPS}
        self.frr = {k: P.frr[k] for k in COMPS}
        self.ft = {k: P.ft[k] for k in COMPS}
        self.ftt = {k: P.ftt[k] for k in COMPS}
        self.fr['p'] = np.zeros_like(P.f['u1'])
        self.ft['p'] = np.zeros_like(P.f['u1'])

UF = profile_F()
PU = Phys(UF, b, t, 'cos')
Uval = {k: PU.f[k] for k in COMPS}
Ubf = {k: dict(f=PU.f[k], fr=PU.fr[k], frr=PU.frr[k],
               ft=PU.ft[k], ftt=PU.ftt[k]) for k in COMPS}
Ng = Nb*Nt
out = Lcols(_Wrap(P), Uval, Ubf, geo, True)
Lg = [out[:Ng].reshape(Nb, Nt), out[Ng:2*Ng].reshape(Nb, Nt),
      out[2*Ng:].reshape(Nb, Nt)]
g = testdot(*Lg)
h = testdot(P.f['u1'], P.f['u2'], P.f['u3'])
if has_p:
    gp1 = P.fr['p']; gp2 = P.ft['p']/rho[:, None]
    d = testdot(gp1, gp2, np.zeros_like(gp1))
else:
    d = np.zeros_like(g)

S1 = np.linalg.norm(g + d - LAM*h)/np.linalg.norm(g)
S2 = np.linalg.norm(d)/np.linalg.norm(g)
S1nop = np.linalg.norm(g - LAM*h)/np.linalg.norm(g)
print(f"S1 ||g + d - lam1 h||/||g|| = {S1:.3e}   (machinery sanity)")
print(f"S1' (pressure dropped)      = {S1nop:.3e}")
print(f"S2 ||d||/||g||              = {S2:.3e}   (dropped-term size)")

# ---- S3: Rayleigh quotient of the fitted coefficients
y1 = (R*wr) @ P.f['u1'] @ (wa*a).T
y2 = (D*wr) @ P.f['u2'] @ (wa*bb).T
y3 = (R*wr) @ P.f['u3'] @ (wa*b3).T
rhs = np.concatenate([(y1 - y2).ravel(), y3.ravel()])
evH = np.linalg.eigvalsh(H)
c = np.linalg.solve(H + 1e-13*evH[-1]*np.eye(len(H)), rhs)
lam_rq = (c @ G @ c)/(c @ H @ c)
print(f"S3 lam_RQ(fitted v1) = {lam_rq:+.6f}   (target {LAM:+.6f})")

# ---- S4: oblique pencil landscape, coarse
print("S4 eta(lam) = min ||(G-lam H)c||/||H c||:")
for lam in (-0.5, -0.25, 0.0, 0.05, 0.113142, 0.15):
    Aop = G - lam*H
    # generalized smallest singular pair via dense solve on whitened basis
    T = np.linalg.eigh(H)
    keep = T[0] > 1e-12*T[0][-1]
    W = T[1][:, keep]/np.sqrt(T[0][keep])
    Mw = Aop @ W
    sv = np.linalg.svd(Mw @ np.linalg.pinv(H @ W), compute_uv=False) \
         if False else None
    # simpler consistent metric: min ||(G-lam H)c|| / ||Hc|| over c
    B1 = Aop @ W; B2 = H @ W
    s = np.linalg.svd(np.linalg.lstsq(B2, B1, rcond=None)[0],
                      compute_uv=False)
    print(f"   lam={lam:+.4f}: eta={s[-1]:.4e}")
