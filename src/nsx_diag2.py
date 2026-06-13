# SPDX-License-Identifier: MIT
# Copyright (c) 2026 P. Salmond — hwy-spectral-survey (see LICENSE, NOTICE)
"""nsx_diag2.py -- attribute the dropped pressure term d_i = <w_i, grad p1>
(truncated-domain quadrature) to boundary flux through the four cut edges.
For exactly div-free w:
  d_i = [ int p w1 rho^2 sin(th) dth ]_{rho_in}^{rho_out}
      + [ int p w2 rho  sin(th) drho ]_{th=T0}^{th=T1}
Poloidal only (toroidal rows of d vanish identically).
Report: which edge dominates, and whether the rho_in flux is carried by
radial functions with O(1) amplitude at the inner cut.
"""
import numpy as np
import nsx_op
from nsx_op import composite_radial, hermite_full, COMPS
from nsx_basis import angular_factors, radial_factors
from ns_part3k import load_family
from ns_part3f import make_geo, B0, T0, T1, gl_nodes
from ns_part12_gate import Phys
from ns_part3_spectrum import S

Z = np.load('nsx_x1_grams.npz')
Nb, Nt = int(Z['Nb']), int(Z['Nt'])
NJH, RHO0, SIG = int(Z['NJH']), float(Z['RHO0']), float(Z['SIG'])
NJL, ELLL = int(Z['NJL']), float(Z['ELLL'])
NKP, NKT = int(Z['NKP']), int(Z['NKT'])
BHI = np.pi/2 - 0.04

b, wb = gl_nodes(Nb, B0, BHI)
t, wt = gl_nodes(Nt, T0, T1)
rho = S*np.tan(b)
nsx_op.RHO = rho
wr = rho**2*S/np.cos(b)**2*wb          # rho^2 drho
wa = np.sin(t)*wt                      # sin(th) dth
wrho1 = rho*S/np.cos(b)**2*wb          # rho drho   (theta-edge integrals)

# ---- raw radial on grid + the two rho edges, then whiten with grid Gram
rho_in, rho_out = S*np.tan(B0), S*np.tan(BHI)
rho_ext = np.concatenate([[rho_in], rho, [rho_out]])
RHraw = hermite_full(range(NJH), RHO0, SIG, rho_ext)[0]
RLraw = radial_factors(range(NJL), ELLL, rho_ext)['R']
Rraw_ext = np.vstack([RHraw, RLraw])
Rg = Rraw_ext[:, 1:-1]
Gr = (Rg*wr) @ Rg.T
ev, Q = np.linalg.eigh(Gr)
keep = ev > 1e-12*ev[-1]
T = (Q[:, keep]/np.sqrt(ev[keep])).T
nr = int(keep.sum())
R_ext = T @ Rraw_ext                   # whitened, with edge columns
R = R_ext[:, 1:-1]
# D = 2R + rho R' on grid (whitened) for theta-edge integrals
rad, nr2 = composite_radial(rho, wr, NJH, RHO0, SIG, NJL, ELLL)
assert nr2 == nr
D = 2*rad[0] + rho[None, :]*rad[1]
assert np.allclose(rad[0], R, atol=1e-9)

ksP = [2*i for i in range(NKP)]
AP, BP = angular_factors(ksP, t)
a, bb = AP['f'], BP['f']               # (nk, Nt): P_{k+1},  sin*P'_{k+1}
kk = np.array(ksP, float)
fac = (kk + 1)*(kk + 2)

# ---- stored eigenpair, interior d (as in diag1)
fams, _ = load_family('odd')
Pg = Phys(fams[-1], b, t, 'cos')
gp1 = Pg.fr['p']; gp2 = Pg.ft['p']/rho[:, None]
y1 = (R*wr) @ gp1 @ (wa*(a*fac[:, None])).T
y2 = (D*wr) @ (-gp2) @ (wa*bb).T       # w2 = -D sin P' ; sign folded here
d = (y1 + y2)                          # (nr, NKP) poloidal block of d
print(f"setup: nr={nr}, NKP={NKP};  ||d||={np.linalg.norm(d):.6e}")

# ---- edge fluxes
def phys_at(bb_, tt_):
    return Phys(fams[-1], np.atleast_1d(bb_), np.atleast_1d(tt_), 'cos')

# rho edges: val(rho*) = rho*^2 R_r(rho*) * fac_k * int p(rho*,th) P sin dth
p_in = phys_at(B0, t).f['p'].reshape(-1)      # (Nt,)
p_out = phys_at(BHI, t).f['p'].reshape(-1)
ang_in = (a*fac[:, None]) @ (wa*p_in)          # (NKP,)
ang_out = (a*fac[:, None]) @ (wa*p_out)
F_rho_in = -rho_in**2 * np.outer(R_ext[:, 0], ang_in)     # minus: lower limit
F_rho_out = rho_out**2 * np.outer(R_ext[:, -1], ang_out)

# theta edges: val(th*) = -sin^2(th*) P'_{k+1}(mu*) int p(rho,th*) D_r rho drho
def theta_edge(th):
    pe = phys_at(b, th).f['p'].reshape(-1)     # (Nb,)
    rint = D @ (wrho1*pe)                      # (nr,)
    _, BPe = angular_factors(ksP, np.atleast_1d(th))
    spe = BPe['f'][:, 0]                       # sin(th)P'_{k+1}(mu)
    return -np.sin(th)*np.outer(rint, spe)
F_th_hi = theta_edge(T1)
F_th_lo = -theta_edge(T0)

F = F_rho_in + F_rho_out + F_th_lo + F_th_hi
nd = np.linalg.norm(d)
print(f"attribution closure ||d - F||/||d|| = {np.linalg.norm(d-F)/nd:.3e}")
for nm, X in (("rho_in ", F_rho_in), ("rho_out", F_rho_out),
              ("th_lo  ", F_th_lo), ("th_hi  ", F_th_hi)):
    print(f"  ||F_{nm}||/||d|| = {np.linalg.norm(X)/nd:.3e}")

# ---- is rho_in flux carried by radial fns alive at the inner cut?
amp = np.abs(R_ext[:, 0])
rn = np.linalg.norm(d, axis=1)
o = np.argsort(amp)[::-1]
print("top-8 whitened radial fns by |R(rho_in)| vs their d row-norm:")
for r in o[:8]:
    print(f"  r={r:3d}  |R(rho_in)|={amp[r]:.3e}   ||d_r||={rn[r]:.3e}")
print(f"corr(|R(rho_in)|, ||d_r||) = "
      f"{np.corrcoef(amp, rn)[0,1]:+.4f}")
print(f"p1 at inner edge: max|p|={np.abs(p_in).max():.3e}; "
      f"outer: {np.abs(p_out).max():.3e}")
