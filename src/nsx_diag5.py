# SPDX-License-Identifier: MIT
# Copyright (c) 2026 P. Salmond — hwy-spectral-survey (see LICENSE, NOTICE)
"""nsx_diag5.py -- diag2/3/4 redone with the CORRECT angular convention
(a = (k+1)(k+2) P already; no extra fac).
T1: synthetic smooth gradient -> pairing must vanish (IBP sanity).
T2: closure of real d = <w, grad p1> against the four edge fluxes.
T3: edge attribution of the real d.
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
Nb, Nt = 1000, 240
b, wb = gl_nodes(Nb, B0, BHI)
t, wt = gl_nodes(Nt, T0, T1)
rho = S*np.tan(b)
wr = rho**2*S/np.cos(b)**2*wb
wrho1 = rho*S/np.cos(b)**2*wb
wa = np.sin(t)*wt
RH = hermite_full(range(NJH), RHO0, SIG, rho)
RL = radial_factors(range(NJL), ELLL, rho)
R = np.vstack([RH[0], RL['R']]); R1 = np.vstack([RH[1], RL['R1']])
D = 2*R + rho[None, :]*R1
ksP = [2*i for i in range(NKP)]
AP, BP = angular_factors(ksP, t)
a, bb = AP['f'], BP['f']            # a includes (k+1)(k+2)

def pair(g1, g2):                   # <w_(r,k), (g1,g2,0)>
    return (R*wr) @ g1 @ (wa*a).T - (D*wr) @ g2 @ (wa*bb).T

# ---- T1 synthetic
q = np.exp(-(rho[:, None]-20.0)**2/8.0)*np.cos(2*t[None, :])
q1 = (-(rho[:, None]-20.0)/4.0)*q
q2 = np.exp(-(rho[:, None]-20.0)**2/8.0)*(-2*np.sin(2*t[None, :]))/rho[:, None]
dsyn = pair(q1, q2)
ref = np.linalg.norm(pair(np.abs(q1), np.abs(q2)))
print(f"T1 synthetic: ratio = {np.linalg.norm(dsyn)/ref:.3e}  "
      f"(||.||={np.linalg.norm(dsyn):.3e}, scale {ref:.3e})")

# ---- real d and edge fluxes
fams, _ = load_family('odd')
P = Phys(fams[-1], b, t, 'cos')
g1 = P.fr['p']; g2 = P.ft['p']/rho[:, None]
d = pair(g1, g2)

rho_in, rho_out = S*np.tan(B0), S*np.tan(BHI)
Re = np.vstack([hermite_full(range(NJH), RHO0, SIG,
                             np.array([rho_in, rho_out]))[0],
                radial_factors(range(NJL), ELLL,
                               np.array([rho_in, rho_out]))['R']])
p_in = Phys(fams[-1], np.array([B0]), t, 'cos').f['p'].reshape(-1)
p_out = Phys(fams[-1], np.array([BHI]), t, 'cos').f['p'].reshape(-1)
F_rin = -rho_in**2*np.outer(Re[:, 0], a @ (wa*p_in))
F_rout = rho_out**2*np.outer(Re[:, 1], a @ (wa*p_out))
def theta_edge(th):
    pe = Phys(fams[-1], b, np.array([th]), 'cos').f['p'].reshape(-1)
    _, BPe = angular_factors(ksP, np.array([th]))
    return -np.sin(th)*BPe['f'][:, 0][None, :]*(D @ (wrho1*pe))[:, None]
F_thi = theta_edge(T1)
F_tlo = -theta_edge(T0)
F = F_rin + F_rout + F_tlo + F_thi
nd = np.linalg.norm(d)
print(f"T2 ||d||={nd:.4e}  ||F||={np.linalg.norm(F):.4e}  "
      f"closure ||d-F||/||d|| = {np.linalg.norm(d-F)/nd:.3e}")
for nm, X in (("rho_in ", F_rin), ("rho_out", F_rout),
              ("th_lo  ", F_tlo), ("th_hi  ", F_thi)):
    print(f"   ||F_{nm}||/||d|| = {np.linalg.norm(X)/nd:.3e}")
amp = np.abs(Re[:, 0])
rn = np.linalg.norm(d, axis=1)
print(f"T3 corr(|R(rho_in)|, ||d_r||) = {np.corrcoef(amp, rn)[0,1]:+.4f}")
o = np.argsort(rn)[::-1][:6]
for r in o:
    blk = 'H' if r < NJH else f'L{r-NJH}'
    print(f"   r={r:2d}({blk}) ||d_r||={rn[r]:.3e}  |R(rho_in)|={amp[r]:.3e}")
