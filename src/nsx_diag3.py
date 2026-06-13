# SPDX-License-Identifier: MIT
# Copyright (c) 2026 P. Salmond — hwy-spectral-survey (see LICENSE, NOTICE)
"""nsx_diag3.py -- quadrature-convergence probe in Nb.
For Nb in a ladder: recompute (raw, unwhitened basis, poloidal only)
  d_i = <w_i, grad p1>      (should -> edge flux F, norm ~1e-4 of current)
  closure ||d - F||/||F||
  H_raw poloidal block       (vs reference at largest Nb)
If d collapses and H settles as Nb grows, X1's failure is quadrature
starvation, not formulation.
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
Nt = 200
fams, _ = load_family('odd')
ksP = [2*i for i in range(NKP)]
kk = np.array(ksP, float); fac = (kk+1)*(kk+2)
rho_in, rho_out = S*np.tan(B0), S*np.tan(BHI)

t, wt = gl_nodes(Nt, T0, T1)
wa = np.sin(t)*wt
AP, BP = angular_factors(ksP, t)
a, bb = AP['f'], BP['f']
aF = a*fac[:, None]

def build(Nb):
    b, wb = gl_nodes(Nb, B0, BHI)
    rho = S*np.tan(b)
    wr = rho**2*S/np.cos(b)**2*wb
    wrho1 = rho*S/np.cos(b)**2*wb
    RH = hermite_full(range(NJH), RHO0, SIG, rho)
    RL = radial_factors(range(NJL), ELLL, rho)
    R = np.vstack([RH[0], RL['R']]); R1 = np.vstack([RH[1], RL['R1']])
    D = 2*R + rho[None, :]*R1
    return b, rho, wr, wrho1, R, D

def dvec_and_F(Nb):
    b, rho, wr, wrho1, R, D = build(Nb)
    Pg = Phys(fams[-1], b, t, 'cos')
    gp1 = Pg.fr['p']; gp2 = Pg.ft['p']/rho[:, None]
    d = (R*wr) @ gp1 @ (wa*aF).T + (D*wr) @ (-gp2) @ (wa*bb).T
    # edge fluxes (rho edges need R at the edge points)
    Re = np.vstack([hermite_full(range(NJH), RHO0, SIG,
                                 np.array([rho_in, rho_out]))[0],
                    radial_factors(range(NJL), ELLL,
                                   np.array([rho_in, rho_out]))['R']])
    p_in = Phys(fams[-1], np.array([B0]), t, 'cos').f['p'].reshape(-1)
    p_out = Phys(fams[-1], np.array([BHI]), t, 'cos').f['p'].reshape(-1)
    F = (-rho_in**2*np.outer(Re[:, 0], aF @ (wa*p_in))
         + rho_out**2*np.outer(Re[:, 1], aF @ (wa*p_out)))
    for th, sgn in ((T1, +1.0), (T0, -1.0)):
        pe = Phys(fams[-1], b, np.array([th]), 'cos').f['p'].reshape(-1)
        _, BPe = angular_factors(ksP, np.array([th]))
        F += sgn*(-np.sin(th))*np.outer(D @ (wrho1*pe), BPe['f'][:, 0])
    return d, F

def Hraw(Nb):
    b, rho, wr, wrho1, R, D = build(Nb)
    W1 = np.einsum('it,jt->ij', wa*aF, aF)
    W2 = np.einsum('it,jt->ij', wa*bb, bb)
    H = (np.einsum('pb,ij,qb->piqj', R*wr, W1, R)
         + np.einsum('pb,ij,qb->piqj', D*wr, W2, D))
    return H.reshape(R.shape[0]*NKP, -1)

ladder = (400, 700, 1000, 1400, 2000)
Href = Hraw(2600)
nH = np.linalg.norm(Href)
print(f"reference Hraw at Nb=2600: ||H||={nH:.6e}")
for Nb in ladder:
    d, F = dvec_and_F(Nb)
    H = Hraw(Nb)
    print(f"Nb={Nb:5d}: ||d||={np.linalg.norm(d):.4e}  "
          f"||F||={np.linalg.norm(F):.4e}  "
          f"closure={np.linalg.norm(d-F)/np.linalg.norm(F):.3e}  "
          f"||H-Href||/||Href||={np.linalg.norm(H-Href)/nH:.3e}")
