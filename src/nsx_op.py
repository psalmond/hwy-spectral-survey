# SPDX-License-Identifier: MIT
# Copyright (c) 2026 P. Salmond — hwy-spectral-survey (see LICENSE, NOTICE)
"""nsx_op.py -- separable assembly of the L_U pencil on the composite
shell-adapted div-free basis.

Zero-transcription principle: the coefficient grids C[c,(sc,d)](b,t) of
    (L v)_c = sum_{sc,d} C[c,(sc,d)] * (d-derivative of v_sc)
are extracted NUMERICALLY from the gate-validated ns_part3k.Lcols by
probing with unit fields (Lcols is linear in the derivative slots).
The separable contraction then assembles G,H in O(10^2) GFlop instead of
O(10^4): for test family (radT, angT) and column family (radC, angC),
    block += einsum('pb,bij,qb->piqj', radT*wr, W, radC),
    W[b,i,j] = einsum('it,jt,bt->bij', angT, angC, wa*C).
Gate T4: separable G columns vs direct batched Lcols on random probes.
"""
import numpy as np
from ns_part3k import profile_F, Lcols
from ns_part3f import make_geo, B0, T0, T1, gl_nodes
from ns_part12_gate import Phys
from ns_part3_spectrum import S
from nsx_basis import radial_factors, angular_factors
from scipy.special import eval_genlaguerre

COMPS = ('u1', 'u2', 'u3')
DERIVS = ('f', 'fr', 'frr', 'ft', 'ftt')
DMAP = {'f': (0, 0), 'fr': (1, 0), 'frr': (2, 0), 'ft': (0, 1), 'ftt': (0, 2)}

class _Probe:
    """Duck-typed field for Lcols with a single unit derivative slot."""
    def __init__(self, shape, sc, d):
        Zs = lambda: np.zeros(shape)
        self.f, self.fr, self.frr, self.ft, self.ftt = {}, {}, {}, {}, {}
        for k in COMPS:
            for dd in DERIVS:
                getattr(self, dd)[k] = Zs()
        getattr(self, d)[sc] = np.ones(shape)
        self.fr['p'] = Zs(); self.ft['p'] = Zs()

def extract_coeffs(b, t, geo, withU=True):
    """C[c][(sc,d)] grids on the (b,t) grid, from Lcols probes."""
    UF = profile_F()
    PU = Phys(UF, b, t, 'cos')
    Uval = {k: PU.f[k] for k in COMPS}
    Ubf = {k: dict(f=PU.f[k], fr=PU.fr[k], frr=PU.frr[k],
                   ft=PU.ft[k], ftt=PU.ftt[k]) for k in COMPS}
    shape = geo['B'].shape
    Ng = geo['B'].size
    C = {c: {} for c in COMPS}
    for sc in COMPS:
        for d in DERIVS:
            out = Lcols(_Probe(shape, sc, d), Uval, Ubf, geo, withU)
            o1, o2, o3 = out[:Ng], out[Ng:2*Ng], out[2*Ng:]
            C['u1'][(sc, d)] = o1.reshape(shape)
            C['u2'][(sc, d)] = o2.reshape(shape)
            C['u3'][(sc, d)] = o3.reshape(shape)
    return C

# --------------------------------------------------- composite radial set
def hermite_full(ns, rho0, sig, rho):
    nmax = max(ns)
    z = (rho - rho0)/sig
    psi = np.zeros((nmax+4, len(rho)))
    psi[0] = np.pi**-0.25*np.exp(-0.5*z*z)
    psi[1] = np.sqrt(2.0)*z*psi[0]
    for n in range(1, nmax+3):
        psi[n+1] = np.sqrt(2.0/(n+1))*z*psi[n] - np.sqrt(n/(n+1.0))*psi[n-1]
    def dp(n):
        lo = np.sqrt(n/2.0)*psi[n-1] if n >= 1 else 0.0
        return lo - np.sqrt((n+1)/2.0)*psi[n+1]
    def d2p(n):
        a = np.sqrt(n/2.0)*dp(n-1) if n >= 1 else 0.0
        return a - np.sqrt((n+1)/2.0)*dp(n+1)
    def d3p(n):
        a = np.sqrt(n/2.0)*d2p(n-1) if n >= 1 else 0.0
        return a - np.sqrt((n+1)/2.0)*d2p(n+1)
    R = psi[list(ns)]/np.sqrt(sig)
    R1 = np.array([dp(n) for n in ns])/(sig*np.sqrt(sig))
    R2 = np.array([d2p(n) for n in ns])/(sig**2*np.sqrt(sig))
    R3 = np.array([d3p(n) for n in ns])/(sig**3*np.sqrt(sig))
    return R, R1, R2, R3

def composite_radial(rho, wr, njH, rho0, sig, njL, ellL, cut=1e-12):
    RH, RH1, RH2, RH3 = hermite_full(range(njH), rho0, sig, rho)
    radL = radial_factors(range(njL), ellL, rho)
    R = np.vstack([RH, radL['R']]); R1 = np.vstack([RH1, radL['R1']])
    R2 = np.vstack([RH2, radL['R2']])
    # R3 for Laguerre: recompute (radial_factors keeps R3 internal via D2)
    x = rho/ellL; E = np.exp(-0.5*x)
    RL3 = []
    for j in range(njL):
        n = 1.0/(ellL**1.5*np.sqrt((j+1.0)*(j+2.0)))
        L0 = eval_genlaguerre(j, 2, x)
        L1 = -eval_genlaguerre(j-1, 3, x) if j >= 1 else 0*x
        L2 = eval_genlaguerre(j-2, 4, x) if j >= 2 else 0*x
        L3 = -eval_genlaguerre(j-3, 5, x) if j >= 3 else 0*x
        RL3.append(n*(L3 - 1.5*L2 + 0.75*L1 - 0.125*L0)*E/ellL**3)
    R3 = np.vstack([RH3, np.array(RL3)])
    # radial pre-whitening under int rho^2 drho
    Gr = (R*wr) @ R.T
    ev, Q = np.linalg.eigh(Gr)
    keep = ev > cut*ev[-1]
    T = (Q[:, keep]/np.sqrt(ev[keep])).T          # (nkeep, nraw)
    return [T @ X for X in (R, R1, R2, R3)], int(keep.sum())

# ------------------------------------------------------------- assembler
def assemble(C, wr, wa, rad, angP, angT_):
    """G block-assembly. rad = whitened [R,R1,R2,R3]; angP=(a,at,att,b,bt,btt)
    poloidal; angT_=(b3,b3t,b3tt) toroidal. Returns G with column/test
    ordering [POL (r x kP), TOR (r x kT)] flattened."""
    R, R1, R2, R3 = rad
    D = 2*R + RHO[None, :]*R1
    D1 = 3*R1 + RHO[None, :]*R2
    D2 = 4*R2 + RHO[None, :]*R3
    a, at, att, bb, bt, btt = angP
    b3, b3t, b3tt = angT_
    nr = R.shape[0]; nkP = a.shape[0]; nkT = b3.shape[0]
    NP, NT = nr*nkP, nr*nkT
    # source-side separable factors per (family, component, deriv)
    POLfac = {'u1': {'f': (R, a), 'fr': (R1, a), 'frr': (R2, a),
                     'ft': (R, at), 'ftt': (R, att)},
              'u2': {'f': (-D, bb), 'fr': (-D1, bb), 'frr': (-D2, bb),
                     'ft': (-D, bt), 'ftt': (-D, btt)},
              'u3': None}
    TORfac = {'u1': None, 'u2': None,
              'u3': {'f': (R, b3), 'fr': (R1, b3), 'frr': (R2, b3),
                     'ft': (R, b3t), 'ftt': (R, b3tt)}}
    # test-side (f only)
    POLtest = {'u1': (R, a), 'u2': (-D, bb), 'u3': None}
    TORtest = {'u1': None, 'u2': None, 'u3': (R, b3)}

    def block(testfac, colfac):
        ntk = [f for f in testfac.values() if f is not None][0][1].shape[0]
        nck = [v for v in colfac.values() if v is not None]
        nck = list(list(c.values())[0] for c in nck)[0][1].shape[0]
        out = np.zeros((nr, ntk, nr, nck))
        for c in COMPS:
            tf = testfac[c]
            if tf is None:
                continue
            tR, tA = tf
            for sc in COMPS:
                cf = colfac[sc]
                if cf is None:
                    continue
                for d in DERIVS:
                    Cg = C[c][(sc, d)]
                    if not np.any(Cg):
                        continue
                    cR, cA = cf[d]
                    W = np.einsum('it,jt,bt->bij', tA, cA,
                                  wa[None, :]*Cg, optimize=True)
                    out += np.einsum('pb,bij,qb->piqj', tR*wr, W, cR,
                                     optimize=True)
        return out.reshape(nr*ntk, nr*nck)

    G = np.zeros((NP+NT, NP+NT))
    G[:NP, :NP] = block(POLtest, POLfac)
    G[:NP, NP:] = block(POLtest, TORfac)
    G[NP:, :NP] = block(TORtest, POLfac)
    G[NP:, NP:] = block(TORtest, TORfac)
    return G

def assemble_H(wr, wa, rad, angP, angT_):
    R, R1 = rad[0], rad[1]
    D = 2*R + RHO[None, :]*R1
    a = angP[0]; bb = angP[3]; b3 = angT_[0]
    Grr = (R*wr) @ R.T; Gdd = (D*wr) @ D.T; Grd = (R*wr) @ D.T
    Gaa = (a*wa) @ a.T; Gbb = (bb*wa) @ bb.T
    G33 = (b3*wa) @ b3.T
    Gab3 = (a*wa) @ b3.T; Gbb3 = (bb*wa) @ b3.T
    nr = R.shape[0]
    NP = nr*a.shape[0]; NT = nr*b3.shape[0]
    H = np.zeros((NP+NT, NP+NT))
    H[:NP, :NP] = np.kron(Grr, Gaa) + np.kron(Gdd, Gbb)
    # POL-TOR velocity overlap is zero (u3 vs u1,u2 disjoint components)
    H[NP:, NP:] = np.kron(Grr, G33)
    return H

RHO = None  # set by driver before assemble()
