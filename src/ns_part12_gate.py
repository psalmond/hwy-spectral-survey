# SPDX-License-Identifier: MIT
# Copyright (c) 2026 P. Salmond — hwy-spectral-survey (see LICENSE, NOTICE)
# Independent implementation of the HWY operator (arXiv:2509.25116, eq. for L_U);
# mirrors their public data formats/conventions; contains no code from their repo.
"""
PART 1+2 -- our own forward-self-similar NS operators, gated on HWY data.

Reconstruction (the ONLY part taken from their code, by reading src/):
  field(beta,theta) = RadialTrig(beta) @ C @ AngularTrig(theta)^T
  with C = freq @ conv  (conv only for u1, u2), bases per BDRY_LST_SET[2]:
    u1 (u_rho):  radial sin((2i-1)b) [300] ; conv=che2len(150)[1:, :] ->
                 angular cos(2(j-1)t) [151]
    u2 (u_th):   radial sin((2i-1)b) [301] ; conv=che2len(150)[1:,1:]@diag(2n)
                 -> angular sin(2j t) [150]
    u3 (u_phi):  radial sin((2i-1)b) [300] ; angular sin((2j-1)t) [150]
    p:           radial cos((2i-1)b) [300] ; angular cos(2(j-1)t) [150]
  rho = s*tan(beta), s = scl_fac = 22 (beta=pi/2 <-> infinity).
  Div-free coupling: u2_freq = DL @ u1_freq @ DR (div_mat/eigs ported).

Physics (OURS, textbook axisymmetric spherical coordinates; independent of
their operator library):
  (a.grad b)_r  = a_r dr(b_r) + (a_t/r) dt(b_r) - (a_t b_t + a_p b_p)/r
  (a.grad b)_t  = a_r dr(b_t) + (a_t/r) dt(b_t) + a_t b_r/r - a_p b_p ct/r
  (a.grad b)_p  = a_r dr(b_p) + (a_t/r) dt(b_p) + a_p b_r/r + a_p b_t ct/r
  grad p        = (dr p, dt p / r, 0)
  Lap f         = drr f + 2 dr f/r + dtt f/r^2 + ct dt f/r^2
  (Lap U)_r     = Lap u_r - 2u_r/r^2 - 2 dt(u_t)/r^2 - 2 u_t ct/r^2
  (Lap U)_t     = Lap u_t + 2 dt(u_r)/r^2 - u_t/(r^2 st^2)
  (Lap U)_p     = Lap u_p - u_p/(r^2 st^2)
  div U         = dr u_r + 2u_r/r + dt(u_t)/r + u_t ct/r
  xi.grad U     = r dr(u_r, u_t, u_p) componentwise   [xi = r e_r]
  Profile eq (paper 1.6, pressure form, up to overall sign / p-sign which
  the gate adjudicates):
      -1/2 U - 1/2 xi.grad U + U.grad U + grad p - Lap U = 0
  Linearized (paper 1.9): L_U v = 1/2 v + 1/2 xi.grad v
      - [U.grad v + v.grad U] - grad q + Lap v = lambda v

VERIFICATION LADDER:
  V1: conversion port sanity -- u1 angular basis ~ even Legendre P_{2j}(cos t)
  V2: stored u2 == DL @ u1 @ DR (representation-internal; no physics)
  V3: OUR divergence on reconstructed fields ~ 0 (selects far-field weight)
  V4: OUR momentum residual on profile ~ 0  <- PART 1 GATE
  V5: OUR linearized operator on their eigenpair: L_U v ~ lambda v,
      lambda = -0.11314203 (their convention)  <- PART 2 GATE
"""
import numpy as np
import scipy.io as sio
from numpy.polynomial import legendre as npleg

import os
REPO = os.environ.get('HWY_REPO',
        '../data/3d-navier-stokes-nonuniqueness')
S = 22.0   # scl_fac

# ----------------------------------------------------- ported: conversion
def double_fact_list(n):
    out = [1.0]; b = 1.0
    for i in range(2, n-1, 2):
        b *= 1.0 - 1.0/i
        out.append(b)
    return np.array(out)

def che2len(n, flag=False):
    b = double_fact_list(4*(n+1))
    A = np.zeros((n+1, n+1))
    for i in range(n+1):
        for j in range(i+1):
            if flag:
                A[i, j] = 2.0*b[i+j+1]*b[i-j]
            else:
                A[i, j] = 2.0*b[i+j]*b[i-j]
                if j == 0: A[i, j] /= 2.0
    return A

def conv_u1(M1):   # conversion_mat(M1,"11",1) = che2len(M1)[2:end, :]
    return che2len(M1)[1:, :]

def conv_u2(M1):   # conversion_mat(M1,"00",1) = che2len[1:,1:] @ diag(2n)
    return che2len(M1)[1:, 1:] @ np.diag(2.0*np.arange(1, M1+1))

def conv_10(M1):   # conversion_mat(M1,"10",1) = che2len(M1-1, flag=true)
    return che2len(M1-1, flag=True)

def conv_01(M1):   # conversion_mat(M1,"01",1) = che2len(M1-1,true) @ diag(2n-1)
    return che2len(M1-1, flag=True) @ np.diag(2.0*np.arange(1, M1+1) - 1.0)

def div_mat(M0, bdry):  # ported from div_mat
    I = np.vstack([np.eye(M0), np.zeros((1, M0))])
    d = (np.arange(1, M0+1) + 0.5)/2.0
    if bdry != "00": d -= 0.25
    L = np.zeros((M0+1, M0+1))
    L += np.diag(d, -1)[:M0+1, :M0+1]
    L -= np.diag(np.append(d, 0.0), 1)[:M0+1, :M0+1]
    L = L[:, :M0]
    return 1.5*I + L

# ----------------------------------------------------- basis synthesis
def basis(x, freqs, kind, deriv=0):
    """Matrix B[i,j] = d^deriv/dx^deriv of sin/cos(freqs[j]*x) at x[i]."""
    X = np.outer(x, freqs)
    f = freqs[None, :]
    if kind == 'sin':
        tab = [np.sin(X), f*np.cos(X), -(f**2)*np.sin(X)]
    else:
        tab = [np.cos(X), -f*np.sin(X), -(f**2)*np.cos(X)]
    return tab[deriv]

class Field:
    """f(beta,theta) = R(beta) @ C @ A(theta)^T with derivative access.
    rspec: either (rfreq, rkind) or a 2-tuple (col1_spec, rest_spec) for the
    two-part radial boundary case (first angular column uses col1_spec)."""
    def __init__(self, C, rspec, afreq, akind, two_part=False, conv=None):
        self.C, self.rspec, self.af, self.ak = C, rspec, afreq, akind
        self.two_part, self.conv = two_part, conv
    def ev(self, b, t, db=0, dt_=0):
        A = basis(t, self.af, self.ak, dt_)
        if not self.two_part:
            tmp = basis(b, *self.rspec, db) @ self.C
        else:
            (rf1, rk1), (rf2, rk2) = self.rspec
            tmp = np.hstack([basis(b, rf1, rk1, db) @ self.C[:, :1],
                             basis(b, rf2, rk2, db) @ self.C[:, 1:]])
        if self.conv is not None:          # conversion AFTER radial split,
            tmp = tmp @ self.conv          # matching their interpolate()
        return tmp @ A.T

def load_profile(path, bdry_index=2):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"HWY data file not found at:\n    {os.path.abspath(path)}\n"
            f"Set HWY_REPO to your clone of the HWY data repository so that\n"
            f"'<HWY_REPO>/data/UP.mat' and '<HWY_REPO>/data/up_eig.mat' exist\n"
            f"(current HWY_REPO default resolves REPO={REPO!r}).\n"
            f"See data/README.md for where to obtain the data; check the\n"
            f"exact filenames (case-sensitive: UP.mat, up_eig.mat).")
    try:
        d = sio.loadmat(path)
    except Exception as e:
        raise RuntimeError(
            f"Found {path} but failed to read it as a MATLAB .mat file "
            f"({type(e).__name__}: {e}). The file may be a newer (v7.3/HDF5) "
            f"format or a different layout than this loader expects; see "
            f"data/README.md and verify the data matches the campaign's.")
    try:
        u1, u2, u3, p = d['u1'], d['u2'], d['u3'], d['p']
    except KeyError as e:
        raise KeyError(
            f"{path} loaded, but expected variable {e} is missing. Keys "
            f"present: {[k for k in d if not k.startswith('__')]}. The data "
            f"layout differs from what this pipeline expects (see "
            f"data/README.md).")
    M1 = u1.shape[1]                                  # 150
    odd = lambda n: 2.0*np.arange(1, n+1) - 1.0
    evn = lambda n: 2.0*np.arange(1, n+1)
    F = {}
    if bdry_index == 2:    # even sector (profile)
        F['u1'] = Field(u1 @ conv_u1(M1), (odd(u1.shape[0]), 'sin'),
                        2.0*np.arange(0, M1+1), 'cos')
        F['u2'] = Field(u2 @ conv_u2(M1), (odd(u2.shape[0]), 'sin'),
                        evn(M1), 'sin')
        F['u3'] = Field(u3, (odd(u3.shape[0]), 'sin'), odd(M1), 'sin')
        F['p']  = Field(p,  (odd(p.shape[0]), 'cos'),
                        2.0*np.arange(0, M1), 'cos')
    elif bdry_index == 1:  # odd sector (eigenfunction)
        # u1: radial ["10","00"] (col1 cos odd, rest sin even); ang "10" w/ conv
        F['u1'] = Field(u1,
                        ((odd(u1.shape[0]), 'cos'), (evn(u1.shape[0]), 'sin')),
                        odd(M1), 'cos', two_part=True, conv=conv_10(M1))
        # u2: radial ["10","00"]; ang "01" w/ conv
        F['u2'] = Field(u2,
                        ((odd(u2.shape[0]), 'cos'), (evn(u2.shape[0]), 'sin')),
                        odd(M1), 'sin', two_part=True, conv=conv_01(M1))
        # u3: "00","00" no conv
        F['u3'] = Field(u3, (evn(u3.shape[0]), 'sin'), evn(M1), 'sin')
        # p: radial ["10","00"]; ang "10" no conv
        F['p']  = Field(p,
                        ((odd(p.shape[0]), 'cos'), (evn(p.shape[0]), 'sin')),
                        odd(M1), 'cos', two_part=True)
    lam = float(d['lambda'][0,0]) if 'lambda' in d else None
    return F, (u1, u2, u3, p), lam

# ----------------------------------------------------- our physics
class Phys:
    """Physical fields + derivatives on grid, from Fields + weight w(beta).
    Stored scalar = w * physical: physical = stored / w. Weight candidates
    handle the compactified far field."""
    def __init__(self, F, b, t, weight):
        self.b, self.t = b, t
        B, T = np.meshgrid(b, t, indexing='ij')
        self.rho = S*np.tan(B)
        self.ct  = 1.0/np.tan(T)
        self.st  = np.sin(T)
        cb = np.cos(B)
        self.dbdr  = cb**2/S                      # d beta / d rho
        self.d2bdr = -2*cb**3*np.sin(B)/S**2      # for second derivative
        if weight == 'raw':      w = np.ones_like(B);          dwdb = np.zeros_like(B)
        elif weight == 'rho':    w = 1.0/self.rho;             dwdb = -(1/(S*np.tan(B)**2))*(1/cb**2)
        elif weight == 'cos':    w = cb;                       dwdb = -np.sin(B)
        elif weight == 'cos2':   w = cb**2;                    dwdb = -2*cb*np.sin(B)
        else: raise ValueError(weight)
        # physical u = stored * w  (interpret 'weight' as multiplier on stored)
        self.f, self.fr, self.frr, self.ft, self.ftt, self.frt = {}, {}, {}, {}, {}, {}
        for k, fld in F.items():
            g    = fld.ev(b, t)
            gb   = fld.ev(b, t, db=1)
            gbb  = fld.ev(b, t, db=2)
            gt   = fld.ev(b, t, dt_=1)
            gtt  = fld.ev(b, t, dt_=2)
            gbt  = fld.ev(b, t, db=1, dt_=1)
            u    = w*g
            ub   = dwdb*g + w*gb                  # d/d beta
            ubb_ = None                           # second beta deriv below
            # d2w/db2 needed: compute numerically-free per weight
            if weight == 'raw':    d2w = np.zeros_like(g)
            elif weight == 'cos':  d2w = -cb
            elif weight == 'cos2': d2w = -2*np.cos(2*B)
            elif weight == 'rho':
                tb = np.tan(B)
                d2w = (2.0/(S*tb**3))*(1/cb**4) + (2.0/(S*tb**2))*(np.sin(B)/cb**3)
            ubb = d2w*g + 2*dwdb*gb + w*gbb
            ut  = w*gt
            utt = w*gtt
            ubt = dwdb*gt + w*gbt
            self.f[k]   = u
            self.fr[k]  = self.dbdr*ub                       # d/d rho
            self.frr[k] = self.dbdr**2*ubb + self.d2bdr*ub
            self.ft[k]  = ut
            self.ftt[k] = utt
            self.frt[k] = self.dbdr*ubt
    def adv(self, a, bfield):
        """(a . grad b) for vector fields given as dicts of component names."""
        r, ct = self.rho, self.ct
        ar, at_, ap = a['u1'], a['u2'], a['u3']
        out = {}
        out['u1'] = (ar*bfield.fr['u1'] + at_/r*bfield.ft['u1']
                     - (at_*bfield.f['u2'] + ap*bfield.f['u3'])/r)
        out['u2'] = (ar*bfield.fr['u2'] + at_/r*bfield.ft['u2']
                     + at_*bfield.f['u1']/r - ap*bfield.f['u3']*ct/r)
        out['u3'] = (ar*bfield.fr['u3'] + at_/r*bfield.ft['u3']
                     + ap*bfield.f['u1']/r + ap*bfield.f['u2']*ct/r)
        return out
    def lap_vec(self):
        r, ct, st = self.rho, self.ct, self.st
        def lap_s(k):
            return (self.frr[k] + 2*self.fr[k]/r + self.ftt[k]/r**2
                    + ct*self.ft[k]/r**2)
        L = {}
        L['u1'] = (lap_s('u1') - 2*self.f['u1']/r**2 - 2*self.ft['u2']/r**2
                   - 2*self.f['u2']*ct/r**2)
        L['u2'] = lap_s('u2') + 2*self.ft['u1']/r**2 - self.f['u2']/(r**2*st**2)
        L['u3'] = lap_s('u3') - self.f['u3']/(r**2*st**2)
        return L
    def div(self):
        r, ct = self.rho, self.ct
        return (self.fr['u1'] + 2*self.f['u1']/r + self.ft['u2']/r
                + self.f['u2']*ct/r)
    def grad_p(self):
        return {'u1': self.fr['p'], 'u2': self.ft['p']/self.rho,
                'u3': np.zeros_like(self.f['p'])}
    def xi_grad(self):
        return {k: self.rho*self.fr[k] for k in ('u1','u2','u3')}

def vnorm(d, keys=('u1','u2','u3')):
    return np.sqrt(sum(np.mean(d[k]**2) for k in keys))

# ----------------------------------------------------- the ladder
def main():
    F, raw, _ = load_profile(f'{REPO}/data/UP.mat')
    u1s, u2s, u3s, ps = raw

    # V1: u1 angular basis vs even Legendre
    t = np.linspace(0.05, np.pi/2-0.05, 200)
    M1 = u1s.shape[1]
    Cu1 = conv_u1(M1)
    okV1 = []
    for j in [0, 1, 4, 20]:
        synth = basis(t, 2.0*np.arange(0, M1+1), 'cos') @ Cu1[j]
        Pl = npleg.legval(np.cos(t), [0]*(2*(j+1)) + [1])    # P_{2(j+1)}(cos t)
        ratio = synth/Pl
        okV1.append(np.ptp(ratio)/np.abs(ratio).mean() < 1e-10)
    print(f"V1 conversion-port vs Legendre P_2j(cos t): "
          f"{'PASS' if all(okV1) else 'FAIL'} {okV1}")

    # V2: stored u2 vs DL @ u1 @ DR
    DL = div_mat(u1s.shape[0], "01")
    l = 2.0*np.arange(1, M1+1)
    DR = -np.diag(1.0/(l*(l+1)))
    u2_rec = DL @ u1s @ DR
    relV2 = np.linalg.norm(u2_rec - u2s)/np.linalg.norm(u2s)
    print(f"V2 div-free coupling u2 = DL.u1.DR: rel diff {relV2:.2e} "
          f"{'PASS' if relV2 < 1e-10 else 'FAIL'}")

    # interior grid (avoid axis/infinity)
    b = np.linspace(0.04, np.pi/2-0.04, 180)
    t = np.linspace(0.04, np.pi/2-0.04, 140)

    # V3: weight scan via OUR divergence
    print("V3 divergence (selects weight): rel ||div U|| / ||U||/rho_scale")
    best = None
    for w in ['raw', 'cos', 'cos2', 'rho']:
        P = Phys(F, b, t, w)
        dv = P.div()
        scale = vnorm(P.f)/np.median(P.rho) + 1e-300
        rel = np.sqrt(np.mean(dv**2))/scale
        print(f"   weight={w:5s}: {rel:.3e}")
        if best is None or rel < best[1]: best = (w, rel)
    wsel, relV3 = best
    print(f"V3 selected weight '{wsel}': {'PASS' if relV3 < 1e-6 else 'FAIL'}")

    # V4: momentum residual, sign conventions scanned
    P = Phys(F, b, t, wsel)
    U = {k: P.f[k] for k in ('u1','u2','u3')}
    ADV  = P.adv(U, P)
    LAP  = P.lap_vec()
    GP   = P.grad_p()
    XG   = P.xi_grad()
    nrm  = max(vnorm(ADV), vnorm(LAP), vnorm(GP))
    bestE = None
    for s1 in (+1.0, -1.0):
        for s2 in (+1.0, -1.0):
            E = {k: s1*(-0.5*U[k] - 0.5*XG[k]) + ADV[k] + s2*GP[k]
                    - s1*LAP[k] for k in U}
            r = vnorm(E)/nrm
            if bestE is None or r < bestE[2]: bestE = (s1, s2, r)
    s1, s2, relV4 = bestE
    print(f"V4 momentum residual: rel {relV4:.3e} with signs "
          f"(eq, gradp)=({s1:+.0f},{s2:+.0f})  "
          f"{'PASS' if relV4 < 1e-5 else 'FAIL'}")

    # V5: linearized operator on eigenpair
    Fv, rawv, lam = load_profile(f'{REPO}/data/up_eig.mat', bdry_index=1)
    # V5a: odd-sector div-free coupling (DL per column rule, odd harmonics)
    v1s, v2s = rawv[0], rawv[1]
    DLo  = div_mat(v1s.shape[0], "00")
    DLo2 = div_mat(v1s.shape[0], "10")
    lo = 2.0*np.arange(1, v1s.shape[1]+1) - 1.0
    DRo = -np.diag(1.0/(lo*(lo+1)))
    v2_rec = DLo @ v1s
    v2_rec[:, :1] = DLo2 @ v1s[:, :1]
    v2_rec = v2_rec @ DRo
    relV5a = np.linalg.norm(v2_rec - v2s)/np.linalg.norm(v2s)
    print(f"V5a odd-sector div-free coupling: rel diff {relV5a:.2e} "
          f"{'PASS' if relV5a < 1e-10 else 'FAIL'}")
    print(f"   stored eigenvalue lambda = {lam}")
    Pv = Phys(Fv, b, t, wsel)
    V  = {k: Pv.f[k] for k in ('u1','u2','u3')}
    UgV = P.adv(U, Pv)        # U.grad v
    VgU = Pv.adv(V, P)        # v.grad U
    LAPv = Pv.lap_vec()
    GQ   = Pv.grad_p()
    XGv  = Pv.xi_grad()
    nrmv = max(vnorm(UgV), vnorm(VgU), vnorm(LAPv), 1e-300)
    bestL = None
    for s1 in (+1.0, -1.0):
      for s2 in (+1.0, -1.0):
        for slam in (+1.0, -1.0):
            Lv = {k: s1*(0.5*V[k] + 0.5*XGv[k] + LAPv[k])
                     - s1*(UgV[k] + VgU[k]) + s2*GQ[k] for k in V}
            E = {k: Lv[k] - slam*lam*V[k] for k in V}
            r = vnorm(E)/nrmv
            if bestL is None or r < bestL[3]: bestL = (s1, s2, slam, r)
    s1, s2, slam, relV5 = bestL
    print(f"V5 eigenpair residual ||L_U v - lambda v||: rel {relV5:.3e} "
          f"signs (op, gradq, lam)=({s1:+.0f},{s2:+.0f},{slam:+.0f})  "
          f"{'PASS' if relV5 < 1e-4 else 'FAIL'}")

if __name__ == '__main__':
    main()
