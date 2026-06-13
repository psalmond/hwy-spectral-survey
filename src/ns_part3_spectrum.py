# SPDX-License-Identifier: MIT
# Copyright (c) 2026 P. Salmond — hwy-spectral-survey (see LICENSE, NOTICE)
# Independent implementation of the HWY operator (arXiv:2509.25116, eq. for L_U);
# mirrors their public data formats/conventions; contains no code from their repo.
"""
PART 3 (rung 1) -- spectrum survey of OUR validated linearized operator L_U.

Strategy (per HANDOVER_NEXT.md): float-level NAVIGATION, not certification.
Discretize L_U on truncated copies of the gate-validated sector bases,
slave u2 via the (V2/V5a bit-exact) div-free coupling, eliminate pressure by
orthogonal projection against the truncated grad-q range, and eigensolve the
projected pencil.  All operator formulas are imported verbatim from
ns_part12_gate.py (V3/V4/V5-validated); ZERO new physics here, only new
linear algebra.

Equation form (signs gate-determined in V5):
    (1/2 v + 1/2 rho dr v + Lap v) - (U.grad v + v.grad U) + grad q = lam v
    expected known mode in ODD sector: lam = +0.11314203...

Ladder:
  W0  truncated conversion sanity (even-sector conv vs Legendre, small Na)
  W1  truncated slaving => pointwise div ~ machine eps (both sectors)
  W2  odd-sector survey reproduces lam = +0.11314   <- ENTRY GATE
  W3  resolution stability scan (spurious-mode filter)
  W4  negative control: U := 0 kills the 0.113 mode
  S   even-sector survey (the unexplored target)
"""
import numpy as np
import scipy.io as sio
from numpy.polynomial import legendre as npleg
from ns_part12_gate import (basis, che2len, conv_u1, conv_u2, conv_10,
                            conv_01, div_mat, Field, load_profile, Phys, S,
                            vnorm)

import os
DATA = os.environ.get('HWY_DATA', '../data/derived')
LAM_KNOWN = 0.11314203274385946

# ---------------------------------------------------------------- grids
def make_grid(Nb=72, Nt=48, bmax=np.pi/2 - 0.04):
    b = np.linspace(0.04, bmax, Nb)
    t = np.linspace(0.04, np.pi/2 - 0.04, Nt)
    B, T = np.meshgrid(b, t, indexing='ij')
    geo = dict(B=B, T=T, rho=S*np.tan(B), ct=1.0/np.tan(T), st=np.sin(T),
               cb=np.cos(B), sb=np.sin(B),
               dbdr=np.cos(B)**2/S, d2bdr=-2*np.cos(B)**3*np.sin(B)/S**2)
    return b, t, geo

# -------------------------------------------- weight 'cos' chain rule
# (verbatim transcription of the validated Phys weight/derivative block)
def phys_from_raw(g, gb, gbb, gt, gtt, geo):
    cb, sb = geo['cb'], geo['sb']
    w, dwdb, d2w = cb, -sb, -cb
    u   = w*g
    ub  = dwdb*g + w*gb
    ubb = d2w*g + 2*dwdb*gb + w*gbb
    return dict(f=u,
                fr=geo['dbdr']*ub,
                frr=geo['dbdr']**2*ubb + geo['d2bdr']*ub,
                ft=w*gt, ftt=w*gtt)

# ------------------------------------- validated operator formulas
# (verbatim from Phys.adv / Phys.lap_vec / Phys.grad_p, dict-based)
def adv(a, bf, geo):
    r, ct = geo['rho'], geo['ct']
    ar, at_, ap = a['u1'], a['u2'], a['u3']
    out = {}
    out['u1'] = (ar*bf['u1']['fr'] + at_/r*bf['u1']['ft']
                 - (at_*bf['u2']['f'] + ap*bf['u3']['f'])/r)
    out['u2'] = (ar*bf['u2']['fr'] + at_/r*bf['u2']['ft']
                 + at_*bf['u1']['f']/r - ap*bf['u3']['f']*ct/r)
    out['u3'] = (ar*bf['u3']['fr'] + at_/r*bf['u3']['ft']
                 + ap*bf['u1']['f']/r + ap*bf['u2']['f']*ct/r)
    return out

def lap_vec(bf, geo):
    r, ct, st = geo['rho'], geo['ct'], geo['st']
    def lap_s(k):
        return (bf[k]['frr'] + 2*bf[k]['fr']/r + bf[k]['ftt']/r**2
                + ct*bf[k]['ft']/r**2)
    L = {}
    L['u1'] = (lap_s('u1') - 2*bf['u1']['f']/r**2 - 2*bf['u2']['ft']/r**2
               - 2*bf['u2']['f']*ct/r**2)
    L['u2'] = lap_s('u2') + 2*bf['u1']['ft']/r**2 - bf['u2']['f']/(r**2*st**2)
    L['u3'] = lap_s('u3') - bf['u3']['f']/(r**2*st**2)
    return L

def div_of(bf, geo):
    r, ct = geo['rho'], geo['ct']
    return (bf['u1']['fr'] + 2*bf['u1']['f']/r + bf['u2']['ft']/r
            + bf['u2']['f']*ct/r)

# ------------------------------------------------ sector descriptions
class Sector:
    """Truncated dof parametrization of one symmetry sector.
    Velocity dofs: c1 (Nr x Na) for u1 [u2 slaved], c3 (Nr x Na) for u3.
    Pressure dofs: cq (Nr x Na).
    Provides per-dof outer-product factors (radial vecs r0,r1,r2 and angular
    vecs a0,a1,a2) for each component touched by that dof."""
    def __init__(self, kind, Nr, Na, b, t):
        self.kind, self.Nr, self.Na = kind, Nr, Na
        odd = lambda n: 2.0*np.arange(1, n+1) - 1.0
        evn = lambda n: 2.0*np.arange(1, n+1)
        # radial basis tables: dict name -> [B0,B1,B2] (Nb x Nmodes)
        def rtab(freqs, kindk):
            return [basis(b, freqs, kindk, d) for d in range(3)]
        # angular tables: (Nt x Nfreq) per deriv
        def atab(freqs, kindk):
            return [basis(t, freqs, kindk, d) for d in range(3)]
        if kind == 'even':
            self.r_u1 = rtab(odd(Nr), 'sin')
            self.r_u2 = rtab(odd(Nr+1), 'sin')
            self.r_u3 = rtab(odd(Nr), 'sin')
            self.r_q  = rtab(odd(Nr), 'cos')
            self.a_u1 = atab(2.0*np.arange(0, Na+1), 'cos')
            self.a_u2 = atab(evn(Na), 'sin')
            self.a_u3 = atab(odd(Na), 'sin')
            self.a_q  = atab(2.0*np.arange(0, Na), 'cos')
            self.cv1 = conv_u1(Na)            # (Na x Na+1)
            self.cv2 = conv_u2(Na)            # (Na x Na)
            self.DL  = div_mat(Nr, "01")      # (Nr+1 x Nr)
            l = evn(Na)
            self.DRd = -1.0/(l*(l+1.0))       # diagonal entries
            self.DL2 = None
        elif kind == 'odd':
            self.r_u1a = rtab(odd(Nr), 'cos')   # coefficient col 1
            self.r_u1b = rtab(evn(Nr), 'sin')   # remaining cols
            self.r_u2a = rtab(odd(Nr+1), 'cos')
            self.r_u2b = rtab(evn(Nr+1), 'sin')
            self.r_u3 = rtab(evn(Nr), 'sin')
            self.r_qa = rtab(odd(Nr), 'cos')
            self.r_qb = rtab(evn(Nr), 'sin')
            self.a_u1 = atab(odd(Na), 'cos')
            self.a_u2 = atab(odd(Na), 'sin')
            self.a_u3 = atab(evn(Na), 'sin')
            self.a_q  = atab(odd(Na), 'cos')
            self.cv1 = conv_10(Na)            # (Na x Na)
            self.cv2 = conv_01(Na)            # (Na x Na)
            self.DL  = div_mat(Nr, "00")
            self.DL2 = div_mat(Nr, "10")
            l = odd(Na)
            self.DRd = -1.0/(l*(l+1.0))
        else:
            raise ValueError(kind)

    # ---- raw derivative grids for a single velocity dof of c1 (slaved u2)
    def vel_dof_u1(self, i, j, geo):
        Na = self.Na
        if self.kind == 'even':
            r1 = [self.r_u1[d][:, i] for d in range(3)]
            avec = [self.cv1[j] @ self.a_u1[d].T for d in range(3)]
            u2col = self.DL[:, i] * self.DRd[j]              # (Nr+1,)
            r2 = [self.r_u2[d] @ u2col for d in range(3)]
            avec2 = [self.cv2[j] @ self.a_u2[d].T for d in range(3)]
        else:
            rt = self.r_u1a if j == 0 else self.r_u1b
            r1 = [rt[d][:, i] for d in range(3)]
            avec = [self.cv1[j] @ self.a_u1[d].T for d in range(3)]
            DLj = self.DL2 if j == 0 else self.DL
            u2col = DLj[:, i] * self.DRd[j]
            rt2 = self.r_u2a if j == 0 else self.r_u2b
            r2 = [rt2[d] @ u2col for d in range(3)]
            avec2 = [self.cv2[j] @ self.a_u2[d].T for d in range(3)]
        comp = {}
        comp['u1'] = phys_from_raw(np.outer(r1[0], avec[0]),
                                   np.outer(r1[1], avec[0]),
                                   np.outer(r1[2], avec[0]),
                                   np.outer(r1[0], avec[1]),
                                   np.outer(r1[0], avec[2]), geo)
        comp['u2'] = phys_from_raw(np.outer(r2[0], avec2[0]),
                                   np.outer(r2[1], avec2[0]),
                                   np.outer(r2[2], avec2[0]),
                                   np.outer(r2[0], avec2[1]),
                                   np.outer(r2[0], avec2[2]), geo)
        comp['u3'] = zero_comp(geo)
        return comp

    def vel_dof_u3(self, i, j, geo):
        r3 = [self.r_u3[d][:, i] for d in range(3)]
        a3 = [self.a_u3[d][:, j] for d in range(3)]
        comp = {'u1': zero_comp(geo), 'u2': zero_comp(geo)}
        comp['u3'] = phys_from_raw(np.outer(r3[0], a3[0]),
                                   np.outer(r3[1], a3[0]),
                                   np.outer(r3[2], a3[0]),
                                   np.outer(r3[0], a3[1]),
                                   np.outer(r3[0], a3[2]), geo)
        return comp

    def q_dof(self, i, j, geo):
        if self.kind == 'even':
            rq = [self.r_q[d][:, i] for d in range(3)]
        else:
            rt = self.r_qa if j == 0 else self.r_qb
            rq = [rt[d][:, i] for d in range(3)]
        aq = [self.a_q[d][:, j] for d in range(3)]
        return phys_from_raw(np.outer(rq[0], aq[0]),
                             np.outer(rq[1], aq[0]),
                             np.outer(rq[2], aq[0]),
                             np.outer(rq[0], aq[1]),
                             np.outer(rq[0], aq[2]), geo)

def zero_comp(geo):
    z = np.zeros_like(geo['B'])
    return dict(f=z, fr=z, frr=z, ft=z, ftt=z)

# ---------------------------------------------------- profile on grid
def profile_grids(b, t):
    F, _, _ = load_profile(f'{DATA}/UP.mat', bdry_index=2)
    P = Phys(F, b, t, 'cos')
    Uval = {k: P.f[k] for k in ('u1', 'u2', 'u3')}
    Ubf = {k: dict(f=P.f[k], fr=P.fr[k], frr=P.frr[k],
                   ft=P.ft[k], ftt=P.ftt[k]) for k in ('u1', 'u2', 'u3')}
    return Uval, Ubf

# ----------------------------------------------------- assembly
def Lu_of(comp, Uval, Ubf, geo):
    """validated linearized operator (velocity part):
       0.5 v + 0.5 rho dr v + Lap v - (U.grad v + v.grad U)"""
    vval = {k: comp[k]['f'] for k in ('u1', 'u2', 'u3')}
    UgV = adv(Uval, comp, geo)
    VgU = adv(vval, Ubf, geo)
    LAP = lap_vec(comp, geo)
    out = {}
    for k in ('u1', 'u2', 'u3'):
        out[k] = (0.5*comp[k]['f'] + 0.5*geo['rho']*comp[k]['fr']
                  + LAP[k] - UgV[k] - VgU[k])
    return out

def assemble(sector, geo, Uval, Ubf, quad_weight=None):
    Nr, Na = sector.Nr, sector.Na
    Ng = geo['B'].size
    nvel = 2*Nr*Na
    nq = Nr*Na
    Av = np.empty((3*Ng, nvel))
    Mv = np.empty((3*Ng, nvel))
    Aq = np.empty((3*Ng, nq))
    w = np.ones(3*Ng) if quad_weight is None else np.tile(quad_weight, 3)

    def stackL(comp):
        L = Lu_of(comp, Uval, Ubf, geo)
        return np.concatenate([L['u1'].ravel(), L['u2'].ravel(),
                               L['u3'].ravel()])
    def stackV(comp):
        return np.concatenate([comp['u1']['f'].ravel(),
                               comp['u2']['f'].ravel(),
                               comp['u3']['f'].ravel()])
    k = 0
    for i in range(Nr):
        for j in range(Na):
            comp = sector.vel_dof_u1(i, j, geo)
            Av[:, k] = w*stackL(comp); Mv[:, k] = w*stackV(comp); k += 1
    for i in range(Nr):
        for j in range(Na):
            comp = sector.vel_dof_u3(i, j, geo)
            Av[:, k] = w*stackL(comp); Mv[:, k] = w*stackV(comp); k += 1
    k = 0
    for i in range(Nr):
        for j in range(Na):
            qc = sector.q_dof(i, j, geo)
            # grad q = (fr, ft/rho, 0)
            Aq[:, k] = w*np.concatenate([qc['fr'].ravel(),
                                         (qc['ft']/geo['rho']).ravel(),
                                         np.zeros(Ng)])
            k += 1
    return Av, Mv, Aq

def project_out(Aq, X):
    """X - Aq Aq^+ X  (orthogonal projection onto complement of range Aq)"""
    coef, *_ = np.linalg.lstsq(Aq, X, rcond=None)
    return X - Aq @ coef

def survey(kind, Nr, Na, b, t, geo, Uval, Ubf, quad_weight=None):
    sec = Sector(kind, Nr, Na, b, t)
    Av, Mv, Aq = assemble(sec, geo, Uval, Ubf, quad_weight)
    G = project_out(Aq, Av)
    H = project_out(Aq, Mv)
    T, *_ = np.linalg.lstsq(H, G, rcond=None)
    lam = np.linalg.eigvals(T)
    return lam, (Av, Mv, Aq, sec)

# ----------------------------------------------------------- ladder
def main():
    b, t, geo = make_grid()
    Uval, Ubf = profile_grids(b, t)

    # W0: truncated conversion sanity (even sector, Na=12)
    Na = 12
    cvs = conv_u1(Na)
    tt = np.linspace(0.05, np.pi/2 - 0.05, 200)
    ok = []
    for j in [0, 3, Na-1]:
        synth = basis(tt, 2.0*np.arange(0, Na+1), 'cos') @ cvs[j]
        Pl = npleg.legval(np.cos(tt), [0]*(2*(j+1)) + [1])
        ratio = synth/Pl
        ok.append(np.ptp(ratio)/np.abs(ratio).mean() < 1e-10)
    print(f"W0 truncated conversion vs Legendre: "
          f"{'PASS' if all(ok) else 'FAIL'} {ok}")

    # W1: truncated slaving => div ~ 0 on grid, both sectors
    rng = np.random.default_rng(0)
    for kind in ('even', 'odd'):
        Nr, Na = 16, 8
        sec = Sector(kind, Nr, Na, b, t)
        acc = zero_comp(geo); acc = {k: dict(acc) for k in ('u1','u2','u3')}
        tot = {k: {f: np.zeros_like(geo['B']) for f in
                   ('f','fr','frr','ft','ftt')} for k in ('u1','u2','u3')}
        for _ in range(12):
            i, j = rng.integers(0, Nr), rng.integers(0, Na)
            c = rng.standard_normal()
            comp = sec.vel_dof_u1(i, j, geo)
            for k in ('u1','u2','u3'):
                for f in tot[k]:
                    tot[k][f] += c*comp[k][f]
        dv = div_of(tot, geo)
        scale = (np.sqrt(sum(np.mean(tot[k]['f']**2) for k in tot))
                 / np.median(geo['rho']) + 1e-300)
        rel = np.sqrt(np.mean(dv**2))/scale
        print(f"W1 [{kind}] truncated slaving div: rel {rel:.2e} "
              f"{'PASS' if rel < 1e-9 else 'FAIL'}")

    # W2 + W3: odd-sector survey, resolution scan
    print("\n-- ODD sector survey (gate: reproduce +0.11314) --")
    results = {}
    for (Nr, Na) in [(16, 8), (22, 11), (28, 14)]:
        lam, _ = survey('odd', Nr, Na, b, t, geo, Uval, Ubf)
        lam = lam[np.isfinite(lam)]
        top = lam[np.argsort(-lam.real)][:8]
        results[(Nr, Na)] = lam
        s = ", ".join(f"{z.real:+.5f}{z.imag:+.4f}j" for z in top)
        print(f"  (Nr,Na)=({Nr},{Na}): top Re: {s}")
        d = np.min(np.abs(lam - LAM_KNOWN))
        print(f"   dist to +0.11314: {d:.2e}  "
              f"{'PASS' if d < 5e-3 else 'FAIL'}")

    # W4: negative control -- U := 0
    print("\n-- W4 negative control (U=0), odd sector (22,11) --")
    Z = {k: np.zeros_like(geo['B']) for k in ('u1','u2','u3')}
    Zbf = {k: {f: np.zeros_like(geo['B']) for f in
               ('f','fr','frr','ft','ftt')} for k in ('u1','u2','u3')}
    lam0, _ = survey('odd', 22, 11, b, t, geo, Z, Zbf)
    lam0 = lam0[np.isfinite(lam0)]
    d0 = np.min(np.abs(lam0 - LAM_KNOWN))
    print(f"  dist of U=0 spectrum to +0.11314: {d0:.2e}  "
          f"{'PASS (mode is profile-driven)' if d0 > 5e-3 else 'FAIL'}")
    top0 = lam0[np.argsort(-lam0.real)][:6]
    print("  U=0 top Re:", ", ".join(f"{z.real:+.4f}{z.imag:+.4f}j"
                                     for z in top0))

    # S: even-sector survey
    print("\n-- EVEN sector survey (unexplored) --")
    for (Nr, Na) in [(16, 8), (22, 11), (28, 14)]:
        lam, _ = survey('even', Nr, Na, b, t, geo, Uval, Ubf)
        lam = lam[np.isfinite(lam)]
        top = lam[np.argsort(-lam.real)][:8]
        s = ", ".join(f"{z.real:+.5f}{z.imag:+.4f}j" for z in top)
        print(f"  (Nr,Na)=({Nr},{Na}): top Re: {s}")

if __name__ == '__main__':
    main()
