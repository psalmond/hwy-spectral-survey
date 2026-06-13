# SPDX-License-Identifier: MIT
# Copyright (c) 2026 P. Salmond — hwy-spectral-survey (see LICENSE, NOTICE)
# Independent implementation of the HWY operator (arXiv:2509.25116, eq. for L_U);
# mirrors their public data formats/conventions; contains no code from their repo.
"""
PART 3 rev K -- Rayleigh-Ritz projection of OUR validated L_U onto the HWY
repo's generalized-eigenfunction subspaces (phi_odd: 24, phi_even: 19),
read via the minimal pure-Python HDF5 reader (hdf5min, gate-validated:
u1_conv_mat == conv_10(150)/conv_u1(150) bit-exact; shared repo data files
byte-identical to handover copies).

The phi's are honest L2 shell functions in the SAME storage convention as
up_eig.mat / UP.mat (validated by Part-2 gate machinery), so the projected
pencil G c = lam H c with
    G_ij = <phi_i, L_U phi_j + grad p_j>_{L2(R^3)},  H_ij = <phi_i, phi_j>
is small (<=25 dim), perfectly conditioned, and free of every pathology in
the rev A-I ladder.  Gates:
  W2-pi : odd 25-dim (24 phi + up_eig) spectrum must contain +0.11314.
  W4-pi : with U := 0 the +0.11314 mode must vanish.
  tail  : per-phi mass fraction beyond beta=1.2 reported (L2 honesty).
Then the EVEN 19-dim pencil = the novel survey output.
"""
import sys
import numpy as np
import scipy.linalg as sla
from numpy.polynomial.legendre import leggauss
from hdf5min import loadmat73
from ns_part12_gate import (Field, Phys, conv_u1, conv_u2, conv_10, conv_01)
from ns_part3_spectrum import LAM_KNOWN, DATA, S
from ns_part3f import make_geo, B0, T0, T1, gl_nodes
from ns_part3_spectrum import adv, lap_vec

import os
REPO = os.environ.get('HWY_REPO',
        '../data/3d-navier-stokes-nonuniqueness') + '/data'

def fields_from_dict(d, kind):
    u1, u2, u3, p = d['u1'], d['u2'], d['u3'], d['p']
    M1 = u1.shape[1]
    odd = lambda n: 2.0*np.arange(1, n+1) - 1.0
    evn = lambda n: 2.0*np.arange(1, n+1)
    F = {}
    if kind == 'even':
        F['u1'] = Field(u1 @ conv_u1(M1), (odd(u1.shape[0]), 'sin'),
                        2.0*np.arange(0, M1+1), 'cos')
        F['u2'] = Field(u2 @ conv_u2(M1), (odd(u2.shape[0]), 'sin'),
                        evn(M1), 'sin')
        F['u3'] = Field(u3, (odd(u3.shape[0]), 'sin'), odd(M1), 'sin')
        F['p']  = Field(p,  (odd(p.shape[0]), 'cos'),
                        2.0*np.arange(0, M1), 'cos')
    else:
        F['u1'] = Field(u1,
                        ((odd(u1.shape[0]), 'cos'), (evn(u1.shape[0]), 'sin')),
                        odd(M1), 'cos', two_part=True, conv=conv_10(M1))
        F['u2'] = Field(u2,
                        ((odd(u2.shape[0]), 'cos'), (evn(u2.shape[0]), 'sin')),
                        odd(M1), 'sin', two_part=True, conv=conv_01(M1))
        F['u3'] = Field(u3, (evn(u3.shape[0]), 'sin'), evn(M1), 'sin')
        F['p']  = Field(p,
                        ((odd(p.shape[0]), 'cos'), (evn(p.shape[0]), 'sin')),
                        odd(M1), 'cos', two_part=True)
    return F

def load_family(kind):
    fams, lams = [], []
    if kind == 'odd':
        for j in range(24):
            d = loadmat73(f'{REPO}/phi_odd/up_phi_{j}.mat')
            fams.append(fields_from_dict(d, 'odd'))
            lams.append(float(np.atleast_1d(d['lambda']).ravel()[0]))
        import scipy.io as sio
        d = sio.loadmat(f'{DATA}/up_eig.mat')
        fams.append(fields_from_dict(d, 'odd'))       # 25th: the eigenpair
        lams.append(float(np.atleast_1d(d['lambda']).ravel()[0]))
    else:
        for j in range(19):
            d = loadmat73(f'{REPO}/phi_even/UP_phi_{j}.mat')
            fams.append(fields_from_dict(d, 'even'))
            lams.append(np.nan)
    return fams, lams

def profile_F():
    from ns_part12_gate import load_profile
    F, _, _ = load_profile(f'{DATA}/UP.mat', bdry_index=2)
    return F

def Lcols(P, Uval, Ubf, geo, withU=True):
    comp = {k: dict(f=P.f[k], fr=P.fr[k], frr=P.frr[k],
                    ft=P.ft[k], ftt=P.ftt[k]) for k in ('u1', 'u2', 'u3')}
    vval = {k: comp[k]['f'] for k in ('u1', 'u2', 'u3')}
    LAP = lap_vec(comp, geo)
    out = {}
    for k in ('u1', 'u2', 'u3'):
        out[k] = (0.5*comp[k]['f'] + 0.5*geo['rho']*comp[k]['fr'] + LAP[k])
    if withU:
        UgV = adv(Uval, comp, geo)
        VgU = adv(vval, Ubf, geo)
        for k in ('u1', 'u2', 'u3'):
            out[k] = out[k] - UgV[k] - VgU[k]
    # + grad p (this phi's own pressure)
    out['u1'] = out['u1'] + P.fr['p']
    out['u2'] = out['u2'] + P.ft['p']/geo['rho']
    return np.concatenate([out['u1'].ravel(), out['u2'].ravel(),
                           out['u3'].ravel()])

def project(kind, Nb=400, Nt=200, nbchunk=20):
    fams, lams = load_family(kind)
    n = len(fams)
    UF = profile_F()
    b, wb = gl_nodes(Nb, B0, T1 if False else (np.pi/2 - 0.04))
    t, wt = gl_nodes(Nt, T0, T1)
    G = np.zeros((n, n)); G0 = np.zeros((n, n)); H = np.zeros((n, n))
    tail = np.zeros(n); tot = np.zeros(n)
    for s in range(0, Nb, nbchunk):
        e = min(s+nbchunk, Nb)
        bc = b[s:e]
        geo = make_geo(bc, t)
        w = (geo['rho']**2*geo['st']*S/geo['cb']**2
             * np.outer(wb[s:e], wt)).ravel()
        sqW = np.sqrt(np.tile(w, 3))
        PU = Phys(UF, bc, t, 'cos')
        Uval = {k: PU.f[k] for k in ('u1', 'u2', 'u3')}
        Ubf = {k: dict(f=PU.f[k], fr=PU.fr[k], frr=PU.frr[k],
                       ft=PU.ft[k], ftt=PU.ftt[k]) for k in ('u1','u2','u3')}
        Ngc = geo['B'].size
        M = np.empty((3*Ngc, n)); A = np.empty((3*Ngc, n))
        A0 = np.empty((3*Ngc, n))
        for j, F in enumerate(fams):
            P = Phys(F, bc, t, 'cos')
            M[:, j] = sqW*np.concatenate([P.f['u1'].ravel(),
                                          P.f['u2'].ravel(),
                                          P.f['u3'].ravel()])
            A[:, j] = sqW*Lcols(P, Uval, Ubf, geo, True)
            A0[:, j] = sqW*Lcols(P, Uval, Ubf, geo, False)
        G += M.T @ A; G0 += M.T @ A0; H += M.T @ M
        msq = (M**2).sum(axis=0)
        tot += msq
        if bc[0] > 1.2:
            tail += msq
        print(f"    chunk {s}-{e}", flush=True)
    return G, G0, H, np.array(lams), tail/np.maximum(tot, 1e-300)

def solve_report(G, H, lams, tag):
    lam = sla.eig(G, H, right=False)
    lam = lam[np.argsort(-lam.real)]
    print(f"  {tag} cond(H) = {np.linalg.cond(H):.3e}")
    print(f"  {tag} Ritz values:")
    for z in lam:
        print(f"    {z.real:+.6f} {z.imag:+.6f}j")
    d = np.min(np.abs(lam - LAM_KNOWN))
    print(f"  {tag} dist to +0.11314: {d:.2e} "
          f"{'PASS' if d < 5e-3 else 'FAIL'}", flush=True)
    return lam

if __name__ == '__main__':
    kind = sys.argv[1] if len(sys.argv) > 1 else 'odd'
    G, G0, H, lams, tailfrac = project(kind)
    np.savez(f'projK_{kind}.npz', G=G, G0=G0, H=H, lams=lams, tail=tailfrac)
    print(f"== rev K {kind}: tail-mass fractions (beta>1.2) ==")
    print(np.array2string(tailfrac, precision=2, max_line_width=100))
    lam = solve_report(G, H, lams, f"[{kind} U-on]")
    lam0 = solve_report(G0, H, lams, f"[{kind} U-OFF]")
    if kind == 'odd':
        print("  their stored lambdas (sign conv as stored):")
        print(np.array2string(lams, precision=6, max_line_width=100))
