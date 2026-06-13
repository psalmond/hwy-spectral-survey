# SPDX-License-Identifier: MIT
# Copyright (c) 2026 P. Salmond — hwy-spectral-survey (see LICENSE, NOTICE)
# Independent implementation of the HWY operator (arXiv:2509.25116, eq. for L_U);
# mirrors their public data formats/conventions; contains no code from their repo.
"""
PART 3 rev F -- row-streamed assembly, Gauss-Legendre quadrature,
asymmetric pressure span, Leray-Schur + whitened eigensolve (rev E core).

Rev E findings: machinery validated (W2b with planted v*+grad p* recovers
+0.11314 to 3.5e-4 at eps_h=1e-10); plain-W2 deficit isolated to the
velocity truncation tail (W2a-red ~ 1.28 at (40,20) even with q doubled).
Also: uniform-grid rectangle quadrature commits ~2% Gram errors on top
modes.  Rev F therefore: (a) Gauss-Legendre nodes/weights in both beta and
theta; (b) beta-chunk streaming so v=(60,30) fits in 3 GB; (c) q span
scaled independently.
"""
import sys
import numpy as np
import scipy.linalg as sla
from numpy.polynomial.legendre import leggauss
from ns_part3_spectrum import Sector, Lu_of, LAM_KNOWN, DATA, S
from ns_part12_gate import load_profile, Phys
from ns_part3e import schur_whiten_eig, report

B0, T0, T1 = 0.04, 0.04, np.pi/2 - 0.04

def gl_nodes(n, a, c):
    x, w = leggauss(n)
    return 0.5*(c-a)*x + 0.5*(c+a), 0.5*(c-a)*w

def make_geo(b, t):
    B, T = np.meshgrid(b, t, indexing='ij')
    return dict(B=B, T=T, rho=S*np.tan(B), ct=1.0/np.tan(T), st=np.sin(T),
                cb=np.cos(B), sb=np.sin(B),
                dbdr=np.cos(B)**2/S, d2bdr=-2*np.cos(B)**3*np.sin(B)/S**2)

def stackV(comp):
    return np.concatenate([comp[k]['f'].ravel() for k in ('u1','u2','u3')])

def stackL(comp, Uval, Ubf, geo):
    L = Lu_of(comp, Uval, Ubf, geo)
    return np.concatenate([L[k].ravel() for k in ('u1','u2','u3')])

def chunk_fields(bc, t):
    """profile + stored eigenpair on a beta-chunk grid"""
    F, _, _ = load_profile(f'{DATA}/UP.mat', bdry_index=2)
    P = Phys(F, bc, t, 'cos')
    Uval = {k: P.f[k] for k in ('u1','u2','u3')}
    Ubf = {k: dict(f=P.f[k], fr=P.fr[k], frr=P.frr[k],
                   ft=P.ft[k], ftt=P.ftt[k]) for k in ('u1','u2','u3')}
    return Uval, Ubf

def stored_chunk(bc, t, geo):
    F, _, _ = load_profile(f'{DATA}/up_eig.mat', bdry_index=1)
    P = Phys(F, bc, t, 'cos')
    comp = {k: dict(f=P.f[k], fr=P.fr[k], frr=P.frr[k],
                    ft=P.ft[k], ftt=P.ftt[k]) for k in ('u1','u2','u3')}
    Ng = geo['B'].size
    gq = np.concatenate([P.fr['p'].ravel(),
                         (P.ft['p']/geo['rho']).ravel(),
                         np.zeros(Ng)])
    return comp, gq

def assemble_stream(kind, Nr, Na, Nrq, Naq, bmax, Nb, Nt, enrich_on,
                    nbchunk=12, wkind='full'):
    b, wb = gl_nodes(Nb, B0, bmax)
    t, wt = gl_nodes(Nt, T0, T1)
    nvel = 2*Nr*Na
    nq = Nrq*Naq + (1 if enrich_on else 0)
    ne = nvel + (1 if enrich_on else 0)
    GvA = np.zeros((ne, ne)); QA = np.zeros((nq, ne))
    GvQ = np.zeros((ne, nq)); Hvv = np.zeros((ne, ne))
    Gqq = np.zeros((nq, nq))
    for s in range(0, Nb, nbchunk):
        e = min(s+nbchunk, Nb)
        bc = b[s:e]
        geo = make_geo(bc, t)
        Ngc = geo['B'].size
        if wkind == 'full':
            wq = (geo['rho']**2*geo['st']*S/geo['cb']**2
                  * np.outer(wb[s:e], wt)).ravel()    # true-L2 x GL weights
        elif wkind == 'tame':                          # L2(cos^2 b dx)
            wq = (S**3*np.tan(geo['B'])**2*geo['st']
                  * np.outer(wb[s:e], wt)).ravel()
        elif wkind.startswith('w'):                    # L2(cos^{2s} b dx)
            sxp = float(wkind[1:])
            wq = (geo['rho']**2*geo['st']*S/geo['cb']**2
                  * geo['cb']**(2.0*sxp)
                  * np.outer(wb[s:e], wt)).ravel()
        else:
            raise ValueError(wkind)
        sqW = np.sqrt(np.tile(wq, 3))
        Uval, Ubf = chunk_fields(bc, t)
        sec = Sector(kind, Nr, Na, bc, t)
        sq_ = Sector(kind, Nrq, Naq, bc, t)
        dofs = ([('u1', i, j) for i in range(Nr) for j in range(Na)]
                + [('u3', i, j) for i in range(Nr) for j in range(Na)])
        Mv = np.empty((3*Ngc, ne)); Av = np.empty((3*Ngc, ne))
        for k, (c, i, j) in enumerate(dofs):
            comp = (sec.vel_dof_u1(i, j, geo) if c == 'u1'
                    else sec.vel_dof_u3(i, j, geo))
            Mv[:, k] = sqW*stackV(comp)
            Av[:, k] = sqW*stackL(comp, Uval, Ubf, geo)
        Aq = np.empty((3*Ngc, nq))
        k = 0
        for i in range(Nrq):
            for j in range(Naq):
                qc = sq_.q_dof(i, j, geo)
                Aq[:, k] = sqW*np.concatenate([qc['fr'].ravel(),
                                               (qc['ft']/geo['rho']).ravel(),
                                               np.zeros(Ngc)])
                k += 1
        if enrich_on:
            comp_e, gq = stored_chunk(bc, t, geo)
            Mv[:, nvel] = sqW*stackV(comp_e)
            Av[:, nvel] = sqW*stackL(comp_e, Uval, Ubf, geo)
            Aq[:, nq-1] = sqW*gq
        GvA += Mv.T @ Av
        QA += Aq.T @ Av
        GvQ += Mv.T @ Aq
        Hvv += Mv.T @ Mv
        Gqq += Aq.T @ Aq
        print(f"    chunk {s}-{e} done", flush=True)
        del Mv, Av, Aq
    return GvA, QA, GvQ, Hvv, Gqq

def main():
    Nr, Na = int(sys.argv[1]), int(sys.argv[2])
    Nrq, Naq = int(sys.argv[3]), int(sys.argv[4])
    bmax = float(sys.argv[5])
    enrich_on = bool(int(sys.argv[6]))
    Nb = int(sys.argv[7]) if len(sys.argv) > 7 else 180
    Nt = int(sys.argv[8]) if len(sys.argv) > 8 else 120
    kind = sys.argv[9] if len(sys.argv) > 9 else 'odd'
    tag0 = (f"{kind} v({Nr},{Na}) q({Nrq},{Naq}) bmax={bmax} GL({Nb},{Nt})"
            + (" ENR" if enrich_on else ""))
    print(f"== rev F {tag0} ==", flush=True)
    blocks = assemble_stream(kind, Nr, Na, Nrq, Naq, bmax, Nb, Nt, enrich_on)
    for eps_h in (1e-5, 3e-6, 1e-6, 3e-7):
        lam, kq, kh, Gred, Hred = schur_whiten_eig(*blocks, eps_h=eps_h)
        np.save(f"lamF_{kind}_{Nr}x{Na}_b{bmax}_e{eps_h:.0e}"
                + ("_ENR" if enrich_on else "") + ".npy", lam)
        report(lam, f"{tag0} eps_h={eps_h:.0e}", kh)

if __name__ == '__main__':
    main()
