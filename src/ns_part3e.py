# SPDX-License-Identifier: MIT
# Copyright (c) 2026 P. Salmond — hwy-spectral-survey (see LICENSE, NOTICE)
"""
PART 3 rev E -- discrete-Leray reduced pencil + whitened eigensolve.

Rev D failure: capped-domain true-L2 mass matrix is horribly conditioned
(full-domain basis restricted to a sub-interval is near-dependent), so the
pressure-augmented pencil leaked huge spurious finite modes from H's
near-null space.

Rev E: Schur-eliminate the pressure block exactly.  With
    GvA = Mv^T W Av, QA = Aq^T W Av, GvQ = Mv^T W Aq,
    Hvv = Mv^T W Mv, Gqq = Aq^T W Aq (SPD),
the rev-C pencil reduces EXACTLY (q = Gqq^{-1}(lam GvQ^T - QA) v) to
    (GvA - GvQ Gqq^{-1} QA) v = lam (Hvv - GvQ Gqq^{-1} GvQ^T) v
i.e. Gred = Mv^T P (L .), Hred = Mv^T P Mv with P the W-orthogonal projector
off the discrete gradient span: the discrete Leray projection.  Hred is
symmetric PSD; whiten with a relative spectral cut eps_h (scanned -- true
modes must be eps-stable, spurious ones dance).

Enrichment diagnostic W2b: append ONE extra trial/test column = the FULL
stored eigenfield (Phys-synthesized, the same machinery that passed the
Part-2 gate at residual 1.7e-5).  If W2b finds +0.11314 but plain W2 does
not, machinery is right and the deficit is pure basis resolution.
NOTE: W2b is a machinery check ONLY (the answer is planted); it can never
count as the survey gate.
"""
import numpy as np
import scipy.linalg as sla
import scipy.io as sio
from ns_part3_spectrum import (make_grid, profile_grids, Sector, Lu_of,
                               LAM_KNOWN, DATA)
from ns_part3c import measure, stored_dofvecs
from ns_part12_gate import load_profile, Phys

def stackV(comp):
    return np.concatenate([comp[k]['f'].ravel() for k in ('u1','u2','u3')])

def stackL(comp, Uval, Ubf, geo):
    L = Lu_of(comp, Uval, Ubf, geo)
    return np.concatenate([L[k].ravel() for k in ('u1','u2','u3')])

def stored_full_comp(b, t, geo):
    F, _, lam = load_profile(f'{DATA}/up_eig.mat', bdry_index=1)
    P = Phys(F, b, t, 'cos')
    comp = {k: dict(f=P.f[k], fr=P.fr[k], frr=P.frr[k],
                    ft=P.ft[k], ftt=P.ftt[k]) for k in ('u1','u2','u3')}
    Ng = geo['B'].size
    gq = np.concatenate([P.fr['p'].ravel(),
                         (P.ft['p']/geo['rho']).ravel(),
                         np.zeros(Ng)])
    return comp, gq

def assemble_leray(sec, geo, Uval, Ubf, w, enrich=None, enrich_gq=None,
                   sec_q=None, blk=160):
    Ng = geo['B'].size
    Nr, Na = sec.Nr, sec.Na
    nvel, nq = 2*Nr*Na, Nr*Na
    sqW = np.sqrt(np.tile(w, 3))
    dofs = ([('u1', i, j) for i in range(Nr) for j in range(Na)]
            + [('u3', i, j) for i in range(Nr) for j in range(Na)])

    Mv = np.empty((3*Ng, nvel))
    for k, (c, i, j) in enumerate(dofs):
        comp = (sec.vel_dof_u1(i, j, geo) if c == 'u1'
                else sec.vel_dof_u3(i, j, geo))
        Mv[:, k] = sqW*stackV(comp)
    sq = sec_q if sec_q is not None else sec
    nq = sq.Nr*sq.Na
    Aq = np.empty((3*Ng, nq))
    k = 0
    for i in range(sq.Nr):
        for j in range(sq.Na):
            qc = sq.q_dof(i, j, geo)
            Aq[:, k] = sqW*np.concatenate([qc['fr'].ravel(),
                                           (qc['ft']/geo['rho']).ravel(),
                                           np.zeros(Ng)])
            k += 1
    if enrich_gq is not None:
        Aq = np.column_stack([Aq, sqW*enrich_gq])
        nq += 1

    if enrich is not None:
        m_e = sqW*stackV(enrich)
        a_e = sqW*stackL(enrich, Uval, Ubf, geo)
        n = nvel + 1
    else:
        m_e = a_e = None
        n = nvel

    GvA = np.zeros((n, n))          # Mv_ext^T Av_ext
    QA = np.zeros((nq, n))          # Aq^T Av_ext
    for s in range(0, nvel, blk):
        e = min(s+blk, nvel)
        Ab = np.empty((3*Ng, e-s))
        for k in range(s, e):
            c, i, j = dofs[k]
            comp = (sec.vel_dof_u1(i, j, geo) if c == 'u1'
                    else sec.vel_dof_u3(i, j, geo))
            Ab[:, k-s] = sqW*stackL(comp, Uval, Ubf, geo)
        GvA[:nvel, s:e] = Mv.T @ Ab
        QA[:, s:e] = Aq.T @ Ab
        if enrich is not None:
            GvA[nvel, s:e] = m_e @ Ab
    if enrich is not None:
        GvA[:nvel, nvel] = Mv.T @ a_e
        GvA[nvel, nvel] = m_e @ a_e
        QA[:, nvel] = Aq.T @ a_e
        Mx = np.column_stack([Mv, m_e])
    else:
        Mx = Mv
    Hvv = Mx.T @ Mx
    GvQ = Mx.T @ Aq
    Gqq = Aq.T @ Aq
    return GvA, QA, GvQ, Hvv, Gqq

def schur_whiten_eig(GvA, QA, GvQ, Hvv, Gqq, eps_q=1e-11, eps_h=1e-8):
    sq, Uq = np.linalg.eigh(Gqq)
    kq = sq > eps_q*sq.max()
    Z = Uq[:, kq]/np.sqrt(sq[kq])              # Gqq^{-1} ~= Z Z^T
    GZ = GvQ @ Z
    Gred = GvA - GZ @ (Z.T @ QA)
    Hred = Hvv - GZ @ GZ.T
    Hred = 0.5*(Hred + Hred.T)
    sh, Uh = np.linalg.eigh(Hred)
    kh = sh > eps_h*max(sh.max(), 1e-300)
    Y = Uh[:, kh]/np.sqrt(sh[kh])
    A = Y.T @ Gred @ Y
    lam = sla.eig(A, right=False)
    return lam, int(kq.sum()), int(kh.sum()), Gred, Hred

def report(lam, tag, kh):
    lam = lam[np.abs(lam) < 1e3]
    top = lam[np.argsort(-lam.real)][:6]
    s = ", ".join(f"{z.real:+.5f}{z.imag:+.4f}j" for z in top)
    d = np.min(np.abs(lam - LAM_KNOWN)) if lam.size else np.inf
    print(f"  {tag} [kept {kh}] top Re: {s}")
    print(f"   dist to +0.11314: {d:.2e} {'PASS' if d < 5e-3 else 'FAIL'}",
          flush=True)
    return d

def run_case(Nr, Na, bmax, enrich_on, qmul=1, eps_list=(1e-6, 1e-8, 1e-10)):
    b, t, geo = make_grid(110, 68, bmax=bmax)
    Uval, Ubf = profile_grids(b, t)
    w = measure(geo, 'full')
    sec = Sector('odd', Nr, Na, b, t)
    sec_q = Sector('odd', qmul*Nr, qmul*Na, b, t) if qmul > 1 else None
    if enrich_on:
        enr, gq = stored_full_comp(b, t, geo)
    else:
        enr = gq = None
    blocks = assemble_leray(sec, geo, Uval, Ubf, w, enrich=enr, enrich_gq=gq,
                            sec_q=sec_q)
    tagb = ('W2b-enriched' if enrich_on else 'W2-plain') + f' q*{qmul}'
    for eps_h in eps_list:
        lam, kq, kh, Gred, Hred = schur_whiten_eig(*blocks, eps_h=eps_h)
        report(lam, f"({Nr},{Na}) bmax={bmax} {tagb} eps_h={eps_h:.0e}", kh)
    # reduced-form residual diagnostic on truncated stored dofs
    if not enrich_on:
        xv, _ = stored_dofvecs(Nr, Na)
        num = np.linalg.norm(Gred@xv - LAM_KNOWN*(Hred@xv))
        den = np.linalg.norm(Hred@xv)
        print(f"  W2a-red ({Nr},{Na}) bmax={bmax}: {num/den:.3e}", flush=True)

if __name__ == '__main__':
    import sys
    Nr, Na = int(sys.argv[1]), int(sys.argv[2])
    bmax = float(sys.argv[3])
    enrich_on = bool(int(sys.argv[4]))
    qmul = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    run_case(Nr, Na, bmax, enrich_on, qmul=qmul)
