# SPDX-License-Identifier: MIT
# Copyright (c) 2026 P. Salmond — hwy-spectral-survey (see LICENSE, NOTICE)
# Independent implementation of the HWY operator (arXiv:2509.25116, eq. for L_U);
# mirrors their public data formats/conventions; contains no code from their repo.
"""
PART 3 (rung 1, rev C) -- Petrov-Galerkin pencil, pressure kept, no IBP.

Rev A failed W2 (representation deficit + skewed implicit inner product).
Rev B failed W2a (weak grad-q annihilation needs boundary terms that do NOT
vanish on the clipped (beta,theta) domain: u2 ~ sin((2j-1)t) = +-1 at the
equator; fields nonzero at the truncated far-field edge).

Rev C: no integration by parts anywhere. Unknowns [v; q] (v div-free by
slaving, q truncated pressure basis). Momentum equation
    L_vel v + grad q = lam v
is tested against span{ div-free basis, gradient basis } in a quadrature
inner product => square pencil
    G [v;q] = lam H [v;q],
    G = [Mv | Aq]^T W [Av | Aq],   H = [Mv | Aq]^T W [Mv | 0].
q-directions give infinite eigenvalues (filtered). Assembly streams Av in
column blocks (3 GB container).

Gates: W2a diagnostic (discrete V5 on truncated stored eigenpair, info only;
operator amplifies the discarded tail so this overstates), W2 hard gate
(eigensolve must contain +0.11314), W3 resolution scan, W4 negative control.
"""
import numpy as np
import scipy.io as sio
import scipy.linalg as sla
from ns_part3_spectrum import (make_grid, profile_grids, Sector, Lu_of,
                               zero_comp, LAM_KNOWN, S)

DATA = '/home/claude/work/realNS_session9/data'

def measure(geo, kind):
    B, T = geo['B'], geo['T']
    if kind == 'flat':    return np.ones_like(B).ravel()
    if kind == 'angular': return np.sin(T).ravel()
    if kind == 'full':    return (geo['rho']**2*np.sin(T)*S/np.cos(B)**2).ravel()
    raise ValueError(kind)

def stackV(comp):
    return np.concatenate([comp[k]['f'].ravel() for k in ('u1','u2','u3')])

def assemble(sector, geo, Uval, Ubf, w, blk=192):
    Nr, Na = sector.Nr, sector.Na
    Ng = geo['B'].size
    nvel, nq = 2*Nr*Na, Nr*Na
    W3 = np.tile(w, 3)
    sqW = np.sqrt(W3)[:, None]

    # Mv (dense, weighted by sqrt W on both sides of all products)
    Mv = np.empty((3*Ng, nvel))
    dofs = ([('u1', i, j) for i in range(Nr) for j in range(Na)]
            + [('u3', i, j) for i in range(Nr) for j in range(Na)])
    for k, (c, i, j) in enumerate(dofs):
        comp = (sector.vel_dof_u1(i, j, geo) if c == 'u1'
                else sector.vel_dof_u3(i, j, geo))
        Mv[:, k] = stackV(comp)
    Mv *= sqW

    Aq = np.empty((3*Ng, nq))
    k = 0
    for i in range(Nr):
        for j in range(Na):
            qc = sector.q_dof(i, j, geo)
            Aq[:, k] = np.concatenate([qc['fr'].ravel(),
                                       (qc['ft']/geo['rho']).ravel(),
                                       np.zeros(Ng)])
            k += 1
    Aq *= sqW

    # stream Av in blocks, accumulate G upper-left/lower-left
    G = np.zeros((nvel+nq, nvel+nq))
    for s in range(0, nvel, blk):
        e = min(s+blk, nvel)
        Ab = np.empty((3*Ng, e-s))
        for k in range(s, e):
            c, i, j = dofs[k]
            comp = (sector.vel_dof_u1(i, j, geo) if c == 'u1'
                    else sector.vel_dof_u3(i, j, geo))
            L = Lu_of(comp, Uval, Ubf, geo)
            Ab[:, k-s] = np.concatenate([L[kk].ravel()
                                         for kk in ('u1','u2','u3')])
        Ab *= sqW
        G[:nvel, s:e] = Mv.T @ Ab
        G[nvel:, s:e] = Aq.T @ Ab
    G[:nvel, nvel:] = Mv.T @ Aq
    G[nvel:, nvel:] = Aq.T @ Aq
    H = np.zeros_like(G)
    H[:nvel, :nvel] = Mv.T @ Mv
    H[nvel:, :nvel] = Aq.T @ Mv
    return G, H, Mv, Aq, dofs

def stored_dofvecs(Nr, Na):
    d = sio.loadmat(f'{DATA}/up_eig.mat')
    xv = np.concatenate([d['u1'][:Nr, :Na].ravel(),
                         d['u3'][:Nr, :Na].ravel()])
    xq = d['p'][:Nr, :Na].ravel()
    return xv, xq

def survey(kind, Nr, Na, b, t, geo, Uval, Ubf, meas, gate=False, blk=192):
    sec = Sector(kind, Nr, Na, b, t)
    w = measure(geo, meas)
    G, H, Mv, Aq, _ = assemble(sec, geo, Uval, Ubf, w, blk)
    nvel = 2*Nr*Na
    if gate:
        xv, xq = stored_dofvecs(Nr, Na)
        x = np.concatenate([xv, xq])
        num = np.linalg.norm(G@x - LAM_KNOWN*(H@x))
        den = np.linalg.norm(H@x)
        print(f"  W2a diag [{meas}] ({Nr},{Na}): {num/den:.3e}")
    del Mv, Aq
    lam = sla.eig(G, H, right=False)
    lam = lam[np.isfinite(lam)]
    lam = lam[np.abs(lam) < 1e3]
    return lam

def report(lam, tag, gate=False):
    top = lam[np.argsort(-lam.real)][:8]
    s = ", ".join(f"{z.real:+.5f}{z.imag:+.4f}j" for z in top)
    print(f"  {tag} top Re: {s}")
    if gate:
        d = np.min(np.abs(lam - LAM_KNOWN)) if lam.size else np.inf
        print(f"   dist to +0.11314: {d:.2e} {'PASS' if d < 5e-3 else 'FAIL'}")

if __name__ == '__main__':
    b, t, geo = make_grid(120, 68)
    Uval, Ubf = profile_grids(b, t)
    print("-- ODD sector, rev C: measure scan at (40,20) --")
    for meas in ('flat', 'angular', 'full'):
        lam = survey('odd', 40, 20, b, t, geo, Uval, Ubf, meas, gate=True)
        report(lam, f"[{meas}] (40,20)", gate=True)
