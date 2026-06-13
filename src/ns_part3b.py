# SPDX-License-Identifier: MIT
# Copyright (c) 2026 P. Salmond — hwy-spectral-survey (see LICENSE, NOTICE)
# Independent implementation of the HWY operator (arXiv:2509.25116, eq. for L_U);
# mirrors their public data formats/conventions; contains no code from their repo.
"""
PART 3 (rung 1, rev B) -- Galerkin spectrum survey of L_U.

Changes vs rev A (which FAILED the W2 entry gate -- see ns_part3_run1.log):
  * No explicit pressure dofs. Test against the div-free trial basis in an
    (approximate) L2(R^3) inner product: <grad q, v_test> = 0 weakly, so the
    Leray projection is applied by the Galerkin pairing itself.
  * Measure variants scanned (the implicit inner product was the suspected
    spurious-mode source): 'flat' grid l2 | 'angular' rho^2-free softened |
    'full' true rho^2 sin(t) drho/dbeta jacobian.
  * New gate W2a (discrete V5): project THEIR stored eigenfunction coeffs
    into the truncated dof space, demand small ||G x - lam H x||/||H x|| at
    lam=+0.11314. Separates assembly bugs from representation deficit.
  * Sizes raised to where the stored eigenfunction is actually representable
    ((40,20) captures 98.9%/95.3% of u1/u3 energy; (52,26) ~99.9%).
"""
import numpy as np
import scipy.io as sio
import scipy.linalg as sla
from ns_part3_spectrum import (make_grid, profile_grids, Sector, Lu_of,
                               zero_comp, S, LAM_KNOWN)

import os
DATA = os.environ.get('HWY_REPO', '../data/3d-navier-stokes-nonuniqueness') + '/data'

def measure(geo, kind):
    B, T = geo['B'], geo['T']
    if kind == 'flat':
        return np.ones_like(B).ravel()
    if kind == 'angular':                       # softened: no rho^2 growth
        return (np.sin(T)).ravel()
    if kind == 'full':                          # true L2(R^3), truncated
        rho = geo['rho']
        return (rho**2*np.sin(T)*S/np.cos(B)**2).ravel()
    raise ValueError(kind)

def assemble_GH(sector, geo, Uval, Ubf, w):
    Nr, Na = sector.Nr, sector.Na
    Ng = geo['B'].size
    nvel = 2*Nr*Na
    Av = np.empty((3*Ng, nvel))
    Mv = np.empty((3*Ng, nvel))
    def stackL(comp):
        L = Lu_of(comp, Uval, Ubf, geo)
        return np.concatenate([L[k].ravel() for k in ('u1','u2','u3')])
    def stackV(comp):
        return np.concatenate([comp[k]['f'].ravel() for k in ('u1','u2','u3')])
    k = 0
    for i in range(Nr):
        for j in range(Na):
            c = sector.vel_dof_u1(i, j, geo)
            Av[:, k] = stackL(c); Mv[:, k] = stackV(c); k += 1
    for i in range(Nr):
        for j in range(Na):
            c = sector.vel_dof_u3(i, j, geo)
            Av[:, k] = stackL(c); Mv[:, k] = stackV(c); k += 1
    W = np.tile(w, 3)[:, None]
    G = Mv.T @ (W*Av)
    H = Mv.T @ (W*Mv)
    return G, H

def stored_eig_dofvec(Nr, Na):
    d = sio.loadmat(f'{DATA}/up_eig.mat')
    c1 = d['u1'][:Nr, :Na]
    c3 = d['u3'][:Nr, :Na]
    return np.concatenate([c1.ravel(), c3.ravel()])

def run_sector(kind, sizes, meas, geo, Uval, Ubf, gate_eig=False, label=''):
    w = measure(geo, meas)
    out = {}
    for (Nr, Na) in sizes:
        sec = Sector(kind, Nr, Na, *gridxy)
        G, H = assemble_GH(sec, geo, Uval, Ubf, w)
        if gate_eig:
            x = stored_eig_dofvec(Nr, Na)
            r = np.linalg.norm(G@x - LAM_KNOWN*(H@x))/np.linalg.norm(H@x)
            print(f"  W2a [{meas}] ({Nr},{Na}): "
                  f"||Gx-l Hx||/||Hx|| = {r:.3e}")
        lam = sla.eig(G, H, right=False)
        lam = lam[np.isfinite(lam)]
        out[(Nr, Na)] = lam
        top = lam[np.argsort(-lam.real)][:8]
        s = ", ".join(f"{z.real:+.5f}{z.imag:+.4f}j" for z in top)
        print(f"  {label}[{meas}] ({Nr},{Na}) top Re: {s}")
        if gate_eig:
            dmin = np.min(np.abs(lam - LAM_KNOWN))
            print(f"   dist to +0.11314: {dmin:.2e} "
                  f"{'PASS' if dmin < 5e-3 else 'FAIL'}")
    return out

if __name__ == '__main__':
    gridxy = make_grid(110, 64)[:2]
    b, t, geo = make_grid(110, 64)
    gridxy = (b, t)
    Uval, Ubf = profile_grids(b, t)
    print("-- ODD sector: measure scan at (40,20), W2a + entry gate --")
    for meas in ('flat', 'angular', 'full'):
        run_sector('odd', [(40, 20)], meas, geo, Uval, Ubf, gate_eig=True)
