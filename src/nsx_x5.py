# SPDX-License-Identifier: MIT
# Copyright (c) 2026 P. Salmond — hwy-spectral-survey (see LICENSE, NOTICE)
"""nsx_x5.py -- the survey basis, designed from the measured v1 spectrum:

  radial (origin-masked, per-block CHOLESKY-orthonormalized => nested spans):
      H60(rho0=20, sig=2.5) (+) H24(sig=1.2) (+) Lag12(ell=3)   = 96
  poloidal: k even 0..120 (61);  full-96 radial for k<=38,
            slim-46 (H36+S6+L4 prefixes) for k>=40        [taper]
  toroidal: k odd 1..103 (52);  rad-30 (H24+S2+L4 prefixes)
  => N = 20*96 + 41*46 + 52*30 = 5366;  predicted v1 fit ~ 2.5e-4.

Assembly: heterogeneous blocks via the T5-gated aa_block kernel plus small
g_block/h_block variants (test side = f-slots); T6 mini-gate vs direct
Lcols on a tiny heterogeneous config before anything full-size.
Stages: gate | full   (full: assemble+mask+save grams, X0r + X1 gates).
"""
import sys, time
import numpy as np
import scipy.linalg as sla
import nsx_op
from nsx_op import extract_coeffs, hermite_full
from nsx_basis import angular_factors, radial_factors
from ns_part3k import load_family, profile_F, Lcols
from ns_part3f import make_geo, gl_nodes
from ns_part12_gate import Phys
from ns_part3_spectrum import S
from nsx_x2_aa import aa_block, b_block, slot_tables, pressure_span

LAM = 0.11314203274385946
COMPS = ('u1', 'u2', 'u3')
t0 = time.time()
def log(s): print(f"[{time.time()-t0:7.1f}s] {s}", flush=True)

# ----------------------------------------------------- radial construction
def origin_mask4(rho, a=2.0):
    den = rho*rho + a*a
    return (rho*rho/den, 2*a*a*rho/den**2, 2*a*a*(a*a - 3*rho*rho)/den**3,
            24*a*a*rho*(rho*rho - a*a)/den**4)

def apply_mask(stk, rho):
    R, R1, R2, R3 = stk
    m0, m1, m2, m3 = origin_mask4(rho)
    return (R*m0, R1*m0 + R*m1, R2*m0 + 2*R1*m1 + R*m2,
            R3*m0 + 3*R2*m1 + 3*R1*m2 + R*m3)

def lag_stack(nm, ell, rho):
    from scipy.special import eval_genlaguerre
    rl = radial_factors(range(nm), ell, rho)
    x = rho/ell; E = np.exp(-0.5*x)
    R3 = []
    for j in range(nm):
        n = 1.0/(ell**1.5*np.sqrt((j+1.0)*(j+2.0)))
        L0 = eval_genlaguerre(j, 2, x)
        L1 = -eval_genlaguerre(j-1, 3, x) if j >= 1 else 0*x
        L2 = eval_genlaguerre(j-2, 4, x) if j >= 2 else 0*x
        L3 = -eval_genlaguerre(j-3, 5, x) if j >= 3 else 0*x
        R3.append(n*(L3 - 1.5*L2 + 0.75*L1 - 0.125*L0)*E/ell**3)
    return (rl['R'], rl['R1'], rl['R2'], np.array(R3))

def chol_block(stk, wr):
    R = stk[0]
    G = (R*wr) @ R.T
    L = sla.cholesky(G + 1e-14*np.trace(G)/len(G)*np.eye(len(G)),
                     lower=True)
    return [sla.solve_triangular(L, X, lower=True) for X in stk]

def build_radial(rho, wr, njA, njS, njL):
    A = chol_block(apply_mask(hermite_full(range(njA), 20.0, 2.5, rho), rho), wr)
    Sb = chol_block(apply_mask(hermite_full(range(njS), 20.0, 1.2, rho), rho), wr)
    Lb = chol_block(apply_mask(lag_stack(njL, 3.0, rho), rho), wr)
    rad = [np.vstack([A[i], Sb[i], Lb[i]]) for i in range(4)]
    return rad, njA, njS, njL

def sub(rad, rows):
    return [X[rows] for X in rad]

# ----------------------------------------------------- pencil block kernels
def g_block(C, wr, wa, testF, slotsB):
    """<test_f, sum_slots C slotB> : test side is plain f-factors."""
    nr = testF[0][1].shape[0]; nkA = testF[0][2].shape[0]
    nrB = slotsB[0][2].shape[0]; nkB = slotsB[0][3].shape[0]
    out = np.zeros((nr*nrB, nkA*nkB))
    Nb = wr.size
    for c, rA, aA in testF:
        for scB, dB, rB, aB in slotsB:
            CB = C[c][(scB, dB)]
            if not np.any(CB):
                continue
            CC = wa[None, :]*CB
            W = np.einsum('it,jt,bt->bij', aA, aB, CC,
                          optimize=True).reshape(Nb, -1)
            Z = ((rA*wr)[:, None, :]*rB[None, :, :]).reshape(-1, Nb)
            out += Z @ W
    out = out.reshape(nr, nrB, nkA, nkB).transpose(0, 2, 1, 3)
    return out.reshape(nr*nkA, nrB*nkB)

def h_block(wr, wa, testF, colF):
    nr = testF[0][1].shape[0]; nkA = testF[0][2].shape[0]
    nrB = colF[0][1].shape[0]; nkB = colF[0][2].shape[0]
    out = np.zeros((nr*nkA, nrB*nkB))
    dA = {c: (rA, aA) for c, rA, aA in testF}
    for c, rB, aB in colF:
        if c not in dA:
            continue
        rA, aA = dA[c]
        out += np.kron((rA*wr) @ rB.T, (aA*wa) @ aB.T)
    return out

# =====================================================================
def run(mode):
    if mode == 'gate':
        Nb, Nt = 90, 70
        njA, njS, njL = 4, 2, 2
        ksP_lo = [0, 2]; ksP_hi = [4, 6]; ksT_ = [1, 3]
        slim = list(range(3)) + [4] + [6]
        torr = list(range(2)) + [4] + [6]
    else:
        Nb, Nt = 480, 260
        njA, njS, njL = 60, 24, 12
        ksP_lo = [2*i for i in range(20)]            # k <= 38
        ksP_hi = [2*i for i in range(20, 61)]        # 40..120
        ksT_ = [2*i+1 for i in range(52)]            # 1..103
        slim = list(range(36)) + list(range(60, 66)) + list(range(84, 88))
        torr = list(range(24)) + list(range(60, 62)) + list(range(84, 88))

    b, wb = gl_nodes(Nb, 0.0, np.pi/2)
    t, wt = gl_nodes(Nt, 0.0, np.pi/2)
    geo = make_geo(b, t)
    rho = S*np.tan(b)
    nsx_op.RHO = rho
    wr = rho**2*S/np.cos(b)**2*wb
    wa = np.sin(t)*wt
    rad, _, _, _ = build_radial(rho, wr, njA, njS, njL)
    radS, radT = sub(rad, slim), sub(rad, torr)
    nf, ns_, ntr = rad[0].shape[0], len(slim), len(torr)

    APl, BPl = angular_factors(ksP_lo, t)
    APh, BPh = angular_factors(ksP_hi, t)
    _, BT = angular_factors(ksT_, t)
    def pol_slots(rd, ang):
        a, at, att, bb, bt, btt = ang
        return slot_tables(rd, ang, (BT['f'], BT['t'], BT['tt']), rho)[0]
    angL = (APl['f'], APl['t'], APl['tt'], BPl['f'], BPl['t'], BPl['tt'])
    angH = (APh['f'], APh['t'], APh['tt'], BPh['f'], BPh['t'], BPh['tt'])
    angT3 = (BT['f'], BT['t'], BT['tt'])
    PL = slot_tables(rad, angL, angT3, rho)[0]
    PH = slot_tables(radS, angH, angT3, rho)[0]
    TO = slot_tables(radT, angL, angT3, rho)[1]
    DL = 2*rad[0] + rho[None, :]*rad[1]
    DH = 2*radS[0] + rho[None, :]*radS[1]
    fL = [('u1', rad[0], APl['f']), ('u2', -DL, BPl['f'])]
    fH = [('u1', radS[0], APh['f']), ('u2', -DH, BPh['f'])]
    fT = [('u3', radT[0], BT['f'])]
    blocks = [(fL, PL), (fH, PH), (fT, TO)]
    sizes = [nf*len(ksP_lo), ns_*len(ksP_hi), ntr*len(ksT_)]
    N = sum(sizes)
    off = np.concatenate([[0], np.cumsum(sizes)]).astype(int)
    log(f"basis: nf={nf} slim={ns_} tor={ntr}; "
        f"block sizes {sizes}, N={N}")

    C = extract_coeffs(b, t, geo, withU=True)
    C0 = extract_coeffs(b, t, geo, withU=False)
    def assemble3(kind, Cg=None):
        M = np.zeros((N, N))
        for i, (fi, si) in enumerate(blocks):
            for j, (fj, sj) in enumerate(blocks):
                if kind == 'H':
                    M[off[i]:off[i+1], off[j]:off[j+1]] = h_block(wr, wa, fi, fj)
                elif kind == 'G':
                    M[off[i]:off[i+1], off[j]:off[j+1]] = g_block(Cg, wr, wa, fi, sj)
                else:
                    M[off[i]:off[i+1], off[j]:off[j+1]] = aa_block(Cg, wr, wa, si, sj)
        return M
    G = assemble3('G', C)
    G0 = assemble3('G', C0)
    H = assemble3('H')
    log("G, G0, H assembled (heterogeneous blocks)")

    if mode == 'gate':
        # T6: direct pencil via Lcols column-by-column
        UF = profile_F()
        PU = Phys(UF, b, t, 'cos')
        Uval = {k: PU.f[k] for k in COMPS}
        Ubf = {k: dict(f=PU.f[k], fr=PU.fr[k], frr=PU.frr[k],
                       ft=PU.ft[k], ftt=PU.ftt[k]) for k in COMPS}
        WGT = (wr[:, None]*wa[None, :]).ravel()
        class F: pass
        DERIVS = ('f', 'fr', 'frr', 'ft', 'ftt')
        cols, Lw, Wf = [], [], []
        for fi, si in blocks:
            nrB = si[0][2].shape[0]; nkB = si[0][3].shape[0]
            for p in range(nrB):
                for i in range(nkB):
                    f = F()
                    f.f, f.fr, f.frr, f.ft, f.ftt = {}, {}, {}, {}, {}
                    for k in COMPS:
                        for dd in DERIVS:
                            getattr(f, dd)[k] = np.zeros((Nb, Nt))
                    for sc, dd, rfac, afac in si:
                        getattr(f, dd)[sc] = (getattr(f, dd)[sc]
                                              + np.outer(rfac[p], afac[i]))
                    f.fr['p'] = np.zeros((Nb, Nt)); f.ft['p'] = np.zeros((Nb, Nt))
                    out = Lcols(f, Uval, Ubf, geo, True)
                    Lw.append(out.reshape(3, -1))
                    Wf.append(np.stack([f.f['u1'], f.f['u2'],
                                        f.f['u3']]).reshape(3, -1))
        Lw = np.array(Lw); Wf = np.array(Wf)
        Gd = np.einsum('icg,jcg,g->ij', Wf, Lw, WGT, optimize=True)
        Hd = np.einsum('icg,jcg,g->ij', Wf, Wf, WGT, optimize=True)
        AA = assemble3('A', C)
        AAd = np.einsum('icg,jcg,g->ij', Lw, Lw, WGT, optimize=True)
        eG = np.linalg.norm(G - Gd)/np.linalg.norm(Gd)
        eH = np.linalg.norm(H - Hd)/np.linalg.norm(Hd)
        eA = np.linalg.norm(AA - AAd)/np.linalg.norm(AAd)
        ok = max(eG, eH, eA) < 1e-11
        log(f"T6  G: {eG:.3e}  H: {eH:.3e}  AA: {eA:.3e}  "
            f"-> {'PASS' if ok else 'FAIL'}")
        return

    # ---------------- full mode: X0r + X1 gates, AA/B, save
    fams, _ = load_family('odd')
    Pv1 = Phys(fams[-1], b, t, 'cos')
    rhs = np.empty(N)
    for i, (fi, si) in enumerate(blocks):
        parts = []
        for c, rA, aA in fi:
            parts.append((rA*wr) @ Pv1.f[c] @ (wa*aA).T)
        rhs[off[i]:off[i+1]] = sum(parts).ravel()
    Etot = ((wr[:, None]*wa[None, :])
            * (Pv1.f['u1']**2 + Pv1.f['u2']**2 + Pv1.f['u3']**2)).sum()
    evH, QH = sla.eigh(H, check_finite=False)
    cfit = np.linalg.solve(H + 1e-13*evH[-1]*np.eye(N), rhs)
    log(f"X0r (tapered span): v1 fit error = "
        f"{np.sqrt(max(Etot - cfit @ rhs, 0)/Etot):.3e}")

    for cut in (1e-10, 1e-12):
        keep = evH > cut*evH[-1]
        T = QH[:, keep]/np.sqrt(evH[keep])
        ev = np.linalg.eigvals(T.T @ G @ T)
        ev0r = np.linalg.eigvals(T.T @ G0 @ T).real.max()
        i1 = np.argmin(np.abs(ev - LAM))
        top = np.sort(ev.real)[-6:]
        log(f"cut={cut:.0e}: kept {keep.sum()}/{N}  lam1={ev[i1]:.8f} "
            f"(|d|={abs(ev[i1]-LAM):.2e})  X3 maxRe(U=0)={ev0r:+.5f}")
        log("   top Re Ritz: " + " ".join(f"{z:+.5f}" for z in top))

    AA = assemble3('A', C)
    asym = np.linalg.norm(AA - AA.T)/np.linalg.norm(AA)
    AA = 0.5*(AA + AA.T)
    log(f"AA assembled (asym {asym:.2e}); max diag {AA.diagonal().max():.3e}")
    SPEC = [('H', 60, 20.0, 2.5), ('H', 30, 20.0, 1.2),
            ('L', 12, 3.0), ('L', 8, 8.0)]
    Sf, S1, Pq, Pq1 = pressure_span(rho, t, SPEC, 72)
    Sor = Sf/rho[None, :]
    B = np.vstack([b_block(C, wr, wa, si, S1, Sor, Pq, Pq1)
                   for fi, si in blocks])
    log("B assembled")
    np.savez('nsx_x5_grams.npz', G=G, G0=G0, H=H, AA=AA, B=B, N=N,
             sizes=np.array(sizes))
    log("saved nsx_x5_grams.npz")

if __name__ == '__main__':
    run(sys.argv[1] if len(sys.argv) > 1 else 'gate')
