# SPDX-License-Identifier: MIT
# Copyright (c) 2026 P. Salmond — hwy-spectral-survey (see LICENSE, NOTICE)
"""nsx_x2_aa.py -- separable assembly of the landscape grams on the
decay-adapted basis (open quarter domain):
    AA[i,j] = <L w_i, L w_j>          (operator-side Gram)
    B [i,q] = <L w_i, grad q_q>       (gradient coupling; <w,grad q>=0
                                       on the open domain to 4e-13, so the
                                       Schur-complemented pencil
                                       N(lam) = AAd - lam G^T - conj(lam) G
                                                + |lam|^2 H
                                       is exactly quadratic in lam)
plus the 1-D pieces of M = <grad q, grad q'> (kron-assembled later).

Pressure span (calibrated by nsx_r1b control-B floor 7.6e-05):
    S in H60(rho0=20,sig=2.5) + H30(sig=1.2) + Lag(ell=3)x12 + Lag(ell=8)x8
    angular P_{k+1}, k even, kq=72   => Nq = 7920.

Modes:  python nsx_x2_aa.py gate   -> T5 mini-gate vs direct Lcols
        python nsx_x2_aa.py full   -> full assembly, save nsx_x2_grams.npz
"""
import sys, time
import numpy as np
import nsx_op
from nsx_op import extract_coeffs, composite_radial, hermite_full
from nsx_basis import angular_factors, _leg_derivs
from ns_part3k import profile_F, Lcols
from ns_part3f import make_geo, gl_nodes
from ns_part12_gate import Phys
from ns_part3_spectrum import S
from scipy.special import eval_genlaguerre

COMPS = ('u1', 'u2', 'u3')
DERIVS = ('f', 'fr', 'frr', 'ft', 'ftt')

# ----------------------------------------------------------- slot tables
def slot_tables(rad, angP, angT_, rho):
    R, R1, R2, R3 = rad
    D = 2*R + rho[None, :]*R1
    D1 = 3*R1 + rho[None, :]*R2
    D2 = 4*R2 + rho[None, :]*R3
    a, at, att, bb, bt, btt = angP
    b3, b3t, b3tt = angT_
    POL = [('u1', 'f', R, a), ('u1', 'fr', R1, a), ('u1', 'frr', R2, a),
           ('u1', 'ft', R, at), ('u1', 'ftt', R, att),
           ('u2', 'f', -D, bb), ('u2', 'fr', -D1, bb), ('u2', 'frr', -D2, bb),
           ('u2', 'ft', -D, bt), ('u2', 'ftt', -D, btt)]
    TOR = [('u3', 'f', R, b3), ('u3', 'fr', R1, b3), ('u3', 'frr', R2, b3),
           ('u3', 'ft', R, b3t), ('u3', 'ftt', R, b3tt)]
    return POL, TOR

# ------------------------------------------------- generic AA block
def aa_block(C, wr, wa, slotsA, slotsB):
    """sum_c sum_{sA,sB} <C_c[sA] slotA_(p,i), C_c[sB] slotB_(q,j)>."""
    nr = slotsA[0][2].shape[0]; nkA = slotsA[0][3].shape[0]
    nrB = slotsB[0][2].shape[0]; nkB = slotsB[0][3].shape[0]
    out = np.zeros((nr*nrB, nkA*nkB))
    Nb = wr.size
    for c in COMPS:
        for scA, dA, rA, aA in slotsA:
            CA = C[c][(scA, dA)]
            if not np.any(CA):
                continue
            for scB, dB, rB, aB in slotsB:
                CB = C[c][(scB, dB)]
                if not np.any(CB):
                    continue
                CC = wa[None, :]*(CA*CB)
                W = np.einsum('it,jt,bt->bij', aA, aB, CC,
                              optimize=True).reshape(Nb, -1)
                Z = ((rA*wr)[:, None, :]*rB[None, :, :]).reshape(-1, Nb)
                out += Z @ W
    out = out.reshape(nr, nrB, nkA, nkB).transpose(0, 2, 1, 3)
    return out.reshape(nr*nkA, nrB*nkB)

# ------------------------------------------------- B block (grad columns)
def b_block(C, wr, wa, slots, Sg1, Sgor, Pq, Pq1):
    """<(L w)_c, (grad q)_c>, grad q = (S' P, (S/rho)(-st P'), 0)."""
    nr = slots[0][2].shape[0]; nkA = slots[0][3].shape[0]
    ns = Sg1.shape[0]; nkq = Pq.shape[0]
    out = np.zeros((nr*ns, nkA*nkq))
    Nb = wr.size
    grad = {'u1': (Sg1, Pq), 'u2': (Sgor, Pq1)}
    for c in ('u1', 'u2'):
        gR, gA = grad[c]
        for scA, dA, rA, aA in slots:
            CA = C[c][(scA, dA)]
            if not np.any(CA):
                continue
            CC = wa[None, :]*CA
            W = np.einsum('it,jt,bt->bij', aA, gA, CC,
                          optimize=True).reshape(Nb, -1)
            Z = ((rA*wr)[:, None, :]*gR[None, :, :]).reshape(-1, Nb)
            out += Z @ W
    out = out.reshape(nr, ns, nkA, nkq).transpose(0, 2, 1, 3)
    return out.reshape(nr*nkA, ns*nkq)

# ------------------------------------------------- pressure-span factors
def lag0(nm, ell, rho):
    x = rho/ell; E = np.exp(-0.5*x)
    Sf, S1 = [], []
    for m in range(nm):
        L0 = eval_genlaguerre(m, 0, x)
        L1 = -eval_genlaguerre(m-1, 1, x) if m >= 1 else 0*x
        Sf.append(L0*E); S1.append((L1 - 0.5*L0)*E/ell)
    return np.array(Sf), np.array(S1)

def pressure_span(rho, t, spec, nkq):
    blocks = []
    for kind, *p in spec:
        if kind == 'H':
            nm, r0, sg = p
            A = hermite_full(range(nm), r0, sg, rho)
            blocks.append((A[0], A[1]))
        else:
            nm, ell = p
            blocks.append(lag0(nm, ell, rho))
    Sf = np.vstack([x[0] for x in blocks]); S1 = np.vstack([x[1] for x in blocks])
    mu, st = np.cos(t), np.sin(t)
    Pq, Pq1 = [], []
    for k in [2*i for i in range(nkq)]:
        P, P1 = _leg_derivs(k+1, mu, 1)[:2]
        Pq.append(P); Pq1.append(-st*P1)
    return Sf, S1, np.array(Pq), np.array(Pq1)

# ===================================================================
def run(mode):
    t0 = time.time()
    if mode == 'gate':
        Nb, Nt = 90, 70
        NJH, RHO0, SIG, NJL, ELLL = 5, 20.0, 2.5, 3, 3.0
        NKP, NKT = 3, 2
        spec = [('H', 3, 20.0, 2.5), ('L', 2, 3.0)]
        NKQ = 4
    else:
        Nb, Nt = 480, 220
        NJH, RHO0, SIG, NJL, ELLL = 50, 20.0, 2.5, 12, 3.0
        NKP, NKT = 40, 24
        spec = [('H', 60, 20.0, 2.5), ('H', 30, 20.0, 1.2),
                ('L', 12, 3.0), ('L', 8, 8.0)]
        NKQ = 72

    b, wb = gl_nodes(Nb, 0.0, np.pi/2)
    t, wt = gl_nodes(Nt, 0.0, np.pi/2)
    geo = make_geo(b, t)
    rho = S*np.tan(b)
    nsx_op.RHO = rho
    wr = rho**2*S/np.cos(b)**2*wb
    wa = np.sin(t)*wt

    rad, nr = composite_radial(rho, wr, NJH, RHO0, SIG, NJL, ELLL)
    ksP = [2*i for i in range(NKP)]; ksT = [2*i+1 for i in range(NKT)]
    AP, BP = angular_factors(ksP, t)
    _,  BT = angular_factors(ksT, t)
    angP = (AP['f'], AP['t'], AP['tt'], BP['f'], BP['t'], BP['tt'])
    angT_ = (BT['f'], BT['t'], BT['tt'])
    POL, TOR = slot_tables(rad, angP, angT_, rho)
    NP, NT = nr*NKP, nr*NKT; N = NP + NT
    Sf, S1, Pq, Pq1 = pressure_span(rho, t, spec, NKQ)
    Sor = Sf/rho[None, :]
    Nq = Sf.shape[0]*NKQ
    print(f"[{time.time()-t0:7.1f}s] setup: N={N} (nr={nr}), Nq={Nq}, "
          f"grid {Nb}x{Nt}", flush=True)

    C = extract_coeffs(b, t, geo, withU=True)
    print(f"[{time.time()-t0:7.1f}s] coefficient grids extracted", flush=True)

    AA = np.zeros((N, N))
    AA[:NP, :NP] = aa_block(C, wr, wa, POL, POL)
    print(f"[{time.time()-t0:7.1f}s] AA POL-POL", flush=True)
    AA[:NP, NP:] = aa_block(C, wr, wa, POL, TOR)
    AA[NP:, :NP] = AA[:NP, NP:].T
    AA[NP:, NP:] = aa_block(C, wr, wa, TOR, TOR)
    asym = np.linalg.norm(AA - AA.T)/np.linalg.norm(AA)
    AA = 0.5*(AA + AA.T)
    print(f"[{time.time()-t0:7.1f}s] AA done; internal symmetry gate "
          f"asym = {asym:.3e}", flush=True)

    B = np.vstack([b_block(C, wr, wa, POL, S1, Sor, Pq, Pq1),
                   b_block(C, wr, wa, TOR, S1, Sor, Pq, Pq1)])
    print(f"[{time.time()-t0:7.1f}s] B done", flush=True)

    A1 = (S1*wr) @ S1.T; A0 = (Sor*wr) @ Sor.T
    P0 = (Pq*wa) @ Pq.T; P1g = (Pq1*wa) @ Pq1.T

    if mode == 'gate':
        # ---- direct reference via batched Lcols, column by column
        UF = profile_F()
        PU = Phys(UF, b, t, 'cos')
        Uval = {k: PU.f[k] for k in COMPS}
        Ubf = {k: dict(f=PU.f[k], fr=PU.fr[k], frr=PU.frr[k],
                       ft=PU.ft[k], ftt=PU.ftt[k]) for k in COMPS}
        WGT = (wr[:, None]*wa[None, :]).ravel()

        class F:
            pass
        Lws = []
        cols = ([('P', p, i) for p in range(nr) for i in range(NKP)]
                + [('T', p, i) for p in range(nr) for i in range(NKT)])
        for fam, p, i in cols:
            f = F()
            f.f, f.fr, f.frr, f.ft, f.ftt = {}, {}, {}, {}, {}
            for k in COMPS:
                for dd in DERIVS:
                    getattr(f, dd)[k] = np.zeros((Nb, Nt))
            slots = POL if fam == 'P' else TOR
            for sc, dd, rfac, afac in slots:
                getattr(f, dd)[sc] = getattr(f, dd)[sc] \
                    + np.outer(rfac[p], afac[i])
            f.fr['p'] = np.zeros((Nb, Nt)); f.ft['p'] = np.zeros((Nb, Nt))
            out = Lcols(f, Uval, Ubf, geo, True)
            Lws.append(out.reshape(3, Nb*Nt))
        Lws = np.array(Lws)                      # (N, 3, Ng)
        AAd_ = np.einsum('icg,jcg,g->ij', Lws, Lws, WGT, optimize=True)
        eAA = np.linalg.norm(AA - AAd_)/np.linalg.norm(AAd_)
        g1 = (S1[:, :, None]*Pq[None, None, :, :].transpose(0, 3, 1, 2)
              ).reshape(-1, Nb, Nt) if False else None
        # gradient columns explicit
        Gcols = []
        for m in range(Sf.shape[0]):
            for k in range(NKQ):
                Gcols.append(np.stack([np.outer(S1[m], Pq[k]),
                                       np.outer(Sor[m], Pq1[k]),
                                       np.zeros((Nb, Nt))]).reshape(3, -1))
        Gcols = np.array(Gcols)
        Bd_ = np.einsum('icg,qcg,g->iq', Lws, Gcols, WGT, optimize=True)
        eB = np.linalg.norm(B - Bd_)/np.linalg.norm(Bd_)
        ok = eAA < 1e-11 and eB < 1e-11
        print(f"T5  AA separable vs direct: {eAA:.3e}", flush=True)
        print(f"T5b B  separable vs direct: {eB:.3e}", flush=True)
        print(f"T5 gate: {'PASS' if ok else 'FAIL'}", flush=True)
    else:
        np.savez('nsx_x2_grams.npz', AA=AA, B=B, A1=A1, A0=A0, P0=P0,
                 P1g=P1g, nr=nr, NKP=NKP, NKT=NKT, NKQ=NKQ, Nq=Nq,
                 Nb=Nb, Nt=Nt, asym=asym)
        print(f"[{time.time()-t0:7.1f}s] saved nsx_x2_grams.npz", flush=True)

if __name__ == '__main__':
    run(sys.argv[1] if len(sys.argv) > 1 else 'gate')
