# SPDX-License-Identifier: MIT
# Copyright (c) 2026 P. Salmond — hwy-spectral-survey (see LICENSE, NOTICE)
"""nsx_basis.py -- decay-adapted, divergence-free velocity basis for the
HWY operator survey (session 12 rebuild lane).

Design (each choice dodges a documented rev A-I failure mode):
* radial family R_j(rho) = n_j L_j^{(2)}(rho/ell) exp(-rho/(2 ell)),
  n_j = 1/(ell^{3/2} sqrt((j+1)(j+2))): ORTHONORMAL in int |R|^2 rho^2 drho
  -> true-L2 Gram healthy by construction (kills the rev C-H catastrophe);
* poloidal fields from the Stokes stream function
      Psi = rho^2 R_j(rho) sin^2(t) P'_{k+1}(mu),  mu = cos(t)
  =>  u1 = R_j (k+1)(k+2) P_{k+1}(mu)
      u2 = -(2R_j + rho R_j') sin(t) P'_{k+1}(mu)
  divergence-free IDENTICALLY (no slaving rederivation; kills rev-A risk);
* toroidal fields u3 = R_j sin(t) P'_{k+1}(mu);
* odd (HWY) sector: poloidal k even (u1 ~ odd-degree Legendre, matching
  their cos((2m-1)t) family), toroidal k odd (matching sin(2mt));
* exponential decay => pressure-gradient terms integrate away exactly in
  the weak form (the rev-B IBP failure cannot occur).

Exposes batched evaluation: dict comp[k][d] of arrays (N, nb, nt) for
k in u1,u2,u3 and d in f,fr,frr,ft,ftt -- the exact duck-type consumed by
ns_part3_spectrum.adv / lap_vec and ns_part3k.Lcols-style code.
"""
import numpy as np
from scipy.special import eval_genlaguerre
from numpy.polynomial import legendre as npleg

# ---------------------------------------------------------------- radial
def radial_factors(js, ell, rho):
    """Return R, R', R'' , and D=2R+rho R', D', D'' for each j in js.
    Shapes (len(js), len(rho))."""
    x = rho/ell
    E = np.exp(-0.5*x)
    out_R, out_R1, out_R2, out_R3 = [], [], [], []
    for j in js:
        n = 1.0/(ell**1.5*np.sqrt((j+1.0)*(j+2.0)))
        L0 = eval_genlaguerre(j, 2, x)
        L1 = -eval_genlaguerre(j-1, 3, x) if j >= 1 else np.zeros_like(x)
        L2 = eval_genlaguerre(j-2, 4, x) if j >= 2 else np.zeros_like(x)
        L3 = -eval_genlaguerre(j-3, 5, x) if j >= 3 else np.zeros_like(x)
        R = n*L0*E
        R1 = n*(L1 - 0.5*L0)*E/ell
        R2 = n*(L2 - L1 + 0.25*L0)*E/ell**2
        R3 = n*(L3 - 1.5*L2 + 0.75*L1 - 0.125*L0)*E/ell**3
        out_R.append(R); out_R1.append(R1); out_R2.append(R2); out_R3.append(R3)
    R = np.array(out_R); R1 = np.array(out_R1)
    R2 = np.array(out_R2); R3 = np.array(out_R3)
    D = 2*R + rho[None, :]*R1
    D1 = 3*R1 + rho[None, :]*R2
    D2 = 4*R2 + rho[None, :]*R3
    return dict(R=R, R1=R1, R2=R2, D=D, D1=D1, D2=D2)

# ---------------------------------------------------------------- angular
def _leg_derivs(deg, mu, nder):
    """P_deg(mu) and its mu-derivatives up to nder. Returns list."""
    c = np.zeros(deg+1); c[deg] = 1.0
    out = [npleg.legval(mu, c)]
    for _ in range(nder):
        c = npleg.legder(c)
        out.append(npleg.legval(mu, c))
    return out

def angular_factors(ks, t):
    """For each k: a = (k+1)(k+2) P_{k+1}(mu) (u1 factor) with theta-derivs,
    and b = sin(t) P'_{k+1}(mu) (u2 / u3 factor) with theta-derivs.
    d/dt = -sin(t) d/dmu. Shapes (len(ks), len(t))."""
    mu, st, ct = np.cos(t), np.sin(t), np.cos(t)
    A = dict(f=[], t=[], tt=[]); B = dict(f=[], t=[], tt=[])
    for k in ks:
        P, P1, P2, P3 = _leg_derivs(k+1, mu, 3)
        c = (k+1.0)*(k+2.0)
        a   = c*P
        at  = -c*st*P1
        att = -c*(ct*P1 - st*st*P2)
        b   = st*P1
        bt  = ct*P1 - st*st*P2
        btt = -st*P1 - 3*st*ct*P2 + st**3*P3
        A['f'].append(a); A['t'].append(at); A['tt'].append(att)
        B['f'].append(b); B['t'].append(bt); B['tt'].append(btt)
    for d in A: A[d] = np.array(A[d]); B[d] = np.array(B[d])
    return A, B

# ------------------------------------------------------------- assembly
class DivFreeBasis:
    """Odd-sector decay-adapted basis: Npol poloidal (j x k_even) +
    Ntor toroidal (j x k_odd) columns."""
    def __init__(self, nj, nk_pol, nk_tor, ell):
        self.ell = ell
        self.js = list(range(nj))
        self.ks_pol = [2*i for i in range(nk_pol)]        # k even
        self.ks_tor = [2*i+1 for i in range(nk_tor)]      # k odd
        self.n_pol = nj*nk_pol
        self.n_tor = nj*nk_tor
        self.N = self.n_pol + self.n_tor

    def eval_chunk(self, b, t, geo):
        """Batched fields on the (len(b), len(t)) grid.
        Returns comp[k][d] arrays of shape (N, nb, nt)."""
        rho_1d = (np.tan(b)*22.0)  # S = 22, matches make_geo
        rad = radial_factors(self.js, self.ell, rho_1d)
        Apol, Bpol = angular_factors(self.ks_pol, t)
        _,    Btor = angular_factors(self.ks_tor, t)
        nb, nt = len(b), len(t)
        N = self.N
        Z = lambda: np.zeros((N, nb, nt))
        comp = {k: dict(f=Z(), fr=Z(), frr=Z(), ft=Z(), ftt=Z())
                for k in ('u1', 'u2', 'u3')}
        i = 0
        for jj in range(len(self.js)):
            for kk in range(len(self.ks_pol)):
                R, R1, R2 = rad['R'][jj], rad['R1'][jj], rad['R2'][jj]
                D, D1, D2 = rad['D'][jj], rad['D1'][jj], rad['D2'][jj]
                a, at, att = Apol['f'][kk], Apol['t'][kk], Apol['tt'][kk]
                bb, bt, btt = Bpol['f'][kk], Bpol['t'][kk], Bpol['tt'][kk]
                comp['u1']['f'][i]   = np.outer(R,  a)
                comp['u1']['fr'][i]  = np.outer(R1, a)
                comp['u1']['frr'][i] = np.outer(R2, a)
                comp['u1']['ft'][i]  = np.outer(R,  at)
                comp['u1']['ftt'][i] = np.outer(R,  att)
                comp['u2']['f'][i]   = -np.outer(D,  bb)
                comp['u2']['fr'][i]  = -np.outer(D1, bb)
                comp['u2']['frr'][i] = -np.outer(D2, bb)
                comp['u2']['ft'][i]  = -np.outer(D,  bt)
                comp['u2']['ftt'][i] = -np.outer(D,  btt)
                i += 1
        for jj in range(len(self.js)):
            for kk in range(len(self.ks_tor)):
                R, R1, R2 = rad['R'][jj], rad['R1'][jj], rad['R2'][jj]
                bb, bt, btt = Btor['f'][kk], Btor['t'][kk], Btor['tt'][kk]
                comp['u3']['f'][i]   = np.outer(R,  bb)
                comp['u3']['fr'][i]  = np.outer(R1, bb)
                comp['u3']['frr'][i] = np.outer(R2, bb)
                comp['u3']['ft'][i]  = np.outer(R,  bt)
                comp['u3']['ftt'][i] = np.outer(R,  btt)
                i += 1
        return comp

# ---------------------------------------------------- pressure gradients
class GradSpan:
    """Gradient columns grad(q), q = S_m(rho) P_{k+1}(mu), k even (odd-
    sector pressure parity, matching HWY's cos((2m-1)t) family). Used only
    to deflate gradients in the residual landscape; the Galerkin pencil is
    pressure-free by the weak form."""
    def __init__(self, nm, nk, ellp):
        self.ms = list(range(nm)); self.ks = [2*i for i in range(nk)]
        self.ellp = ellp
        self.N = nm*nk

    def eval_chunk(self, b, t):
        rho = 22.0*np.tan(b)
        x = rho/self.ellp
        E = np.exp(-0.5*x)
        mu, st = np.cos(t), np.sin(t)
        cols1, cols2 = [], []      # d/drho q , (1/rho) d/dtheta q
        for m in self.ms:
            L0 = eval_genlaguerre(m, 0, x)
            L1 = -eval_genlaguerre(m-1, 1, x) if m >= 1 else np.zeros_like(x)
            Sm = L0*E
            Sm1 = (L1 - 0.5*L0)*E/self.ellp
            for k in self.ks:
                P, P1 = _leg_derivs(k+1, mu, 1)[:2]
                cols1.append(np.outer(Sm1, P))
                cols2.append(np.outer(Sm/rho, -st*P1))
        g1 = np.array(cols1); g2 = np.array(cols2)
        return g1, g2   # (N, nb, nt) each; u3-component of gradient is 0

# ----------------------------------------------------------- self tests
if __name__ == '__main__':
    from ns_part3f import make_geo, gl_nodes, B0, T0, T1
    from ns_part3_spectrum import div_of
    rng = np.random.default_rng(0)
    bas = DivFreeBasis(nj=6, nk_pol=3, nk_tor=2, ell=5.0)
    b, _ = gl_nodes(40, B0, np.pi/2 - 0.04)
    t, _ = gl_nodes(30, T0, T1)
    geo = make_geo(b, t)
    comp = bas.eval_chunk(b, t, geo)

    # T1: divergence-free identically (through the bundle's own div_of)
    bf = {k: {'f': comp[k]['f'], 'fr': comp[k]['fr'],
              'ft': comp[k]['ft']} for k in ('u1', 'u2', 'u3')}
    dv = div_of(bf, geo)
    scale = max(np.abs(comp[k]['f']).max() for k in ('u1', 'u2', 'u3'))
    print(f"T1 div-free: max|div|/scale = {np.abs(dv).max()/scale:.3e} "
          f"({'PASS' if np.abs(dv).max()/scale < 1e-11 else 'FAIL'})")

    # T2: rho/theta derivative consistency by central FD on a fine line
    eps = 1e-5
    bF, tF = b[20:21], t[15:16]
    for db, dt_ in ((eps, 0.0), (0.0, eps)):
        # FD in rho: vary beta consistently (rho = 22 tan beta)
        pass
    rho0 = 22*np.tan(b[20])
    bp = np.array([np.arctan((rho0+eps)/22)]); bm = np.array([np.arctan((rho0-eps)/22)])
    cp = bas.eval_chunk(bp, t, None); cm = bas.eval_chunk(bm, t, None)
    c0 = bas.eval_chunk(np.array([b[20]]), t, None)
    e1 = np.abs((cp['u1']['f']-cm['u1']['f'])/(2*eps) - c0['u1']['fr']).max()
    e2 = np.abs((cp['u2']['f']-cm['u2']['f'])/(2*eps) - c0['u2']['fr']).max()
    tp = t + eps; tm = t - eps
    ctp = bas.eval_chunk(np.array([b[20]]), tp, None)
    ctm = bas.eval_chunk(np.array([b[20]]), tm, None)
    e3 = np.abs((ctp['u2']['f']-ctm['u2']['f'])/(2*eps) - c0['u2']['ft']).max()
    e4 = np.abs((ctp['u3']['f']-ctm['u3']['f'])/(2*eps) - c0['u3']['ft']).max()
    ok = max(e1, e2, e3, e4) < 1e-6*scale
    print(f"T2 derivative FD check: max err = {max(e1,e2,e3,e4):.3e} "
          f"({'PASS' if ok else 'FAIL'})")

    # T3: radial orthonormality of R_j under rho^2 drho (fine quadrature)
    rr = np.linspace(1e-4, 60*5.0, 40000)
    rad = radial_factors(range(8), 5.0, rr)
    Gr = (rad['R']*rr**2) @ rad['R'].T * (rr[1]-rr[0])
    err = np.abs(Gr - np.eye(8)).max()
    print(f"T3 radial orthonormality: max|G-I| = {err:.3e} "
          f"({'PASS' if err < 1e-6 else 'FAIL'})")
