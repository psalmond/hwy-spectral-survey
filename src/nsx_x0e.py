# SPDX-License-Identifier: MIT
# Copyright (c) 2026 P. Salmond — hwy-spectral-survey (see LICENSE, NOTICE)
"""nsx_x0e.py -- v1 fit-convergence scan: can the fit error reach <=3e-4?
Direct weighted LSQ of the exact v1 in the div-free family (open domain),
scanning radial count/width and angular counts. Origin mask applied
(landscape-compatible family). Verdict for the survey route.
"""
import numpy as np, time
from nsx_op import hermite_full
from nsx_basis import angular_factors, radial_factors
from ns_part3k import load_family
from ns_part3f import gl_nodes
from ns_part12_gate import Phys
from ns_part3_spectrum import S
from scipy.special import eval_genlaguerre
t0=time.time()
Nb, Nt = 560, 260
b, wb = gl_nodes(Nb, 0.0, np.pi/2)
t, wt = gl_nodes(Nt, 0.0, np.pi/2)
rho = S*np.tan(b)
wr = rho**2*S/np.cos(b)**2*wb
wa = np.sin(t)*wt
fams,_ = load_family('odd')
P = Phys(fams[-1], b, t, 'cos')
U1, U2, U3 = P.f['u1'], P.f['u2'], P.f['u3']
Etot = ((wr[:,None]*wa[None,:])*(U1*U1+U2*U2+U3*U3)).sum()
def mask(R, R1):
    m0 = rho*rho/(rho*rho+4.0); m1 = 8.0*rho/(rho*rho+4.0)**2
    return R*m0, R1*m0 + R*m1
def lagblk(nm, ell):
    rl = radial_factors(range(nm), ell, rho)
    return rl['R'], rl['R1']
def fit(njH, sig, njL, NKP, NKT, tag):
    RH, RH1 = hermite_full(range(njH), 20.0, sig, rho)[:2]
    RL, RL1 = lagblk(njL, 3.0)
    R = np.vstack([RH, RL]); R1 = np.vstack([RH1, RL1])
    R, R1 = mask(R, R1)
    D = 2*R + rho[None,:]*R1
    ksP=[2*i for i in range(NKP)]; ksT=[2*i+1 for i in range(NKT)]
    AP, BP = angular_factors(ksP, t); _, BT = angular_factors(ksT, t)
    a, bb, b3 = AP['f'], BP['f'], BT['f']
    nr = R.shape[0]
    Grr=(R*wr)@R.T; Gdd=(D*wr)@D.T
    Gaa=(a*wa)@a.T; Gbb=(bb*wa)@bb.T; G33=(b3*wa)@b3.T
    HP = np.kron(Grr,Gaa)+np.kron(Gdd,Gbb)
    HT = np.kron(Grr,G33)
    y1=(R*wr)@U1@(wa*a).T; y2=(D*wr)@U2@(wa*bb).T; y3=(R*wr)@U3@(wa*b3).T
    rP=(y1-y2).ravel(); rT=y3.ravel()
    def sol(Hm, rm):
        ev = np.linalg.eigvalsh(Hm)
        c = np.linalg.solve(Hm+1e-13*ev[-1]*np.eye(len(Hm)), rm)
        return c@rm
    cap = sol(HP,rP)+sol(HT,rT)
    err = np.sqrt(max(Etot-cap,0)/Etot)
    print(f"{tag:42s} N={nr*NKP+nr*NKT:5d}  err={err:.4e}", flush=True)
    return err
fit(50, 2.5, 12, 40, 24, "baseline njH50 s2.5 kP40 kT24")
fit(70, 2.5, 12, 40, 24, "njH 50->70")
fit(50, 2.5, 12, 56, 32, "kP40->56 kT24->32")
fit(70, 2.5, 12, 56, 32, "njH70 kP56 kT32")
fit(90, 2.5, 12, 56, 32, "njH90 kP56 kT32")
fit(70, 3.5, 12, 56, 32, "njH70 SIG2.5->3.5 kP56 kT32")
fit(90, 3.5, 12, 72, 40, "njH90 s3.5 kP72 kT40")
print(f"[{time.time()-t0:.1f}s]")
