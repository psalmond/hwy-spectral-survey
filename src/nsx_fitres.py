# SPDX-License-Identifier: MIT
# Copyright (c) 2026 P. Salmond — hwy-spectral-survey (see LICENSE, NOTICE)
import numpy as np, time
import nsx_op
from nsx_op import composite_radial, hermite_full
from nsx_basis import angular_factors, _leg_derivs
from ns_part3k import load_family, profile_F, Lcols
from ns_part3f import make_geo, gl_nodes
from ns_part12_gate import Phys
from ns_part3_spectrum import S
from scipy.special import eval_genlaguerre
LAM = 0.11314203274385946
t0=time.time()
Z = np.load('nsx_x1_grams.npz'); H = Z['H']
nr,NKP,NKT = int(Z['nr']),int(Z['NKP']),int(Z['NKT'])
Nb,Nt = int(Z['Nb']),int(Z['Nt'])
b,wb = gl_nodes(Nb,0.0,np.pi/2); t,wt = gl_nodes(Nt,0.0,np.pi/2)
geo = make_geo(b,t); rho = S*np.tan(b); nsx_op.RHO = rho
wr = rho**2*S/np.cos(b)**2*wb; wa = np.sin(t)*wt
WGT = wr[:,None]*wa[None,:]
rad,_ = composite_radial(rho,wr,int(Z['NJH']),float(Z['RHO0']),float(Z['SIG']),int(Z['NJL']),float(Z['ELLL']))
R,R1,R2,R3 = rad
D = 2*R+rho[None,:]*R1; D1 = 3*R1+rho[None,:]*R2; D2 = 4*R2+rho[None,:]*R3
ksP=[2*i for i in range(NKP)]; ksT=[2*i+1 for i in range(NKT)]
AP,BP = angular_factors(ksP,t); _,BT = angular_factors(ksT,t)
NP,NT = nr*NKP,nr*NKT; N=NP+NT
# fit vector (X0r)
fams,_ = load_family('odd'); Pv1 = Phys(fams[-1],b,t,'cos')
y1=(R*wr)@Pv1.f['u1']@(wa*AP['f']).T; y2=(D*wr)@Pv1.f['u2']@(wa*BP['f']).T
y3=(R*wr)@Pv1.f['u3']@(wa*BT['f']).T
rhs = np.concatenate([(y1-y2).ravel(), y3.ravel()])
evH = np.linalg.eigvalsh(H)
c = np.linalg.solve(H+1e-13*evH[-1]*np.eye(N), rhs)
# field + Lcols
class F: pass
cP=c[:NP].reshape(nr,NKP); cT=c[NP:].reshape(nr,NKT)
mix=lambda Rm,cm,Am:(Rm.T@cm)@Am
f=F(); f.f,f.fr,f.frr,f.ft,f.ftt={},{},{},{},{}
f.f['u1']=mix(R,cP,AP['f']); f.fr['u1']=mix(R1,cP,AP['f']); f.frr['u1']=mix(R2,cP,AP['f'])
f.ft['u1']=mix(R,cP,AP['t']); f.ftt['u1']=mix(R,cP,AP['tt'])
f.f['u2']=-mix(D,cP,BP['f']); f.fr['u2']=-mix(D1,cP,BP['f']); f.frr['u2']=-mix(D2,cP,BP['f'])
f.ft['u2']=-mix(D,cP,BP['t']); f.ftt['u2']=-mix(D,cP,BP['tt'])
f.f['u3']=mix(R,cT,BT['f']); f.fr['u3']=mix(R1,cT,BT['f']); f.frr['u3']=mix(R2,cT,BT['f'])
f.ft['u3']=mix(R,cT,BT['t']); f.ftt['u3']=mix(R,cT,BT['tt'])
f.fr['p']=np.zeros((Nb,Nt)); f.ft['p']=np.zeros((Nb,Nt))
UF=profile_F(); PU=Phys(UF,b,t,'cos')
Uval={k:PU.f[k] for k in ('u1','u2','u3')}
Ubf={k:dict(f=PU.f[k],fr=PU.fr[k],frr=PU.frr[k],ft=PU.ft[k],ftt=PU.ftt[k]) for k in ('u1','u2','u3')}
out=Lcols(f,Uval,Ubf,geo,True); Ng=Nb*Nt
r1=out[:Ng].reshape(Nb,Nt)-LAM*f.f['u1']; r2=out[Ng:2*Ng].reshape(Nb,Nt)-LAM*f.f['u2']
r3=out[2*Ng:].reshape(Nb,Nt)-LAM*f.f['u3']
vn2=(WGT*(f.f['u1']**2+f.f['u2']**2+f.f['u3']**2)).sum()
print(f"[{time.time()-t0:.1f}s] fit field, raw r = {np.sqrt((WGT*(r1*r1+r2*r2+r3*r3)).sum()/vn2):.4e}", flush=True)
# deflate with the calibrated span
def lag0(nm,ell):
    x=rho/ell; E=np.exp(-0.5*x); Sf,S1=[],[]
    for m in range(nm):
        L0=eval_genlaguerre(m,0,x); L1=-eval_genlaguerre(m-1,1,x) if m>=1 else 0*x
        Sf.append(L0*E); S1.append((L1-0.5*L0)*E/ell)
    return np.array(Sf),np.array(S1)
HH=hermite_full(range(60),20.0,2.5,rho); H2=hermite_full(range(30),20.0,1.2,rho)
L3=lag0(12,3.0); L8=lag0(8,8.0)
Sf=np.vstack([HH[0],H2[0],L3[0],L8[0]]); S1g=np.vstack([HH[1],H2[1],L3[1],L8[1]])
Sor=Sf/rho[None,:]
mu,st=np.cos(t),np.sin(t); Pq,Pq1=[],[]
for k in [2*i for i in range(72)]:
    P,P1=_leg_derivs(k+1,mu,1)[:2]; Pq.append(P); Pq1.append(-st*P1)
Pq=np.array(Pq); Pq1=np.array(Pq1)
M=np.kron((S1g*wr)@S1g.T,(Pq*wa)@Pq.T)+np.kron((Sor*wr)@Sor.T,(Pq1*wa)@Pq1.T)
evM,QM=np.linalg.eigh(M)
rhsq=((S1g*wr)@r1@(wa[:,None]*Pq.T)+(Sor*wr)@r2@(wa[:,None]*Pq1.T)).ravel()
y=QM.T@rhsq; n2=(WGT*(r1*r1+r2*r2+r3*r3)).sum()
for cut in (1e-10,1e-12):
    kp=evM>cut*evM[-1]
    print(f"  fit-vector deflated r(lam1) cut{cut:.0e}: {np.sqrt(max(n2-(y[kp]**2/evM[kp]).sum(),0)/vn2):.4e}", flush=True)
print(f"[{time.time()-t0:.1f}s] done")
