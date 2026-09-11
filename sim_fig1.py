import numpy as np, json
from jc import *
from scipy.linalg import expm
N=30; nbar=5.0; alpha=np.sqrt(nbar)
A,SP,SM,SZ,PI=ops(N)
H=A@SP+A.conj().T@SM
p0=psi0(N,alpha,'e'); rho0=np.outer(p0,p0.conj())
Nex=A.conj().T@A+SP@SM; N2=np.real(np.vdot(p0,Nex@Nex@p0))
ez=lambda r: np.real(np.trace(SZ@r))
kap0=0.0167; gam0=0.06   # C = g^2/(kappa*gamma) ~ 1.0e3
out={'N2':N2,'nbar':nbar}
# (a) trajectories T=5
T=5.0
ts=np.linspace(0,2*T,401)
out['a_t']=ts.tolist()
out['a_bare']=[np.real(np.vdot(expm(-1j*H*t)@p0,SZ@expm(-1j*H*t)@p0)) for t in ts]
def echo_traj(Ls,T,npts=201):
    t1,r1=evolve(rho0,H,Ls,0,T,npts)
    r=SZ@r1[-1]@SZ
    t2,r2=evolve(r,H,Ls,T,2*T,npts)
    return list(t1)+list(t2),[ez(x) for x in r1]+[ez(x) for x in r2],r2[-1]
for lab,s in (('lossless',None),('C1e3',1.0),('C1e4',10**-0.5),('C1e5',0.1)):
    Ls=[] if s is None else [np.sqrt(kap0*s)*A,np.sqrt(gam0*s)*SM]
    tt,ss,_=echo_traj(Ls,T)
    out['a_'+lab]=[tt,ss]
# (b) echo return vs 2T
Ts=np.linspace(0.5,12,24)
for lab,s in (('C1e3',1.0),('C1e4',10**-0.5),('C1e5',0.1)):
    kap,gam=kap0*s,gam0*s
    Ls=[np.sqrt(kap)*A,np.sqrt(gam)*SM]
    vals=[];Fs=[]
    for T in Ts:
        _,_,rf=echo_traj(Ls,T,npts=2)
        vals.append(ez(rf)); ideal=SZ@p0; Fs.append(np.real(np.vdot(ideal,rf@ideal)))
    bound=[max(0,1-T*(kap*np.sqrt(N2)+gam))**2 for T in Ts]
    out['b_'+lab]={'T2':(2*Ts).tolist(),'sz':vals,'F':Fs,'Fbound':bound,'kap':kap,'gam':gam}
    print(lab,'F>=bound?',all(f>=b-1e-9 for f,b in zip(Fs,bound)), 'sz(2T) first/last',vals[0],vals[-1])
json.dump(out,open('fig1.json','w'))
