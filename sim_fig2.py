import numpy as np, json
from jc import *
from scipy.linalg import expm
N=30; nbar=5.0; alpha=np.sqrt(nbar)
A,SP,SM,SZ,PI=ops(N)
H=A@SP+A.conj().T@SM
SX=SP+SM; SY=-1j*(SP-SM)
p0=psi0(N,alpha,'e'); rho0=np.outer(p0,p0.conj())
ez=lambda r: np.real(np.trace(SZ@r))
kap=0.0167; gam=0.06
Ls=[np.sqrt(kap)*A,np.sqrt(gam)*SM]
out={}
tmax=40.0
# bare lossy
_,rb=evolve(rho0,H,Ls,0,tmax,npts=401); out['bare_t']=np.linspace(0,tmax,401).tolist(); out['bare']=[ez(r) for r in rb]
# kick trains: kicks at T,3T,5T...; stroboscopic record at 2kT
for T in (2.0,0.5,0.125):
    r=rho0; tcur=0; st=[0.0]; ss=[1.0]
    # first T, then alternate 2T segments with kick
    r=evolve(r,H,Ls,0,T)[1][-1]; tcur=T
    while tcur+2*T<=tmax+1e-9:
        r=SZ@r@SZ
        # record at echo point tcur+T
        r=evolve(r,H,Ls,tcur,tcur+T)[1][-1]; tcur+=T; st.append(tcur); ss.append(ez(r))
        r=evolve(r,H,Ls,tcur,tcur+T)[1][-1]; tcur+=T
    out['kick_%g'%T]=[st,ss]
    tt=np.array(st); print('T',T,'max dev from 2e^{-gam t}-1:',np.max(np.abs(np.array(ss)-(2*np.exp(-gam*tt)-1))))
# frame bound: bare Bloch length (lossless) and a product-frame example
ts=np.linspace(0,30,601)
def bare_state(t): return expm(-1j*H*t)@p0
bl=[];szb=[]
for t in ts:
    v=bare_state(t); s=[np.real(np.vdot(v,O@v)) for O in (SX,SY,SZ)]; bl.append(np.linalg.norm(s)); szb.append(s[2])
out['frame_t']=ts.tolist(); out['bloch']=bl; out['sz_bare']=szb
# product frame W(t)= w(t) atom rotation (classical control). Optimal atom-only frame achieves sz_1 = |s|.
# explicit example: w(t)=exp(-i theta(t) (cos phi sx + sin phi sy)/2), theta=1.3 sin(0.4 t), phi=0.7 t
def w(t):
    th=1.3*np.sin(0.4*t); ph=0.7*t
    n=np.cos(ph)*SX+np.sin(ph)*SY
    return expm(-1j*th*n/2)
def h1(t,eps=1e-6):
    W=w(t); Wd=(w(t+eps)-w(t-eps))/(2*eps)
    return W@H@W.conj().T+1j*Wd@W.conj().T
from scipy.integrate import solve_ivp
y0=w(0)@p0
sol=solve_ivp(lambda t,y:-1j*h1(t)@y,(0,30),y0,t_eval=ts,rtol=1e-10,atol=1e-12,method='DOP853')
sz1=[np.real(np.vdot(sol.y[:,k],SZ@sol.y[:,k])) for k in range(len(ts))]
pred=[]
for k,t in enumerate(ts):
    v=w(t)@bare_state(t); pred.append(np.real(np.vdot(v,SZ@v)))
print('frame lemma max err',np.max(np.abs(np.array(sz1)-np.array(pred))),' bound violated?',np.any(np.abs(sz1)>np.array(bl)+1e-7))
out['sz_frame']=sz1
json.dump(out,open('fig2.json','w'))
