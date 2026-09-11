import numpy as np
from scipy.integrate import solve_ivp
from jc import *
# forced oscillator H_F = w a^dag a + f(t) a + f*(t) a^dag ; L = a - alpha(t), i alpha' = w alpha + f*
N=90; w=1.3
a=np.diag(np.sqrt(np.arange(1,N+1)),1); ad=a.conj().T; n=ad@a
f=lambda t: 0.4*np.exp(1j*0.7*t)+0.2j*np.cos(1.9*t)
HF=lambda t: w*n+f(t)*a+np.conj(f(t))*ad
c0=coherent(N,1.2+0.5j); fock1=np.zeros(N+1,complex); fock1[1]=1
psi_init=(c0+0.6*np.roll(c0,1)); psi_init/=np.linalg.norm(psi_init)
alpha0=0.3-0.2j
def rhs(t,y):
    psi=y[:-1]; al=y[-1]
    return np.concatenate([-1j*HF(t)@psi,[-1j*(w*al+np.conj(f(t)))]])
ts=np.linspace(0,6,7)
s=solve_ivp(rhs,(0,6),np.concatenate([psi_init,[alpha0]]),t_eval=ts,rtol=1e-11,atol=1e-13,method='DOP853')
phi_path=[(a-s.y[-1,k]*np.eye(N+1))@s.y[:-1,k] for k in range(len(ts))]
s2=solve_ivp(lambda t,y:-1j*(HF(t)+w*np.eye(N+1))@y,(0,6),phi_path[0],t_eval=ts,rtol=1e-11,atol=1e-13,method='DOP853')
print('intertwiner max err',max(np.abs(phi_path[k]-s2.y[:,k]).max() for k in range(len(ts))))
# Mollow displacement with loss: H = gD + Om(t)(a+a^dag), L=sqrt(k) a  vs  H' = gD + g(beta sp + beta* sm), i beta' = Om - i k beta/2, state displaced
Nf=40; A,SP,SM,SZ,PI=ops(Nf); D=A@SP+A.conj().T@SM
k=0.2; gam=0.05; Om=lambda t:0.6*np.cos(0.8*t)
Ls=[np.sqrt(k)*A,np.sqrt(gam)*SM]
p=psi0(Nf,0.0,'e'); rho=np.outer(p,p.conj())
t_,rs=evolve(rho,lambda t:D+Om(t)*(A+A.conj().T),Ls,0,8,npts=9)
# frame: rho = Dis(beta) rho' Dis^dag, rho'(0)=rho(0) with beta(0)=0
sb=solve_ivp(lambda t,y:[-1j*Om(t)-k/2*y[0]],(0,8),[0j],t_eval=t_,rtol=1e-12,atol=1e-14)
from scipy.interpolate import interp1d
bfun=lambda t: solve_ivp(lambda tt,y:[-1j*Om(tt)-k/2*y[0]],(0,t if t>0 else 1e-12),[0j],rtol=1e-12,atol=1e-14).y[0,-1]
# precompute beta on fine grid
tg=np.linspace(0,8,4001); bg=solve_ivp(lambda t,y:[-1j*Om(t)-k/2*y[0]],(0,8),[0j],t_eval=tg,rtol=1e-12,atol=1e-14).y[0]
br=interp1d(tg,bg.real,kind='cubic'); bi=interp1d(tg,bg.imag,kind='cubic')
beta=lambda t: br(t)+1j*bi(t)
t2,rs2=evolve(rho,lambda t:D+(beta(t)*SP+np.conj(beta(t))*SM),Ls,0,8,npts=9)
print('Mollow sigma_z max diff',max(abs(np.trace(SZ@rs[j]).real-np.trace(SZ@rs2[j]).real) for j in range(9)))
