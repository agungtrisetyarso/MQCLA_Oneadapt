import numpy as np
from scipy.integrate import solve_ivp
from jc import *
N=40; A,SP,SM,SZ,PI=ops(N); D=A@SP+A.conj().T@SM
T=4.0; Om=lambda t: 0.5*np.cos(0.9*(t-T))   # palindromic about T
H=lambda t: D+Om(t)*(A+A.conj().T)
p=psi0(N,np.sqrt(3),'e')
def U(p,t0,t1): return solve_ivp(lambda t,y:-1j*H(t)@y,(t0,t1),p,rtol=1e-11,atol=1e-13,method='DOP853').y[:,-1]
for name,K in (('Pi',PI),('sz',SZ)):
    q=U(K@U(p,0,T),T,2*T)
    print(name,'|<K psi0|psi(2T)>|=',abs(np.vdot(K@p,q)))
