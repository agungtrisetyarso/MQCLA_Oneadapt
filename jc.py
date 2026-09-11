import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from math import factorial
# basis: field n=0..N (x) atom {e,g}; index = 2*n + (0 for e, 1 for g)
def ops(N):
    a=np.diag(np.sqrt(np.arange(1,N+1)),1)
    I_f=np.eye(N+1); I_a=np.eye(2)
    sp=np.array([[0,1],[0,0]],complex)  # |e><g| with e=index0
    sz=np.diag([1.,-1.])
    A=np.kron(a,I_a); SP=np.kron(I_f,sp); SM=SP.conj().T; SZ=np.kron(I_f,sz)
    PI=np.kron(np.diag((-1.)**np.arange(N+1)),I_a)
    return A,SP,SM,SZ,PI
def coherent(N,alpha):
    n=np.arange(N+1); c=np.array([alpha**k/np.sqrt(float(factorial(k))) for k in n],complex)*np.exp(-abs(alpha)**2/2)
    return c/np.linalg.norm(c)
def psi0(N,alpha,atom='e'):
    at=np.array([1,0],complex) if atom=='e' else np.array([0,1],complex)
    return np.kron(coherent(N,alpha),at)
def lindblad_rhs(H,Ls):
    def f(t,y,Hfun=None):
        d=int(round(np.sqrt(y.size))); r=y.reshape(d,d)
        Ht=H(t) if callable(H) else H
        out=-1j*(Ht@r-r@Ht)
        for L in Ls:
            LdL=L.conj().T@L
            out+=L@r@L.conj().T-0.5*(LdL@r+r@LdL)
        return out.ravel()
    return f
def evolve(rho,H,Ls,t0,t1,npts=2,rtol=1e-9,atol=1e-11):
    d=rho.shape[0]
    ts=np.linspace(t0,t1,npts)
    sol=solve_ivp(lindblad_rhs(H,Ls),(t0,t1),rho.ravel().astype(complex),t_eval=ts,rtol=rtol,atol=atol,method='DOP853')
    return ts,[sol.y[:,k].reshape(d,d) for k in range(len(ts))]
