import numpy as np
from jc import *
N=30; A,SP,SM,SZ,PI=ops(N)
D=A@SP+A.conj().T@SM
Nexc=A.conj().T@A+SP@SM
# restrict checks away from truncation edge: project onto n<=N-1 manifold
P=np.kron(np.diag([1.]*N+[0.]),np.eye(2))
print('{sz,D}',np.abs(SZ@D+D@SZ).max())
print('{Pi,D}',np.abs(PI@D+PI.T@D*0+D@PI).max())
print('D^2-N (interior)',np.abs(P@(D@D-Nexc)@P).max())
Q=A@SP; print('Q^2',np.abs(Q@Q).max())
ev=np.sort(np.linalg.eigvalsh(D)); print('eig sample',ev[:3],ev[N-1:N+3])
# MBQC: X M(theta) X = M(-theta)
X=np.array([[0,1],[1,0]]);Y=np.array([[0,-1j],[1j,0]]);Z=np.diag([1,-1])
th=0.37;M=lambda t:np.cos(t)*X+np.sin(t)*Y
print('XMX-M(-th)',np.abs(X@M(th)@X-M(-th)).max(),' ZMZ+M',np.abs(Z@M(th)@Z+M(th)).max())
# graded-frame lemma random check
d=6;rng=np.random.default_rng(1)
G=np.diag([1,1,1,-1,-1,-1]);B=rng.normal(size=(3,3))+1j*rng.normal(size=(3,3))
K=np.zeros((6,6),complex);K[:3,3:]=B;K[3:,:3]=B.conj().T
from scipy.linalg import expm
print('graded lemma',np.abs(G@expm(-1j*K*0.9)@G-expm(1j*K*0.9)).max())
