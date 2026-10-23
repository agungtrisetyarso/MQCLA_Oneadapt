"""Numerical checks for the second-round revision of the IJTP manuscript.

(9)  symmetric intertwining forces h1 = h0; corrected (asymmetric) relation
(5,6) explicit Hermitian family of B(t) (Lemma 2(iii))
(2)  Strang/BCH error of e^{TL+} e^{2TL-} e^{TL+} versus e^{4TD}
(7)  displaced dissipator identity with the kappa factor
(11) horizon numbers
(12) round-budget corollary numbers
"""
import numpy as np
from scipy.linalg import expm, sqrtm, polar
from scipy.integrate import solve_ivp

rng = np.random.default_rng(7)
d = 4

def herm(n):
    A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    return (A + A.conj().T) / 2

# ---------- (9): symbolic-free check with matrix-valued polynomials in t ----------
# Represent operators acting on f(t) = f0 + f1 t + f2 t^2 (vector coefficients) is clumsy;
# instead check the coefficient identity directly: for L = d/dt + B and Lt = L + i(h1-h0),
#   (i d/dt - h1) L f - Lt (i d/dt - h0) f = [iB' - h1 B + h0' + Lt_B h0] f   (f-dot terms cancel)
# with h1 = i M' M^{-1} + M h0 M^{-1}, M = B - i h0.  Evaluate at random t.
H0a, H0b = herm(d), herm(d)
Ba, Bb, Bc = [rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d)) for _ in range(3)]
h0 = lambda t: H0a + np.sin(t) * H0b
h0d = lambda t: np.cos(t) * H0b
B = lambda t: Ba + t * Bb + t**2 * Bc
Bd = lambda t: Bb + 2 * t * Bc
M = lambda t: B(t) - 1j * h0(t)
Md = lambda t: Bd(t) - 1j * h0d(t)
def h1(t):
    Mi = np.linalg.inv(M(t))
    return 1j * Md(t) @ Mi + M(t) @ h0(t) @ Mi
res = []
for t in rng.uniform(0, 2, 5):
    Bt = B(t) + 1j * (h1(t) - h0(t))           # zeroth-order part of L~
    fdot_coef = (1j * B(t) - h1(t)) - (1j * Bt - h0(t))
    f_coef = (1j * Bd(t) - h1(t) @ B(t)) - (-h0d(t) - Bt @ h0(t))
    res.append(max(np.abs(fdot_coef).max(), np.abs(f_coef).max()))
print("(9) corrected relation residual:", max(res))
t = 0.7
print("(9) symmetric form would need h1=h0; ||h1-h0|| =", np.linalg.norm(h1(t) - h0(t)))

# on-shell check: propagate psi under h0, phi = M psi must solve h1
def rhs0(t, y): return (-1j * h0(t) @ y)
y0 = rng.normal(size=d) + 1j * rng.normal(size=d)
ts = np.linspace(0, 1.5, 7)
sol = solve_ivp(rhs0, (0, 1.5), y0, t_eval=ts, rtol=1e-12, atol=1e-13, method="DOP853")
phi0 = M(0) @ y0
sol1 = solve_ivp(lambda t, y: -1j * h1(t) @ y, (0, 1.5), phi0, t_eval=ts, rtol=1e-12, atol=1e-13, method="DOP853")
err = max(np.linalg.norm(M(tt) @ sol.y[:, k] - sol1.y[:, k]) for k, tt in enumerate(ts))
print("(9) on-shell: M psi solves h1-equation, max error:", err)

# ---------- (5,6) Hermitian family: M = W(t) U0(t) G0^{1/2} U0(t)^dag ----------
K = herm(d); A = herm(d); Hc = herm(d)
h0c = lambda t: Hc                     # time-independent for an explicit U0
U0 = lambda t: expm(-1j * Hc * t)
W = lambda t: expm(-1j * (A * t + K * t**2))
G0h = sqrtm((lambda X: X @ X.conj().T + np.eye(d))(rng.normal(size=(d, d))))
def Mf(t): return W(t) @ U0(t) @ G0h @ U0(t).conj().T
def h1f(t, e=1e-6):
    Mdot = (Mf(t + e) - Mf(t - e)) / (2 * e)
    Mi = np.linalg.inv(Mf(t))
    return 1j * Mdot @ Mi + Mf(t) @ Hc @ Mi
t = 0.9
Wd = (W(t + 1e-6) - W(t - 1e-6)) / 2e-6
target = W(t) @ Hc @ W(t).conj().T + 1j * Wd @ W(t).conj().T
print("(5,6) Hermitian family: ||h1 - (W h0 W^+ + i W' W^+)|| =", np.linalg.norm(h1f(t) - target),
      " non-Hermiticity:", np.linalg.norm(h1f(t) - h1f(t).conj().T))

# ---------- JC model on truncated space ----------
nmax = 30
a = np.diag(np.sqrt(np.arange(1, nmax + 1)), 1)
I_f = np.eye(nmax + 1); I_a = np.eye(2)
sp = np.array([[0, 1], [0, 0]], dtype=complex)  # |e><g| with basis (e,g)
sm = sp.conj().T
sz = np.diag([1., -1.])
A_ = np.kron(a, I_a); Sp = np.kron(I_f, sp); Sm = np.kron(I_f, sm); Sz = np.kron(I_f, sz)
Dop = A_ @ Sp + A_.conj().T @ Sm
dim = Dop.shape[0]
Id = np.eye(dim)

def lind_super(Hm, Ls):
    """column-stacking vec: vec(AXB) = (B^T kron A) vec X"""
    S = -1j * (np.kron(Id, Hm) - np.kron(Hm.T, Id))
    for L in Ls:
        LdL = L.conj().T @ L
        S += np.kron(L.conj(), L) - 0.5 * np.kron(Id, LdL) - 0.5 * np.kron(LdL.T, Id)
    return S

# ---------- (2) Strang error: use a small cutoff to keep superoperators manageable ----------
def small_jc(nm):
    a = np.diag(np.sqrt(np.arange(1, nm + 1)), 1)
    A_ = np.kron(a, I_a); Sp = np.kron(np.eye(nm + 1), sp); Sm = Sp.conj().T
    return A_, Sp, Sm, A_ @ Sp + A_.conj().T @ Sm
nm = 6
As, Sps, Sms, Ds = small_jc(nm)
dm = Ds.shape[0]
def lsup(Hm, Ls, I=None):
    I = np.eye(Hm.shape[0])
    S = -1j * (np.kron(I, Hm) - np.kron(Hm.T, I))
    for L in Ls:
        LdL = L.conj().T @ L
        S += np.kron(L.conj(), L) - 0.5 * np.kron(I, LdL) - 0.5 * np.kron(LdL.T, I)
    return S
kap, gam = 0.05, 0.1
Ls = [np.sqrt(kap) * As, np.sqrt(gam) * Sms]
Lp = lsup(Ds, Ls); Lm = lsup(-Ds, Ls); Dsup = lsup(0 * Ds, Ls)
comm = Lp @ Lm - Lm @ Lp
print("(2) ||[L+,L-]|| =", np.linalg.norm(comm, 2), "(nonzero: no commutation assumed)")
print("(2) ||(L+ + L-)/2 - D|| =", np.linalg.norm((Lp + Lm) / 2 - Dsup))
for T in [0.2, 0.1, 0.05, 0.025]:
    P = expm(T * Lp) @ expm(2 * T * Lm) @ expm(T * Lp)
    E = expm(4 * T * Dsup)
    print(f"(2) T={T:6.3f}  ||product - e^(4TD)|| = {np.linalg.norm(P - E, 2):.3e}   /T^3 = {np.linalg.norm(P - E, 2)/T**3:.3f}")
# leading BCH term for Strang e^{X/2} e^{Y} e^{X/2}, X = 2T L+, Y = 2T L-:
# log = X + Y - (1/24)[X,[X,Y]] + (1/12)[Y,[Y,X]] + O(T^5)
T = 0.025
X, Y = 2 * T * Lp, 2 * T * Lm
c3 = -(1/24) * (X @ (X @ Y - Y @ X) - (X @ Y - Y @ X) @ X) + (1/12) * (Y @ (Y @ X - X @ Y) - (Y @ X - X @ Y) @ Y)
P = expm(T * Lp) @ expm(2 * T * Lm) @ expm(T * Lp)
pred = expm(X + Y + c3)
print("(2) BCH third-order prediction residual at T=0.025:", np.linalg.norm(P - pred, 2))

# ---------- (7) displaced dissipator identity ----------
beta = 0.3 - 0.2j
rho = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim)); rho = rho @ rho.conj().T; rho /= np.trace(rho)
def Dx(X, r, k=1.0): return k * (X @ r @ X.conj().T - 0.5 * (X.conj().T @ X @ r + r @ X.conj().T @ X))
k = 0.7
lhs = Dx(A_ + beta * Id, rho, k)
rhs = Dx(A_, rho, k) + (k / 2) * ((np.conj(beta) * A_ - beta * A_.conj().T) @ rho - rho @ (np.conj(beta) * A_ - beta * A_.conj().T))
print("(7) D_{a+beta} = D_a + (kappa/2)[beta* a - beta a^+, .] residual:", np.abs(lhs - rhs).max())

# ---------- (11) horizon numbers ----------
g = 2 * np.pi * 50e6; kp = 2 * np.pi * 0.83e6; gm = 2 * np.pi * 3e6
N2 = np.sqrt(25 + 5 + 2 * 5 + 1)
tstar = 2 / (kp * N2 + gm)
print(f"(11) C = {g**2/(kp*gm):.0f}, <N^2>^1/2 = {N2:.2f}, t* = {tstar*1e9:.1f} ns, g t* = {g*tstar:.1f}, 1/gamma = {1e9/gm:.1f} ns")
for F in [0.99, 0.9, 0.5]:
    print(f"(11) bound guarantees F >= {F} for t <= {(1-np.sqrt(F)):.4f} t* = {(1-np.sqrt(F))*tstar*1e9:.2f} ns")
print("(11) bound at t = 0.99 t*:", (1 - 0.99)**2)

# ---------- (12) round budget of unencoded storage ----------
# F_store >= [1 - R tau/t*]^2  =>  R <= (1-sqrt(F0)) t*/tau
for tau_ns in [0.5]:
    for F0 in [0.9]:
        Rmax = (1 - np.sqrt(F0)) * tstar / (tau_ns * 1e-9)
        print(f"(12) tau={tau_ns} ns, F0={F0}: R_max = {Rmax:.2f}")
