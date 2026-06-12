import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from qutip import wigner
import warnings
warnings.filterwarnings("ignore")

print("QuTiP version:", qt.__version__)

# ========================= STABLE PARAMETERS =========================
g = 2 * np.pi * 50e6
Omega0 = 0.25 * g
N_fock = 6                     # Significantly reduced
t_target = 5e-6                # Shorter time (still shows revival effect)

alpha0 = np.sqrt(5.0)
psi0 = qt.tensor(qt.coherent(N_fock, alpha0), qt.basis(2, 0))

a = qt.tensor(qt.destroy(N_fock), qt.qeye(2))
sm = qt.tensor(qt.qeye(N_fock), qt.destroy(2))

def H_darboux(t, args=None):
    H_jc = g * (a.dag() * sm + a * sm.dag())
    drive = Omega0 * np.cos(g * t) * (a + a.dag())
    return H_jc + drive

H_bare = g * (a.dag() * sm + a * sm.dag())

opts = {
    'nsteps': 500000,      # Very high
    'atol': 1e-9,
    'rtol': 1e-9,
    'method': 'bdf'
}

# ====================== EVOLUTION ======================
print("Evolving systems... (this may take 20-60s)")

result_d = qt.mesolve(H_darboux, psi0, [0, t_target], [], options=opts)
rho_d = result_d.states[-1].ptrace(0)

result_b = qt.mesolve(H_bare, psi0, [0, t_target], [], options=opts)
rho_b = result_b.states[-1].ptrace(0)

# ====================== WIGNER ======================
print("Computing Wigner functions...")
x = np.linspace(-6, 6, 80)
y = x
W_b = wigner(rho_b, x, y)
W_d = wigner(rho_d, x, y)

# Plot
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

im1 = axs[0].contourf(x, y, W_b, 60, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
axs[0].set_title('Bare Vacuum-Rabi\n(Collapsed at t=5 µs)')
axs[0].set_xlabel('Re(α)')
axs[0].set_ylabel('Im(α)')
plt.colorbar(im1, ax=axs[0])

im2 = axs[1].contourf(x, y, W_d, 60, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
axs[1].set_title('Darboux-Engineered\n(Refocused at t=5 µs)')
axs[1].set_xlabel('Re(α)')
axs[1].set_ylabel('Im(α)')
plt.colorbar(im2, ax=axs[1])

plt.tight_layout()
plt.savefig('wigner_darboux_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Wigner plots saved!")
