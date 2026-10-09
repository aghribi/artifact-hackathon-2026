"""Rough finite-element solution of the 1D heat equation, for the "problem" slide of Part I.

    d_t u = D d_x^2 u  on [0, 1] x [0, T],  u(0, t) = u(1, t) = 0,  u(x, 0) = u0(x)

P1 elements on a deliberately coarse mesh (12 elements), implicit Euler in time.
Writes figures/fig_I_fem_heat.pdf.
"""
import matplotlib.pyplot as plt
import numpy as np
from skfem import Basis, BilinearForm, ElementLineP1, MeshLine, asm, condense, solve
from skfem.helpers import dot, grad

D, T, N_ELEM, N_STEPS = 0.05, 1.0, 12, 20


def u0(x):
    return np.sin(np.pi * x) + 0.5 * np.sin(3 * np.pi * x)


@BilinearForm
def mass(u, v, _):
    return u * v


@BilinearForm
def laplace(u, v, _):
    return dot(grad(u), grad(v))


mesh = MeshLine(np.linspace(0, 1, N_ELEM + 1))
basis = Basis(mesh, ElementLineP1())
M, K = asm(mass, basis), asm(laplace, basis)
x = mesh.p[0]
order = np.argsort(x)
boundary = mesh.boundary_nodes()

dt = T / N_STEPS
A = M + dt * D * K
u = u0(x)
history = [u[order]]
for _ in range(N_STEPS):
    u = solve(*condense(A, M @ u, D=boundary))
    history.append(u[order])
U = np.array(history)                      # (time, space)
t = np.linspace(0, T, N_STEPS + 1)
xs = x[order]

exact = lambda x, t: (np.exp(-D * np.pi**2 * t) * np.sin(np.pi * x)
                      + 0.5 * np.exp(-9 * D * np.pi**2 * t) * np.sin(3 * np.pi * x))
X, Tg = np.meshgrid(xs, t)
print(f"max error of the rough FEM solution: {np.abs(U - exact(X, Tg)).max():.3f}")

plt.rcParams.update({"font.size": 7})
fig, ax = plt.subplots(figsize=(2.3, 1.75))
pc = ax.pcolormesh(xs, t, U, shading="gouraud", cmap="viridis")
for xn in xs:                                 # element boundaries
    ax.axvline(xn, color="white", lw=0.35, alpha=0.6)
for tn in t:                                  # time steps
    ax.axhline(tn, color="white", lw=0.25, alpha=0.35)
ax.set_xlabel("$x$")
ax.set_ylabel("$t$")
ax.set_title(f"finite elements: {N_ELEM} elements, {N_STEPS} time steps", fontsize=6.5)
fig.colorbar(pc, ax=ax, label="$u(x,t)$", fraction=0.05, pad=0.03)
fig.tight_layout()
fig.savefig("figures/fig_I_fem_heat.pdf")
