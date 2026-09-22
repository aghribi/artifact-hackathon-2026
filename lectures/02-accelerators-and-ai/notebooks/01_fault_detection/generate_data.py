"""
Synthetic RF cavity IQ waveform generator.

Produces four classes:
  0 — normal      trapezoidal fill / flat-top / decay, small noise
  1 — quench      sudden amplitude collapse at random time
  2 — detuning    cavity goes off-resonance → Q signal becomes nonzero
  3 — multipacting amplitude oscillations during flat-top

Usage:
    python generate_data.py           # writes ../../data/rf_waveforms.npz
    from generate_data import generate_dataset
"""

import numpy as np


# ── Pulse envelope (trapezoidal) ───────────────────────────────────────────────

def _envelope(t: np.ndarray, t_rise: float = 0.10, t_fall: float = 0.85) -> np.ndarray:
    """Normalised trapezoidal pulse envelope on t ∈ [0, 1]."""
    return np.piecewise(
        t.astype(float),
        [t < t_rise, (t >= t_rise) & (t <= t_fall), t > t_fall],
        [
            lambda t: t / t_rise,
            1.0,
            lambda t: (1.0 - t) / (1.0 - t_fall),
        ],
    )


# ── Fault-class generators ────────────────────────────────────────────────────

def generate_normal(n_points: int = 256, noise: float = 0.025, seed: int = 0) -> np.ndarray:
    """Normal RF pulse: trapezoidal I, near-zero Q."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n_points)
    I = _envelope(t) + rng.normal(0, noise, n_points)
    Q = rng.normal(0, noise, n_points)
    return np.stack([I, Q]).astype(np.float32)           # (2, n_points)


def generate_quench(n_points: int = 256, noise: float = 0.025, seed: int = 1) -> np.ndarray:
    """Quench: sudden amplitude collapse at a random time in the flat-top."""
    rng = np.random.default_rng(seed)
    wf = generate_normal(n_points, noise, seed)
    t_q = rng.integers(int(0.30 * n_points), int(0.75 * n_points))
    wf[:, t_q:] = rng.normal(0, noise / 2, (2, n_points - t_q))
    return wf


def generate_detuning(n_points: int = 256, noise: float = 0.025, seed: int = 2) -> np.ndarray:
    """Detuning: off-resonance operation → oscillation in IQ plane."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n_points)
    env = _envelope(t)
    freq = rng.uniform(3.0, 8.0)                          # detuning frequency (a.u.)
    I = env * np.cos(2 * np.pi * freq * t) + rng.normal(0, noise, n_points)
    Q = env * np.sin(2 * np.pi * freq * t) + rng.normal(0, noise, n_points)
    return np.stack([I, Q]).astype(np.float32)


def generate_multipacting(n_points: int = 256, noise: float = 0.025, seed: int = 3) -> np.ndarray:
    """Multipacting: amplitude oscillations during flat-top."""
    rng = np.random.default_rng(seed)
    wf = generate_normal(n_points, noise, seed)
    s = int(0.15 * n_points)
    e = int(0.80 * n_points)
    t_mp = np.linspace(0, 1, e - s)
    freq = rng.uniform(6.0, 14.0)
    amp  = rng.uniform(0.10, 0.25)
    wf[0, s:e] += amp * np.sin(2 * np.pi * freq * t_mp)
    wf[1, s:e] += amp * np.cos(2 * np.pi * freq * t_mp)
    return wf


# ── Dataset builder ───────────────────────────────────────────────────────────

GENERATORS = [generate_normal, generate_quench, generate_detuning, generate_multipacting]
CLASS_NAMES = ["normal", "quench", "detuning", "multipacting"]


def generate_dataset(
    n_per_class: int = 1_000,
    n_points: int = 256,
    noise: float = 0.025,
    rng_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Returns
    -------
    X : float32 array, shape (4 * n_per_class, 2, n_points)
    y : int64 array,   shape (4 * n_per_class,)
    class_names : list of str
    """
    X, y = [], []
    for label, gen in enumerate(GENERATORS):
        for i in range(n_per_class):
            X.append(gen(n_points=n_points, noise=noise, seed=label * n_per_class + i))
            y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    rng = np.random.default_rng(rng_seed)
    idx = rng.permutation(len(X))
    return X[idx], y[idx], CLASS_NAMES


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pathlib, sys

    out = pathlib.Path(__file__).parent.parent.parent / "data" / "rf_waveforms.npz"
    out.parent.mkdir(parents=True, exist_ok=True)

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000
    X, y, names = generate_dataset(n_per_class=n)
    np.savez(out, X=X, y=y, class_names=names)
    print(f"Saved {X.shape} dataset → {out}")
