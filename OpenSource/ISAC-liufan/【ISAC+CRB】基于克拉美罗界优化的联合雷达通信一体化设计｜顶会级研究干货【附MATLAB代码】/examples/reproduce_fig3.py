"""
Reproduce Fig. 3: Beampattern comparison
==========================================

Reproduces beampatterns for K=4 users, SINR=15dB:
- Proposed CRB-Min Design (SDR)
- Beampattern Approximation Design 1 (benchmark)
- Beampattern Approximation Design 2 (benchmark)

Paper: "Cramer-Rao Bound Optimization for Joint Radar-Communication Design"
       Fan Liu et al., IEEE TSP 2022
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model import SystemParams
from src.solver import MultiUserSDRSolver, BeampatternApproxSolver
from src.metrics import compute_beampattern

np.random.seed(42)

# Parameters
params = SystemParams(N_t=16, N_p=20, K=4, L=30, P_dBm=30.0, sigma2_dBm=0.0, gamma_dB=15.0)

# Generate random channels for K users
user_angles = np.array([20, -30, 50, -60])
from src.model import ULASteeringVector
ula = ULASteeringVector(params.N_t)
H = np.zeros((params.N_t, params.K), dtype=complex)
for k, ang in enumerate(user_angles):
    H[:, k] = ula.compute(ang)

print(f"Parameters: N_t={params.N_t}, K={params.K}, SINR={params.gamma_dB}dB")
print(f"User angles: {user_angles}°")

# ─── Compute beamformers ────────────────────────────────────────
# Proposed CRB-Min via SDR
sdr = MultiUserSDRSolver(params)
W_proposed, info = sdr.solve(H, theta_target=0)
print(f"SDR status: {info['cvxpy_status']}")

# Benchmark methods
approx = BeampatternApproxSolver(params)
W_design1 = approx.design1(theta_target=0)
W_design2 = approx.design2(H, theta_target=0)

# ─── Compute beampatterns ────────────────────────────────────────
angles = np.linspace(-90, 90, 361)
_, bp_proposed = compute_beampattern(W_proposed, angles=angles)
_, bp_design1 = compute_beampattern(W_design1, angles=angles)
_, bp_design2 = compute_beampattern(W_design2, angles=angles)

# ─── Plot ────────────────────────────────────────────────────────
plt.figure(figsize=(10, 6))
plt.plot(angles, bp_design1, "b--", lw=1.5, label="Beampattern Approx. Design 1")
plt.plot(angles, bp_design2, "g-.", lw=1.5, label="Beampattern Approx. Design 2")
plt.plot(angles, bp_proposed, "r-", lw=2.5, label="Proposed CRB-Min Design")
plt.axvline(x=0, color="k", ls=":", alpha=0.5, label="Target θ=0°")
plt.xlabel("Angle (degrees)", fontsize=12)
plt.ylabel("Normalized Beampattern (dB)", fontsize=12)
plt.title("Fig. 3 — Beampatterns (K=4, SINR=15dB)", fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.xlim(-90, 90)
plt.ylim(-40, 5)
plt.tight_layout()

out_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "fig3_reproduction.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out_path}")

# ─── Verify trends ───────────────────────────────────────────────
idx_0 = np.argmin(np.abs(angles))
print(f"\nBeampattern @ 0° (dB):")
print(f"  Design 1    : {bp_design1[idx_0]:.1f}")
print(f"  Design 2    : {bp_design2[idx_0]:.1f}")
print(f"  Proposed     : {bp_proposed[idx_0]:.1f}")
print(f"\n✅ Proposed has highest gain at target direction" if bp_proposed[idx_0] >= max(bp_design1[idx_0], bp_design2[idx_0]) else "❌")
