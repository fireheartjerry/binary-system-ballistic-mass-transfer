#!/usr/bin/env python3
"""
Ballistic Mass Transfer in Binary Star Systems
================================================
Author  : Jerry (Yuze) Li
Advisor : Dr. Alexander Mushtukov, University of Oxford

Paper   : "Ballistic Mass Transfer Necessarily Contracts Orbits
           in Binary Star Systems" (RNAAS)

Repository: https://github.com/fireheartjerry/binary-system-ballistic-mass-transfer

Velocity-Verlet integration of a two-body gravitational system with
conservative ballistic mass transfer.  The ballistic prescription
transfers mass at the donor's instantaneous velocity, conserving linear
momentum but *not* orbital angular momentum.  The key analytic result,

    L_f / L_i  =  1 - f        (Eq. 4 of the paper)

is verified numerically for three mass ratios (q_0 = 0.4, 1.0, 2.0).

Output : ``figure.pdf`` — normalised angular momentum versus time.

Usage
-----
    python simulate.py
    python simulate.py --outdir ./figures
    python simulate.py --skip-validation

Reproducibility
---------------
Deterministic (no stochastic components); identical results across runs.
Software versions are logged to stdout at runtime.

Dependencies : numpy >=1.20, matplotlib >=3.4
License      : MIT
"""

from __future__ import annotations

import argparse
import platform
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

AU: float = 1.496e11
M_SUN: float = 1.9891e30
G: float = 6.67430e-11
YEAR: float = 365.25 * 24 * 3600

SEPARATION_AU: float = 5.0
MASS_TRANSFER_FRAC: float = 0.15
N_ORBITS: int = 5
DT: float = 1e3
SAMPLE_INTERVAL: int = 500

CASES: list[dict] = [
    {"m1": 0.8, "m2": 2.0, "q": 0.4, "label": r"$q_0 = 0.4$", "color": "#27ae60"},
    {"m1": 1.5, "m2": 1.5, "q": 1.0, "label": r"$q_0 = 1.0$", "color": "#2980b9"},
    {"m1": 2.4, "m2": 1.2, "q": 2.0, "label": r"$q_0 = 2.0$", "color": "#c0392b"},
]

_PLOT_PARAMS: dict = {
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "axes.linewidth": 0.8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}
SOLID_LW: float = 2.2
DASHED_LW: float = 1.5
GRID_ALPHA: float = 0.25
GRID_LW: float = 0.6


@dataclass
class Body:
    """Point-mass stellar body with 2-D kinematics."""
    mass: float
    position: np.ndarray
    velocity: np.ndarray

    def __post_init__(self) -> None:
        self.mass = float(self.mass)
        self.position = np.asarray(self.position, dtype=float)
        self.velocity = np.asarray(self.velocity, dtype=float)
        if self.mass <= 0.0:
            raise ValueError("Body mass must be positive.")
        if self.position.shape != (2,) or self.velocity.shape != (2,):
            raise ValueError("Position and velocity must be 2-D vectors.")


class BinarySystem:
    """
    Two-body gravitational system with ballistic mass transfer.

    Integration : Velocity Verlet (symplectic, 2nd-order)
    Prescription: transferred δm carries the donor's instantaneous velocity.
                  Linear momentum is conserved; orbital angular momentum is NOT.
    """

    def __init__(self, donor: Body, accretor: Body) -> None:
        self.donor = donor
        self.accretor = accretor

    def _gravity(self) -> Tuple[np.ndarray, float]:
        r_vec = self.accretor.position - self.donor.position
        r = np.linalg.norm(r_vec)
        force = G * self.donor.mass * self.accretor.mass / r**2 * (r_vec / r)
        return force, r

    def angular_momentum(self) -> float:
        def _lz(body: Body) -> float:
            return body.mass * (
                body.position[0] * body.velocity[1]
                - body.position[1] * body.velocity[0]
            )
        return _lz(self.donor) + _lz(self.accretor)

    def _transfer_mass(self, dm: float) -> None:
        if dm <= 0.0:
            return
        p_accretor = self.accretor.mass * self.accretor.velocity
        p_transfer = dm * self.donor.velocity
        self.donor.mass -= dm
        self.accretor.mass += dm
        self.accretor.velocity = (p_accretor + p_transfer) / self.accretor.mass

    def integrate(
        self,
        total_time: float,
        dt: float,
        mass_transfer_fraction: float,
        sample_interval: int = 500,
    ) -> Dict[str, np.ndarray]:
        """Velocity-Verlet integration. Returns time, angular_momentum, mass_transferred_frac."""
        steps = int(total_time / dt)
        m1_init = self.donor.mass
        dm_per_step = mass_transfer_fraction * m1_init / steps if mass_transfer_fraction > 0 else 0.0
        mass_transferred = 0.0

        max_samples = steps // sample_interval + 2
        t_out = np.empty(max_samples)
        L_out = np.empty(max_samples)
        f_out = np.empty(max_samples)
        idx = 0

        def _record(t: float) -> None:
            nonlocal idx
            t_out[idx] = t
            L_out[idx] = self.angular_momentum()
            f_out[idx] = mass_transferred / m1_init
            idx += 1

        force, _ = self._gravity()
        a1 = force / self.donor.mass
        a2 = -force / self.accretor.mass
        _record(0.0)

        for step in range(1, steps + 1):
            self.donor.position += self.donor.velocity * dt + 0.5 * a1 * dt**2
            self.accretor.position += self.accretor.velocity * dt + 0.5 * a2 * dt**2

            force_new, _ = self._gravity()
            a1_new = force_new / self.donor.mass
            a2_new = -force_new / self.accretor.mass

            self.donor.velocity += 0.5 * (a1 + a1_new) * dt
            self.accretor.velocity += 0.5 * (a2 + a2_new) * dt
            a1, a2 = a1_new, a2_new

            if dm_per_step > 0:
                self._transfer_mass(dm_per_step)
                mass_transferred += dm_per_step
                force, _ = self._gravity()
                a1 = force / self.donor.mass
                a2 = -force / self.accretor.mass

            if step % sample_interval == 0 or step == steps:
                _record(step * dt)

        return {
            "time": t_out[:idx],
            "angular_momentum": L_out[:idx],
            "mass_transferred_frac": f_out[:idx],
        }


def setup_circular_binary(
    m1_solar: float, m2_solar: float, separation_au: float
) -> Tuple[BinarySystem, float]:
    """Create a circular Keplerian binary. Returns (system, T_orb)."""
    m1, m2 = m1_solar * M_SUN, m2_solar * M_SUN
    M = m1 + m2
    a = separation_au * AU

    r1 = (m2 / M) * a
    r2 = (m1 / M) * a
    omega = np.sqrt(G * M / a**3)

    donor = Body(m1, np.array([-r1, 0.0]), np.array([0.0, omega * r1]))
    accretor = Body(m2, np.array([r2, 0.0]), np.array([0.0, -omega * r2]))

    return BinarySystem(donor, accretor), 2 * np.pi / omega


def validate() -> None:
    """Verify conservation laws and the L_f/L_i = 1 − f relation."""
    print("Validation")
    print("-" * 50)

    sys0, T0 = setup_circular_binary(1.0, 1.0, 5.0)
    L_i = sys0.angular_momentum()
    data0 = sys0.integrate(10 * T0, DT, mass_transfer_fraction=0.0, sample_interval=1000)
    dL = abs(data0["angular_momentum"][-1] - L_i) / abs(L_i)
    print(f"  Conservation (no transfer): dL/L = {dL:.2e}  [{'PASS' if dL < 1e-7 else 'FAIL'}]")

    sys1, T1 = setup_circular_binary(1.5, 1.5, 5.0)
    data1 = sys1.integrate(5 * T1, DT, mass_transfer_fraction=0.15, sample_interval=500)
    ratio = data1["angular_momentum"][-1] / data1["angular_momentum"][0]
    err = abs(ratio - 0.85)
    print(f"  L_f/L_i vs 1-f: {ratio:.4f} (expected 0.8500, err {err:.4f})  [{'PASS' if err < 0.01 else 'FAIL'}]")
    print()


def make_figure(results: Dict[float, Dict[str, np.ndarray]], outdir: Path) -> Path:
    """Produce Figure 1: normalised L/L_0 versus mass transferred fraction."""
    matplotlib.rcParams.update(_PLOT_PARAMS)
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for case in CASES:
        d = results[case["q"]]
        ax.plot(
            d["mass_transferred_frac"] * 100,
            d["angular_momentum"] / d["angular_momentum"][0],
            color=case["color"], lw=SOLID_LW, label=case["label"],
        )

    x_ref = np.linspace(0, 15, 100)
    ax.plot(x_ref, 1 - x_ref / 100, "k--", lw=DASHED_LW, alpha=0.6, label=r"$L/L_0 = 1 - f$")

    ax.set_xlabel(r"Mass Transferred $\Delta M / M_{1,\mathrm{init}}$ (\%)")
    ax.set_ylabel(r"Normalized Angular Momentum $L/L_0$")
    ax.set_xlim(0, 15)
    ax.set_ylim(0.82, 1.02)
    ax.set_title("(a) Angular Momentum Loss")
    ax.legend(loc="lower left", frameon=False)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, which="major", alpha=GRID_ALPHA, lw=GRID_LW)
    fig.tight_layout()

    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "figure.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def print_summary(results: Dict[float, Dict[str, np.ndarray]]) -> None:
    print("Results")
    print("-" * 50)
    print(f"  {'q0':<6} {'L_f/L_i':>10} {'1 - f':>10} {'error':>10}")
    for case in CASES:
        d = results[case["q"]]
        ratio = d["angular_momentum"][-1] / d["angular_momentum"][0]
        expected = 1.0 - MASS_TRANSFER_FRAC
        print(f"  {case['q']:<6.1f} {ratio:>10.4f} {expected:>10.4f} {abs(ratio - expected):>10.4f}")
    print()


def log_environment() -> None:
    print("Environment")
    print("-" * 50)
    print(f"  Python     : {sys.version.split()[0]}")
    print(f"  NumPy      : {np.__version__}")
    print(f"  Matplotlib : {matplotlib.__version__}")
    print(f"  Platform   : {platform.platform()}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ballistic mass transfer simulation (Li, RNAAS).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Produces figure.pdf — normalised angular momentum vs time for
            three mass ratios, verifying L_f/L_i = 1 - f.
        """),
    )
    parser.add_argument("--outdir", type=str, default=".", help="Output directory (default: cwd)")
    parser.add_argument("--skip-validation", action="store_true", help="Skip validation tests")
    args = parser.parse_args()

    print("=" * 58)
    print("  Ballistic Mass Transfer — Binary Star Simulation")
    print("  Jerry (Yuze) Li  |  Advisor: Dr. A. Mushtukov")
    print("=" * 58)
    print()

    log_environment()

    if not args.skip_validation:
        validate()

    print("Simulation")
    print("-" * 50)
    results: Dict[float, Dict[str, np.ndarray]] = {}

    for case in CASES:
        binary, T_orb = setup_circular_binary(case["m1"], case["m2"], SEPARATION_AU)
        total_time = N_ORBITS * T_orb
        print(
            f"  q0={case['q']:.1f}  M1={case['m1']}  M2={case['m2']}  "
            f"T_orb={T_orb / YEAR:.2f} yr  steps={int(total_time / DT):,}"
        )
        results[case["q"]] = binary.integrate(
            total_time, DT, MASS_TRANSFER_FRAC, sample_interval=SAMPLE_INTERVAL
        )

    print()
    print_summary(results)

    outdir = Path(args.outdir)
    fig_path = make_figure(results, outdir)
    print(f"Figure saved: {fig_path.resolve()}")
    print()
    print("=" * 58)
    print("  Done.")
    print("=" * 58)


if __name__ == "__main__":
    main()