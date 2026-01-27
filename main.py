"""
Binary Star Mass Transfer Simulation
====================================
Author: Jerry (Yuze) Li
Research Supervisor: Dr. Alexander Mushtukov, University of Oxford

This module accompanies the paper:
"Evolution of Orbital Parameters in Binary Star Systems Due to Mass Transfer:
 The Role of Angular Momentum Transport".

Model summary:
- Two-body gravitational dynamics integrated with velocity Verlet (2nd order).
- Conservative mass transfer using a ballistic prescription: transferred mass
  carries the donor's instantaneous velocity.
- Diagnostics: orbital separation, angular momentum, energy, and masses.

All calculations use SI units unless noted. User-facing inputs for masses are in
solar masses and distances in astronomical units.

Key result: ballistic transfer conserves linear momentum but not angular
momentum, leading to orbital contraction for all mass ratios. The fractional
angular momentum change is approximately dL/L ~ -f, where f is the fraction of
donor mass transferred.
"""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import AutoMinorLocator
from matplotlib.lines import Line2D

from plot_style import apply_plot_style
from progress_utils import build_progress, progress_stride

# =============================================================================
# Physical Constants
# =============================================================================
AU = 1.496e11       # Astronomical unit [m]
SUN = 1.9891e30     # Solar mass [kg]
G = 6.67430e-11     # Gravitational constant [m^3 kg^-1 s^-2]
YEAR = 365.25 * 24 * 3600  # Year [s]

# =============================================================================
# Publication-quality plot settings
# =============================================================================
apply_plot_style()
SOLID_LINEWIDTH = 2.2
DASHED_LINEWIDTH = 1.5
GRID_ALPHA = 0.25
GRID_LINEWIDTH = 0.6

# =============================================================================
# Core Classes
# =============================================================================

@dataclass
class Body:
    """
    Represents a point-mass stellar body.

    Attributes:
        mass: Mass in kg
        position: 2D position vector [x, y] in m
        velocity: 2D velocity vector [vx, vy] in m/s
    """

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
            raise ValueError("Position and velocity must be length-2 vectors.")


class BinarySystem:
    """
    Simulates a binary star system with mass transfer.
    
    Implements:
    - Gravitational two-body dynamics
    - Velocity Verlet integration (symplectic, 2nd order)
    - Ballistic mass transfer prescription
    
    The ballistic prescription transfers mass from body1 (donor) to body2 (accretor),
    where the transferred mass carries the donor's instantaneous velocity.
    Linear momentum is conserved; angular momentum is NOT conserved.
    """
    
    def __init__(self, body1: Body, body2: Body) -> None:
        """
        Initialize binary system.
        
        Args:
            body1: Body object for the donor (M1, loses mass)
            body2: Body object for the accretor (M2, gains mass)
        """
        self.body1 = body1  # Donor
        self.body2 = body2  # Accretor
    
    def compute_gravitational_force(self) -> Tuple[np.ndarray, float]:
        """
        Compute gravitational force on body1 due to body2.
        
        Returns:
            force: Force vector on body1 (force on body2 is -force)
            separation: Scalar distance between bodies
        """
        r_vec = self.body2.position - self.body1.position
        r = np.linalg.norm(r_vec)
        if r == 0.0:
            raise ValueError("Zero separation between bodies is not physical.")
        force_magnitude = G * self.body1.mass * self.body2.mass / r**2
        force = force_magnitude * (r_vec / r)
        return force, r
    
    def compute_angular_momentum(self) -> float:
        """
        Compute total orbital angular momentum about the center of mass.
        
        Returns:
            L: Total angular momentum (scalar, z-component for 2D motion)
        """
        L1 = self.body1.mass * (
            self.body1.position[0] * self.body1.velocity[1] -
            self.body1.position[1] * self.body1.velocity[0]
        )
        L2 = self.body2.mass * (
            self.body2.position[0] * self.body2.velocity[1] -
            self.body2.position[1] * self.body2.velocity[0]
        )
        return L1 + L2
    
    def compute_energy(self) -> float:
        """
        Compute total mechanical energy.
        
        Returns:
            E: Total energy (kinetic + potential)
        """
        r = np.linalg.norm(self.body1.position - self.body2.position)
        KE = (0.5 * self.body1.mass * np.dot(self.body1.velocity, self.body1.velocity) +
              0.5 * self.body2.mass * np.dot(self.body2.velocity, self.body2.velocity))
        PE = -G * self.body1.mass * self.body2.mass / r
        return KE + PE
    
    def transfer_mass(self, dm: float) -> None:
        """
        Transfer mass dm from body1 (donor) to body2 (accretor) using
        the ballistic prescription.
        
        The transferred mass carries the donor's velocity. Linear momentum
        is conserved; angular momentum is NOT conserved.
        
        This implements Equation (18) from the paper:
        v2_new = (M2 * v2 + dm * v1) / (M2 + dm)
        
        Args:
            dm: Mass to transfer [kg]. Must be positive and less than donor mass.
        """
        if dm <= 0.0:
            return
        if dm >= self.body1.mass:
            raise ValueError("Mass transfer exceeds or equals donor mass.")
        
        # Store momentum of accretor before transfer
        p2 = self.body2.mass * self.body2.velocity
        
        # Momentum carried by transferred mass (at donor's velocity)
        p_transfer = dm * self.body1.velocity
        
        # Update masses: donor loses, accretor gains
        self.body1.mass -= dm
        self.body2.mass += dm
        
        # Update accretor velocity to conserve linear momentum
        self.body2.velocity = (p2 + p_transfer) / self.body2.mass
    
    def simulate(
        self,
        total_time: float,
        dt: float,
        mass_transfer_fraction: float,
        sample_interval: int = 100,
        show_progress: bool = True,
        progress_label: str = "Simulating",
    ) -> Dict[str, np.ndarray]:
        """
        Run the simulation using Velocity Verlet integration.
        
        Args:
            total_time: Total simulation time [s]
            dt: Timestep [s]
            mass_transfer_fraction: Fraction of initial donor mass to transfer,
                with 0 <= f < 1 to keep donor mass positive.
            sample_interval: Record diagnostics every N steps (final step included)
            show_progress: Whether to display a progress bar
            progress_label: Label used in the progress bar
            
        Returns:
            Dictionary containing time series of:
            - time: Time values [s]
            - separation: Orbital separation [m]
            - angular_momentum: Total L [kg m^2/s]
            - energy: Total E [J]
            - mass1, mass2: Component masses [kg]
            - mass_transferred_frac: Cumulative mass transferred / M1_init
        """
        if dt <= 0.0:
            raise ValueError("dt must be positive.")
        if total_time <= 0.0:
            raise ValueError("total_time must be positive.")
        if sample_interval <= 0:
            raise ValueError("sample_interval must be positive.")
        if mass_transfer_fraction < 0.0:
            raise ValueError("mass_transfer_fraction must be >= 0.")
        if mass_transfer_fraction >= 1.0:
            raise ValueError("mass_transfer_fraction must be < 1 to keep donor mass positive.")

        steps = int(total_time / dt)
        if steps < 1:
            raise ValueError("total_time must be at least one time step.")

        data: Dict[str, List[float]] = {
            'time': [],
            'separation': [],
            'angular_momentum': [],
            'energy': [],
            'mass1': [],
            'mass2': [],
            'mass_transferred_frac': []
        }
        
        # Mass transfer setup
        # Body1 is the DONOR (loses mass), Body2 is the ACCRETOR (gains mass)
        m1_initial = self.body1.mass
        dm_total = mass_transfer_fraction * m1_initial
        dm = dm_total / steps if mass_transfer_fraction > 0.0 else 0.0
        
        # Initial force and acceleration
        force, _ = self.compute_gravitational_force()
        a1 = force / self.body1.mass
        a2 = -force / self.body2.mass
        
        mass_transferred = 0.0
        progress, progress_enabled = build_progress(show_progress)
        progress_every = progress_stride(steps)
        last_progress = 0

        def record_sample(time_value: float) -> None:
            data['time'].append(time_value)
            data['separation'].append(
                np.linalg.norm(self.body1.position - self.body2.position)
            )
            data['angular_momentum'].append(self.compute_angular_momentum())
            data['energy'].append(self.compute_energy())
            data['mass1'].append(self.body1.mass)
            data['mass2'].append(self.body2.mass)
            data['mass_transferred_frac'].append(mass_transferred / m1_initial)

        record_sample(0.0)

        with progress:
            task_id = progress.add_task(
                progress_label,
                total=steps,
                detail="mass 0.00%",
            )
            for step in range(1, steps + 1):
                # ----- Velocity Verlet Integration -----
                # Step 1: Update positions
                new_pos1 = (self.body1.position +
                           self.body1.velocity * dt +
                           0.5 * a1 * dt**2)
                new_pos2 = (self.body2.position +
                           self.body2.velocity * dt +
                           0.5 * a2 * dt**2)

                # Step 2: Compute new forces at new positions
                self.body1.position = new_pos1
                self.body2.position = new_pos2
                force_new, _ = self.compute_gravitational_force()
                a1_new = force_new / self.body1.mass
                a2_new = -force_new / self.body2.mass

                # Step 3: Update velocities
                self.body1.velocity = self.body1.velocity + 0.5 * (a1 + a1_new) * dt
                self.body2.velocity = self.body2.velocity + 0.5 * (a2 + a2_new) * dt

                # Update accelerations for next step
                a1, a2 = a1_new, a2_new

                # ----- Mass Transfer -----
                if dm > 0:
                    self.transfer_mass(dm)
                    mass_transferred += dm

                    # Recompute accelerations after mass change
                    force, _ = self.compute_gravitational_force()
                    a1 = force / self.body1.mass
                    a2 = -force / self.body2.mass

                # Record diagnostics at sample intervals and at the final step
                if step % sample_interval == 0 or step == steps:
                    record_sample(step * dt)

                if progress_enabled and (step - last_progress >= progress_every or step == steps):
                    detail = f"mass {mass_transferred / m1_initial:6.2%}"
                    progress.update(
                        task_id,
                        advance=step - last_progress,
                        detail=detail,
                    )
                    last_progress = step

        return {key: np.asarray(values, dtype=float) for key, values in data.items()}


# =============================================================================
# Simulation Setup Functions
# =============================================================================

def setup_circular_binary(
    m1_solar: float,
    m2_solar: float,
    separation_au: float,
) -> Tuple[BinarySystem, float]:
    """
    Initialize a binary system in a circular orbit.
    
    Args:
        m1_solar: Mass of body 1 (donor) in solar masses
        m2_solar: Mass of body 2 (accretor) in solar masses
        separation_au: Initial orbital separation in AU
        
    Returns:
        binary: BinarySystem object
        T_orb: Orbital period [s]
    """
    if m1_solar <= 0.0 or m2_solar <= 0.0:
        raise ValueError("Stellar masses must be positive.")
    if separation_au <= 0.0:
        raise ValueError("Initial separation must be positive.")

    m1 = m1_solar * SUN
    m2 = m2_solar * SUN
    M_total = m1 + m2
    a = separation_au * AU
    
    # Distances from center of mass
    r1 = (m2 / M_total) * a  # Donor distance from COM
    r2 = (m1 / M_total) * a  # Accretor distance from COM
    
    # Initial positions (COM at origin, bodies on x-axis)
    pos1 = np.array([-r1, 0.0])
    pos2 = np.array([r2, 0.0])
    
    # Circular orbit velocities (perpendicular to position)
    omega = np.sqrt(G * M_total / a**3)
    vel1 = np.array([0.0, omega * r1])   # Donor velocity (+y)
    vel2 = np.array([0.0, -omega * r2])  # Accretor velocity (-y)
    
    # Create bodies and binary system
    body1 = Body(m1, pos1, vel1)
    body2 = Body(m2, pos2, vel2)
    binary = BinarySystem(body1, body2)
    
    # Orbital period
    T_orb = 2 * np.pi * np.sqrt(a**3 / (G * M_total))
    
    return binary, T_orb


def compute_analytical_prediction(
    m1_init: float,
    m2_init: float,
    a_init: float,
    f_transfer: float,
) -> float:
    """
    Compute analytical prediction for final separation assuming
    angular momentum conservation.
    
    For conservative mass transfer with L conserved:
    a_final / a_init = (M1_init * M2_init / M1_final / M2_final)^2
    
    Args:
        m1_init, m2_init: Initial masses [solar masses]
        a_init: Initial separation [AU]
        f_transfer: Fraction of M1 transferred to M2, with 0 <= f < 1
        
    Returns:
        a_final: Predicted final separation [AU]
    """
    if m1_init <= 0.0 or m2_init <= 0.0:
        raise ValueError("Initial masses must be positive.")
    if a_init <= 0.0:
        raise ValueError("Initial separation must be positive.")
    if f_transfer < 0.0 or f_transfer >= 1.0:
        raise ValueError("f_transfer must be in the range [0, 1).")

    # Final masses after transfer
    dm = f_transfer * m1_init
    m1_final = m1_init - dm
    m2_final = m2_init + dm
    
    # Angular momentum conservation gives this scaling
    a_final = a_init * (m1_init * m2_init / m1_final / m2_final)**2
    
    return a_final


def compute_analytical_separation_series(
    mass1_series: np.ndarray,
    mass2_series: np.ndarray,
) -> np.ndarray:
    """
    Compute normalized separation series assuming angular momentum conservation.

    For conservative transfer with L conserved:
    a/a0 = (M1_0 * M2_0 / (M1 * M2))^2
    """
    if mass1_series.size == 0 or mass2_series.size == 0:
        raise ValueError("Mass series must be non-empty.")
    if mass1_series.shape != mass2_series.shape:
        raise ValueError("Mass series must have matching shapes.")
    if np.any(mass1_series <= 0.0) or np.any(mass2_series <= 0.0):
        raise ValueError("Mass series must be positive.")

    mu0 = mass1_series[0] * mass2_series[0]
    return (mu0 / (mass1_series * mass2_series))**2


# =============================================================================
# Main Simulation Runner
# =============================================================================

def run_simulation_suite() -> Tuple[Dict[float, Dict[str, np.ndarray]], List[Dict[str, object]]]:
    """
    Run the complete simulation suite as presented in the paper.
    
    Cases:
    A: q0 = 0.4 (M1=0.8, M2=2.0) - Classical prediction: expansion
    B: q0 = 1.0 (M1=1.5, M2=1.5) - Classical prediction: marginal
    C: q0 = 2.0 (M1=2.4, M2=1.2) - Classical prediction: contraction
    """
    
    print("=" * 70)
    print("Binary Star Mass Transfer Simulation")
    print("=" * 70)
    print("\nThis simulation accompanies the paper:")
    print("'Evolution of Orbital Parameters in Binary Star Systems Due to Mass Transfer'")
    print("=" * 70)
    
    # Define simulation cases
    cases = [
        {'m1': 0.8, 'm2': 2.0, 'q': 0.4, 'label': r'$q_0 = 0.4$', 'color': '#27ae60'},
        {'m1': 1.5, 'm2': 1.5, 'q': 1.0, 'label': r'$q_0 = 1.0$', 'color': '#2980b9'},
        {'m1': 2.4, 'm2': 1.2, 'q': 2.0, 'label': r'$q_0 = 2.0$', 'color': '#c0392b'},
    ]
    
    # Simulation parameters
    separation_au = 5.0      # Initial separation [AU]
    f_transfer = 0.15        # Mass transfer fraction (15%)
    n_orbits = 5             # Number of orbits to simulate
    dt = 1e3                 # Timestep [s]
    
    # Run simulations
    results = {}
    
    for case in cases:
        print(
            f"\n--- Case q0 = {case['q']}: M1 = {case['m1']} M_sun, "
            f"M2 = {case['m2']} M_sun ---"
        )
        
        # Setup binary
        binary, T_orb = setup_circular_binary(case['m1'], case['m2'], separation_au)
        total_time = n_orbits * T_orb
        
        # Compute expected number of steps
        n_steps = int(total_time / dt)
        print(f"  Orbital period: {T_orb/YEAR:.2f} years")
        print(f"  Total steps: {n_steps:,}")
        print(f"  Steps per orbit: {n_steps/n_orbits:,.0f}")
        
        # Run simulation
        data = binary.simulate(
            total_time,
            dt,
            f_transfer,
            sample_interval=500,
            progress_label=f"Simulating q0={case['q']}",
        )
        results[case['q']] = data
        
        # Report results
        a_init = data['separation'][0] / AU
        a_final = data['separation'][-1] / AU
        L_init = data['angular_momentum'][0]
        L_final = data['angular_momentum'][-1]
        
        a_theory = compute_analytical_prediction(case['m1'], case['m2'], separation_au, f_transfer)
        
        print(f"\n  Results:")
        print(f"    Initial separation: {a_init:.3f} AU")
        print(f"    Final separation:   {a_final:.3f} AU ({(a_final/a_init-1)*100:+.1f}%)")
        print(f"    Analytical (L conserved): {a_theory:.3f} AU ({(a_theory/a_init-1)*100:+.1f}%)")
        print(f"    Angular momentum change: {(L_final/L_init-1)*100:+.1f}%")
    
    return results, cases


# =============================================================================
# Plotting Functions
# =============================================================================

def save_figure(
    fig: Figure,
    output_path: Path,
    base_name: str,
) -> None:
    """
    Save a matplotlib figure as a vector PDF.
    """
    fig.savefig(output_path / f"{base_name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {base_name}")


def generate_figures(
    results: Dict[float, Dict[str, np.ndarray]],
    cases: List[Dict[str, object]],
    output_dir: str = '.',
) -> None:
    """
    Generate all publication figures.
    
    Args:
        results: Dictionary of simulation results keyed by q
        cases: List of case dictionaries with plotting info
        output_dir: Directory for saving figures
    """
    
    print("\n" + "=" * 70)
    print("Generating publication figures...")
    print("=" * 70)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ----- Figure 1: Separation vs Time -----
    fig1, ax = plt.subplots(figsize=(8, 5.5))
    
    for case in cases:
        q = case['q']
        t = results[q]['time'] / YEAR
        a = results[q]['separation'] / results[q]['separation'][0]
        ax.plot(t, a, color=case['color'], linewidth=SOLID_LINEWIDTH, label=case['label'])
    
    ax.axhline(1.0, color='gray', linestyle=':', linewidth=DASHED_LINEWIDTH, alpha=0.5)
    ax.set_xlabel('Time (years)')
    ax.set_ylabel(r'Normalized Separation $a/a_0$')
    ax.legend(loc='best', frameon=False)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, which='major', alpha=GRID_ALPHA, linewidth=GRID_LINEWIDTH)
    fig1.tight_layout()
    save_figure(fig1, output_path, 'fig2_normalized_separation')

    # ----- Figure 1b: Separation vs Time with Analytical Overlay -----
    fig1b, ax = plt.subplots(figsize=(8, 5.5))

    for case in cases:
        q = case['q']
        t = results[q]['time'] / YEAR
        a_ballistic = results[q]['separation'] / results[q]['separation'][0]
        a_theory = compute_analytical_separation_series(
            results[q]['mass1'],
            results[q]['mass2'],
        )
        ax.plot(
            t,
            a_theory,
            color=case['color'],
            linewidth=DASHED_LINEWIDTH,
            linestyle='--',
            alpha=0.6,
        )
        ax.plot(t, a_ballistic, color=case['color'], linewidth=SOLID_LINEWIDTH, label=case['label'])

    ax.axhline(1.0, color='gray', linestyle=':', linewidth=DASHED_LINEWIDTH, alpha=0.5)
    ax.set_xlabel('Time (years)')
    ax.set_ylabel(r'Normalized Separation $a/a_0$')

    q_handles, q_labels = ax.get_legend_handles_labels()
    style_handles = [
        Line2D([0], [0], color='black', linewidth=SOLID_LINEWIDTH, linestyle='-', label='Ballistic'),
        Line2D([0], [0], color='black', linewidth=DASHED_LINEWIDTH, linestyle='--', label='Analytical'),
    ]
    ax.legend(
        q_handles + style_handles,
        q_labels + [handle.get_label() for handle in style_handles],
        loc='upper left',
        frameon=False,
    )
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, which='major', alpha=GRID_ALPHA, linewidth=GRID_LINEWIDTH)
    fig1b.tight_layout()
    save_figure(fig1b, output_path, 'fig2_normalized_separation_overlay')
    
    # ----- Figure 2: Angular Momentum vs Time -----
    fig2, ax = plt.subplots(figsize=(8, 5.5))
    
    for case in cases:
        q = case['q']
        t = results[q]['time'] / YEAR
        L = results[q]['angular_momentum'] / results[q]['angular_momentum'][0]
        ax.plot(t, L, color=case['color'], linewidth=SOLID_LINEWIDTH, label=case['label'])
    
    ax.axhline(
        1.0,
        color='black',
        linestyle='--',
        linewidth=DASHED_LINEWIDTH,
        alpha=0.6,
        label='Conserved (theory)',
    )
    ax.set_xlabel('Time (years)')
    ax.set_ylabel(r'Normalized Angular Momentum $L/L_0$')
    ax.legend(loc='best', frameon=False)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, which='major', alpha=GRID_ALPHA, linewidth=GRID_LINEWIDTH)
    fig2.tight_layout()
    save_figure(fig2, output_path, 'fig3_angular_momentum')
    
    # ----- Figure 3: Energy vs Time -----
    fig3, ax = plt.subplots(figsize=(8, 5.5))
    
    for case in cases:
        q = case['q']
        t = results[q]['time'] / YEAR
        E = results[q]['energy'] / results[q]['energy'][0]
        ax.plot(t, E, color=case['color'], linewidth=SOLID_LINEWIDTH, label=case['label'])
    
    ax.axhline(1.0, color='gray', linestyle=':', linewidth=DASHED_LINEWIDTH, alpha=0.5)
    ax.set_xlabel('Time (years)')
    ax.set_ylabel(r'Normalized Energy $E/E_0$')
    ax.legend(loc='best', frameon=False)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, which='major', alpha=GRID_ALPHA, linewidth=GRID_LINEWIDTH)
    fig3.tight_layout()
    save_figure(fig3, output_path, 'fig4_energy')
    
    # ----- Figure 4: Mass Ratio vs Time -----
    fig4, ax = plt.subplots(figsize=(8, 5.5))
    
    for case in cases:
        q = case['q']
        t = results[q]['time'] / YEAR
        q_evol = results[q]['mass1'] / results[q]['mass2']
        ax.plot(t, q_evol, color=case['color'], linewidth=SOLID_LINEWIDTH, label=case['label'])
    
    ax.axhline(
        1.0,
        color='black',
        linestyle=':',
        linewidth=DASHED_LINEWIDTH,
        alpha=0.6,
        label=r'$q = 1$ (critical)',
    )
    ax.set_xlabel('Time (years)')
    ax.set_ylabel(r'Mass Ratio $q = M_1/M_2$')
    ax.legend(loc='best', frameon=False)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, which='major', alpha=GRID_ALPHA, linewidth=GRID_LINEWIDTH)
    fig4.tight_layout()
    save_figure(fig4, output_path, 'fig5_mass_ratio')
    
    # ----- Figure 5: L and a vs Mass Transferred (KEY FIGURE) -----
    fig5, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
    
    # Left panel: Angular momentum vs mass transferred
    for case in cases:
        q = case['q']
        x = results[q]['mass_transferred_frac'] * 100
        y = results[q]['angular_momentum'] / results[q]['angular_momentum'][0]
        ax1.plot(x, y, color=case['color'], linewidth=SOLID_LINEWIDTH, label=case['label'])
    
    # Reference line: L/L0 = 1 - f
    x_ref = np.linspace(0, 15, 100)
    ax1.plot(
        x_ref,
        1 - x_ref/100,
        'k--',
        linewidth=DASHED_LINEWIDTH,
        alpha=0.6,
        label=r'$L/L_0 = 1 - f$',
    )
    
    ax1.set_xlabel(r'Mass Transferred $\Delta M / M_{1,\mathrm{init}}$ (\%)')
    ax1.set_ylabel(r'Normalized Angular Momentum $L/L_0$')
    ax1.set_xlim(0, 15)
    ax1.set_ylim(0.82, 1.02)
    ax1.legend(loc='lower left', frameon=False)
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.grid(True, which='major', alpha=GRID_ALPHA, linewidth=GRID_LINEWIDTH)
    ax1.set_title('(a) Angular Momentum Loss')
    
    # Right panel: Separation vs mass transferred
    for case in cases:
        q = case['q']
        x = results[q]['mass_transferred_frac'] * 100
        y = results[q]['separation'] / results[q]['separation'][0]
        ax2.plot(x, y, color=case['color'], linewidth=SOLID_LINEWIDTH, label=case['label'])
    
    ax2.axhline(1.0, color='gray', linestyle=':', linewidth=DASHED_LINEWIDTH, alpha=0.5)
    ax2.set_xlabel(r'Mass Transferred $\Delta M / M_{1,\mathrm{init}}$ (\%)')
    ax2.set_ylabel(r'Normalized Separation $a/a_0$')
    ax2.set_xlim(0, 15)
    ax2.legend(loc='lower left', frameon=False)
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.grid(True, which='major', alpha=GRID_ALPHA, linewidth=GRID_LINEWIDTH)
    ax2.set_title('(b) Orbital Separation')
    
    fig5.tight_layout()
    save_figure(fig5, output_path, 'fig7_vs_mass_transferred')
    
    print("\nAll figures generated successfully!")


# =============================================================================
# Data Export / Import Helpers
# =============================================================================

def _q_to_key(q_value: float) -> str:
    """Create a filesystem-friendly key for a given q value."""
    return f"q{q_value}".replace('.', 'p')


def _key_to_q(key: str) -> float:
    """Recover q value from a serialized key."""
    return float(key[1:].replace('p', '.'))


def save_simulation_data(
    results: Dict[float, Dict[str, np.ndarray]],
    cases: List[Dict[str, object]],
    output_dir: str = '.',
    filename: str = 'simulation_data',
) -> Path:
    """
    Serialize simulation outputs for reuse without rerunning expensive runs.

    Writes two files in output_dir:
    - `<filename>.npz`: compressed NumPy archive with all time series
    - `<filename>_cases.json`: case metadata (masses, labels, colors)
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    flat: Dict[str, np.ndarray] = {}
    for q_value, series in results.items():
        prefix = _q_to_key(q_value)
        for name, arr in series.items():
            flat[f"{prefix}_{name}"] = np.asarray(arr)

    npz_path = out_path / f"{filename}.npz"
    np.savez_compressed(npz_path, **flat)

    cases_path = out_path / f"{filename}_cases.json"
    cases_path.write_text(json.dumps(cases, indent=2))

    print(f"Saved simulation data: {npz_path}")
    print(f"Saved case metadata:    {cases_path}")
    return npz_path


def load_simulation_data(
    output_dir: str = '.',
    filename: str = 'simulation_data',
) -> Tuple[Dict[float, Dict[str, np.ndarray]], List[Dict[str, object]]]:
    """
    Load previously saved simulation outputs and metadata.
    """
    out_path = Path(output_dir)
    npz_path = out_path / f"{filename}.npz"
    cases_path = out_path / f"{filename}_cases.json"

    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")
    if not cases_path.exists():
        raise FileNotFoundError(f"Case metadata file not found: {cases_path}")

    archive = np.load(npz_path)
    grouped: Dict[float, Dict[str, np.ndarray]] = defaultdict(dict)
    for key in archive.files:
        if '_' not in key:
            continue
        prefix, name = key.split('_', 1)
        q_value = _key_to_q(prefix)
        grouped[q_value][name] = archive[key]

    cases: List[Dict[str, object]] = json.loads(cases_path.read_text())
    return dict(grouped), cases


def print_summary_table(
    results: Dict[float, Dict[str, np.ndarray]],
    cases: List[Dict[str, object]],
) -> None:
    """Print the summary table from the paper."""
    
    print("\n" + "=" * 70)
    print("SUMMARY TABLE (Table 2 from paper)")
    print("=" * 70)
    print(
        f"{'Case':<6} {'q0':<6} {'a_theory':<12} {'a_sim':<12} "
        f"{'da/a0 (th)':<14} {'da/a0 (sim)':<14} {'dL/L0':<10}"
    )
    print("-" * 74)
    
    a_init = 5.0  # AU
    f = 0.15
    
    for i, case in enumerate(cases):
        q = case['q']
        
        # Get simulation results
        a_sim = results[q]['separation'][-1] / AU
        L_init = results[q]['angular_momentum'][0]
        L_final = results[q]['angular_momentum'][-1]
        
        # Compute analytical prediction
        a_theory = compute_analytical_prediction(case['m1'], case['m2'], a_init, f)
        
        # Compute percentage changes
        da_theory = (a_theory / a_init - 1) * 100
        da_sim = (a_sim / a_init - 1) * 100
        dL = (L_final / L_init - 1) * 100
        
        case_label = chr(ord('A') + i)
        print(f"{case_label:<6} {q:<6.1f} {a_theory:<12.2f} {a_sim:<12.2f} {da_theory:+<14.0f}% {da_sim:+<14.0f}% {dL:+<10.0f}%")
    
    print("=" * 70)
    print("\nKey result: dL/L0 ~= -15% for all cases, confirming dL/L ~= -f")
    print("This is NOT a coincidence - it follows from Equation (26) in the paper.")


# =============================================================================
# Validation Tests
# =============================================================================

def run_validation_tests() -> None:
    """
    Run validation tests to verify code correctness.
    """
    print("\n" + "=" * 70)
    print("VALIDATION TESTS")
    print("=" * 70)
    
    # Test 1: Energy conservation without mass transfer
    print("\nTest 1: Energy conservation (no mass transfer)")
    binary, T_orb = setup_circular_binary(1.0, 1.0, 5.0)
    data = binary.simulate(
        10 * T_orb,
        1e3,
        0.0,
        sample_interval=1000,
        progress_label="Validating energy conservation",
    )
    
    E_init = data['energy'][0]
    E_final = data['energy'][-1]
    dE_rel = abs(E_final - E_init) / abs(E_init)
    
    print(f"  Initial energy: {E_init:.6e} J")
    print(f"  Final energy:   {E_final:.6e} J")
    print(f"  Relative error: {dE_rel:.2e}")
    print(f"  PASS: {dE_rel < 1e-7}")
    
    # Test 2: Angular momentum conservation without mass transfer
    print("\nTest 2: Angular momentum conservation (no mass transfer)")
    L_init = data['angular_momentum'][0]
    L_final = data['angular_momentum'][-1]
    dL_rel = abs(L_final - L_init) / abs(L_init)
    
    print(f"  Initial L: {L_init:.6e} kg m^2/s")
    print(f"  Final L:   {L_final:.6e} kg m^2/s")
    print(f"  Relative error: {dL_rel:.2e}")
    print(f"  PASS: {dL_rel < 1e-7}")
    
    # Test 3: Verify L/L0 = 1 - f relationship
    print("\nTest 3: Verify dL/L ~= -f relationship")
    binary2, T_orb2 = setup_circular_binary(1.5, 1.5, 5.0)
    data2 = binary2.simulate(
        5 * T_orb2,
        1e3,
        0.15,
        sample_interval=500,
        progress_label="Validating dL/L ~= -f",
    )
    
    L_init2 = data2['angular_momentum'][0]
    L_final2 = data2['angular_momentum'][-1]
    L_ratio = L_final2 / L_init2
    expected = 1 - 0.15
    error = abs(L_ratio - expected)
    
    print(f"  L_final/L_init: {L_ratio:.4f}")
    print(f"  Expected (1-f): {expected:.4f}")
    print(f"  Absolute error: {error:.4f}")
    print(f"  PASS: {error < 0.01}")
    
    print("\n" + "=" * 70)


# =============================================================================
# Main Entry Point
# =============================================================================

def main() -> None:
    """Run validation tests, the simulation suite, and all figures."""
    run_validation_tests()

    results, cases = run_simulation_suite()

    # Persist results to allow cheap downstream figure regeneration
    save_simulation_data(results, cases, output_dir='.', filename='simulation_data')

    generate_figures(results, cases, output_dir='.')
    print_summary_table(results, cases)

    print("\n" + "=" * 70)
    print("Simulation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

