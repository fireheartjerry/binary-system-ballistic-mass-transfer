"""
Regenerate publication figures from saved simulation data.

This script loads pre-computed simulation results and regenerates all matplotlib
figures without re-running the expensive simulation. Useful for adjusting plot
styles, tweaking figure parameters, or generating output in different formats.

Usage:
    python regenerate_figures.py                    # Use default data file
    python regenerate_figures.py --data other_name  # Use 'other_name.npz'
    python regenerate_figures.py --output ./figs    # Save to ./figs directory
"""

import argparse
from pathlib import Path

from plot_style import apply_plot_style
from main import load_simulation_data, generate_figures, print_summary_table


def main() -> None:
    """Load saved data and regenerate figures."""
    # Apply publication-quality styling
    apply_plot_style()

    parser = argparse.ArgumentParser(
        description="Regenerate publication figures from saved simulation data."
    )
    parser.add_argument(
        '--data',
        type=str,
        default='simulation_data',
        help='Name of the saved data file (without extension). Default: simulation_data',
    )
    parser.add_argument(
        '--output',
        type=str,
        default='.',
        help='Output directory for generated figures. Default: current directory',
    )
    args = parser.parse_args()

    # Load data
    print("=" * 70)
    print("Loading simulation data...")
    print("=" * 70)
    try:
        results, cases = load_simulation_data(
            output_dir='.',
            filename=args.data,
        )
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print(f"Make sure {args.data}.npz and {args.data}_cases.json exist in current directory.")
        return

    print(f"✓ Loaded {len(results)} simulation cases")
    for q_val in sorted(results.keys()):
        n_points = len(results[q_val]['time'])
        print(f"  q = {q_val}: {n_points} time samples")

    # Generate figures
    generate_figures(results, cases, output_dir=args.output)

    # Print summary
    print_summary_table(results, cases)

    print("\n" + "=" * 70)
    print("Done! Check the output directory for generated PDFs.")
    print("=" * 70)


if __name__ == "__main__":
    main()
