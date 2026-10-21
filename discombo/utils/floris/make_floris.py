from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from floris import FlorisModel


def build_horns_rev_layout(n_rows: int, n_cols: int, spacing_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Build a regular grid layout with rows aligned along x-direction."""
    x_coords = []
    y_coords = []
    for r in range(n_rows):
        for c in range(n_cols):
            x_coords.append(c * spacing_m)
            y_coords.append(r * spacing_m)
    return np.asarray(x_coords, dtype=float), np.asarray(y_coords, dtype=float)


def evaluate_random_yaw_configs(
    model: FlorisModel,
    n_turbines: int,
    n_evals: int,
    yaw_min_deg: float,
    yaw_max_deg: float,
    rng: np.random.Generator,
) -> list[list[float]]:
    """Evaluate random yaw configurations and return rows for CSV export."""
    rows: list[list[float]] = []
    for eval_idx in range(n_evals):
        yaw_angles = rng.uniform(yaw_min_deg, yaw_max_deg, size=n_turbines)
        model.set(yaw_angles=yaw_angles[None, :])
        model.run()

        # For one wind condition, this resolves to a single farm power value.
        total_power = float(np.sum(np.asarray(model.get_farm_power())))
        rows.append([eval_idx, total_power, *yaw_angles.tolist()])
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate FLORIS yaw-power dataset for Horns Rev setup.")
    parser.add_argument("--config", type=str, default="./gch.yaml", help="Path to FLORIS config (gch.yaml).")
    parser.add_argument("--output", type=str, default="horns_rev_yaw_10k.csv", help="Output CSV path.")
    parser.add_argument("--n-evals", type=int, default=5000, help="Number of random yaw evaluations.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--yaw-min", type=float, default=-30.0, help="Minimum yaw angle in degrees.")
    parser.add_argument("--yaw-max", type=float, default=30.0, help="Maximum yaw angle in degrees.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    fmodel = FlorisModel(args.config)

    # Turbine diameter from gch.yaml turbine type (nrel_5MW), used for 7D spacing.
    rotor_diameter = float(np.asarray(fmodel.core.farm.rotor_diameters).ravel()[0])
    spacing_m = 7.0 * rotor_diameter

    # Horns Rev-style grid: 8 x 10 = 80 turbines.
    n_rows = 8
    n_cols = 10
    n_turbines = n_rows * n_cols
    layout_x, layout_y = build_horns_rev_layout(n_rows=n_rows, n_cols=n_cols, spacing_m=spacing_m)

    # Wind aligned with turbine rows (x-direction), speed 8.2 m/s, TI 8.6%.
    fmodel.set(
        layout_x=layout_x,
        layout_y=layout_y,
        wind_directions=[270.0],
        wind_speeds=[8.2],
        turbulence_intensities=[0.086],
    )

    rows = evaluate_random_yaw_configs(
        model=fmodel,
        n_turbines=n_turbines,
        n_evals=args.n_evals,
        yaw_min_deg=args.yaw_min,
        yaw_max_deg=args.yaw_max,
        rng=rng,
    )

    output_path = Path(args.output)
    header = ["eval_id", "total_power_w"] + [f"yaw_t{idx:03d}_deg" for idx in range(n_turbines)]
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"Wrote {len(rows)} evaluations to: {output_path}")


if __name__ == "__main__":
    main()