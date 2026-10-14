"""
Tests and benchmarks for the Haack-series nosecone geometry.

Checks volume, planform area, and wetted area for different C values
against reference values from Desmos and the analytical volume formula.

Measures how relative error changes with the number of profile points,
and times profile generation and geometry calculations to help choose
a suitable balance between accuracy and runtime.

Reference calculations and printing are excluded from timed sections.
"""

import math
from statistics import median
from time import perf_counter

import numpy as np
import matplotlib.pyplot as plt

from noseCone import NoseCone


# All reference values must use these dimensions.
REFERENCE_LENGTH = 0.6179       # m
REFERENCE_DIAMETER = 0.167
REFERENCE_RADIUS = REFERENCE_DIAMETER / 2.0       # m


DESMOS_REFERENCES = {
    0.0: {
        "volume":           0.00676723133964,  # m³
        "planform_area":    0.0671988190085,   # m²
        "wetted_area":      0.212511968496,    # m²
    },
    1.0 / 6.0: {
        "volume":           0.00719018329837,
        "planform_area":    0.0697988872929,
        "wetted_area":      0.220740287847,
    },
    1.0 / 3.0: {
        "volume":           0.00761313525709,
        "planform_area":    0.0722577649595,
        "wetted_area":      0.228551744915,
    },
    2.0 / 3.0: {
        "volume":           0.00845903917455,
        "planform_area":    0.0768402832602,
        "wetted_area":      0.243193455807,
    },
}


def forward_relative_error(computed, ref, abs_tol):

    if(np.abs(ref) < abs_tol): raise ValueError("Reference magnitude too small for relative error")


    return np.abs(computed - ref) / np.abs(ref)


def forward_absolute_error(computed, ref):

    return np.abs(computed - ref)


def uniform_scale_backward_error(computed, ref, power):
    if computed <= 0.0 or ref <= 0.0:
        raise ValueError("Computed and reference values must be positive")

    if power not in (2, 3):
        raise ValueError("Use power=2 for area or power=3 for volume")
    
    relative_difference = (computed - ref) / ref
    return abs(math.expm1(math.log1p(relative_difference) / power))


def evaluate_haack_case(C, num_points=2001):

    # Invalid Error checks

    if C not in DESMOS_REFERENCES: raise ValueError((f"No reference values available for C={C}"))
    if not isinstance(num_points, int) or num_points < 2: raise ValueError("num_points must be an integer of at least 2")


    # Assign reference values

    reference = DESMOS_REFERENCES[C]

    ref_volume = reference["volume"]
    ref_planform_area = reference["planform_area"]
    ref_wetted_area = reference["wetted_area"]

    # create nosecone profile and get paramters 

    nosecone = NoseCone.__new__(NoseCone)
    nosecone.shape = "Haack"
    nosecone.baseDiameter = REFERENCE_DIAMETER
    nosecone.length = REFERENCE_LENGTH
    nosecone.C = C
    nosecone.haackX, nosecone.haackR = (nosecone._getHaackProfile(numPoints=num_points))

    volume = nosecone._getVolume()
    planform_area = nosecone._getPlanformArea()
    wetted_area = nosecone._getWettedArea()

    # Volume errors

    volume_rel_error = forward_relative_error(volume, ref_volume, abs_tol=0.0)
    volume_abs_error = forward_absolute_error(volume, ref_volume)
    volume_back_error = uniform_scale_backward_error(volume, ref_volume, 3)


    # Planform errors

    planform_area_rel_error = forward_relative_error(planform_area, ref_planform_area, abs_tol=0.0)
    planform_area_abs_error = forward_absolute_error(planform_area, ref_planform_area)
    planform_area_back_error = uniform_scale_backward_error(planform_area, ref_planform_area, 2)


    # Wetted errors

    wetted_area_rel_error = forward_relative_error(wetted_area, ref_wetted_area, abs_tol=0.0)
    wetted_area_abs_error = forward_absolute_error(wetted_area, ref_wetted_area)
    wetted_area_back_error = uniform_scale_backward_error(wetted_area, ref_wetted_area, 2)

    return {
    "C": C,
    "num_points": num_points,
    "volume": {
        "relative_error": volume_rel_error,
        "absolute_error": volume_abs_error,
        "scaling_backward_error": volume_back_error,
    },
    "planform_area": {
        "relative_error": planform_area_rel_error,
        "absolute_error": planform_area_abs_error,
        "scaling_backward_error": planform_area_back_error,
    },
    "wetted_area": {
        "relative_error": wetted_area_rel_error,
        "absolute_error": wetted_area_abs_error,
        "scaling_backward_error": wetted_area_back_error,
    },
}


def benchmark_haack_case(C, repeats=20, batch_size=10):

    # Validate inputs

    if C not in DESMOS_REFERENCES:
        raise ValueError(f"No reference values available for C={C}")

    if not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")

    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")

    # Double the number of intervals from 500 to 32000

    interval_counts = [500, 1000, 2000, 4000, 8000, 16000, 32000]

    # Create nosecone and assign parameters outside the timed section

    nosecone = NoseCone.__new__(NoseCone)
    nosecone.shape = "Haack"
    nosecone.baseDiameter = REFERENCE_DIAMETER
    nosecone.length = REFERENCE_LENGTH
    nosecone.C = C

    # Define the work being timed:
    # generate the profile once, then calculate volume and both areas

    def calculate_geometry(num_points):
        nosecone.haackX, nosecone.haackR = (nosecone._getHaackProfile(numPoints=num_points))

        nosecone._getVolume()
        nosecone._getPlanformArea()
        nosecone._getWettedArea()

    # Store timing and accuracy results for each resolution

    results = []

    for num_intervals in interval_counts:

        # N intervals require N + 1 points

        num_points = num_intervals + 1

        # Calculate errors separately so they are excluded from timing

        errors = evaluate_haack_case(C, num_points=num_points)

        # Run five warm-up calculations before collecting timings

        for _ in range(5):
            calculate_geometry(num_points)

        # Store the average time per call from each timed batch

        timings = []

        for _ in range(repeats):

            # Time several calculations together to reduce timer overhead

            start = perf_counter()

            for _ in range(batch_size):
                calculate_geometry(num_points)

            elapsed = perf_counter() - start

            # Divide total batch time by the number of calculations

            timings.append(elapsed / batch_size)

        # Store the median per-call time and errors for this resolution

        results.append({
            "C": C,
            "num_intervals": num_intervals,
            "num_points": num_points,
            "median_seconds": median(timings),
            "errors": {
                "volume": errors["volume"],
                "planform_area": errors["planform_area"],
                "wetted_area": errors["wetted_area"],
            },
        })

    # Return all results; printing and tolerance checks happen elsewhere

    return results

def run_haack_tests(num_points=2001, rel_tol=1e-6):

    # Store descriptions of failed tests

    failures = []
    tests_run = 0

    # Test every C value with available reference data

    for C in DESMOS_REFERENCES:
        result = evaluate_haack_case(C, num_points=num_points)

        # Check volume, planform area, and wetted area separately

        for quantity in ("volume", "planform_area", "wetted_area"):
            tests_run += 1
            error = result[quantity]["relative_error"]

            # Non-finite errors also count as failures.
            # Checking only error > rel_tol would miss NaN.

            if not math.isfinite(error) or error > rel_tol:
                failures.append(
                    f"C={C}, points={num_points}, {quantity}: "
                    f"relative error={error:.6e}, "
                    f"tolerance={rel_tol:.6e}"
                )

    # Print the summary after all cases have been checked

    if failures:
        print(f"{len(failures)} of {tests_run} checks failed:")

        for failure in failures:
            print(f"  FAIL: {failure}")
    else:
        print(f"All {tests_run} checks passed.")

    return failures

def _format_benchmark_axis(ax):
    # Keep the grid light and remove minor tick clutter

    ax.grid(False, which="both")
    ax.grid(True, which="major", color="0.9", linewidth=0.7)
    ax.minorticks_off()

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)


def _c_label(C):
    # Display familiar fractions instead of long decimals

    labels = {
        0.0: "C = 0",
        1.0 / 6.0: "C = 1/6",
        1.0 / 3.0: "C = 1/3",
        2.0 / 3.0: "C = 2/3",
    }

    return labels.get(C, f"C = {C:g}")


def _positive_errors(results, quantity):
    # Zero cannot appear on a logarithmic axis.
    # Leave gaps rather than replacing errors with invented values.

    errors = np.array([
        row["errors"][quantity]["relative_error"]
        for row in results
    ])

    return np.where(
        np.isfinite(errors) & (errors > 0.0),
        errors,
        np.nan
    )


def plot_haack_benchmark(all_results):
    # Figure 1: convergence as the number of points increases

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.suptitle("Haack geometry: relative error vs profile points",
                 fontsize=15)

    quantities = (
        ("volume", "Volume"),
        ("planform_area", "Planform area"),
        ("wetted_area", "Wetted area"),
    )

    for C, results in sorted(all_results.items()):
        results = sorted(results, key=lambda row: row["num_points"])
        n = np.array([row["num_points"] for row in results])

        for ax, (quantity, title) in zip(axes, quantities):
            errors = _positive_errors(results, quantity)

            ax.loglog(
                n, errors,
                marker="o", markersize=4, linewidth=1.5,
                label=_c_label(C)
            )

            ax.set_title(title)
            ax.set_xlabel("Profile points (N)")
            ax.set_ylabel("Relative forward error")

    for ax in axes:
        _format_benchmark_axis(ax)

    # Use one legend for the whole figure

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 0.91), ncol=4, frameon=False)

    fig.text(
        0.5, 0.025,
        "Volume uses an exact formula and is independent of N. "
        "Zero and non-finite errors are omitted.",
        ha="center", fontsize=9
    )

    fig.tight_layout(rect=(0, 0.07, 1, 0.83))

    return fig


def plot_accuracy_vs_time(all_results, rel_tol=1e-6):
    # Figure 2: compare accuracy with computational cost

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.suptitle("Haack geometry: accuracy and runtime", fontsize=15)

    # Label only one C curve to avoid overlapping numbers

    labelled_C = max(all_results)

    for C, results in sorted(all_results.items()):
        results = sorted(results, key=lambda row: row["num_points"])

        n = np.array([row["num_points"] for row in results])
        times = np.array([
            row["median_seconds"] * 1e6
            for row in results
        ])

        # Runtime includes profile generation and all three quantities

        axes[0].loglog(
            n, times,
            marker="o", markersize=4, linewidth=1.5,
            label=_c_label(C)
        )

        for ax, quantity in zip(
            axes[1:], ("planform_area", "wetted_area")
        ):
            errors = _positive_errors(results, quantity)

            ax.loglog(
                times, errors,
                marker="o", markersize=4, linewidth=1.5,
                label=_c_label(C)
            )

            # Label alternate resolutions on only one curve

            if C == labelled_C:
                for i in range(0, len(n), 2):
                    if np.isfinite(errors[i]):
                        ax.annotate(
                            f"N={n[i]:,}",
                            (times[i], errors[i]),
                            xytext=(0, 10),
                            textcoords="offset points",
                            ha="center",
                            fontsize=8,
                            color="0.3"
                        )

    axes[0].set_title("Runtime vs point count")
    axes[0].set_xlabel("Profile points (N)")
    axes[0].set_ylabel("Median runtime (µs)")

    for ax, title in zip(
        axes[1:], ("Planform area", "Wetted area")
    ):
        ax.set_title(title)
        ax.set_xlabel("Median runtime (µs)")
        ax.set_ylabel("Relative forward error")

        ax.axhline(
            rel_tol, color="0.4", linestyle="--", linewidth=1,
            label=f"Tolerance = {rel_tol:g}"
        )

        # Leave room around the point annotations

        ax.margins(x=0.15, y=0.15)

    for ax in axes:
        _format_benchmark_axis(ax)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 0.91), ncol=5, frameon=False)

    fig.text(
        0.5, 0.025,
        f"N labels shown for {_c_label(labelled_C)} only. "
        "Runtime includes profile generation, volume, and both areas.",
        ha="center", fontsize=9
    )

    fig.tight_layout(rect=(0, 0.07, 1, 0.83))

    return fig

def select_best_num_points(all_results, rel_tol=1e-6):

    # Require a positive, finite accuracy target

    if not math.isfinite(rel_tol) or rel_tol <= 0:
        raise ValueError("rel_tol must be positive and finite")

    if not all_results:
        raise ValueError("No benchmark results supplied")

    # Group measurements by point count and C

    grouped = {}

    for C, results in all_results.items():
        for row in results:
            n = row["num_points"]
            grouped.setdefault(n, {})[C] = row

    candidates = []

    for n, cases in sorted(grouped.items()):

        # Only consider resolutions tested for every C

        if set(cases) != set(all_results):
            continue

        errors = []
        times = []

        for row in cases.values():
            times.append(row["median_seconds"])

            for quantity in ("planform_area", "wetted_area"):
                errors.append(
                    row["errors"][quantity]["relative_error"]
                )

        # Reject invalid measurements

        if not all(math.isfinite(e) and e >= 0 for e in errors):
            continue

        if not all(math.isfinite(t) and t > 0 for t in times):
            continue

        worst_error = max(errors)

        # Keep resolutions satisfying the accuracy requirement

        if worst_error <= rel_tol:
            candidates.append({
                "num_points": n,
                "worst_relative_error": worst_error,
                "median_seconds": median(times),
                "accuracy_margin": (
                    rel_tol / worst_error
                    if worst_error > 0 else math.inf
                ),
            })

    if not candidates:
        raise ValueError(
            "No tested resolution meets the requested tolerance "
            "for every C"
        )

    # Choose the fastest qualifying resolution.
    # Prefer fewer points if measured times are exactly equal.

    return min(
        candidates,
        key=lambda case: (
            case["median_seconds"],
            case["num_points"]
        )
    )


def main():

    # Run correctness tests

    failures = run_haack_tests(
        num_points=2001,
        rel_tol=1e-6
    )

    # Run benchmarks once and reuse the results for both figures

    all_results = {
        C: benchmark_haack_case(C)
        for C in DESMOS_REFERENCES
    }

    # Create both figures before displaying them

    plot_haack_benchmark(all_results)
    plot_accuracy_vs_time(all_results, rel_tol=1e-6)

    # Best numpoints

    best = select_best_num_points(all_results, rel_tol=1e-6)

    print(f"Selected points: {best['num_points']}")
    print(f"Worst area error: {best['worst_relative_error']:.3e}")
    print(f"Median runtime: {best['median_seconds'] * 1e6:.1f} µs")
    print(f"Accuracy margin: {best['accuracy_margin']:.1f}×")

    # Display both windows together

    plt.show()

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())


# Run main only when this file is executed directly

if __name__ == "__main__":
    raise SystemExit(main())









