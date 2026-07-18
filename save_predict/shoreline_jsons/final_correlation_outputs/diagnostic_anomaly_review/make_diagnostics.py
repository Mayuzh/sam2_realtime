from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "final_correlation_outputs" / "diagnostic_anomaly_review"
OUT.mkdir(parents=True, exist_ok=True)

CASES = [
    {
        "site": "Seabright",
        "pair": "seabright_close_40_1500",
        "a_label": "Offshore bin 10",
        "a_points": ROOT / "final_correlation_outputs/seabright/final_correlation_points.csv",
        "a_match": {
            "final_plot_group": "offshore",
            "direction_group": "offshore",
            "sea_state_bin": "10",
        },
        "b_label": "Onshore SW bin 8",
        "b_points": ROOT / "final_correlation_outputs/seabright/final_correlation_points.csv",
        "b_match": {
            "final_plot_group": "onshore",
            "direction_group": "south_westerly_onshore",
            "sea_state_bin": "8",
        },
    },
    {
        "site": "Jennette's Pier",
        "pair": "jennettes_close_6p4_31000",
        "a_label": "Offshore bin 10",
        "a_points": ROOT / "final_correlation_outputs/jennettes_pier/final_correlation_points.csv",
        "a_match": {
            "final_plot_group": "offshore",
            "direction_group": "offshore",
            "sea_state_bin": "10",
        },
        "b_label": "Onshore SE bin 5",
        "b_points": ROOT / "final_correlation_outputs/jennettes_pier/final_correlation_points.csv",
        "b_match": {
            "final_plot_group": "onshore",
            "direction_group": "south_easterly_onshore",
            "sea_state_bin": "5",
        },
    },
]


def select_row(points_path, match):
    df = pd.read_csv(points_path)
    mask = pd.Series(True, index=df.index)
    for key, value in match.items():
        mask &= df[key].astype(str).eq(str(value))
    rows = df[mask]
    if rows.empty:
        raise RuntimeError(f"No row for {match} in {points_path}")
    return rows.iloc[0]


def curve_columns(df):
    columns = []
    for column in df.columns:
        if column.startswith(("y_", "s_")):
            try:
                columns.append((column, float(column[2:])))
            except ValueError:
                pass
    return sorted(columns, key=lambda item: item[1])


def box_data(curves_csv):
    df = pd.read_csv(curves_csv)
    columns = curve_columns(df)
    xs = np.array([x for _, x in columns], dtype=float)
    values = df[[column for column, _ in columns]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    with np.errstate(all="ignore"):
        median = np.nanmedian(values, axis=0)
        q25 = np.nanpercentile(values, 25, axis=0)
        q75 = np.nanpercentile(values, 75, axis=0)
    return xs, values, median, q25, q75


def plot_pair(case):
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), sharey=False)
    for ax, prefix in zip(axes, ["a", "b"]):
        row = select_row(case[f"{prefix}_points"], case[f"{prefix}_match"])
        label = case[f"{prefix}_label"]
        curves_csv = Path(row["curves_csv"])
        xs, values, median, q25, q75 = box_data(curves_csv)

        rng = np.random.default_rng(7)
        n = values.shape[0]
        sample = np.arange(n) if n <= 120 else rng.choice(n, 120, replace=False)
        for i in sample:
            ax.plot(xs, values[i], color="0.55", lw=0.45, alpha=0.14)

        ax.fill_between(xs, q25, q75, color="#8ab6d6", alpha=0.45, label="Middle 50%")
        ax.plot(xs, median, color="#0b4f6c", lw=2.0, label="Median")
        win_min = float(row["variance_window_y_min"])
        win_max = float(row["variance_window_y_max"])
        ax.axvspan(win_min, win_max, color="#f2c14e", alpha=0.30, label="Variance window")
        ax.set_title(
            f"{label}\nsea-state={float(row['sea_state_for_plot']):.2f}, "
            f"variance={float(row['shoreline_variance']):,.0f}"
        )
        ax.set_xlabel("Baseline coordinate (pixels)")
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=8)

        valid_fraction = np.isfinite(values).mean(axis=0)
        in_window = (xs >= win_min) & (xs <= win_max)
        rows.append(
            {
                "site": case["site"],
                "pair": case["pair"],
                "label": label,
                "final_plot_group": row["final_plot_group"],
                "direction_group": row["direction_group"],
                "sea_state_bin": row["sea_state_bin"],
                "sea_state_for_plot": float(row["sea_state_for_plot"]),
                "shoreline_variance": float(row["shoreline_variance"]),
                "n_usable_curves": int(row["n_usable_curves"]),
                "outlier_count": int(row.get("outlier_count", 0)),
                "included_in_correlation": row.get("included_in_correlation", ""),
                "variance_window_min": win_min,
                "variance_window_max": win_max,
                "window_valid_fraction_mean": float(np.nanmean(valid_fraction[in_window])),
                "median_in_window_mean": float(np.nanmean(median[in_window])),
                "iqr_in_window_mean": float(np.nanmean(q75[in_window] - q25[in_window])),
                "global_iqr_mean": float(np.nanmean(q75 - q25)),
                "curves_csv": str(curves_csv),
                "curve_boxplot_png": str(curves_csv.with_name("curve_boxplot.png")),
            }
        )

    axes[0].set_ylabel("Shoreline intersection distance (pixels)")
    fig.suptitle(f"{case['site']}: diagnostic curve boxplots for close/reversed scatter points", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    outpath = OUT / f"{case['pair']}_curve_boxplot_diagnostic.png"
    fig.savefig(outpath, dpi=180)
    plt.close(fig)
    for row in rows:
        row["diagnostic_png"] = str(outpath)
    return rows


def main():
    rows = []
    for case in CASES:
        rows.extend(plot_pair(case))
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "diagnostic_summary.csv", index=False)
    print(
        summary[
            [
                "site",
                "label",
                "sea_state_for_plot",
                "shoreline_variance",
                "n_usable_curves",
                "iqr_in_window_mean",
                "global_iqr_mean",
                "curve_boxplot_png",
                "diagnostic_png",
            ]
        ].to_string(index=False)
    )
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
