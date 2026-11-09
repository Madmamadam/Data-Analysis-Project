#!/usr/bin/env python3
"""Compute the phase 1 HVFHV report values and generate its chart PDFs.

Requires pandas 2+, NumPy, and Matplotlib. Run from the report directory:
  python phase1_plots_v1.20.29.py hvfhv_2024_random_sample.csv hvfhv_2025_random_sample.csv
The default lookup is taxi_zone_lookup.csv. Use --zone-lookup to specify its path.
Use --tlc-monthly with the official monthly indicators CSV to include the
comparison of daily trip counts across service categories.
"""

import argparse
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
from matplotlib.dates import DateFormatter
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator


WEIGHT = "_sample_weight"
MONTH = "_sample_source_month"
SOURCE_ROW = "_sample_source_row"
SERVICE = "hvfhs_license_num"
SHARED_MATCH = "shared_match_flag"
SERVICES = ("HV0003", "HV0005")
TIMESTAMPS = ("request_datetime", "pickup_datetime", "dropoff_datetime")
NUMERIC = (
    "PULocationID",
    "DOLocationID",
    "trip_time",
    "trip_miles",
    "base_passenger_fare",
)
COLUMNS = (*TIMESTAMPS, *NUMERIC, WEIGHT, MONTH, SOURCE_ROW, SERVICE, SHARED_MATCH)
YEARS = (2024, 2025)
YEAR_KEYS = ("first", "second")
PICKUP_WINDOW_DAYS = 7
REFERENCE_PER_MONTH = 10_000
REDUCED_PER_MONTH = 5_000
FIRST_SEED = 42
DRAWS = 20
BIN_TOLERANCE_PP = 1.0
P95_TOLERANCE_PERCENT = 5.0
UPPER_PERCENTILE = 95
IQR_MULTIPLIER = 1.5
LONG_WAIT_MINUTES = 10
TOP_ORIGINS = 6
REQUEST_HOUR_EXAMPLES = (7, 23)
PICKUP_HOUR_EXAMPLE = 19
OD_HOURS = (3, 11)
EXAMPLE_REQUEST = "18:05"
EXAMPLE_PICKUP = "18:12"
DISTANCE_EDGES = (0, 1, 2, 5, 10, np.inf)
DURATION_TOLERANCE_SECONDS = 1
CONSISTENCY_DISTANCE_MAX = 35
CONSISTENCY_PANELS = (
    ("duration_minutes", "Distance and duration", "Duration (minutes)", 90),
    ("base_passenger_fare", "Distance and base fare", "Base fare (USD)", 150),
)
UNIVARIATE_SPECS = (
    ("wait_minutes", "Passenger wait", "Minutes", 2, 30, 5),
    ("duration_minutes", "Trip duration", "Minutes", 5, 90, 15),
    ("trip_miles", "Trip distance", "Miles", 2, 30, 5),
    ("base_passenger_fare", "Base passenger fare", "US dollars", 10, 100, 20),
)
REPORT_VARIABLES = {
    "wait_minutes": "wait",
    "duration_minutes": "duration",
    "trip_miles": "distance",
    "base_passenger_fare": "fare",
}
HOURS = np.arange(24)
DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
PERSIAN_DAYS = ("دوشنبه", "سه‌شنبه", "چهارشنبه", "پنج‌شنبه", "جمعه", "شنبه", "یکشنبه")
PERSIAN_MONTHS = (
    "ژانویه", "فوریه", "مارس", "آوریل", "مه", "ژوئن", "ژوئیه", "اوت",
    "سپتامبر", "اکتبر", "نوامبر", "دسامبر",
)
CITY_BOROUGHS = ("Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island")
BOROUGH_GROUPS = (*CITY_BOROUGHS, "Newark Airport", "Outside NYC", "Unknown")
VALUES_FILE = "phase1_values_v1.20.29.tex"
TEAL = "#15727b"
GOLD = "#9e5a20"
BLUE = "#355d99"
INK = "#162b38"
GRID = "#dbe5e6"
EMPTY_CELL = "#eef1f3"


def invalid_weights(weights):
    return ~np.isfinite(weights) | weights.le(0)


def load_trips(paths):
    frames = [pd.read_csv(path, usecols=COLUMNS) for path in paths]
    trips = pd.concat(frames, ignore_index=True)
    for column in TIMESTAMPS:
        trips[column] = pd.to_datetime(trips[column], format="ISO8601", errors="coerce")
    weights = pd.to_numeric(trips[WEIGHT], errors="coerce")
    invalid = invalid_weights(weights)
    if invalid.any():
        raise ValueError(f"{int(invalid.sum()):,} rows have invalid {WEIGHT} values; expected finite, positive weights.")
    trips[WEIGHT] = weights

    for column in NUMERIC:
        values = pd.to_numeric(trips[column], errors="coerce")
        trips[column] = values.where(np.isfinite(values))

    waiting = (trips["pickup_datetime"] - trips["request_datetime"]).dt.total_seconds() / 60
    trips["wait_minutes"] = waiting.where(waiting >= 0)
    trips["duration_minutes"] = trips["trip_time"].where(trips["trip_time"] > 0) / 60
    trips["trip_miles"] = trips["trip_miles"].where(trips["trip_miles"] >= 0)
    return trips


def weighted_quantiles(values, weights, probabilities):
    """Return the first value reaching each probability in the weighted empirical CDF."""
    if len(values) == 0:
        return np.full(len(probabilities), np.nan)
    values = np.asarray(values)
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(np.asarray(weights, dtype=float)[order])
    targets = np.asarray(probabilities) * cumulative[-1]
    return values[order[np.searchsorted(cumulative, targets, side="left")]]


def style_axis(ax):
    formatter = FuncFormatter(lambda value, _: f"{value:g}")
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)


def save(fig, folder, filename):
    fig.savefig(folder / filename, facecolor="white")
    plt.close(fig)


def univariate_profiles(trips):
    pickup_year = trips["pickup_datetime"].dt.year
    profiles = {}
    for column, _, _, step, end, _ in UNIVARIATE_SPECS:
        edges = np.arange(0, end + step, step)
        for year in YEARS:
            valid = trips.loc[pickup_year.eq(year), [column, WEIGHT]].dropna(subset=[column])
            values = valid[column].to_numpy()
            weights = valid[WEIGHT].to_numpy()
            displayed = (values >= 0) & (values < end)
            counts, _ = np.histogram(values[displayed], bins=edges, weights=weights[displayed])
            total_weight = weights.sum()
            percentages = 100 * counts / total_weight if total_weight else counts
            p95 = weighted_quantiles(values, weights, [UPPER_PERCENTILE / 100])[0]
            profiles[column, year] = (percentages, p95)
    return profiles


def resolution_study(trips):
    strata = trips.groupby(MONTH, sort=True).indices
    expected = {f"{year}-{month:02d}" for year in YEARS for month in range(1, 13)}
    if set(strata) != expected or any(len(rows) != REFERENCE_PER_MONTH for rows in strata.values()):
        raise ValueError(f"Expected {REFERENCE_PER_MONTH:,} rows in each of the 24 source months.")
    if sum(map(len, strata.values())) != len(trips):
        raise ValueError(f"Some rows have a missing {MONTH}.")

    full = univariate_profiles(trips)
    worst = dict.fromkeys(full, (0.0, 0.0))
    for seed in range(FIRST_SEED, FIRST_SEED + DRAWS):
        rng = np.random.default_rng(seed)
        selected = np.concatenate([
            rng.choice(rows, size=REDUCED_PER_MONTH, replace=False) for rows in strata.values()
        ])
        reduced = trips.iloc[np.sort(selected)].copy()
        reduced[WEIGHT] *= REFERENCE_PER_MONTH / REDUCED_PER_MONTH
        profile = univariate_profiles(reduced)
        if seed == FIRST_SEED:
            example = profile
        for key, (base_bins, base_p95) in full.items():
            bins, p95 = profile[key]
            bin_delta = float(np.max(np.abs(bins - base_bins)))
            p95_delta = float(100 * abs(p95 - base_p95) / abs(base_p95)) if base_p95 else np.inf
            if not np.isfinite(p95_delta):
                raise ValueError(f"Cannot compare the relative P95 change for {key}.")
            old_bin, old_p95 = worst[key]
            worst[key] = (max(old_bin, bin_delta), max(old_p95, p95_delta))
    return full, example, worst


def plot_univariate(profiles, limits, folder, filename):
    fig, axes = plt.subplots(4, 1, figsize=(8.6, 10.5), layout="constrained")
    for ax, (column, title, unit, step, end, tick_step) in zip(axes, UNIVARIATE_SPECS):
        style_axis(ax)
        edges = np.arange(0, end + step, step)
        group_width = 0.88 * step
        bar_width = group_width / len(YEARS)
        group_left = edges[:-1] + (step - group_width) / 2
        for position, (year, color) in enumerate(zip(YEARS, (TEAL, GOLD))):
            percentages, _ = profiles[column, year]
            ax.bar(group_left + position * bar_width, percentages, width=bar_width,
                   align="edge", color=color, edgecolor="white", linewidth=0.55,
                   label=str(year))
        ax.set(xlim=(0, end), ylim=(0, limits[column]), title=title, xlabel=unit,
               ylabel="Weighted share (%)")
        ax.set_xticks(np.arange(0, end + tick_step, tick_step))
    axes[0].legend(title="Pickup year", loc="upper right", ncol=2, frameon=False)
    save(fig, folder, filename)


def hourly_wait_medians(trips):
    valid = trips.dropna(subset=["wait_minutes", "request_datetime"])
    requests = valid["request_datetime"].dt
    groups = valid.groupby([requests.dayofweek >= 5, requests.hour])
    medians = {
        key: weighted_quantiles(group["wait_minutes"], group[WEIGHT], [.5])[0]
        for key, group in groups
    }
    hours = pd.MultiIndex.from_product([[False, True], HOURS])
    return pd.Series(medians, dtype=float).reindex(hours).unstack(0)


def plot_wait_by_hour(trips, folder):
    by_hour = hourly_wait_medians(trips)
    fig, ax = plt.subplots(figsize=(9.2, 2.5), layout="constrained")
    style_axis(ax)
    ax.plot(HOURS, by_hour[False], color=TEAL, marker="o", markersize=3,
            linewidth=1.5, label="Monday to Friday")
    ax.plot(HOURS, by_hour[True], color=GOLD, marker="s", markersize=3,
            linewidth=1.5, linestyle="--", label="Saturday and Sunday")
    ax.set(xlim=(0, 23), xlabel="Request hour (local time)", ylabel="Weighted median wait (min)")
    ax.set_xticks([0, 3, 6, 9, 12, 15, 18, 21, 23])
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1), ncol=2, frameon=False)
    save(fig, folder, "wait_by_hour.pdf")


def zone_wait_shares(trips):
    top_zones = trips.groupby("PULocationID")[WEIGHT].sum().nlargest(TOP_ORIGINS).index
    valid = trips.loc[trips["PULocationID"].isin(top_zones)].dropna(subset=["wait_minutes"])
    total = valid.groupby("PULocationID")[WEIGHT].sum()
    long_wait = valid.loc[valid["wait_minutes"] > LONG_WAIT_MINUTES].groupby("PULocationID")[WEIGHT].sum()
    return long_wait.reindex(total.index, fill_value=0).div(total).reindex(top_zones)


def distance_wait_quantiles(trips):
    valid = trips.dropna(subset=["trip_miles", "wait_minutes"])
    groups = pd.cut(valid["trip_miles"], DISTANCE_EDGES, right=False)
    probabilities = [.25, .5, .75]
    rows = [
        weighted_quantiles(group["wait_minutes"], group[WEIGHT], probabilities)
        for _, group in valid.groupby(groups, observed=False)
    ]
    return pd.DataFrame(rows, index=groups.cat.categories, columns=probabilities)


def plot_wait_by_zone_and_distance(trips, folder):
    fig, (zones_ax, miles_ax) = plt.subplots(1, 2, figsize=(9.2, 2.35), layout="constrained")
    style_axis(zones_ax)
    style_axis(miles_ax)

    share = zone_wait_shares(trips)
    top_zones = share.index
    positions = np.arange(len(top_zones))
    zones_ax.bar(positions, 100 * share, color=TEAL, width=0.46)
    zones_ax.set_xticks(positions, [f"{zone:g}" for zone in top_zones])
    zones_ax.set(title=f"Wait over {LONG_WAIT_MINUTES} min by zone", xlabel="Pickup zone ID",
                 ylabel="Weighted share (%)", ylim=(0, None))

    quantiles = distance_wait_quantiles(trips)
    median = quantiles[.5]
    positions = np.arange(len(quantiles))
    miles_ax.errorbar(positions, median, yerr=[median - quantiles[.25], quantiles[.75] - median],
                      fmt="o", color=BLUE, capsize=3, markersize=4, linewidth=1.2,
                      label="Weighted median and P25–P75")
    miles_ax.set_xticks(positions, ("0-1", "1-2", "2-5", "5-10", "10+"))
    miles_ax.set(title="Wait by recorded distance", xlabel="Trip distance (mi)", ylabel="Wait (min)")
    miles_ax.legend(loc="upper left", frameon=False, fontsize=8)
    save(fig, folder, "wait_by_zone_and_distance.pdf")


def pickup_counts(trips):
    pickups = trips["pickup_datetime"]
    valid = pickups.dt.year.isin(YEARS)
    times = pickups[valid].dt
    totals = trips.loc[valid, WEIGHT].groupby([times.dayofweek, times.hour]).sum()
    return totals.unstack(fill_value=0).reindex(index=range(7), columns=HOURS, fill_value=0)


def plot_trips_heatmap(counts, folder):
    fig, ax = plt.subplots(figsize=(10.0, 3.4), layout="constrained")
    # One square of color for each weekday-hour combination.
    mesh = ax.pcolormesh(counts.to_numpy(), cmap="cividis", vmin=0,
                         edgecolors="white", linewidth=0.5)
    ax.set_xticks(HOURS + .5, [str(hour) for hour in HOURS])
    ax.set_yticks(np.arange(7) + .5, DAYS)
    ax.invert_yaxis()
    ax.set(xlabel="Pickup hour (local time)", ylabel="Day of week")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    colorbar = fig.colorbar(mesh, ax=ax, pad=.022, fraction=.035)
    colorbar.set_label("Estimated trips (both years)")
    colorbar.locator = MaxNLocator(nbins=5, integer=True)
    colorbar.formatter = FuncFormatter(lambda value, _: f"{value:,.0f}")
    colorbar.update_ticks()
    colorbar.outline.set_visible(False)
    save(fig, folder, "trips_heatmap.pdf")


def weekly_pickups(trips):
    """Weighted pickup volumes in seven-day windows starting January 1 each year."""
    totals = trips.groupby(trips["pickup_datetime"].dt.normalize())[WEIGHT].sum()
    windows = []
    for year in YEARS:
        dates = pd.date_range(f"{year}-01-01", f"{year}-12-31")
        days = pd.DataFrame({"date": dates, "weight": totals.reindex(dates).to_numpy()})
        groups = days.groupby(np.arange(len(days)) // PICKUP_WINDOW_DAYS)
        bins = groups.agg(start=("date", "min"), end=("date", "max"),
                          days=("date", "size"), observed=("weight", "count"))
        bins["total"] = groups["weight"].sum().where(bins["days"].eq(bins["observed"]))
        bins["daily"] = bins["total"] / bins["days"]
        bins["year"] = year
        windows.append(bins)
    return pd.concat(windows, ignore_index=True)


def plot_weekly_pickups(windows, folder):
    fig, axes = plt.subplots(2, 1, figsize=(10, 3.65), sharey=True, layout="constrained")
    for ax, year, color in zip(axes, YEARS, (TEAL, GOLD)):
        style_axis(ax)
        data = windows.loc[windows["year"].eq(year)]
        centers = data["start"] + (data["end"] - data["start"]) / 2
        complete = data["days"].eq(PICKUP_WINDOW_DAYS)
        ax.plot(centers[complete], data.loc[complete, "daily"] / 1000,
                color=color, linewidth=1.4)
        ax.scatter(centers[~complete], data.loc[~complete, "daily"] / 1000,
                   s=24, facecolors="white", edgecolors=color, zorder=3)
        ax.set(xlim=(pd.Timestamp(year, 1, 1), pd.Timestamp(year + 1, 1, 2)),
               title=str(year))
        ax.set_xticks(pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="2MS"))
        ax.xaxis.set_major_formatter(DateFormatter("%b"))
    handles = [
        Line2D([], [], color=INK, linewidth=1.4, label=f"Full {PICKUP_WINDOW_DAYS}-day window"),
        Line2D([], [], color=INK, marker="o", markerfacecolor="white", markersize=4,
               linestyle="none", label="Partial year-end window"),
    ]
    axes[0].legend(handles=handles, loc="lower right", bbox_to_anchor=(1, 1),
                   ncol=2, frameon=False, fontsize=8)
    fig.supylabel("Estimated trips per day (thousands)")
    save(fig, folder, "pickups_seven_day.pdf")


def duration_difference(trips):
    elapsed = (trips["dropoff_datetime"] - trips["pickup_datetime"]).dt.total_seconds()
    return elapsed - trips["trip_time"]


def trip_consistency(trips):
    duration = trips["trip_time"]
    miles = trips["trip_miles"]
    fare = trips["base_passenger_fare"]
    speed = 3600 * miles / duration.where(duration.gt(0))
    flags = pd.DataFrame({
        "duration_mismatch": duration_difference(trips).abs().gt(DURATION_TOLERANCE_SECONDS),
        "negative_fare": fare.lt(0),
        "zero_fare": fare.eq(0),
        "zero_distance": miles.eq(0) & duration.gt(0),
    })
    return speed, flags


def consistency_visible(trips, column, upper):
    return trips["trip_miles"].between(0, CONSISTENCY_DISTANCE_MAX) & trips[column].between(0, upper)


def plot_trip_consistency(trips, folder):
    x = trips["trip_miles"]
    xedges = np.linspace(0, CONSISTENCY_DISTANCE_MAX, 71)
    grids = []
    for column, _, _, upper in CONSISTENCY_PANELS:
        y = trips[column]
        visible = consistency_visible(trips, column, upper)
        yedges = np.linspace(0, upper, 61)
        grid, _, _ = np.histogram2d(x[visible], y[visible], bins=[xedges, yedges],
                                    weights=trips.loc[visible, WEIGHT])
        grids.append((grid.T, yedges))

    positive = np.concatenate([grid[grid > 0] for grid, _ in grids])
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), layout="constrained")
    norm = LogNorm(vmin=positive.min(), vmax=positive.max())
    for ax, (_, title, ylabel, _), (grid, yedges) in zip(axes, CONSISTENCY_PANELS, grids):
        style_axis(ax)
        ax.grid(False)
        ax.set_facecolor(EMPTY_CELL)
        mesh = ax.imshow(np.ma.masked_equal(grid, 0), origin="lower",
                         extent=(xedges[0], xedges[-1], yedges[0], yedges[-1]),
                         aspect="auto", interpolation="nearest", cmap="cividis", norm=norm)
        ax.set(title=title, xlabel="Distance (miles)", ylabel=ylabel)
    colorbar = fig.colorbar(mesh, ax=axes, pad=.025, fraction=.04, format=FuncFormatter(
        lambda value, _: f"{value:,.0f}"))
    colorbar.set_label("Estimated trips per cell (log scale)")
    save(fig, folder, "trip_consistency.pdf")


def load_zone_boroughs(path):
    lookup = pd.read_csv(path, usecols=["LocationID", "Borough"],
                         dtype={"LocationID": "Int64"}, keep_default_na=False)
    if lookup["LocationID"].isna().any() or not lookup["LocationID"].is_unique:
        raise ValueError("The taxi zone lookup must have unique, nonempty LocationID values.")
    boroughs = lookup.set_index("LocationID")["Borough"].replace(
        {"EWR": "Newark Airport", "N/A": "Outside NYC"})
    if boroughs.empty or not boroughs.isin(BOROUGH_GROUPS).all():
        raise ValueError("The taxi zone lookup contains missing or unrecognized boroughs.")
    return boroughs


def assign_boroughs(trips, zone_boroughs):
    valid = trips["pickup_datetime"].dt.year.isin(YEARS)
    selected = trips.loc[valid, ["pickup_datetime", WEIGHT]].copy()
    selected["origin"] = trips.loc[valid, "PULocationID"].map(zone_boroughs).fillna("Unknown")
    selected["destination"] = trips.loc[valid, "DOLocationID"].map(zone_boroughs).fillna("Unknown")
    return selected


def origin_destination(trips):
    totals = trips.groupby(["destination", "origin"])[WEIGHT].sum()
    return totals.unstack(fill_value=0).reindex(
        index=BOROUGH_GROUPS, columns=BOROUGH_GROUPS, fill_value=0)


def od_by_pickup_period(trips):
    hours = trips["pickup_datetime"].dt.hour
    start, end = OD_HOURS
    morning = hours.ge(start) & hours.lt(end)
    return {
        "All pickup hours": origin_destination(trips),
        f"{start:02d}:00 to {end:02d}:00": origin_destination(trips.loc[morning]),
        f"{end:02d}:00 to {start:02d}:00 next day": origin_destination(trips.loc[~morning]),
    }


def compact_trip_count(value):
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}".removesuffix(".0") + "M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}".removesuffix(".0") + "k"
    return f"{value:.0f}"


def origin_destination_figure(matrix, period, norm):
    data = matrix.to_numpy()
    fig, ax = plt.subplots(figsize=(9.2, 6.6), layout="constrained")
    ax.set_facecolor(EMPTY_CELL)
    image = ax.pcolormesh(np.ma.masked_equal(data, 0), cmap="viridis", norm=norm,
                         edgecolors="white", linewidth=.6)
    ticks = np.arange(len(BOROUGH_GROUPS)) + .5
    ax.set_xticks(ticks, [label.replace(" ", "\n") for label in matrix.columns])
    ax.set_yticks(ticks, matrix.index)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set(xlabel="Pickup group (origin)", ylabel="Drop-off group (destination)")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_color("#cad7da")
    for y, x in zip(*np.nonzero(data)):
        value = data[y, x]
        red, green, blue, _ = image.cmap(norm(value))
        brightness = .2126 * red + .7152 * green + .0722 * blue
        ax.text(x + .5, y + .5, compact_trip_count(value), ha="center", va="center",
                fontsize=9, color="white" if brightness < .50 else INK)

    bar = fig.colorbar(image, ax=ax, shrink=.84, pad=.035,
                       format=FuncFormatter(lambda value, _: compact_trip_count(value)))
    bar.set_label("Estimated trips (log scale)")
    bar.outline.set_visible(False)
    fig.suptitle(f"Origin–destination flows | {period}",
                 fontsize=14, fontweight="bold")
    return fig


def plot_origin_destination(matrices, folder):
    weights = np.concatenate([matrix.to_numpy().ravel() for matrix in matrices.values()])
    positive = weights[weights > 0]
    norm = LogNorm(vmin=positive.min(), vmax=positive.max())
    with PdfPages(folder / "origin_destination.pdf") as pdf:
        for period, matrix in matrices.items():
            fig = origin_destination_figure(matrix, period, norm)
            pdf.savefig(fig, facecolor="white")
            plt.close(fig)


def borough_pickup_counts(trips):
    hours = trips["pickup_datetime"].dt.hour
    totals = trips.groupby([trips["origin"], hours])[WEIGHT].sum()
    return totals.unstack(fill_value=0).reindex(
        index=BOROUGH_GROUPS, columns=HOURS, fill_value=0)


def plot_borough_pickups(counts, folder):
    fig, ax = plt.subplots(figsize=(10, 4.4), layout="constrained")
    data = counts.to_numpy()
    positive = data[data > 0]
    ax.set_facecolor(EMPTY_CELL)
    image = ax.pcolormesh(np.ma.masked_equal(data, 0), cmap="viridis",
                         norm=LogNorm(vmin=positive.min(), vmax=positive.max()),
                         edgecolors="white", linewidth=.55)
    ax.set_xticks(HOURS + .5, [str(hour) for hour in HOURS])
    ax.set_yticks(np.arange(len(BOROUGH_GROUPS)) + .5, counts.index)
    ax.invert_yaxis()
    ax.set(xlabel="Pickup hour (local time)", ylabel="Pickup group")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(image, ax=ax, pad=.022, fraction=.035,
                       format=FuncFormatter(lambda value, _: compact_trip_count(value)))
    bar.set_label("Estimated trips (log scale)")
    bar.outline.set_visible(False)
    save(fig, folder, "borough_pickups_by_hour.pdf")


def report_count(value):
    return f"{value:,.0f}"


def report_percent(numerator, denominator, places=2):
    return f"{100 * numerator / denominator:.{places}f}" + r"\%" if denominator else "--"


def schema_report_values(paths):
    sampling_fields = {MONTH, SOURCE_ROW, WEIGHT}
    counts_by_year = {year: set() for year in YEARS}
    for path in paths:
        header = pd.read_csv(path, nrows=0).columns
        tlc_count = len(set(header) - sampling_fields)
        source_months = pd.read_csv(path, usecols=[MONTH], dtype=str)[MONTH]
        for year in YEARS:
            if source_months.str.startswith(f"{year}-", na=False).any():
                counts_by_year[year].add(tlc_count)
    values = {"sampling-columns": report_count(len(sampling_fields))}
    for year, key in zip(YEARS, YEAR_KEYS):
        counts = sorted(counts_by_year[year])
        values[f"{key}-tlc-columns"] = ", ".join(map(report_count, counts))
    return values


def report_settings(trips, differences):
    request = datetime.strptime(EXAMPLE_REQUEST, "%H:%M")
    pickup = datetime.strptime(EXAMPLE_PICKUP, "%H:%M")
    od_start, od_end = OD_HOURS
    month_count = trips[MONTH].nunique()
    max_bin = max(bin_difference for bin_difference, _ in differences.values())
    max_p95 = max(p95_difference for _, p95_difference in differences.values())
    sufficient = max_bin <= BIN_TOLERANCE_PP and max_p95 <= P95_TOLERANCE_PERCENT
    return {
        "first-year": str(YEARS[0]),
        "second-year": str(YEARS[1]),
        "sample-months": report_count(month_count),
        "reference-per-month": report_count(REFERENCE_PER_MONTH),
        "reduced-per-month": report_count(REDUCED_PER_MONTH),
        "reduced-rows": report_count(month_count * REDUCED_PER_MONTH),
        "reduced-rows-per-year": report_count(month_count * REDUCED_PER_MONTH / len(YEARS)),
        "sample-probability": report_percent(REDUCED_PER_MONTH, REFERENCE_PER_MONTH, 0),
        "sample-weight-factor": f"{REFERENCE_PER_MONTH / REDUCED_PER_MONTH:g}",
        "draw-count": report_count(DRAWS),
        "seed-first": str(FIRST_SEED),
        "seed-last": str(FIRST_SEED + DRAWS - 1),
        "bin-tolerance": f"{BIN_TOLERANCE_PP:g}",
        "p95-tolerance": report_percent(P95_TOLERANCE_PERCENT, 100, 0),
        "upper-percentile": str(UPPER_PERCENTILE),
        "upper-percentile-share": report_percent(UPPER_PERCENTILE, 100, 0),
        "max-bin-difference": f"{max_bin:.3f}",
        "max-p95-difference": report_percent(max_p95, 100, 3),
        "resolution-verdict": "کافی است" if sufficient else "کافی نیست",
        "example-request": EXAMPLE_REQUEST,
        "example-pickup": EXAMPLE_PICKUP,
        "example-wait": f"{(pickup - request).total_seconds() / 60:g}",
        "long-wait-minutes": str(LONG_WAIT_MINUTES),
        "iqr-multiplier": f"{IQR_MULTIPLIER:g}",
        "service-first": SERVICES[0],
        "service-second": SERVICES[1],
        "week-days": str(PICKUP_WINDOW_DAYS),
        "od-start": f"{od_start:02d}:00",
        "od-end": f"{od_end:02d}:00",
        "od-first-hours": str(od_end - od_start),
        "od-second-hours": str(len(HOURS) - (od_end - od_start)),
        "pickup-example-hour": str(PICKUP_HOUR_EXAMPLE),
        "pickup-example-start": f"{PICKUP_HOUR_EXAMPLE:02d}:00",
        "pickup-example-end": f"{PICKUP_HOUR_EXAMPLE + 1:02d}:00",
        "duration-tolerance": str(DURATION_TOLERANCE_SECONDS),
    }


def audit_report_values(trips, zone_boroughs):
    pickup = trips["pickup_datetime"]
    request = trips["request_datetime"]
    negative = pickup.lt(request)
    wait = trips["wait_minutes"].dropna()
    q1, q3 = wait.quantile([.25, .75])
    iqr = q3 - q1
    potential_outliers = ~wait.between(q1 - IQR_MULTIPLIER * iqr, q3 + IQR_MULTIPLIER * iqr)
    unknown_zone = ~trips[["PULocationID", "DOLocationID"]].isin(zone_boroughs.index).all(axis=1)
    outside_origin = trips["PULocationID"].map(zone_boroughs).eq("Outside NYC")
    bad_eda = (
        trips[["dropoff_datetime", "trip_time", "trip_miles", "base_passenger_fare"]].isna().any(axis=1)
        | ~trips[SHARED_MATCH].isin(("Y", "N"))
    )
    last_dropoff = trips["dropoff_datetime"].max()
    values = {
        "sample-rows": report_count(len(trips)),
        "bad-wait-times": report_count((request.isna() | pickup.isna()).sum()),
        "month-mismatches": report_count((pickup.notna() & pickup.dt.strftime("%Y-%m").ne(trips[MONTH])).sum()),
        "negative-waits": report_count(negative.sum()),
        "negative-wait-share": report_percent(negative.sum(), len(trips)),
        "zero-waits": report_count(pickup.eq(request).sum()),
        "bad-location-rows": report_count(unknown_zone.sum()),
        "outside-origin-rows": report_count(outside_origin.sum()),
        "other-service-rows": report_count((~trips[SERVICE].isin(SERVICES)).sum()),
        "invalid-weight-rows": report_count(invalid_weights(trips[WEIGHT]).sum()),
        "duplicate-source-keys": report_count(trips.duplicated([MONTH, SOURCE_ROW]).sum()),
        "potential-wait-outliers": report_count(potential_outliers.sum()),
        "bad-eda-rows": report_count(bad_eda.sum()),
        "unreadable-timestamp-rows": report_count(trips[list(TIMESTAMPS)].isna().any(axis=1).sum()),
        "last-dropoff-date": last_dropoff.strftime("%Y-%m-%d") if pd.notna(last_dropoff) else "--",
    }
    for key, year, service in zip(YEAR_KEYS, YEARS, SERVICES):
        service_rows = trips[SERVICE].eq(service)
        values[f"negative-waits-{key}-year"] = report_count((negative & pickup.dt.year.eq(year)).sum())
        values[f"negative-wait-share-{key}-service"] = report_percent(
            (negative & service_rows).sum(), service_rows.sum())
    return values


def univariate_report_values(trips, profiles, differences):
    values = {}
    pickup_year = trips["pickup_datetime"].dt.year
    for year_key, year in zip(YEAR_KEYS, YEARS):
        annual = trips.loc[pickup_year.eq(year)]
        total = annual[WEIGHT].sum()
        values[f"{year_key}-year-rows"] = report_count(len(annual))
        values[f"{year_key}-year-trips"] = report_count(total)
        category_masks = {
            "service-first": annual[SERVICE].eq(SERVICES[0]),
            "service-second": annual[SERVICE].eq(SERVICES[1]),
            "shared": annual[SHARED_MATCH].eq("Y"),
        }
        for name, mask in category_masks.items():
            amount = annual.loc[mask, WEIGHT].sum()
            values[f"{year_key}-{name}"] = f"{report_count(amount)} ({report_percent(amount, total)})"

        for column, name in REPORT_VARIABLES.items():
            valid = annual[[column, WEIGHT]].dropna(subset=[column])
            series, weights = valid[column], valid[WEIGHT]
            mean = np.average(series, weights=weights)
            sd = np.sqrt(np.average((series - mean) ** 2, weights=weights))
            q1, median, q3 = weighted_quantiles(series, weights, [.25, .5, .75])
            bins, p95 = profiles[column, year]
            bin_difference, p95_difference = differences[column, year]
            prefix = f"{year_key}-{name}"
            values.update({
                f"{prefix}-valid": report_count(len(valid)),
                f"{prefix}-mean": f"{mean:.2f}",
                f"{prefix}-sd": f"{sd:.2f}",
                f"{prefix}-median": f"{median:.2f}",
                f"{prefix}-quartile-span": f"{q1:.2f}-{q3:.2f}",
                f"{prefix}-p95": f"{p95:.2f}",
                f"{prefix}-outside-display": report_percent(max(0.0, 100 - bins.sum()), 100),
                f"{prefix}-bin-difference": f"{bin_difference:.3f}",
                f"{prefix}-p95-difference": f"{p95_difference:.3f}",
            })
        valid_wait = annual.dropna(subset=["wait_minutes"])
        long_wait = valid_wait.loc[valid_wait["wait_minutes"].gt(LONG_WAIT_MINUTES), WEIGHT].sum()
        values[f"{year_key}-long-wait-share"] = report_percent(long_wait, valid_wait[WEIGHT].sum())
    return values


def waiting_report_values(trips):
    hourly = hourly_wait_medians(trips)
    zone_shares = zone_wait_shares(trips)
    zones = zone_shares.dropna()
    distance = distance_wait_quantiles(trips)
    values = {
        "hourly-wait-rows": report_count(trips.dropna(subset=["wait_minutes", "request_datetime"]).shape[0]),
        "top-origin-count": report_count(len(zone_shares)),
        "smallest-zone-long-wait-share": report_percent(zones.min(), 1),
        "largest-zone-long-wait-share": report_percent(zones.max(), 1),
        "smallest-zone-id": f"{zones.idxmin():g}",
        "largest-zone-id": f"{zones.idxmax():g}",
        "short-distance-limit": f"{DISTANCE_EDGES[1]:g}",
        "long-distance-limit": f"{DISTANCE_EDGES[-2]:g}",
        "short-distance-wait-median": f"{distance[.5].iloc[0]:.2f}",
        "long-distance-wait-median": f"{distance[.5].iloc[-1]:.2f}",
    }
    for key, hour in zip(YEAR_KEYS, REQUEST_HOUR_EXAMPLES):
        values[f"request-hour-{key}"] = str(hour)
        values[f"weekday-wait-{key}"] = f"{hourly.loc[hour, False]:.2f}"
        values[f"weekend-wait-{key}"] = f"{hourly.loc[hour, True]:.2f}"
    return values


def weekly_extrema(windows):
    complete = windows.loc[windows["days"].eq(PICKUP_WINDOW_DAYS)].dropna(subset=["daily"])
    return {
        year: (group.loc[group["daily"].idxmin()], group.loc[group["daily"].idxmax()])
        for year, group in complete.groupby("year")
    }


def week_span(start, end):
    first = rf"\pn{{{start.day}}}"
    if start.month != end.month:
        first += f" {PERSIAN_MONTHS[start.month - 1]}"
    return first + rf" تا \pn{{{end.day}}} {PERSIAN_MONTHS[end.month - 1]}"


def timing_report_values(trips, counts, windows, extrema):
    cells = counts.stack()
    peak_day, peak_hour = cells.idxmax()
    low_day, low_hour = cells.idxmin()
    values = {
        "heatmap-sample-rows": report_count(trips["pickup_datetime"].dt.year.isin(YEARS).sum()),
        "heatmap-total": report_count(cells.sum()),
        "heatmap-peak-day": PERSIAN_DAYS[peak_day],
        "heatmap-peak-hour": str(peak_hour),
        "heatmap-peak-trips": report_count(cells.loc[peak_day, peak_hour]),
        "heatmap-low-day": PERSIAN_DAYS[low_day],
        "heatmap-low-hour": str(low_hour),
        "heatmap-low-trips": report_count(cells.loc[low_day, low_hour]),
        "covered-pickup-days": report_count(windows["observed"].sum()),
    }
    for key, year in zip(YEAR_KEYS, YEARS):
        tail = windows.loc[windows["year"].eq(year)].iloc[-1]
        values[f"{key}-year-partial-days"] = str(tail["days"] % PICKUP_WINDOW_DAYS)
        for name, row in zip(("low", "high"), extrema[year]):
            values[f"{key}-week-{name}-span"] = week_span(row["start"], row["end"])
            values[f"{key}-week-{name}-daily"] = report_count(row["daily"])
    return values


def daylight_saving_days():
    dates = pd.date_range(f"{YEARS[0]}-01-01", f"{YEARS[-1]}-12-31")
    local_noons = (dates + pd.Timedelta(hours=12)).tz_localize("America/New_York")
    offsets = pd.Series([time.utcoffset() for time in local_noons], index=dates)
    changes = offsets.diff().dropna()
    return changes.loc[changes.ne(pd.Timedelta(0))].index


def consistency_report_values(trips, speed, flags, summary):
    weights = trips[WEIGHT]
    speed_median, speed_p95 = weighted_quantiles(
        speed.dropna(), weights[speed.notna()], [.5, UPPER_PERCENTILE / 100])
    near_hour = duration_difference(trips).abs().sub(3600).abs().le(DURATION_TOLERANCE_SECONDS)
    transition = trips["pickup_datetime"].dt.normalize().isin(daylight_saving_days())
    values = {
        "speed-median": f"{speed_median:.2f}",
        "speed-p95": f"{speed_p95:.2f}",
        "consistency-distance-max": str(CONSISTENCY_DISTANCE_MAX),
        "dst-duration-mismatches": report_count((flags["duration_mismatch"] & transition & near_hour).sum()),
    }
    for name, (column, _, _, upper) in zip(("duration", "fare"), CONSISTENCY_PANELS):
        visible = consistency_visible(trips, column, upper)
        values[f"consistency-{name}-max"] = str(upper)
        values[f"consistency-{name}-coverage"] = report_percent(weights[visible].sum(), weights.sum())
    for group, mask in (("short", trips["trip_miles"].lt(DISTANCE_EDGES[1])),
                        ("long", trips["trip_miles"].ge(DISTANCE_EDGES[-2]))):
        for name, series in (("duration", trips["duration_minutes"]),
                             ("fare", trips["base_passenger_fare"]), ("speed", speed)):
            valid = mask & series.notna()
            median = weighted_quantiles(series[valid], weights[valid], [.5])[0]
            values[f"{group}-distance-{name}-median"] = f"{median:.2f}"
    for flag, row in summary.iterrows():
        name = flag.replace("_", "-")
        values[f"{name}-rows"] = report_count(row["sample_rows"])
        values[f"{name}-share"] = report_percent(row["weighted_pct"], 100, 4)
    return values


def geography_report_values(zone_boroughs, all_day):
    within_city = all_day.loc[list(CITY_BOROUGHS), list(CITY_BOROUGHS)].to_numpy()
    return {
        "city-group-count": str(len(CITY_BOROUGHS)),
        "od-group-count": str(len(BOROUGH_GROUPS)),
        "zone-unknown": ", ".join(map(str, zone_boroughs.index[zone_boroughs.eq("Unknown")])),
        "zone-outside": ", ".join(map(str, zone_boroughs.index[zone_boroughs.eq("Outside NYC")])),
        "same-borough-share": report_percent(np.trace(within_city), within_city.sum()),
        "outside-destination-share": report_percent(all_day.loc["Outside NYC"].sum(), all_day.to_numpy().sum()),
    }


def tlc_monthly_report_values(path):
    fhv_classes = ("FHV - Black Car", "FHV - Livery", "FHV - Lux Limo")
    classes = ("Green", "FHV - High Volume", *fhv_classes)
    months = pd.period_range(f"{YEARS[0]}-01", f"{YEARS[-1]}-12", freq="M")
    raw = pd.read_csv(path, usecols=["Month/Year", "License Class", "Trips Per Day"], dtype=str)
    selected = raw.loc[raw["Month/Year"].isin(months.astype(str)) & raw["License Class"].isin(classes)].copy()
    selected["Trips Per Day"] = pd.to_numeric(selected["Trips Per Day"].str.replace(",", "", regex=False))
    by_month = selected.pivot(index="Month/Year", columns="License Class", values="Trips Per Day")
    by_month = by_month.reindex(index=months.astype(str), columns=classes)
    if by_month.isna().any().any() or not np.isfinite(by_month).all().all() or by_month.lt(0).any().any():
        raise ValueError("The TLC indicators must include valid daily trip counts for every month and class.")
    days = pd.Series(months.days_in_month, index=by_month.index)
    daily_average = by_month.mul(days, axis=0).sum() / days.sum()
    return {
        "tlc-green-daily": report_count(daily_average["Green"]),
        "tlc-nonhv-daily": report_count(daily_average.loc[list(fhv_classes)].sum()),
        "tlc-hv-daily": report_count(daily_average["FHV - High Volume"]),
    }


def write_report_values(path, values):
    lines = ["% Generated by phase1_plots_v1.20.29.py. Regenerate from the CSV inputs."]
    lines.extend(rf"\reportdatum{{{key}}}{{{value}}}" for key, value in sorted(values.items()))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def publish_report_outputs(staging, output_dir):
    """Publish a completed run and make the values file available last."""
    output_dir.mkdir(parents=True, exist_ok=True)
    lock = output_dir / ".phase1-publishing"
    lock.mkdir()
    try:
        values_path = output_dir / VALUES_FILE
        values_path.unlink(missing_ok=True)
        for chart in sorted(staging.glob("*.pdf")):
            chart.replace(output_dir / chart.name)
        (staging / VALUES_FILE).replace(values_path)
    finally:
        lock.rmdir()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, nargs=2, help="HVFHV samples for 2024 and 2025")
    parser.add_argument("--output-dir", type=Path, default=Path("charts"))
    parser.add_argument("--zone-lookup", type=Path, default=Path("taxi_zone_lookup.csv"),
                        help="Taxi zone lookup CSV with LocationID and Borough columns")
    parser.add_argument("--tlc-monthly", type=Path,
                        help="Optional TLC data_reports_monthly.csv for the service-volume comparison")
    args = parser.parse_args()

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "pdf.fonttype": 42,
    })

    try:
        trips = load_trips(args.csv)
        zone_boroughs = load_zone_boroughs(args.zone_lookup)
        full, example, differences = resolution_study(trips)
        counts = pickup_counts(trips)
        if counts.to_numpy().sum() == 0:
            raise ValueError(f"No valid pickup times for {YEARS[0]} or {YEARS[1]} in the input files.")
        windows = weekly_pickups(trips)
        extrema = weekly_extrema(windows)
        speed, flags = trip_consistency(trips)
        weights = trips[WEIGHT]
        consistency_summary = pd.DataFrame({
            "sample_rows": flags.sum(),
            "weighted_pct": 100 * flags.mul(weights, axis=0).sum() / weights.sum(),
        })
        borough_trips = assign_boroughs(trips, zone_boroughs)
        matrices = od_by_pickup_period(borough_trips)
        values = (
            report_settings(trips, differences)
            | schema_report_values(args.csv)
            | audit_report_values(trips, zone_boroughs)
            | univariate_report_values(trips, full, differences)
            | waiting_report_values(trips)
            | timing_report_values(trips, counts, windows, extrema)
            | consistency_report_values(trips, speed, flags, consistency_summary)
            | geography_report_values(zone_boroughs, matrices["All pickup hours"])
        )
        if args.tlc_monthly is not None:
            values.update(tlc_monthly_report_values(args.tlc_monthly))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    limits = {
        column: max(1.0, 1.05 * max(p[column, year][0].max()
                                   for p in (full, example) for year in YEARS))
        for column, *_ in UNIVARIATE_SPECS
    }
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".phase1-", dir=args.output_dir.parent) as directory:
        staging = Path(directory)
        plot_univariate(full, limits, staging, "univariate.pdf")
        plot_univariate(example, limits, staging, "univariate_5000.pdf")
        plot_wait_by_hour(trips, staging)
        plot_wait_by_zone_and_distance(trips, staging)
        plot_trips_heatmap(counts, staging)
        plot_weekly_pickups(windows, staging)
        plot_trip_consistency(trips, staging)
        plot_origin_destination(matrices, staging)
        plot_borough_pickups(borough_pickup_counts(borough_trips), staging)
        write_report_values(staging / VALUES_FILE, values)
        try:
            publish_report_outputs(staging, args.output_dir)
        except OSError as exc:
            parser.error(f"Could not publish the report outputs: {exc}")

    peak_day, peak_hour = counts.stack().idxmax()
    print(f"Wrote 9 chart PDFs to {args.output_dir}.")
    print(f"Wrote the calculated report values to {args.output_dir / VALUES_FILE}.")
    print(f"Monthly sample-size check: {DRAWS} draws, seeds {FIRST_SEED}-{FIRST_SEED + DRAWS - 1}.")
    print(f"Worst deviations from the {REFERENCE_PER_MONTH:,}-row monthly reference:")
    for (column, year), (bin_pp, p95_percent) in differences.items():
        print(f"  {year} {column}: bin {bin_pp:.3f} pp; P95 {p95_percent:.3f}%")
    sufficient = all(bin_pp <= BIN_TOLERANCE_PP and p95_percent <= P95_TOLERANCE_PERCENT
                     for bin_pp, p95_percent in differences.values())
    verdict = "within" if sufficient else "outside"
    print(f"Observed differences are {verdict} the thresholds "
          f"({BIN_TOLERANCE_PP:g} pp per bin and {P95_TOLERANCE_PERCENT:g}% relative P95).")
    print(f"Heatmap total: {counts.to_numpy().sum():,.0f} trips represented by the sample.")
    print(f"Peak: {DAYS[peak_day]} {peak_hour:02d}:00, {counts.loc[peak_day, peak_hour]:,.0f} estimated trips.")
    print("Complete seven-day window extrema (estimated trips/day):")
    for year, rows in extrema.items():
        for label, row in zip(("minimum", "maximum"), rows):
            print(f"  {year} {label}: {row['start']:%Y-%m-%d} to {row['end']:%Y-%m-%d}, "
                  f"{row['daily']:,.0f} trips/day.")
    print("Initial consistency checks (categories may overlap):")
    print(consistency_summary.to_string(float_format=lambda value: f"{value:.4f}"))
    for period, matrix in matrices.items():
        print(f"OD {period}: {matrix.to_numpy().sum():,.0f} estimated trips, "
              f"{np.count_nonzero(matrix.to_numpy()):,} observed pairs.")


if __name__ == "__main__":
    main()
