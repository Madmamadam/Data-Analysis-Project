# Version 2.0
"""Plot pickup waits, monthly license shares, and origin–destination flows.

Run this module after main.py to save the plots in output/. The heatmaps use
all sampled trips as their denominator. The license charts show the individual
and combined shares of HV0003 and HV0005 in each pickup month.
The trip-volume heatmap uses the new sample's weights from merged.parquet.
Lyft and combined provider plots use the original license codes in merged.parquet.
"""

from pathlib import Path

from matplotlib.colors import LogNorm, Normalize
from matplotlib.dates import AutoDateLocator, DateFormatter
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, PercentFormatter
import numpy as np
import pandas as pd

import config
import io_part
from standardize_part import (
    INTEGER_COLUMNS,
    IS_UBER_COLUMN,
    LICENSE_COLUMN,
    TIMESTAMP_PARTS,
    UBER_LICENSE,
    _parse_integer,
    parse_tlc_timestamp,
)

EVENTS = ("request_datetime", "on_scene_datetime", "pickup_datetime")
EVENT_COLUMNS = [f"{event}_{part}" for event in EVENTS for part in TIMESTAMP_PARTS]
REQUEST_WAIT = "request_to_on_scene_minutes"
PICKUP_WAIT = "on_scene_to_pickup_minutes"
TOTAL_WAIT = "request_to_pickup_minutes"
PICKUP_YEAR = "pickup_datetime_year"
PICKUP_MONTH = "pickup_datetime_month"
OD_GROUPS = (
    "Bronx",
    "Brooklyn",
    "Manhattan",
    "Queens",
    "Staten Island",
    "Newark Airport",
    "Outside NYC",
    "Unknown",
)


def calculate_pickup_waits(frame: pd.DataFrame, *, include_total: bool = False) -> pd.DataFrame:
    """Calculate signed minutes per trip from standardized timestamp components."""
    timestamps = []
    for event in EVENTS:
        columns = [f"{event}_{part}" for part in TIMESTAMP_PARTS]
        components = frame[columns].set_axis(TIMESTAMP_PARTS, axis=1)
        complete = components.notna().all(axis=1)
        # Arrow may deliver nullable integer columns as floats. Mask missing parts after assembly.
        timestamp = pd.to_datetime(components.fillna(0).astype("int64"), errors="coerce")
        timestamps.append(timestamp.where(complete))

    request, on_scene, pickup = timestamps
    waits = pd.DataFrame({
        REQUEST_WAIT: (on_scene - request).dt.total_seconds() / 60,
        PICKUP_WAIT: (pickup - on_scene).dt.total_seconds() / 60,
    })
    if include_total:
        waits[TOTAL_WAIT] = (pickup - request).dt.total_seconds() / 60
    return waits


def plot_pickup_waits(
    working_path: Path = config.WORKING_PARQUET,
    output_path: Path = config.CODE_DIR / "output" / "pickup_wait_times.png",
    *,
    total_output_path: Path | None = None,
) -> Path:
    """Compare waits with arrival-to-pickup and optionally with total time to pickup."""
    include_total = total_output_path is not None
    if total_output_path is not None and output_path.resolve() == total_output_path.resolve():
        raise ValueError("Choose different paths for the two pickup wait plots.")

    request_bins = np.linspace(0, 30, 61)
    pickup_bins = np.linspace(0, 10, 41)
    total_bins = np.linspace(0, 40, 81)
    pickup_counts = np.zeros((len(request_bins) - 1, len(pickup_bins) - 1))
    total_counts = np.zeros((len(request_bins) - 1, len(total_bins) - 1))
    missing = negative = chronological = 0

    with io_part.read_parquet_batches(
        working_path, config.CHUNK_ROWS, columns=EVENT_COLUMNS
    ) as (schema, _, batches):
        absent = sorted(set(EVENT_COLUMNS).difference(schema.names))
        if absent:
            raise ValueError(
                f"Expected standardized timestamp columns in {working_path}. Missing {', '.join(absent)}"
            )
        for batch in batches:
            waits = calculate_pickup_waits(batch.to_pandas(), include_total=include_total)
            complete = waits.dropna()
            usable = complete[complete.ge(0).all(axis=1)]
            missing += len(waits) - len(complete)
            negative += len(complete) - len(usable)
            chronological += len(usable)
            pickup_counts += np.histogram2d(
                usable[REQUEST_WAIT], usable[PICKUP_WAIT], bins=(request_bins, pickup_bins)
            )[0]
            if include_total:
                total_counts += np.histogram2d(
                    usable[REQUEST_WAIT], usable[TOTAL_WAIT], bins=(request_bins, total_bins)
                )[0]

    excluded = (
        f"Excluded trips  |  {missing:,} missing a timestamp  |  {negative:,} with negative intervals"
    )
    all_trips = missing + negative + chronological
    _save_wait_density(
        output_path, pickup_counts, request_bins, pickup_bins, chronological, excluded,
        all_trips=all_trips, title="Wait times before pickup", y_label="On-scene to pickup (minutes)",
    )
    if total_output_path is not None:
        _save_wait_density(
            total_output_path, total_counts, request_bins, total_bins, chronological, excluded,
            all_trips=all_trips, title="Total wait time to pickup", y_label="Request to pickup (minutes)",
            show_diagonal=True,
        )
    return output_path


def _save_wait_density(
    output_path: Path,
    counts: np.ndarray,
    request_bins: np.ndarray,
    y_bins: np.ndarray,
    chronological: int,
    excluded: str,
    *,
    all_trips: int,
    title: str,
    y_label: str,
    show_diagonal: bool = False,
) -> None:
    plotted = int(counts.sum())
    x_bin_seconds = round(60 * (request_bins[1] - request_bins[0]))
    y_bin_seconds = round(60 * (y_bins[1] - y_bins[0]))
    coverage = (
        f"{plotted:,} of {all_trips:,} trips shown in {x_bin_seconds}-second by {y_bin_seconds}-second bins\n"
        f"{excluded}  |  {chronological - plotted:,} beyond the axes"
    )

    percent_per_trip = 100 / max(all_trips, 1)
    percentages = counts * percent_per_trip
    maximum = max(1, counts.max()) * percent_per_trip
    minimum = min(percent_per_trip, maximum / 2)

    figure = Figure(figsize=(9, 6))
    axes = figure.subplots()
    figure.subplots_adjust(left=0.11, right=0.96, bottom=0.20, top=0.91)
    density = axes.pcolormesh(
        request_bins, y_bins, np.ma.masked_equal(percentages.T, 0),
        cmap="magma", norm=LogNorm(vmin=minimum, vmax=maximum),
    )
    if show_diagonal:
        endpoint = min(request_bins[-1], y_bins[-1])
        axes.plot([0, endpoint], [0, endpoint], "--", color="#64748b", label="Pickup at arrival")
        axes.legend(loc="lower right")

    figure.colorbar(
        density,
        ax=axes,
        label="Share of all trips per bin (log scale)",
        format=FuncFormatter(lambda value, _: f"{value:.8f}".rstrip("0").rstrip(".") + "%"),
    )
    axes.set_title(title, loc="left", fontsize=15, fontweight="bold")
    axes.set_xlabel("Request to on-scene (minutes)")
    axes.set_ylabel(y_label)
    figure.text(0.11, 0.055, coverage, fontsize=9, color="#475569", linespacing=1.5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    print(f"Saved pickup wait plot to {output_path}\n{coverage}")


def _iter_license_trips(
    path: Path,
    license_codes: tuple[str, ...],
    location_columns: tuple[str, ...] = (),
):
    """Read the fields for a provider plot from the raw or standardized schema."""
    legacy_flags = [f"{LICENSE_COLUMN}_{code}" for code in license_codes]
    pickup_column = "pickup_datetime"
    columns = [
        PICKUP_YEAR, PICKUP_MONTH, pickup_column, LICENSE_COLUMN, IS_UBER_COLUMN,
        *legacy_flags, *location_columns,
    ]
    with io_part.read_parquet_batches(path, config.CHUNK_ROWS, columns=columns) as (schema, _, batches):
        raw = LICENSE_COLUMN in schema.names
        flags = legacy_flags
        if license_codes == (UBER_LICENSE,) and IS_UBER_COLUMN in schema.names:
            flags = [IS_UBER_COLUMN]
        required = [pickup_column, LICENSE_COLUMN] if raw else [PICKUP_YEAR, PICKUP_MONTH, *flags]
        absent = sorted(set([*required, *location_columns]).difference(schema.names))
        if absent:
            raise ValueError(f"Missing license plot fields in {path}: {', '.join(absent)}")

        for batch in batches:
            trips = batch.to_pandas()
            if raw:
                pickup = parse_tlc_timestamp(trips[pickup_column].astype("string").str.strip())
                trips[PICKUP_YEAR] = pickup.dt.year
                trips[PICKUP_MONTH] = pickup.dt.month
                licenses = trips[LICENSE_COLUMN].astype("string").str.strip()
                trips["matches_license"] = licenses.isin(license_codes)
                for column in location_columns:
                    values = trips[column].astype("string").str.strip()
                    trips[column] = _parse_integer(values, INTEGER_COLUMNS[column])
            else:
                trips["matches_license"] = trips[flags].eq(1).any(axis=1)
            yield trips


def plot_hv0005_monthly_share(
    working_path: Path = config.MERGED_PARQUET,
    output_path: Path = config.CODE_DIR / "output" / "hv0005_monthly_share.png",
) -> Path:
    """Plot the percentage of sampled trips with HV0005 (Lyft) by pickup month."""
    return plot_license_monthly_share("HV0005", working_path, output_path)


def plot_license_monthly_share(
    license_code: str | tuple[str, ...],
    working_path: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    """Plot monthly shares using the Uber flag or the raw license codes by default."""
    license_codes = (license_code,) if isinstance(license_code, str) else license_code
    if not license_codes:
        raise ValueError("Provide at least one license code.")
    license_label = " + ".join(license_codes)

    if working_path is None:
        working_path = (
            config.WORKING_PARQUET if license_codes == (UBER_LICENSE,) else config.MERGED_PARQUET
        )
    year, month = PICKUP_YEAR, PICKUP_MONTH
    batch_counts = []
    trip_count = 0

    for trips in _iter_license_trips(working_path, license_codes):
        trip_count += len(trips)
        counts = trips.groupby([year, month])["matches_license"].agg(
            license_trips="sum", all_trips="size"
        )
        batch_counts.append(counts)

    if not batch_counts or all(counts.empty for counts in batch_counts):
        raise ValueError(f"No trips with a pickup month in {working_path}")

    monthly = pd.concat(batch_counts).groupby(level=[0, 1]).sum()
    months = pd.PeriodIndex.from_fields(
        year=monthly.index.get_level_values(year).to_numpy(dtype=int),
        month=monthly.index.get_level_values(month).to_numpy(dtype=int),
        freq="M",
    )
    share = monthly["license_trips"].div(monthly["all_trips"]).mul(100)
    share.index = months
    share = share.reindex(pd.period_range(months.min(), months.max(), freq="M"))

    figure = Figure(figsize=(9, 5))
    axes = figure.subplots()
    figure.subplots_adjust(left=0.11, right=0.96, bottom=0.23, top=0.88)
    axes.plot(share.index.to_timestamp(), share, color="#146b83", marker="o", markersize=4)
    axes.set_title(f"{license_label} share of trips by month", loc="left", fontsize=15, fontweight="bold")
    axes.set_xlabel("Pickup month")
    axes.set_ylabel("Share of trips")
    axes.xaxis.set_major_locator(AutoDateLocator(minticks=4, maxticks=10))
    axes.xaxis.set_major_formatter(DateFormatter("%b %Y"))
    axes.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    axes.margins(x=0.02)
    if share.nunique() == 1:
        axes.set_ylim(-5, 105)
    else:
        lower, upper = axes.get_ylim()
        axes.set_ylim(max(0, lower), min(100, upper))
    axes.grid(axis="y", alpha=0.25)

    missing_month = trip_count - monthly["all_trips"].sum()
    coverage = "Share of all sampled trips in each pickup month"
    if missing_month:
        coverage += f"  |  {missing_month:,} trips missing a pickup month"
    figure.text(0.11, 0.055, coverage, fontsize=9, color="#475569")

    if output_path is None:
        license_filename = "_".join(code.lower() for code in license_codes)
        output_path = config.CODE_DIR / "output" / f"{license_filename}_monthly_share.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    print(f"Saved {license_label} monthly share plot to {output_path}")
    return output_path


def _load_od_location_groups(zone_lookup_path: Path) -> pd.Series:
    lookup = pd.read_csv(
        zone_lookup_path, usecols=["LocationID", "Borough"], keep_default_na=False
    )
    if lookup["LocationID"].isna().any() or lookup["LocationID"].duplicated().any():
        raise ValueError(f"Invalid or duplicate LocationID values in {zone_lookup_path}")

    borough_groups = {
        "EWR": "Newark Airport",
        "N/A": "Outside NYC",
        "Unknown": "Unknown",
    }
    lookup["od_group"] = lookup["Borough"].replace(borough_groups)
    unexpected = sorted(set(lookup["od_group"]).difference(OD_GROUPS))
    if unexpected:
        raise ValueError(f"Unexpected borough groups in {zone_lookup_path}: {', '.join(unexpected)}")
    return lookup.set_index("LocationID")["od_group"]


def calculate_lyft_od_shares(
    working_path: Path = config.MERGED_PARQUET,
    zone_lookup_path: Path = config.TAXI_ZONE_LOOKUP,
) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    """Return full-period Lyft shares and monthly share ranges by OD group."""
    location_groups = _load_od_location_groups(zone_lookup_path)

    year, month = PICKUP_YEAR, PICKUP_MONTH
    grouped_batches = []
    all_trips = 0

    for trips in _iter_license_trips(working_path, ("HV0005",), ("PULocationID", "DOLocationID")):
        all_trips += len(trips)
        trips = trips.dropna(subset=[year, month]).copy()
        trips["origin"] = trips["PULocationID"].map(location_groups).fillna("Unknown")
        trips["destination"] = trips["DOLocationID"].map(location_groups).fillna("Unknown")
        grouped_batches.append(
            trips.groupby([year, month, "origin", "destination"], observed=True)["matches_license"]
            .agg(lyft_trips="sum", all_trips="size")
        )

    if not grouped_batches or all(group.empty for group in grouped_batches):
        raise ValueError(f"No trips with a pickup month in {working_path}")

    monthly = pd.concat(grouped_batches).groupby(level=[0, 1, 2, 3]).sum()
    months = monthly.index.droplevel(["origin", "destination"]).unique()
    period_total = monthly.groupby(level=["origin", "destination"]).sum()
    period_share = period_total["lyft_trips"].div(period_total["all_trips"]).mul(100)
    period_matrix = period_share.unstack("origin").reindex(index=OD_GROUPS, columns=OD_GROUPS)

    monthly_share = monthly["lyft_trips"].div(monthly["all_trips"]).mul(100)
    extrema = monthly_share.groupby(level=["origin", "destination"]).agg(
        minimum="min", maximum="max", observed_months="count"
    )
    extrema["range"] = extrema["maximum"] - extrema["minimum"]
    extrema.loc[extrema["observed_months"].lt(2), "range"] = np.nan
    range_matrix = extrema["range"].unstack("origin").reindex(
        index=OD_GROUPS, columns=OD_GROUPS
    )
    return period_matrix, range_matrix, all_trips, len(months)


def plot_lyft_od_shares(
    working_path: Path = config.MERGED_PARQUET,
    zone_lookup_path: Path = config.TAXI_ZONE_LOOKUP,
    output_path: Path = config.CODE_DIR / "output" / "lyft_od_share.png",
    range_output_path: Path = config.CODE_DIR / "output" / "lyft_od_monthly_share_range.png",
) -> tuple[Path, Path]:
    """Plot Lyft share by OD group and its max-minus-min monthly variation."""
    if output_path.resolve() == range_output_path.resolve():
        raise ValueError("Choose different paths for the two Lyft OD plots.")
    period_share, monthly_range, all_trips, month_count = calculate_lyft_od_shares(
        working_path, zone_lookup_path
    )
    footer = f"HV0005 (Lyft) share within each OD cell  |  {all_trips:,} sampled trips"
    _save_od_heatmap(
        period_share,
        output_path,
        title="Lyft share of all trips by origin and destination",
        colorbar_label="Lyft share of trips",
        annotation_suffix="%",
        footer=footer,
    )
    _save_od_heatmap(
        monthly_range,
        range_output_path,
        title="Monthly range of Lyft share by origin and destination",
        colorbar_label="Maximum − minimum monthly share (percentage points)",
        annotation_suffix=" pp",
        footer=(
            f"Maximum minus minimum Lyft share across {month_count} pickup months"
            "  |  Months with no trips in a cell are excluded"
        ),
    )
    return output_path, range_output_path


def _save_od_heatmap(
    values: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
    colorbar_label: str,
    annotation_suffix: str,
    footer: str,
    maximum: float | None = None,
) -> None:
    """Save an annotated origin-destination percentage heatmap."""
    data = values.to_numpy(dtype=float)
    finite = data[np.isfinite(data)]
    if not len(finite):
        raise ValueError("The origin-destination table has no values to plot.")
    upper = maximum if maximum is not None else max(float(finite.max()), 1)
    norm = Normalize(vmin=0, vmax=upper)
    color_map = __import__("matplotlib").colormaps["viridis"].copy()
    color_map.set_bad("#e5e7eb")

    figure = Figure(figsize=(10, 8))
    axes = figure.subplots()
    figure.subplots_adjust(left=0.20, right=0.88, bottom=0.20, top=0.88)
    image = axes.imshow(np.ma.masked_invalid(data), cmap=color_map, norm=norm, aspect="equal")
    figure.colorbar(
        image,
        ax=axes,
        label=colorbar_label,
        format=FuncFormatter(lambda value, _: f"{value:g}%" if annotation_suffix == "%" else f"{value:g}"),
        fraction=0.046,
        pad=0.04,
    )

    tick_labels = [label.replace(" ", "\n") for label in OD_GROUPS]
    axes.set_xticks(range(len(OD_GROUPS)), labels=tick_labels)
    axes.set_yticks(range(len(OD_GROUPS)), labels=OD_GROUPS)
    axes.set_xlabel("Pickup group (origin)")
    axes.set_ylabel("Drop-off group (destination)")
    axes.set_title(title, loc="left", fontsize=15, fontweight="bold", pad=12)
    axes.tick_params(axis="x", labelsize=9)
    axes.tick_params(axis="y", labelsize=9)
    axes.set_xticks(np.arange(-0.5, len(OD_GROUPS), 1), minor=True)
    axes.set_yticks(np.arange(-0.5, len(OD_GROUPS), 1), minor=True)
    axes.grid(which="minor", color="white", linewidth=1)
    axes.tick_params(which="minor", bottom=False, left=False)

    for row in range(data.shape[0]):
        for column in range(data.shape[1]):
            value = data[row, column]
            if not np.isfinite(value):
                label, color = "—", "#475569"
            else:
                label = f"{value:.1f}{annotation_suffix}"
                color = "white" if norm(value) < 0.63 else "#111827"
            axes.text(column, row, label, ha="center", va="center", fontsize=8, color=color)

    figure.text(0.20, 0.055, footer, fontsize=9, color="#475569")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    print(f"Saved Lyft OD plot to {output_path}")


def _compact_trip_count(value: float) -> str:
    """Format an estimated trip count for the reference plot."""
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}".removesuffix(".0") + "M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}".removesuffix(".0") + "k"
    return f"{value:.0f}"


def plot_origin_destination_flows(
    merged_path: Path = config.MERGED_PARQUET,
    zone_lookup_path: Path = config.TAXI_ZONE_LOOKUP,
    output_path: Path = config.CODE_DIR / "output" / "origin_destination_flows.png",
) -> Path:
    """Plot all-hours estimated trip volumes using the merged sample weights."""
    location_groups = _load_od_location_groups(zone_lookup_path)
    weight = "_sample_weight"
    columns = ["PULocationID", "DOLocationID", "pickup_datetime", weight]
    grouped_batches = []
    starts = []
    ends = []
    with io_part.read_parquet_batches(
        merged_path, config.CHUNK_ROWS, columns=columns
    ) as (schema, _, batches):
        absent = sorted(set(columns).difference(schema.names))
        if absent:
            raise ValueError(f"Expected raw sample fields in {merged_path}. Missing {', '.join(absent)}")
        for batch in batches:
            trips = batch.to_pandas()
            weights = pd.to_numeric(trips[weight], errors="coerce")
            if not (np.isfinite(weights) & weights.gt(0)).all():
                raise ValueError(f"Sample weights in {merged_path} must be finite and positive.")
            pickup = pd.to_datetime(trips["pickup_datetime"], format="ISO8601", errors="coerce")
            valid = pickup.notna()
            if not valid.any():
                continue
            starts.append(pickup.min())
            ends.append(pickup.max())
            origins = pd.to_numeric(trips.loc[valid, "PULocationID"], errors="coerce")
            destinations = pd.to_numeric(trips.loc[valid, "DOLocationID"], errors="coerce")
            od_trips = pd.DataFrame({
                "origin": origins.map(location_groups).fillna("Unknown"),
                "destination": destinations.map(location_groups).fillna("Unknown"),
                "weight": weights[valid],
                "morning": pickup[valid].dt.hour.between(3, 10),
            })
            grouped_batches.append(
                od_trips.groupby(["destination", "origin", "morning"])["weight"].sum()
            )
    if not grouped_batches:
        raise ValueError(f"No valid pickup timestamps in {merged_path}")

    # Retain the reference plot's shared scale across all hours and its two time windows.
    window_totals = pd.concat(grouped_batches).groupby(level=[0, 1, 2]).sum()
    totals = window_totals.groupby(level=["destination", "origin"]).sum()
    matrix = totals.unstack("origin").reindex(index=OD_GROUPS, columns=OD_GROUPS)
    minimum, maximum = window_totals.min(), totals.max()
    norm = LogNorm(vmin=minimum, vmax=maximum if maximum > minimum else minimum * 2)

    figure = Figure(figsize=(9.2, 6.6), layout="constrained")
    axes = figure.subplots()
    axes.set_facecolor("#eef1f3")
    data = matrix.to_numpy()
    mesh = axes.pcolormesh(
        np.ma.masked_invalid(data), cmap="viridis", norm=norm, edgecolors="white", linewidth=0.6
    )
    ticks = np.arange(len(OD_GROUPS)) + 0.5
    axes.set_xticks(ticks, [group.replace(" ", "\n") for group in OD_GROUPS])
    axes.set_yticks(ticks, OD_GROUPS)
    axes.invert_yaxis()
    axes.set_aspect("equal")
    axes.set_xlabel("Pickup group (origin)")
    axes.set_ylabel("Drop-off group (destination)")
    axes.tick_params(length=0, labelsize=9)
    for spine in axes.spines.values():
        spine.set_color("#cad7da")
    for row, column in zip(*np.nonzero(np.isfinite(data))):
        value = data[row, column]
        red, green, blue, _ = mesh.cmap(norm(value))
        brightness = 0.2126 * red + 0.7152 * green + 0.0722 * blue
        axes.text(
            column + 0.5, row + 0.5, _compact_trip_count(value), ha="center", va="center",
            fontsize=9, color="white" if brightness < 0.5 else "#162b38",
        )
    colorbar = figure.colorbar(
        mesh, ax=axes, shrink=0.84, pad=0.035,
        format=FuncFormatter(lambda value, _: _compact_trip_count(value)),
    )
    colorbar.set_label("Estimated trips (log scale)")
    colorbar.outline.set_visible(False)
    date_range = f"{min(starts):%b %Y} – {max(ends):%b %Y}"
    figure.suptitle(
        f"Origin–destination flows | All pickup hours\n{date_range}", fontsize=14, fontweight="bold"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, facecolor="white")
    print(f"Saved origin-destination flow plot to {output_path}")
    return output_path


if __name__ == "__main__":
    plot_pickup_waits(total_output_path=config.CODE_DIR / "output" / "request_to_pickup_wait.png")
    plot_license_monthly_share(UBER_LICENSE)
    plot_hv0005_monthly_share()
    plot_license_monthly_share((UBER_LICENSE, "HV0005"))
    plot_lyft_od_shares()
    if config.MERGED_PARQUET.is_file():
        plot_origin_destination_flows()
