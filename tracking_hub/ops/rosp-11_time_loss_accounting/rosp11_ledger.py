"""Shared functions for the ROSP-11 time-loss ledger.

The ledger assigns every second of the observable night (Sun -12 deg to -12 deg, ROSP-22) to exactly one
category of the ROSP-24 loss-cause taxonomy, by the ROSP-24 precedence

    B1 -> C -> B2 -> D (D2, D1, D4, D3) -> E -> A -> F

It is used by ``rosp-11_time_loss_accounting.ipynb``. The state handling (seeding each night from the last
Scheduler ``observatoryStatus`` message before twilight, the 30 h freshness limit, the FAULT-clause parser and
the ROSP-42 override reader) is adapted from the supporting notebook ``fault_downtime.ipynb``.

Every rule that is still pending a decision is a field of :class:`Rules` (or a row of its Weather Constraints
table), not a literal in the code, so the notebook can print the full rule set it ran with and a change of rule
is a change of one field.

Times are TAI MJD floats throughout. The EFD is UTC and is converted on read; ConsDB ``obs_start_mjd`` and
``obs_end_mjd`` are already TAI.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import re
import urllib.request
from collections import OrderedDict

import numpy as np
import pandas as pd
from astropy.time import Time, TimeDelta

# Bumped whenever a function signature or a returned column changes, so the notebook can refuse a module
# that does not match it (on Times Square the module is fetched from GitHub, not read from the checkout).
MODULE_API_VERSION = 2

TAXONOMY_VERSION = "v1.0 (pending ROSP-24 sign-off; rule updates of 2026-11-16/17)"

GITHUB_REPO = "lsst/sp_notebooks"
OVERRIDES_PATH = (
    "tracking_hub/ops/observatory_state_overrides/observatory_state_overrides.csv"
)

SECONDS_PER_DAY = 86400.0
MINUTE = 1.0 / 1440.0

# ---------------------------------------------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------------------------------------------

CATEGORIES = OrderedDict(
    [
        ("A1", "Science: survey open-shutter time"),
        ("A2", "Science: overhead within the slew model"),
        ("A3", "Science: start-up within the allowance"),
        ("B1", "Scheduled: planned maintenance"),
        ("B2a", "Scheduled: on-sky engineering"),
        ("B2b", "Scheduled: closed-dome calibration / engineering, good conditions"),
        ("C1", "Weather: closed by conditions"),
        ("C2", "Weather: open but unusable"),
        ("C3", "Weather: standby"),
        ("C4", "Weather: summit access closed"),
        ("C5", "Weather: attributed from neighbouring observatories"),
        ("D1", "Fault: short (recovered within the night)"),
        ("D2", "Fault: long (whole night)"),
        ("D3", "Fault: unreported (visit-gap excess)"),
        ("D4", "Fault: unplanned downtime"),
        ("E", "Operational idle"),
        ("F", "Unattributed"),
    ]
)
CATEGORY_ORDER = list(CATEGORIES)
GROUP_NAMES = OrderedDict(
    [
        ("A", "Science operations (not lost)"),
        ("B", "Scheduled"),
        ("C", "Weather"),
        ("D", "Technical fault"),
        ("E", "Operational idle"),
        ("F", "Unattributed"),
    ]
)
PRECEDENCE = ("B1", "C", "B2", "D2", "D1", "D4", "D3", "E", "A", "F")
# Override categories. B1 / C1 / C4 / C5 / D4 are the ROSP-42 file as merged; D1 / D2 / B2a are the ROSP-58
# decision-2 proposal (attribution and recorded-state corrections), read if present.
OVERRIDE_CATEGORIES = ("B1", "C1", "C4", "C5", "D4", "D1", "D2", "B2a")

# Scheduler logevent_observatoryStatus bitmask (salIndex 1). status == 0 is UNKNOWN.
STATUS_BITS = {
    "DAYTIME": 1,
    "OPERATIONAL": 2,
    "FAULT": 4,
    "WEATHER": 8,
    "DOWNTIME": 16,
    "IDLE": 32,
}
NO_STATE = -1  # no fresh status message (telemetry gap, or before the first message)
SUMMARY_STATE_FAULT = 3

# ROSP-24 fault attribution groups (Q5, the mapping check by the subsystem leads, is still open).
PENDING = "Pending attribution"
UNATTRIBUTED_FAULT = "F (unattributed)"
FAULT_GROUPS = (
    "Telescope mount and structure",
    "Optics and active optics",
    "Camera",
    "Dome and enclosure",
    "Facility utilities",
    "Safety systems",
    "Observatory control software",
    "IT, networks and data services",
    "Operations decision",
    "Site and external",
    PENDING,
    UNATTRIBUTED_FAULT,
)
CSC_GROUPS = {
    "MTMount": "Telescope mount and structure",
    "MTRotator": "Telescope mount and structure",
    # MTPtg is not in the ROSP-24 table; it is grouped with the mount, as fault_downtime does (Q5).
    "MTPtg": "Telescope mount and structure",
    "MTM1M3": "Optics and active optics",
    "MTM1M3TS": "Optics and active optics",
    "MTM2": "Optics and active optics",
    "MTHexapod": "Optics and active optics",
    "MTAOS": "Optics and active optics",
    "MTCamera": "Camera",
    "MTOODS": "Camera",
    "MTHeaderService": "Camera",
    "MTDome": "Dome and enclosure",
    "MTDomeTrajectory": "Dome and enclosure",
    "GIS": "Safety systems",
    "Scheduler": "Observatory control software",
    "ScriptQueue": "Observatory control software",
    "Watcher": "Observatory control software",
}

# Only the machine-generated clause, and only its current ("are in") form: "were in" is sticky and repeats long
# after recovery. Operator text before the clause must not be searched. (From fault_downtime.ipynb.)
FAULT_CSC_RE = re.compile(r"components are in <State\.FAULT: \d+> state:\s*([^.]*)\.")
FAULT_CSC_WERE_RE = re.compile(
    r"components were in <State\.FAULT: \d+> state:\s*([^.]*)\."
)

# Weather Constraints versions (Confluence page 39696225), applied by the date in force. The close limits are the
# ROSP-24 C1 limits. Every date and every re-opening value here is a PLACEHOLDER until Brian supplies the page
# history: the 2026-10-01 wind switch is assumed (ROSP-11, 2026-11-17), and the 2026-11-06 row marks the "11/06
# adopted version" named on ROSP-58, with values not yet known.
DEFAULT_WEATHER_CONSTRAINTS = (
    {
        "version": "WC-1 (placeholder)",
        "from_dayobs": 0,
        "rh_close": 80.0,
        "dpd_close": 2.0,
        "wind_close": 15.0,
        "gust_close": 18.0,
        "rh_reopen": 75.0,
        "dpd_reopen": 3.0,
        "wind_reopen": 15.0,
        "gust_reopen": 18.0,
        "reopen_hold_min": 30.0,
        "precip_hold_min": 60.0,
        "icing_temp_c": 0.0,
        "icing_rh": 90.0,
        "status": "placeholder: re-opening values and date to confirm",
    },
    {
        "version": "WC-2 (placeholder, wind 20/25 m/s)",
        "from_dayobs": 20261001,
        "rh_close": 80.0,
        "dpd_close": 2.0,
        "wind_close": 20.0,
        "gust_close": 25.0,
        "rh_reopen": 75.0,
        "dpd_reopen": 3.0,
        "wind_reopen": 20.0,
        "gust_reopen": 25.0,
        "reopen_hold_min": 30.0,
        "precip_hold_min": 60.0,
        "icing_temp_c": 0.0,
        "icing_rh": 90.0,
        "status": "placeholder: wind-switch date 20261001 unverified",
    },
    {
        "version": "WC-3 (placeholder, adopted 2026-11-06)",
        "from_dayobs": 20261106,
        "rh_close": 80.0,
        "dpd_close": 2.0,
        "wind_close": 20.0,
        "gust_close": 25.0,
        "rh_reopen": 75.0,
        "dpd_reopen": 3.0,
        "wind_reopen": 20.0,
        "gust_reopen": 25.0,
        "reopen_hold_min": 30.0,
        "precip_hold_min": 60.0,
        "icing_temp_c": 0.0,
        "icing_rh": 90.0,
        "status": "placeholder: values of the 11/06 version not yet known",
    },
)
CONSTRAINT_FIELDS = (
    "rh_close",
    "dpd_close",
    "wind_close",
    "gust_close",
    "rh_reopen",
    "dpd_reopen",
    "wind_reopen",
    "gust_reopen",
    "reopen_hold_min",
    "precip_hold_min",
    "icing_temp_c",
    "icing_rh",
)


@dataclasses.dataclass(frozen=True)
class Rules:
    """Every ledger rule that is a parameter rather than a literal. RULE_NOTES says which are decided."""

    taxonomy_version: str = TAXONOMY_VERSION
    twilight_alt: float = -12.0
    survey_programs: tuple = (
        "BLOCK-407",
        "BLOCK-408",
        "BLOCK-416",
        "BLOCK-417",
        "BLOCK-419",
        "BLOCK-421",
    )
    a1_img_types: tuple = ("science",)
    # Start-up (A3). T539 (initial alignment) is the start-up procedure, not engineering (Brian, 2026-11-17).
    startup_programs: tuple = ("BLOCK-T539",)
    startup_allowance_min: float = 10.0
    # Where the allowance starts: "twilight" (-12 deg, ROSP-50 as written) or "dome_release" (the ROSP-58 Q6
    # amendment: from dome release, with -12 deg -> release charged as D4). Release = the dome opening after it
    # has been closed for at least `release_min_closed_min`.
    startup_origin: str = "twilight"
    release_min_closed_min: float = 15.0
    startup_overrun_category: str = "D1"
    # A2 vs D3: a visit gap is overhead up to the slew model plus this scatter.
    gap_scatter_s: float = 10.0
    gap_fallback_s: float = 60.0
    # D3 cap (ROSP-23): a gap longer than this, with no FAULT and no engineering inside, is E, not D3. None means
    # the `d3_cap_quantile` of the non-fault survey-to-survey gaps in the range.
    d3_cap_s: float | None = None
    d3_cap_quantile: float = 0.99
    activity_join_min: float = 10.0
    weather_constraints: tuple = DEFAULT_WEATHER_CONSTRAINTS
    cloud_index_max: float = 0.5
    dome_open_pct: float = 5.0
    # A rain report is trusted for at most this long without a following report.
    rain_stale_h: float = 3.0
    # Weather standby is evaluated from this many hours before evening twilight, so daytime rain or icing count.
    standby_lead_h: float = 6.0
    d2_tolerance_min: float = 30.0
    max_state_age_h: float = 30.0
    fault_onset_tolerance_s: float = 120.0
    fault_onset_lookback_h: float = 24.0
    # A FAULT set by hand with no CSC: its hours stay in D1/D2 with group "F (unattributed)" ("D"), or move to
    # category F ("F").
    no_csc_fault_category: str = "D"

    def constraints(self, day_obs):
        """Per-element Weather Constraints values in force on day_obs (array-like), plus the version name."""
        day_obs = np.asarray(day_obs)
        out = {k: np.full(day_obs.shape, np.nan) for k in CONSTRAINT_FIELDS}
        version = np.full(day_obs.shape, "", dtype=object)
        for row in sorted(self.weather_constraints, key=lambda r: r["from_dayobs"]):
            on = day_obs >= row["from_dayobs"]
            for k in CONSTRAINT_FIELDS:
                out[k][on] = row[k]
            version[on] = row["version"]
        out["version"] = version
        return out

    def weather_table(self):
        return pd.DataFrame(list(self.weather_constraints)).set_index("version")

    def as_table(self):
        rows = []
        for f in dataclasses.fields(self):
            if f.name == "weather_constraints":
                value = f"{len(self.weather_constraints)} versions (table below)"
            else:
                value = getattr(self, f.name)
            status, note = RULE_NOTES.get(f.name, ("", ""))
            rows.append(
                {
                    "rule": f.name,
                    "value": value,
                    "status": status,
                    "source / note": note,
                }
            )
        return pd.DataFrame(rows).set_index("rule")


RULE_NOTES = {
    "taxonomy_version": (
        "pending",
        "ROSP-24 v1.0 + Decision Log v32 rule updates (2026-11-16/17)",
    ),
    "twilight_alt": ("decided", "Q1, ROSP-22"),
    "survey_programs": ("parameter", "Early Operations science blocks, as ROSP-2"),
    "a1_img_types": ("parameter", "acq/cwfs inside a survey block count as A2"),
    "startup_programs": (
        "decided",
        "T539 is always the start-up (Brian, ROSP-11 2026-11-17)",
    ),
    "startup_allowance_min": ("decided", "Q6 / ROSP-50"),
    "startup_origin": ("pending", "ROSP-58 Q6 amendment: 'dome_release' if accepted"),
    "release_min_closed_min": (
        "parameter",
        "dome release = opening after a closure this long",
    ),
    "startup_overrun_category": ("decided", "Q6 / ROSP-50"),
    "gap_scatter_s": ("pending", "'slew model plus normal scatter' has no number yet"),
    "gap_fallback_s": ("pending", "only where the slew model cannot be evaluated"),
    "d3_cap_s": (
        "pending",
        "ROSP-23: Erik to set; None = d3_cap_quantile of non-fault gaps",
    ),
    "d3_cap_quantile": ("pending", "ROSP-23 proposal: 99th percentile"),
    "activity_join_min": ("pending", "how B2 activity is bridged between exposures"),
    "weather_constraints": (
        "pending",
        "Weather Constraints versions: dates and re-opening values from Brian",
    ),
    "cloud_index_max": ("decided", "ROSP-24 C1/C2"),
    "dome_open_pct": ("parameter", "MTDome.apertureShutter positionActual0/1"),
    "rain_stale_h": ("parameter", "ESS:301 logevent_precipitation"),
    "standby_lead_h": ("parameter", "daytime rain / icing carried into the night"),
    "d2_tolerance_min": (
        "pending",
        "ROSP-49 'whole-night fault' with Lynne's 30-minute tolerance",
    ),
    "max_state_age_h": ("parameter", "from fault_downtime"),
    "fault_onset_tolerance_s": (
        "parameter",
        "first CSC into FAULT, from CSC summaryState",
    ),
    "fault_onset_lookback_h": (
        "parameter",
        "look-back for suggested_group only (ROSP-58)",
    ),
    "no_csc_fault_category": (
        "pending",
        "hand-set FAULT with no CSC 'goes to F': group F ('D') or category F ('F')",
    ),
}


# ---------------------------------------------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------------------------------------------


def git_blob_sha(data: bytes) -> str:
    """The git blob SHA-1 of a file's bytes (what `git hash-object` prints)."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _http_get(url, timeout=30, accept=None):
    request = urllib.request.Request(url, headers={"User-Agent": "rosp-11-ledger"})
    if accept:
        request.add_header("Accept", accept)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def utc_index_to_tai_mjd(index) -> np.ndarray:
    """A tz-aware (UTC) pandas DatetimeIndex as TAI MJD floats."""
    if len(index) == 0:
        return np.array([], dtype=float)
    values = (
        pd.DatetimeIndex(index)
        .tz_convert("UTC")
        .tz_localize(None)
        .values.astype("datetime64[ns]")
    )
    return Time(values, format="datetime64", scale="utc").tai.mjd


def tai_mjd_to_utc_iso(mjd) -> str:
    return Time(mjd, format="mjd", scale="tai").utc.isot


def day_obs_range(first, last):
    days = pd.date_range(pd.to_datetime(str(first)), pd.to_datetime(str(last)))
    return [int(d.strftime("%Y%m%d")) for d in days]


def day_obs_chunks(first, last, chunk_days):
    lo, end = pd.Timestamp(str(first)), pd.Timestamp(str(last))
    while lo <= end:
        hi = min(lo + pd.Timedelta(days=chunk_days - 1), end)
        yield int(lo.strftime("%Y%m%d")), int(hi.strftime("%Y%m%d"))
        lo = hi + pd.Timedelta(days=1)


class Intervals:
    """A set of half-open [start, end) intervals (TAI MJD), merged, with vectorised membership."""

    def __init__(self, starts, ends):
        starts = np.asarray(starts, dtype=float)
        ends = np.asarray(ends, dtype=float)
        keep = ends > starts
        starts, ends = starts[keep], ends[keep]
        order = np.argsort(starts)
        starts, ends = starts[order], ends[order]
        merged_s, merged_e = [], []
        for s, e in zip(starts, ends):
            if merged_e and s <= merged_e[-1]:
                merged_e[-1] = max(merged_e[-1], e)
            else:
                merged_s.append(s)
                merged_e.append(e)
        self.starts = np.array(merged_s, dtype=float)
        self.ends = np.array(merged_e, dtype=float)

    @classmethod
    def from_pairs(cls, pairs):
        pairs = list(pairs)
        if not pairs:
            return cls([], [])
        a = np.array(pairs, dtype=float)
        return cls(a[:, 0], a[:, 1])

    def __len__(self):
        return len(self.starts)

    def edges(self):
        return np.concatenate([self.starts, self.ends])

    def contains(self, t):
        t = np.asarray(t, dtype=float)
        if len(self.starts) == 0:
            return np.zeros(t.shape, dtype=bool)
        i = np.searchsorted(self.starts, t, side="right") - 1
        ok = i >= 0
        out = np.zeros(t.shape, dtype=bool)
        out[ok] = t[ok] < self.ends[i[ok]]
        return out

    def overlaps(self, start, end):
        """True if any interval overlaps [start, end)."""
        if len(self.starts) == 0:
            return False
        i = np.searchsorted(self.starts, end, side="left") - 1
        return bool(i >= 0 and self.ends[i] > start)


def _lookup_bins(bin_starts, bin_width_days, values, t):
    """Value of the fixed-width bin containing each t, NaN outside every bin."""
    t = np.asarray(t, dtype=float)
    out = np.full(t.shape, np.nan)
    if len(bin_starts) == 0:
        return out
    i = np.searchsorted(bin_starts, t, side="right") - 1
    ok = (i >= 0) & (t < bin_starts[np.clip(i, 0, None)] + bin_width_days)
    out[ok] = np.asarray(values, dtype=float)[i[ok]]
    return out


# ---------------------------------------------------------------------------------------------------------------
# Night windows
# ---------------------------------------------------------------------------------------------------------------


def night_windows(day_obs_list, twilight_alt=-12.0, now=None):
    """One row per dayObs whose observable night has started: -12 deg evening to -12 deg morning (TAI MJD)."""
    import rubin_nights.dayobs_utils as rn_dayobs

    now_mjd = (Time.now() if now is None else now).tai.mjd
    rows = []
    for day_obs in day_obs_list:
        sunset, sunrise = rn_dayobs.day_obs_sunset_sunrise(int(day_obs), twilight_alt)
        start, end = sunset.tai.mjd, sunrise.tai.mjd
        if start >= now_mjd:
            continue
        rows.append(
            {
                "day_obs": int(day_obs),
                "start": start,
                "end": min(end, now_mjd),
                "full_end": end,
                "complete": end <= now_mjd,
            }
        )
    out = pd.DataFrame(
        rows, columns=["day_obs", "start", "end", "full_end", "complete"]
    )
    out["night_hours"] = (out.end - out.start) * 24.0
    return out.set_index("day_obs")


# ---------------------------------------------------------------------------------------------------------------
# Observatory state overrides (ROSP-42)
# ---------------------------------------------------------------------------------------------------------------


def resolve_overrides_source(source=None, ref="main"):
    """Where to read the override file from, and the commit it is pinned to (see read_overrides)."""
    if source:
        return {
            "url": source,
            "ref": None,
            "commit_sha": "n/a (ROSP_STATE_OVERRIDES_CSV set)",
            "error": None,
        }
    api = f"https://api.github.com/repos/{GITHUB_REPO}/commits?sha={ref}&path={OVERRIDES_PATH}&per_page=1"
    try:
        commits = json.loads(_http_get(api, accept="application/vnd.github+json"))
        sha = commits[0]["sha"]
        url = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{sha}/{OVERRIDES_PATH}"
        return {"url": url, "ref": ref, "commit_sha": sha, "error": None}
    except (
        Exception
    ) as err:  # rate limit or no network: fall back to the branch tip, SHA unresolved
        url = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{ref}/{OVERRIDES_PATH}"
        return {
            "url": url,
            "ref": ref,
            "commit_sha": "UNRESOLVED",
            "error": f"{type(err).__name__}: {err}",
        }


def prepare_overrides(ov):
    """Validate an override table and add the columns the ledger uses (TAI window, flags).

    Rows may cover a whole dayObs (12:00 UT to 12:00 UT) or a window inside it; windows must not overlap.
    `category` must be one of OVERRIDE_CATEGORIES; `group` (optional) is the attribution group of D rows.
    """
    ov = ov.copy()
    if "group" not in ov:
        ov["group"] = ""
    ov["group"] = ov["group"].fillna("").astype(str)
    ov["planned_downtime"] = (
        ov.planned_downtime.astype(str).str.upper().isin(["TRUE", "1"])
    )
    ov["start_utc"] = pd.to_datetime(ov.start_utc)
    ov["end_utc"] = pd.to_datetime(ov.end_utc)
    day_start = pd.to_datetime(ov.day_obs.astype(str), format="%Y%m%d") + pd.Timedelta(
        hours=12
    )
    outside = ov[
        (ov.start_utc < day_start)
        | (ov.end_utc > day_start + pd.Timedelta(days=1))
        | (ov.end_utc <= ov.start_utc)
    ]
    if len(outside):
        raise ValueError(
            f"override windows must lie inside 12:00 UT to 12:00 UT of their dayObs: {outside.day_obs.tolist()}"
        )
    unknown = ov[~ov.category.isin(OVERRIDE_CATEGORIES)]
    if len(unknown):
        raise ValueError(
            f"override category not in {OVERRIDE_CATEGORIES}: {unknown.day_obs.tolist()}"
        )
    ov["whole_night"] = (ov.start_utc == day_start) & (
        ov.end_utc == day_start + pd.Timedelta(days=1)
    )
    labels = (
        ov.states.fillna("")
        .astype(str)
        .str.split("|")
        .apply(lambda s: {x.strip() for x in s})
    )
    ov["has_weather"] = labels.apply(lambda s: "WEATHER" in s)
    ov["has_downtime"] = labels.apply(lambda s: "DOWNTIME" in s)
    ov["t0"] = Time(
        ov.start_utc.values.astype("datetime64[ns]"), format="datetime64", scale="utc"
    ).tai.mjd
    ov["t1"] = Time(
        ov.end_utc.values.astype("datetime64[ns]"), format="datetime64", scale="utc"
    ).tai.mjd
    ov = ov.sort_values("t0").reset_index(drop=True)
    if (ov.t0.values[1:] < ov.t1.values[:-1]).any():
        raise ValueError("override windows overlap")
    return ov


def read_overrides(source=None, ref="main"):
    """The ROSP-42 override table (one row per window) and a provenance dict (URL, commit SHA, blob SHA)."""
    prov = resolve_overrides_source(source, ref)
    url = prov["url"]
    if re.match(r"https?://", url):
        data = _http_get(url)
    else:
        with open(url, "rb") as f:
            data = f.read()
    prov["blob_sha"] = git_blob_sha(data)
    ov = pd.read_csv(
        io.BytesIO(data),
        dtype={"day_obs": int, "states": str, "planned_downtime": str, "category": str},
        keep_default_na=False,
    )
    ov = prepare_overrides(ov)
    # README mapping for the merged categories: planned DOWNTIME -> B1; otherwise WEATHER -> C1/C4/C5; else D4.
    expected = np.where(
        ov.planned_downtime & ov.has_downtime,
        "B1",
        np.where(ov.has_weather, "C", np.where(ov.has_downtime, "D4", "?")),
    )
    consistent = [
        c in ("D1", "D2", "B2a")
        or (e == "B1" and c == "B1")
        or (e == "C" and c in ("C1", "C4", "C5"))
        or (e == "D4" and c == "D4")
        for e, c in zip(expected, ov.category)
    ]
    prov["inconsistent_rows"] = ov.day_obs[~np.array(consistent)].tolist()
    prov["n_rows"] = len(ov)
    prov["first"] = int(ov.day_obs.min()) if len(ov) else None
    prov["last"] = int(ov.day_obs.max()) if len(ov) else None
    prov["has_group_column"] = bool(ov.group.ne("").any())
    prov["partial_windows"] = int((~ov.whole_night).sum())
    return ov, prov


# ---------------------------------------------------------------------------------------------------------------
# Observatory state (EFD)
# ---------------------------------------------------------------------------------------------------------------


def fetch_status_messages(efd_client, t_start, t_end, lead_days=3):
    """Scheduler observatoryStatus messages, with a lead-in so the state at the first twilight is known."""
    import rubin_nights.observatory_status as observatory_status

    messages = observatory_status.get_obs_status_messages(
        t_start - TimeDelta(lead_days, format="jd"), t_end, efd_client
    )
    if len(messages) == 0:
        return pd.DataFrame(columns=["status", "note", "statusLabels"])
    return messages.sort_index()


def state_intervals(messages, windows, max_state_age_h=30.0):
    """Partition each night into state runs, each clipped to the night and carrying its status bitmask."""
    msg_t = utc_index_to_tai_mjd(messages.index)
    status = (
        messages["status"].astype(int).values
        if len(messages)
        else np.array([], dtype=int)
    )
    notes = messages["note"].values if len(messages) else np.array([], dtype=object)
    max_age = max_state_age_h / 24.0
    rows = []

    def add(day_obs, start, end, st, note, i):
        if end > start:
            rows.append(
                {
                    "day_obs": day_obs,
                    "start": start,
                    "end": end,
                    "status": st,
                    "note": note,
                    "msg_index": i,
                    "msg_time": msg_t[i] if i >= 0 else np.nan,
                }
            )

    for day_obs, w in windows.iterrows():
        lo, hi = w.start, w.end
        if len(msg_t) == 0:
            add(day_obs, lo, hi, NO_STATE, None, -1)
            continue
        first = max(np.searchsorted(msg_t, lo, side="right") - 1, 0)
        last = np.searchsorted(msg_t, hi, side="left")
        if msg_t[first] > lo:
            add(day_obs, lo, min(msg_t[first], hi), NO_STATE, None, -1)
        for i in range(first, last):
            start = max(msg_t[i], lo)
            end = min(msg_t[i + 1], hi) if i + 1 < len(msg_t) else hi
            fresh_end = min(end, msg_t[i] + max_age)
            add(day_obs, start, fresh_end, int(status[i]), notes[i], i)
            add(day_obs, max(start, fresh_end), end, NO_STATE, None, -1)
    cols = ["day_obs", "start", "end", "status", "note", "msg_index", "msg_time"]
    out = (
        pd.DataFrame(rows, columns=cols)
        .sort_values(["day_obs", "start"])
        .reset_index(drop=True)
    )
    out["hours"] = (out.end - out.start) * 24.0
    return out


def parse_fault_cscs(note, ordered=False, regex=FAULT_CSC_RE):
    """Faulting CSC names (SAL index kept) from a status note's FAULT clause (both 2026 payload formats)."""
    if not isinstance(note, str):
        return []
    names = []
    for match in regex.finditer(note):
        payload = match.group(1).strip().strip("[]")
        for name in payload.split(","):
            name = name.strip().strip("'\"").strip()
            if name and name not in names:
                names.append(name)
    return names if ordered else sorted(names)


def csc_group(csc):
    return CSC_GROUPS.get(str(csc).split(":")[0], PENDING)


def fault_runs(intervals, messages, lookback_h=24.0):
    """Consecutive FAULT-labelled state runs within a night, one row per run, with the CSCs behind each.

    `onset_source` is where `onset_cscs` came from, all from the current ("are in") clause: "run start" (the
    run's first message), "FAULT stretch" (the earliest clause of the unbroken FAULT stretch the run continues)
    or "later in run". A run with none of these is a FAULT with no CSC ("no CSC", typically set by hand): it goes
    to F (unattributed), and `suggested_cscs` keeps the last clause within `lookback_h` before it, or else the
    sticky "were in" clause of its own messages, as a suggestion only (ROSP-58).
    """
    iv = intervals.copy()
    iv["is_fault"] = (iv.status >= 0) & ((iv.status & STATUS_BITS["FAULT"]) > 0)
    new_run = (iv.is_fault != iv.is_fault.shift()) | (iv.day_obs != iv.day_obs.shift())
    iv["fault_run"] = new_run.cumsum().where(iv.is_fault)
    msg_t = utc_index_to_tai_mjd(messages.index)
    msg_status = (
        messages["status"].astype(int).values
        if len(messages)
        else np.array([], dtype=int)
    )
    msg_names = (
        [parse_fault_cscs(n, ordered=True) for n in messages["note"].values]
        if len(messages)
        else []
    )
    records = []
    for run_id, g in iv.dropna(subset=["fault_run"]).groupby("fault_run"):
        onset, onset_time, source = [], np.nan, ""
        first_msg = int(g.msg_index.iloc[0])
        if first_msg >= 0 and msg_names[first_msg]:
            onset, onset_time, source = (
                msg_names[first_msg],
                msg_t[first_msg],
                "run start",
            )
        if not onset and first_msg >= 0:
            j = first_msg - 1
            while j >= 0 and (msg_status[j] & STATUS_BITS["FAULT"]) > 0:
                if msg_names[j]:
                    onset, onset_time, source = msg_names[j], msg_t[j], "FAULT stretch"
                j -= 1
        if not onset:
            for note, t in zip(g.note, g.msg_time):
                names = parse_fault_cscs(note, ordered=True)
                if names:
                    onset, onset_time, source = names, t, "later in run"
                    break
        suggested, suggested_from = [], ""
        if not onset:
            source = "no CSC"
            if first_msg >= 0:
                j = first_msg - 1
                while j >= 0 and msg_t[j] >= msg_t[first_msg] - lookback_h / 24.0:
                    if msg_names[j]:
                        suggested, suggested_from = msg_names[j], "look-back"
                        break
                    j -= 1
            if not suggested:
                for note in g.note:
                    names = parse_fault_cscs(
                        note, ordered=True, regex=FAULT_CSC_WERE_RE
                    )
                    if names:
                        suggested, suggested_from = names, "were in"
                        break
        all_cscs = sorted(set().union(*(set(parse_fault_cscs(n)) for n in g.note)))
        records.append(
            {
                "fault_run": int(run_id),
                "day_obs": int(g.day_obs.iloc[0]),
                "start": g.start.min(),
                "end": g.end.max(),
                "hours": g.hours.sum(),
                "onset_time": (
                    onset_time if np.isfinite(onset_time) else g.msg_time.iloc[0]
                ),
                "onset_cscs": onset,
                "onset_source": source or "none",
                "suggested_cscs": suggested,
                "suggested_from": suggested_from,
                "all_cscs": all_cscs or list(onset),
                "first_note": g.note.iloc[0],
            }
        )
    cols = [
        "fault_run",
        "day_obs",
        "start",
        "end",
        "hours",
        "onset_time",
        "onset_cscs",
        "onset_source",
        "suggested_cscs",
        "suggested_from",
        "all_cscs",
        "first_note",
    ]
    return iv, pd.DataFrame(records, columns=cols)


def fetch_summary_states(efd_client, cscs, t_start, t_end):
    """CSC summaryState events (TAI MJD, state) per CSC name ("MTMount", "MTHexapod:1", ...)."""
    out = {}
    for csc in sorted(set(cscs)):
        base, _, index = csc.partition(":")
        try:
            df = efd_client.select_time_series(
                f"lsst.sal.{base}.logevent_summaryState",
                ["summaryState"],
                t_start,
                t_end,
                index=int(index) if index else None,
            )
        except Exception as err:
            out[csc] = (np.array([]), np.array([]), f"{type(err).__name__}: {err}")
            continue
        if len(df) == 0:
            out[csc] = (np.array([]), np.array([]), "no events")
            continue
        df = df.sort_index()
        out[csc] = (
            utc_index_to_tai_mjd(df.index),
            df.summaryState.astype(int).values,
            None,
        )
    return out


def assign_primary_csc(runs, summary_states, rules):
    """The primary CSC of each fault run: the first CSC of the onset set to have entered FAULT.

    A FAULT with no CSC (no current clause; typically set by hand) has no primary CSC: its group is
    "F (unattributed)" and the look-back / "were in" suggestion is kept as `suggested_group` (ROSP-58).
    """
    tol = rules.fault_onset_tolerance_s / SECONDS_PER_DAY
    lookback = rules.fault_onset_lookback_h / 24.0
    primary, rule, entered, suggested = [], [], [], []
    for r in runs.itertuples():
        if r.onset_source == "no CSC":
            primary.append(None)
            rule.append(
                f"FAULT with no CSC (suggestion from the {r.suggested_from})"
                if r.suggested_from
                else "FAULT with no CSC"
            )
            entered.append(np.nan)
            suggested.append(csc_group(r.suggested_cscs[0]) if r.suggested_cscs else "")
            continue
        suggested.append("")
        best, best_t = None, np.inf
        for csc in r.onset_cscs:
            times, states, _ = summary_states.get(
                csc, (np.array([]), np.array([]), None)
            )
            if len(times) == 0:
                continue
            ok = (
                (states == SUMMARY_STATE_FAULT)
                & (times <= r.onset_time + tol)
                & (times >= r.onset_time - lookback)
            )
            if ok.any():
                t = times[ok].max()
                if t < best_t:
                    best, best_t = csc, t
        where = f" ({r.onset_source})"
        if best is not None:
            primary.append(best)
            rule.append(
                (
                    "first to enter FAULT (summaryState)"
                    if len(r.onset_cscs) > 1
                    else "only CSC named"
                )
                + where
            )
            entered.append(best_t)
        elif r.onset_cscs:
            primary.append(r.onset_cscs[0])
            rule.append(
                (
                    "first named, no summaryState match"
                    if len(r.onset_cscs) > 1
                    else "only CSC named"
                )
                + where
            )
            entered.append(np.nan)
        else:
            primary.append(None)
            rule.append("no CSC in the status notes")
            entered.append(np.nan)
    runs = runs.copy()
    runs["primary_csc"] = primary
    runs["primary_rule"] = rule
    runs["fault_entered"] = entered
    runs["group"] = [csc_group(c) if c else UNATTRIBUTED_FAULT for c in primary]
    runs["suggested_group"] = suggested
    return runs


# ---------------------------------------------------------------------------------------------------------------
# Environment (EFD): weather tower, dome shutter, precipitation
# ---------------------------------------------------------------------------------------------------------------


def _influx_minutes(efd_client, select, measurement, where, t0_iso, t1_iso):
    conditions = ([where] if where else []) + [
        f"time >= '{t0_iso}'",
        f"time < '{t1_iso}'",
    ]
    query = f'SELECT {select} FROM "efd"."autogen"."{measurement}" WHERE {" AND ".join(conditions)} GROUP BY time(1m)'
    df = efd_client.query(query)
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()


def fetch_environment(efd_client, first_day_obs, last_day_obs, chunk_days=10):
    """1-minute weather-tower (ESS:301) and dome-shutter bins over dayObs first..last (chunked: InfluxDB refuses
    more than 20 000 GROUP BY buckets per query)."""
    import rubin_nights.dayobs_utils as rn_dayobs

    frames = []
    specs = [
        (
            'mean("speed") AS wind, max("speed") AS gust',
            "lsst.sal.ESS.airFlow",
            '"salIndex" = 301',
        ),
        (
            'mean("relativeHumidityItem") AS rh',
            "lsst.sal.ESS.relativeHumidity",
            '"salIndex" = 301',
        ),
        (
            'mean("dewPointItem") AS dew_point',
            "lsst.sal.ESS.dewPoint",
            '"salIndex" = 301',
        ),
        (
            'mean("temperatureItem0") AS air_temp',
            "lsst.sal.ESS.temperature",
            '"salIndex" = 301',
        ),
        (
            'max("positionActual0") AS shutter0, max("positionActual1") AS shutter1',
            "lsst.sal.MTDome.apertureShutter",
            None,
        ),
    ]
    for lo, hi in day_obs_chunks(first_day_obs, last_day_obs, chunk_days):
        t0 = rn_dayobs.day_obs_to_time(lo).utc.isot + "Z"
        t1 = (rn_dayobs.day_obs_to_time(hi) + TimeDelta(1, format="jd")).utc.isot + "Z"
        parts = []
        for select, measurement, where in specs:
            try:
                parts.append(
                    _influx_minutes(efd_client, select, measurement, where, t0, t1)
                )
            except Exception as err:
                print(f"  EFD {measurement} {lo}-{hi}: {type(err).__name__}: {err}")
        parts = [p for p in parts if len(p)]
        if parts:
            frames.append(pd.concat(parts, axis=1))
    cols = ["wind", "gust", "rh", "dew_point", "air_temp", "shutter0", "shutter1"]
    if not frames:
        env = pd.DataFrame(columns=cols)
    else:
        env = pd.concat(frames).sort_index()
        env = env[~env.index.duplicated()].reindex(columns=cols)
    env["dpd"] = env.air_temp - env.dew_point
    env["shutter"] = env[["shutter0", "shutter1"]].max(axis=1, skipna=True)
    env["t"] = utc_index_to_tai_mjd(env.index)
    return env


def fetch_precipitation(efd_client, t_start, t_end, stale_h=3.0):
    """Rain intervals (TAI MJD) from ESS:301 logevent_precipitation: raining from a True report to the next
    False report, or for at most `stale_h` without a further report."""
    try:
        df = efd_client.select_time_series(
            "lsst.sal.ESS.logevent_precipitation",
            ["raining"],
            t_start,
            t_end,
            index=301,
        )
    except Exception as err:
        print(f"  precipitation: {type(err).__name__}: {err}")
        return Intervals([], []), 0
    if len(df) == 0:
        return Intervals([], []), 0
    df = df.sort_index()
    t = utc_index_to_tai_mjd(df.index)
    raining = df.raining.astype(bool).values
    pairs = []
    for i in np.where(raining)[0]:
        end = t[i + 1] if i + 1 < len(t) else t[i] + stale_h / 24.0
        pairs.append((t[i], min(end, t[i] + stale_h / 24.0)))
    return Intervals.from_pairs(pairs), int(raining.sum())


def weather_minutes(env, cam, rain, windows, rules):
    """Per-minute weather evidence from the weather tower, the allskycam and the rain gauge, per Weather
    Constraints version.

    `bad`: a closing condition (RH, dew-point depression, wind or gust beyond the close limit; cloud index above
    `cloud_index_max`; rain; icing). `standby` (C3) applies only with the dome closed, after a weather closure
    (closing conditions while the dome is shut): from when the conditions clear until the re-opening criteria
    (every re-open limit met, no cloud, no rain) have held for `reopen_hold_min`; for `precip_hold_min` after
    rain; and, after icing, until the dome is next opened (inspection; the melt-out / icing rule is open,
    RSO-1183). Opening the dome ends the standby. Evaluated from `standby_lead_h` before twilight so daytime
    rain or icing carry over; no dome telemetry counts as closed.
    """
    cols = ["t", "day_obs", "known", "bad", "standby", "reason", "version"]
    if len(env) == 0:
        return pd.DataFrame(columns=cols)
    t_env = env.t.values
    lead = rules.standby_lead_h / 24.0
    out = []
    for day_obs, w in windows.iterrows():
        sel = (t_env >= w.start - lead) & (t_env < w.end)
        if not sel.any():
            continue
        e = env[sel]
        t = e.t.values
        c = rules.constraints(np.full(len(t), day_obs))
        with np.errstate(invalid="ignore"):
            lim = (
                (e.rh.values > c["rh_close"])
                | (e.dpd.values < c["dpd_close"])
                | (e.wind.values > c["wind_close"])
                | (e.gust.values > c["gust_close"])
            )
            reopen_ok = (
                (e.rh.values <= c["rh_reopen"])
                & (e.dpd.values >= c["dpd_reopen"])
                & (e.wind.values <= c["wind_reopen"])
                & (e.gust.values <= c["gust_reopen"])
            )
            icing = (e.air_temp.values <= c["icing_temp_c"]) & (
                e.rh.values >= c["icing_rh"]
            )
        if cam is not None and len(cam):
            cloud = _lookup_bins(
                cam.t.values - 0.5 * MINUTE,
                MINUTE,
                cam.cloud_idx.values,
                t + 0.5 * MINUTE,
            )
        else:
            cloud = np.full(len(t), np.nan)
        with np.errstate(invalid="ignore"):
            cloudy = cloud > rules.cloud_index_max
        wet = rain.contains(t + 0.5 * MINUTE)
        with np.errstate(invalid="ignore"):
            dome_open = e.shutter.values > rules.dome_open_pct
        known = (
            np.isfinite(e.rh.values) | np.isfinite(e.wind.values) | np.isfinite(cloud)
        )
        bad = lim | cloudy | wet | icing
        reopen_ok = reopen_ok & ~cloudy & ~wet
        standby = np.zeros(len(t), dtype=bool)
        reason = np.full(len(t), "", dtype=object)
        closed, held, iced, precip_until = False, 0.0, False, -np.inf
        hold_n = c["reopen_hold_min"][0]
        for i in range(len(t)):
            if dome_open[i]:
                # Open: no standby, and an opening ends any closure, inspection or post-rain hold.
                closed, held, iced, precip_until = False, 0.0, False, -np.inf
                continue
            if icing[i]:
                iced = True
            if wet[i]:
                precip_until = t[i] + c["precip_hold_min"][i] * MINUTE
            if bad[i]:
                closed, held = True, 0.0
                continue
            if iced:
                standby[i], reason[i] = True, "icing inspection"
                continue
            if closed:
                held = held + 1 if reopen_ok[i] else 0.0
                if held >= hold_n:
                    closed = False
                else:
                    standby[i], reason[i] = True, "re-opening criteria not yet met"
            if not standby[i] and t[i] < precip_until:
                standby[i], reason[i] = True, "post-precipitation inspection"
        night = t >= w.start
        out.append(
            pd.DataFrame(
                {
                    "t": t[night],
                    "day_obs": day_obs,
                    "known": known[night],
                    "bad": bad[night],
                    "standby": standby[night] & known[night],
                    "reason": reason[night],
                    "version": c["version"][night],
                }
            )
        )
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=cols)


# ---------------------------------------------------------------------------------------------------------------
# Allskycam (optional; SDF only)
# ---------------------------------------------------------------------------------------------------------------


def load_allskycam(path, first_day_obs, last_day_obs, rms_max=1.4):
    """Allskycam frames (TAI MJD of mid-frame, cloud_idx) for the range, or None if not readable; applies the
    night-level astrometric gate of weather_downtime_analysis. Also returns the last night in the file.
    """
    try:
        cam = pd.read_parquet(
            path, columns=["day_obs", "mjd", "cloud_idx", "ast_mode", "ast_rms_px"]
        )
    except Exception as err:
        return None, f"not available ({type(err).__name__}: {err})", None
    last_in_file = int(cam.day_obs.max()) if len(cam) else None
    cam = cam[(cam.day_obs >= first_day_obs) & (cam.day_obs <= last_day_obs)].copy()
    rms = cam[cam.ast_mode == "refine"].groupby("day_obs").ast_rms_px.median()
    bad = rms.index[rms > rms_max]
    cam = cam[~cam.day_obs.isin(bad)]
    cam["t"] = (
        Time(cam.mjd.values, format="mjd", scale="utc").tai.mjd if len(cam) else []
    )
    cam = cam.sort_values("t")
    note = f"{path}; {cam.day_obs.nunique()} nights in range, file ends {last_in_file}"
    if len(bad):
        note += f"; astrometric gate dropped {len(bad)} night(s)"
    return cam[["day_obs", "t", "cloud_idx"]].reset_index(drop=True), note, last_in_file


# ---------------------------------------------------------------------------------------------------------------
# Exposures (ConsDB) and the slew model
# ---------------------------------------------------------------------------------------------------------------

EXPOSURE_COLUMNS = (
    "exposure_id",
    "day_obs",
    "seq_num",
    "img_type",
    "science_program",
    "observation_reason",
    "obs_start_mjd",
    "obs_end_mjd",
    "exp_time",
    "can_see_sky",
    "band",
    "s_ra",
    "s_dec",
    "sky_rotation",
)


def fetch_exposures(consdb_client, first_day_obs, last_day_obs, chunk_days=31):
    """All LSSTCam exposures (science, engineering and calibrations), chunked by month."""
    frames = []
    for lo, hi in day_obs_chunks(first_day_obs, last_day_obs, chunk_days):
        sql = f"SELECT {', '.join(EXPOSURE_COLUMNS)} FROM cdb_lsstcam.exposure WHERE day_obs >= {lo} AND day_obs <= {hi}"
        df = consdb_client.query(sql)
        if len(df) and not set(EXPOSURE_COLUMNS).issubset(df.columns):
            raise RuntimeError(
                f"ConsDB returned an unexpected schema for {lo}-{hi}; refusing partial results"
            )
        if len(df):
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=list(EXPOSURE_COLUMNS))
    exp = pd.concat(frames, ignore_index=True)
    for c in (
        "obs_start_mjd",
        "obs_end_mjd",
        "exp_time",
        "s_ra",
        "s_dec",
        "sky_rotation",
    ):
        exp[c] = pd.to_numeric(exp[c], errors="coerce")
    for c in ("img_type", "science_program", "observation_reason", "band"):
        exp[c] = exp[c].fillna("").astype(str)
    exp = exp.dropna(subset=["obs_start_mjd", "obs_end_mjd"])
    return exp.sort_values("obs_start_mjd").reset_index(drop=True)


def add_slew_model(survey, efd_client):
    """Model slew time *to* each survey exposure (rubin_nights.add_model_slew_times). Returns (survey, status)."""
    survey = survey.copy()
    survey["slew_model"] = np.nan
    if len(survey) == 0:
        return survey, "no survey exposures"
    try:
        from rubin_nights.rubin_scheduler_addons import (
            HAS_RUBIN_SCHEDULER,
            add_model_slew_times,
        )
    except Exception as err:
        return survey, f"not available ({type(err).__name__}: {err})"
    if not HAS_RUBIN_SCHEDULER:
        return survey, "not available (rubin_scheduler not importable)"
    frames, problems = [], []
    for day_obs, night in survey.groupby("day_obs"):
        night = night.sort_values("obs_start_mjd").copy()
        night["prev_obs_end_mjd"] = night.obs_end_mjd.shift(1).fillna(0.0)
        night["visit_gap"] = (
            night.obs_start_mjd - night.prev_obs_end_mjd
        ) * SECONDS_PER_DAY
        night["band"] = night.band.str[:1]
        try:
            modelled, _ = add_model_slew_times(
                night.drop(columns=["slew_model"]), efd_client
            )
            night["slew_model"] = modelled["slew_model"].reindex(night.index).values
        except Exception as err:
            problems.append(f"{day_obs}: {type(err).__name__}")
        frames.append(night)
    out = pd.concat(frames).sort_values("obs_start_mjd")
    status = f"rubin_nights add_model_slew_times: {int(out.slew_model.notna().sum())}/{len(out)} survey exposures modelled"
    if problems:
        status += f"; failed on {len(problems)} night(s): {', '.join(problems[:5])}"
    return out, status


def _runs(starts, ends, join_days):
    runs = []
    for s, e in zip(starts, ends):
        if runs and s - runs[-1][1] <= join_days:
            runs[-1][1] = max(runs[-1][1], e)
        else:
            runs.append([s, e])
    return Intervals.from_pairs(runs)


def survey_sequence(exposures, rules):
    """Split exposures into the evidence the ledger needs.

    Returns Intervals a1 (survey open shutter), a2_exp (other survey exposures), a2_gap (the allowed part of each
    gap between consecutive survey exposures with nothing in between), activity (non-survey, non-start-up
    exposures joined into B2 activity runs), startup (start-up-program runs, T539), the per-night first survey
    science start, and the gap table (whose excess the ledger splits into D3 and E with the cap).
    """
    exp = exposures.copy()
    exp["survey"] = exp.science_program.isin(rules.survey_programs)
    exp["startup"] = exp.science_program.isin(rules.startup_programs)
    exp["a1"] = exp.survey & exp.img_type.isin(rules.a1_img_types)
    survey = exp[exp.survey].sort_values("obs_start_mjd")
    other = exp[~exp.survey].sort_values("obs_start_mjd")
    eng = other[~other.startup]
    start = other[other.startup]

    gap_allowed, gap_rows = [], []
    other_starts = other.obs_start_mjd.values
    for day_obs, night in survey.groupby("day_obs"):
        night = night.sort_values("obs_start_mjd")
        ends, starts = night.obs_end_mjd.values, night.obs_start_mjd.values
        model = (
            night.slew_model.values
            if "slew_model" in night
            else np.full(len(night), np.nan)
        )
        for i in range(1, len(night)):
            g0, g1 = ends[i - 1], starts[i]
            if g1 <= g0:
                continue
            k = np.searchsorted(other_starts, g0, side="left")
            if k < len(other_starts) and other_starts[k] < g1:
                continue  # a non-survey exposure inside the gap: the survey sequence was interrupted
            allowance_s = (
                (model[i] + rules.gap_scatter_s)
                if np.isfinite(model[i])
                else rules.gap_fallback_s
            )
            cut = min(g0 + allowance_s / SECONDS_PER_DAY, g1)
            gap_allowed.append((g0, cut))
            gap_rows.append(
                {
                    "day_obs": day_obs,
                    "g0": g0,
                    "cut": cut,
                    "g1": g1,
                    "gap_s": (g1 - g0) * SECONDS_PER_DAY,
                    "model_s": model[i],
                    "allowance_s": allowance_s,
                    "excess_s": max(g1 - cut, 0.0) * SECONDS_PER_DAY,
                }
            )
    join = rules.activity_join_min / 1440.0
    a1 = survey[survey.a1]
    return {
        "a1": Intervals(a1.obs_start_mjd, a1.obs_end_mjd),
        "a2_exp": Intervals(
            survey[~survey.a1].obs_start_mjd, survey[~survey.a1].obs_end_mjd
        ),
        "a2_gap": Intervals.from_pairs(gap_allowed),
        "activity": _runs(eng.obs_start_mjd.values, eng.obs_end_mjd.values, join),
        "startup": _runs(start.obs_start_mjd.values, start.obs_end_mjd.values, join),
        "first_science": a1.groupby("day_obs").obs_start_mjd.min(),
        "gaps": pd.DataFrame(
            gap_rows,
            columns=[
                "day_obs",
                "g0",
                "cut",
                "g1",
                "gap_s",
                "model_s",
                "allowance_s",
                "excess_s",
            ],
        ),
        "exposures": exp,
    }


def fault_state_intervals(intervals):
    f = intervals[
        (intervals.status >= 0) & ((intervals.status & STATUS_BITS["FAULT"]) > 0)
    ]
    return Intervals(f.start.values, f.end.values)


def d3_cap_default(gaps, intervals, quantile=0.99):
    """The `quantile` of the survey-to-survey gap lengths (s) that do not overlap a FAULT state (ROSP-23)."""
    if len(gaps) == 0:
        return np.nan
    faults = fault_state_intervals(intervals)
    clean = [
        not faults.overlaps(g0, g1) for g0, g1 in zip(gaps.g0.values, gaps.g1.values)
    ]
    g = gaps.gap_s.values[np.array(clean, dtype=bool)]
    return float(np.quantile(g, quantile)) if len(g) else np.nan


def split_gap_excess(gaps, intervals, activity, cap_s):
    """D3 and E intervals from the gap excess: a gap longer than `cap_s` with no FAULT and no engineering inside is
    E (ROSP-23 proposal); every other excess is D3. cap_s None or NaN means no cap."""
    faults = fault_state_intervals(intervals)
    d3, e = [], []
    for g0, cut, g1, gs in zip(
        gaps.g0.values, gaps.cut.values, gaps.g1.values, gaps.gap_s.values
    ):
        if g1 <= cut:
            continue
        long = cap_s is not None and np.isfinite(cap_s) and gs > cap_s
        if long and not faults.overlaps(g0, g1) and not activity.overlaps(g0, g1):
            e.append((cut, g1))
        else:
            d3.append((cut, g1))
    return Intervals.from_pairs(d3), Intervals.from_pairs(e)


def dome_releases(env, windows, rules):
    """Per night, the dome-release times (TAI MJD): each opening after the dome had been closed for at least
    `release_min_closed_min` (an opening before evening twilight releases the night at twilight).
    """
    out = {}
    if len(env) == 0:
        return out
    t = env.t.values
    with np.errstate(invalid="ignore"):
        open_ = env.shutter.values > rules.dome_open_pct
    known = np.isfinite(env.shutter.values)
    for day_obs, w in windows.iterrows():
        sel = (t >= w.start - 0.5) & (t < w.end) & known
        if not sel.any():
            continue
        releases, closed_since, open_at_start = [], None, False
        for ti, oi in zip(t[sel], open_[sel]):
            if ti <= w.start < ti + MINUTE:
                open_at_start = bool(oi)
            if not oi:
                closed_since = ti if closed_since is None else closed_since
                continue
            if (
                closed_since is not None
                and (ti - closed_since) * 1440 >= rules.release_min_closed_min
            ):
                releases.append(max(ti, w.start))
            closed_since = None
        releases = sorted({r for r in releases if r < w.end})
        if open_at_start and (not releases or releases[0] > w.start):
            releases.insert(0, w.start)
        out[day_obs] = releases
    return out


# ---------------------------------------------------------------------------------------------------------------
# The ledger
# ---------------------------------------------------------------------------------------------------------------


def build_ledger(
    windows, intervals, runs, overrides, env, cam, wxm, seq, rules, d3_cap_s=None
):
    """Assign every part of every observable night to one category.

    `overrides` is the prepared override table (read_overrides / prepare_overrides); `wxm` the per-minute weather
    evidence (weather_minutes). Returns a segment table: one row per elementary segment with day_obs, start, end,
    hours, category, detail, the evidence used, and the fault attribution.
    """
    gaps = seq["gaps"]
    d3_iv, e_gap_iv = split_gap_excess(gaps, intervals, seq["activity"], d3_cap_s)
    releases = (
        dome_releases(env, windows, rules)
        if rules.startup_origin == "dome_release"
        else {}
    )

    # 1. Elementary segments: every evidence boundary inside a night.
    edges = [
        windows.start.values,
        windows.end.values,
        intervals.start.values,
        intervals.end.values,
    ]
    for key in ("a1", "a2_exp", "a2_gap", "activity", "startup"):
        edges.append(seq[key].edges())
    edges += [d3_iv.edges(), e_gap_iv.edges()]
    if len(env):
        edges += [env.t.values, env.t.values + MINUTE]
    if cam is not None and len(cam):
        edges += [cam.t.values - 0.5 * MINUTE, cam.t.values + 0.5 * MINUTE]
    if len(overrides):
        edges += [overrides.t0.values, overrides.t1.values]
    allowance = rules.startup_allowance_min / 1440.0
    edges += [
        windows.start.values + allowance,
        seq["first_science"].values.astype(float),
    ]
    for rel in releases.values():
        edges += [np.array(rel), np.array(rel) + allowance]
    b = np.unique(np.concatenate([np.asarray(e, dtype=float) for e in edges]))
    lo_all, hi_all = windows.start.values, windows.end.values
    b = b[(b >= lo_all.min()) & (b <= hi_all.max())]
    s0, s1 = b[:-1], b[1:]
    mid = 0.5 * (s0 + s1)
    w_index = np.searchsorted(lo_all, mid, side="right") - 1
    inside = (w_index >= 0) & (mid < hi_all[np.clip(w_index, 0, None)])
    s0, s1, mid, w_index = s0[inside], s1[inside], mid[inside], w_index[inside]
    day_obs = windows.index.values[w_index]
    n = len(mid)

    # 2. Overrides: category, group and row of each segment (windows do not overlap).
    ov_cat = np.full(n, "", dtype=object)
    ov_group = np.full(n, "", dtype=object)
    ov_weather = np.zeros(n, dtype=bool)
    ov_downtime = np.zeros(n, dtype=bool)
    if len(overrides):
        k = np.searchsorted(overrides.t0.values, mid, side="right") - 1
        hit = (k >= 0) & (mid < overrides.t1.values[np.clip(k, 0, None)])
        kk = k[hit]
        ov_cat[hit] = overrides.category.values[kk]
        ov_group[hit] = overrides.group.values[kk]
        ov_weather[hit] = overrides.has_weather.values[kk]
        ov_downtime[hit] = overrides.has_downtime.values[kk]
    on_override = ov_cat != ""

    # 3. Evidence. The override replaces the recorded state (not the weather record) inside its window.
    iv_index = np.searchsorted(intervals.start.values, mid, side="right") - 1
    iv_ok = (iv_index >= 0) & (mid < intervals.end.values[np.clip(iv_index, 0, None)])
    status = np.where(
        iv_ok, intervals.status.values[np.clip(iv_index, 0, None)], NO_STATE
    )
    status = np.where(on_override, NO_STATE, status)
    fresh = status >= 0

    def bit(name):
        return fresh & ((status & STATUS_BITS[name]) > 0)

    st_W, st_F, st_D, st_I, st_O = (
        bit(k) for k in ("WEATHER", "FAULT", "DOWNTIME", "IDLE", "OPERATIONAL")
    )
    st_U = fresh & (status == 0)
    st_day_only = fresh & (status == STATUS_BITS["DAYTIME"])

    if len(env):
        shutter = _lookup_bins(env.t.values, MINUTE, env.shutter.values, mid)
    else:
        shutter = np.full(n, np.nan)
    dome_known = np.isfinite(shutter)
    with np.errstate(invalid="ignore"):
        dome_open = dome_known & (shutter > rules.dome_open_pct)
    dome_closed = dome_known & ~dome_open
    if cam is not None and len(cam):
        cloud = _lookup_bins(
            cam.t.values - 0.5 * MINUTE, MINUTE, cam.cloud_idx.values, mid
        )
    else:
        cloud = np.full(n, np.nan)
    cam_known = np.isfinite(cloud)
    with np.errstate(invalid="ignore"):
        cloudy = cam_known & (cloud > rules.cloud_index_max)
    if len(wxm):
        wt = wxm.t.values.astype(float)
        wx_known = _lookup_bins(wt, MINUTE, wxm.known.values.astype(float), mid) == 1
        wx_bad = (
            _lookup_bins(wt, MINUTE, wxm.bad.values.astype(float), mid) == 1
        ) & wx_known
        standby = _lookup_bins(wt, MINUTE, wxm.standby.values.astype(float), mid) == 1
        reason_idx = np.searchsorted(wt, mid, side="right") - 1
        standby_reason = np.where(
            standby, wxm.reason.values[np.clip(reason_idx, 0, None)], ""
        )
    else:
        wx_known = wx_bad = standby = np.zeros(n, dtype=bool)
        standby_reason = np.full(n, "", dtype=object)
    wx_bad = wx_bad | cloudy
    weather_data = wx_known | cam_known

    a1 = seq["a1"].contains(mid)
    a2 = seq["a2_exp"].contains(mid) | seq["a2_gap"].contains(mid)
    d3 = d3_iv.contains(mid)
    e_gap = e_gap_iv.contains(mid)
    activity = seq["activity"].contains(mid)
    startup_run = seq["startup"].contains(mid)
    survey_active = a1 | a2
    available = ~(st_F | st_D)

    # 4. Which categories apply (before precedence).
    good = wx_known & ~wx_bad & ~st_W & ~standby
    applies = OrderedDict()
    applies["B1"] = ov_cat == "B1"
    applies["C_override"] = np.isin(ov_cat, ["C1", "C4"]) | (
        (ov_cat == "C5") & ~weather_data
    )
    applies["C1"] = ~survey_active & ~dome_open & (st_W | wx_bad)
    applies["C2"] = ~survey_active & dome_open & cloudy
    applies["C3"] = ~survey_active & ((dome_open & (st_W | wx_bad)) | standby)
    applies["C5_to_D4"] = (ov_cat == "C5") & weather_data
    applies["B2a_override"] = ov_cat == "B2a"
    applies["B2a"] = activity & ~dome_closed & available & ~on_override
    applies["B2b"] = activity & dome_closed & available & ~on_override
    applies["D_override"] = np.isin(ov_cat, ["D1", "D2", "D4"])
    applies["D1"] = st_F
    applies["D4_state"] = st_D
    applies["D4_closed"] = (
        dome_closed & good & ~activity & ~startup_run & ~survey_active & ~on_override
    )
    applies["D3"] = d3
    applies["E"] = (((st_I | st_O) & ~survey_active) | e_gap) & ~on_override
    applies["A1"] = a1
    applies["A2"] = a2

    # 5. Precedence B1 -> C -> B2 -> D (D2, D1, D4, D3) -> E -> A -> F.
    choices = [
        (applies["B1"], "B1", "override B1"),
        (applies["C_override"], ov_cat, "override"),
        (applies["C1"], "C1", ""),
        (applies["C2"], "C2", "allskycam index"),
        (applies["C3"], "C3", ""),
        (applies["B2a_override"], "B2a", "override B2a"),
        (applies["B2a"], "B2a", ""),
        (applies["B2b"], "B2b", ""),
        (applies["D_override"], ov_cat, "override"),
        (applies["C5_to_D4"], "D4", "override C5 with EFD weather data within limits"),
        (applies["D1"], "D1", "FAULT state"),
        (applies["D4_state"], "D4", "DOWNTIME state"),
        (applies["D4_closed"], "D4", "closed dome, good conditions, nothing running"),
        (applies["D3"], "D3", "visit-gap excess"),
        (applies["E"], "E", ""),
        (applies["A1"], "A1", ""),
        (applies["A2"], "A2", ""),
    ]
    category = np.full(n, "F", dtype=object)
    detail = np.full(n, "", dtype=object)
    done = np.zeros(n, dtype=bool)
    for mask, cat, why in choices:
        take = mask & ~done
        category[take] = cat[take] if isinstance(cat, np.ndarray) else cat
        detail[take] = why
        done |= take
    c_ev = np.where(st_W, "WEATHER state", "")
    c_ev = np.where(~st_W & wx_bad, "measured (tower / allskycam / rain)", c_ev)
    c_ev = np.where((c_ev == "") & standby, standby_reason, c_ev)
    sel = np.isin(category, ["C1", "C3"]) & (detail == "")
    detail[sel] = [
        f"measured, inside override {o}" if on else e
        for o, on, e in zip(ov_cat[sel], on_override[sel], c_ev[sel])
    ]
    detail[(category == "E") & e_gap] = "visit gap beyond the D3 cap"
    f = category == "F"
    detail[f & st_U] = "F1 UNKNOWN state"
    detail[f & ~fresh & ~on_override] = "no fresh state"
    detail[f & st_day_only] = "DAYTIME state at night"
    detail[f & (detail == "")] = "other"

    seg = pd.DataFrame(
        {
            "day_obs": day_obs,
            "start": s0,
            "end": s1,
            "hours": (s1 - s0) * 24.0,
            "category": category,
            "detail": detail,
            "status": status,
            "interval": np.where(iv_ok & ~on_override, iv_index, -1),
            "dome_open": np.where(dome_known, dome_open, np.nan),
            "wx_bad": np.where(wx_known | cam_known, wx_bad, np.nan),
            "cloudy": np.where(cam_known, cloudy, np.nan),
            "standby": standby,
            "survey_active": survey_active,
            "a1_evidence": a1,
            "activity": activity,
            "startup_run": startup_run,
            "on_override": on_override,
            "override_category": ov_cat,
            "override_group": ov_group,
        }
    )

    # 6. Start-up (A3 / overrun / before release) and whole-night faults (D2).
    seg = _apply_startup(seg, windows, seq["first_science"], releases, rules)
    seg = _apply_d2(seg, rules)

    # 7. Shadow evidence (kept for statistics, never added to the totals).
    seg["sh_weather_curated"] = ov_weather
    seg["sh_weather_measured"] = wx_bad | st_W | standby
    seg["sh_fault"] = st_F
    seg["sh_downtime"] = st_D | ov_downtime
    seg["sh_unavailable_activity"] = activity & ~available
    for key, val in applies.items():
        seg[f"applies_{key}"] = val

    # 8. Fault attribution.
    seg = _attribute_faults(seg, intervals, runs, rules)
    seg.attrs["d3_cap_s"] = d3_cap_s
    seg.attrs["releases"] = releases
    return seg


def _apply_startup(seg, windows, first_science, releases, rules):
    """Start-up: T539 runs and the E / F time before the first survey science visit.

    Allowance windows start at evening twilight ("twilight") or at each dome release ("dome_release"). Inside an
    allowance window, T539 time and (before the night's first survey science visit) E / F time is A3. T539 time
    outside every allowance window, and the E / F stretch after the allowance up to the first survey visit
    (stopping at any other category), is the start-up overrun (D1). With "dome_release", the E / F / closed-dome
    D4 time from twilight to the first release is D4 "before dome release" (ROSP-58 Q6 amendment).
    """
    allowance = rules.startup_allowance_min / 1440.0
    seg = seg.copy()
    cat = seg.category.values.copy()
    det = seg.detail.values.copy()
    mids = 0.5 * (seg.start.values + seg.end.values)
    is_startup = seg.startup_run.values & ~seg.on_override.values
    free = np.isin(cat, ["E", "F"]) & ~seg.on_override.values
    by_night = {
        d: np.where(seg.day_obs.values == d)[0] for d in np.unique(seg.day_obs.values)
    }
    for day_obs, idx in by_night.items():
        w = windows.loc[day_obs]
        if rules.startup_origin == "dome_release":
            origins = list(releases.get(day_obs) or [w.start])  # no dome data: twilight
        else:
            origins = [w.start]
        m = mids[idx]
        in_allow = np.zeros(len(idx), dtype=bool)
        for o in origins:
            in_allow |= (m >= o) & (m < o + allowance)
        t_first = first_science.get(day_obs, np.nan)
        has_first = np.isfinite(t_first) and w.start <= t_first < w.end
        first_origin = origins[0] if origins else np.inf
        if rules.startup_origin == "dome_release" and origins:
            before = (m < first_origin) & (
                np.isin(cat[idx], ["E", "F"])
                | (
                    (cat[idx] == "D4")
                    & (det[idx] == "closed dome, good conditions, nothing running")
                )
            )
            before &= ~seg.on_override.values[idx]
            cat[idx[before]] = "D4"
            det[idx[before]] = "before dome release (start-up delay)"
        # A3: T539 inside an allowance window; E/F inside the first window, before the first survey visit.
        a3 = (is_startup[idx] & in_allow & np.isin(cat[idx], ["E", "F"])) | (
            free[idx]
            & in_allow
            & (m >= first_origin)
            & (m < (t_first if has_first else -np.inf))
        )
        cat[idx[a3]] = "A3"
        det[idx[a3]] = "start-up allowance"
        # Overrun: T539 outside the allowance windows.
        t539_over = is_startup[idx] & ~in_allow & np.isin(cat[idx], ["E", "F"])
        cat[idx[t539_over]] = rules.startup_overrun_category
        det[idx[t539_over]] = "start-up overrun (T539)"
        # Overrun: the E/F stretch from the end of the first allowance to the first survey visit.
        if has_first and np.isfinite(first_origin):
            over = np.where((m >= first_origin + allowance) & (m < t_first))[0]
            for j in over:
                i = idx[j]
                if cat[i] in ("E", "F") and not seg.on_override.values[i]:
                    cat[i] = rules.startup_overrun_category
                    det[i] = "start-up overrun"
                elif det[i] == "start-up overrun (T539)":
                    continue
                else:
                    break
    seg["category"] = cat
    seg["detail"] = det
    return seg


def _apply_d2(seg, rules):
    """Whole-night faults: a non-override night with no survey science visit whose FAULT state covers the night to
    within `d2_tolerance_min`. Its time not claimed by B1 or C becomes D2 (B2 cannot apply under a FAULT).
    """
    seg = seg.copy()
    claimed = seg.category.str.match(r"^(B1|B2|C)")
    fault_state = (seg.status >= 0) & ((seg.status & STATUS_BITS["FAULT"]) > 0)
    per_night = (
        seg.assign(
            night=seg.hours,
            fault=np.where(fault_state, seg.hours, 0.0),
            a1=np.where(seg.a1_evidence, seg.hours, 0.0),
            override=seg.on_override.astype(float),
        )
        .groupby("day_obs")[["night", "fault", "a1", "override"]]
        .sum()
    )
    d2_nights = per_night.index[
        (per_night.override <= 0)
        & (per_night.a1 <= 0)
        & (per_night.fault > 0)
        & (per_night.fault >= per_night.night - rules.d2_tolerance_min / 60.0)
    ]
    promote = seg.day_obs.isin(d2_nights) & ~claimed & ~seg.on_override
    seg.loc[promote, "detail"] = (
        "whole-night fault (was " + seg.loc[promote, "category"] + ")"
    )
    seg.loc[promote, "category"] = "D2"
    seg.attrs["d2_nights"] = list(map(int, d2_nights))
    return seg


def _attribute_faults(seg, intervals, runs, rules):
    """Fault group of D1 / D2 / D4 segments.

    FAULT-state D1/D2: the primary CSC's group, or "F (unattributed)" for a hand-set FAULT with no CSC
    (`suggested_group` keeps the look-back). D2 without a run of its own: the night's largest run. Override D
    rows: the row's `group` (pending if empty). D4 closed dome with nothing running: Operations decision. Other D4,
    the start-up overrun and the pre-release delay: pending (the cause is not in the data). D3: not attributed.
    """
    seg = seg.copy()
    run_of_interval = {}
    if len(runs) and "fault_run" in intervals:
        for i, r in intervals.dropna(subset=["fault_run"]).iterrows():
            run_of_interval[i] = int(r.fault_run)
    seg["fault_run"] = [
        run_of_interval.get(int(i), np.nan) if i >= 0 else np.nan
        for i in seg.interval.values
    ]
    rg = (
        runs.set_index("fault_run")
        if len(runs)
        else pd.DataFrame(columns=["group", "primary_csc", "suggested_group"])
    )
    run_group, run_csc, run_sugg = (
        rg.group.to_dict(),
        rg.primary_csc.to_dict(),
        rg.suggested_group.to_dict(),
    )
    n = len(seg)
    group = np.full(n, "", dtype=object)
    csc = np.full(n, "", dtype=object)
    sugg = np.full(n, "", dtype=object)
    cat, det = seg.category.values, seg.detail.values
    fr = seg.fault_run.values
    state_fault = np.isin(cat, ["D1", "D2"]) & np.isfinite(fr)
    for i in np.where(state_fault)[0]:
        r = int(fr[i])
        group[i], csc[i], sugg[i] = (
            run_group.get(r, PENDING),
            run_csc.get(r) or "",
            run_sugg.get(r, ""),
        )
    d2 = cat == "D2"
    if d2.any():
        big = (
            seg[d2 & np.isfinite(fr)]
            .groupby(["day_obs", "fault_run"])
            .hours.sum()
            .reset_index()
            .sort_values("hours")
            .groupby("day_obs")
            .last()
        )
        for i in np.where(d2 & (group == ""))[0]:
            d = seg.day_obs.values[i]
            if d in big.index:
                r = int(big.loc[d, "fault_run"])
                group[i], csc[i], sugg[i] = (
                    run_group.get(r, PENDING),
                    run_csc.get(r) or "",
                    run_sugg.get(r, ""),
                )
    ov_d = (det == "override") & np.isin(cat, ["D1", "D2", "D4"])
    group[ov_d] = np.where(
        seg.override_group.values[ov_d] != "", seg.override_group.values[ov_d], PENDING
    )
    c5d4 = det == "override C5 with EFD weather data within limits"
    group[c5d4] = np.where(
        seg.override_group.values[c5d4] != "", seg.override_group.values[c5d4], PENDING
    )
    group[(cat == "D4") & (det == "closed dome, good conditions, nothing running")] = (
        "Operations decision"
    )
    group[np.isin(cat, ["D1", "D2", "D4"]) & (group == "")] = PENDING
    seg["fault_group"] = group
    seg["fault_csc"] = csc
    seg["suggested_group"] = sugg
    if rules.no_csc_fault_category == "F":
        move = np.isin(cat, ["D1", "D2"]) & (group == UNATTRIBUTED_FAULT)
        seg.loc[move, "detail"] = (
            "hand-set FAULT, no CSC (was " + seg.loc[move, "category"] + ")"
        )
        seg.loc[move, "category"] = "F"
    return seg


def apply_group_corrections(seg, corrections):
    """Re-attribute the FAULT-state D1/D2 hours of whole nights: corrections = {day_obs: group}."""
    seg = seg.copy()
    for day_obs, group in corrections.items():
        m = (
            (seg.day_obs == day_obs)
            & seg.category.isin(["D1", "D2"])
            & seg.fault_run.notna()
        )
        seg.loc[m, "fault_group"] = group
    return seg


# ---------------------------------------------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------------------------------------------


def per_night_table(seg, windows):
    """Hours per category per night, closing to the night length."""
    table = seg.pivot_table(
        index="day_obs",
        columns="category",
        values="hours",
        aggfunc="sum",
        fill_value=0.0,
    )
    table = table.reindex(
        index=windows.index, columns=CATEGORY_ORDER, fill_value=0.0
    ).fillna(0.0)
    table["night_hours"] = windows.night_hours
    table["closure_s"] = (
        table[CATEGORY_ORDER].sum(axis=1) - table.night_hours
    ) * 3600.0
    return table


def cycle_summary(per_night):
    """Hours and fraction of the observable night by category, by group, and the short losses (D1 + E)."""
    total = per_night.night_hours.sum()
    hours = per_night[CATEGORY_ORDER].sum()
    out = pd.DataFrame(
        {"hours": hours, "fraction": hours / total if total > 0 else np.nan}
    )
    out.insert(0, "description", [CATEGORIES[c] for c in out.index])
    groups = pd.DataFrame(
        {
            "description": list(GROUP_NAMES.values()),
            "hours": [
                hours[[c for c in CATEGORY_ORDER if c[0] == g]].sum()
                for g in GROUP_NAMES
            ],
        },
        index=list(GROUP_NAMES),
    )
    groups["fraction"] = groups.hours / total if total > 0 else np.nan
    short = pd.DataFrame(
        {
            "description": [
                "D1 short fault",
                "E operational idle",
                "Short losses: D1 + E (reported together, Q7)",
            ],
            "hours": [hours["D1"], hours["E"], hours["D1"] + hours["E"]],
        },
        index=["D1", "E", "D1+E"],
    )
    short["fraction"] = short.hours / total if total > 0 else np.nan
    return out, groups, short, total


def fault_by_group(seg, runs):
    """Fault-category hours by attribution group (D1, D2, D4), plus fault_downtime's former equal split."""
    d = seg[seg.category.isin(["D1", "D2", "D4"])]
    table = d.pivot_table(
        index="fault_group",
        columns="category",
        values="hours",
        aggfunc="sum",
        fill_value=0.0,
    )
    table = table.reindex(columns=["D1", "D2", "D4"], fill_value=0.0)
    table["total"] = table.sum(axis=1)
    state_fault = seg[seg.category.isin(["D1", "D2"]) & seg.fault_run.notna()]
    by_run = state_fault.groupby("fault_run").hours.sum()
    split = {}
    run_cscs = runs.set_index("fault_run").all_cscs.to_dict() if len(runs) else {}
    for r, h in by_run.items():
        groups = [csc_group(c) for c in (run_cscs.get(int(r)) or [])] or [PENDING]
        for g in groups:
            split[g] = split.get(g, 0.0) + h / len(groups)
    primary = state_fault.groupby("fault_group").hours.sum()
    for g in list(split) + list(primary.index):
        if g not in table.index:
            table.loc[g] = 0.0
    table["FAULT-state hours, primary CSC"] = primary.reindex(table.index).fillna(0.0)
    table["FAULT-state hours, equal split (old fault_downtime)"] = (
        pd.Series(split).reindex(table.index).fillna(0.0)
    )
    order = [g for g in FAULT_GROUPS if g in table.index] + [
        g for g in table.index if g not in FAULT_GROUPS
    ]
    return table.reindex(order)


def shadow_table(seg):
    """Shadow record: evidence masked by a higher-precedence category. Never added to the ledger totals."""
    rows = []

    def add(name, mask):
        rows.append(
            {
                "shadow record": name,
                "hours": float(seg.hours[mask].sum()),
                "nights": int(seg.day_obs[mask].nunique()),
            }
        )

    b1 = seg.category == "B1"
    c = seg.category.str.startswith("C")
    d = seg.category.str.startswith("D")
    add(
        "Weather during planned maintenance (B1), from the override WEATHER state",
        b1 & seg.sh_weather_curated,
    )
    add(
        "Weather during planned maintenance (B1), measured",
        b1 & seg.sh_weather_measured,
    )
    add("FAULT state under weather (C)", c & seg.sh_fault)
    add("DOWNTIME (state or override) under weather (C)", c & seg.sh_downtime)
    add("Engineering / calibration activity under weather (C)", c & seg.activity)
    add(
        "Engineering / calibration during FAULT or DOWNTIME (charged to D)",
        d & seg.sh_unavailable_activity,
    )
    return pd.DataFrame(rows).set_index("shadow record")


def shadow_by_night(seg):
    b1 = seg.category == "B1"
    c = seg.category.str.startswith("C")
    d = seg.category.str.startswith("D")
    frame = pd.DataFrame(
        {
            "day_obs": seg.day_obs,
            "wx_under_B1_curated": np.where(
                b1 & seg.sh_weather_curated, seg.hours, 0.0
            ),
            "wx_under_B1_measured": np.where(
                b1 & seg.sh_weather_measured, seg.hours, 0.0
            ),
            "fault_under_C": np.where(c & seg.sh_fault, seg.hours, 0.0),
            "downtime_under_C": np.where(c & seg.sh_downtime, seg.hours, 0.0),
            "activity_under_C": np.where(c & seg.activity, seg.hours, 0.0),
            "activity_under_D": np.where(
                d & seg.sh_unavailable_activity, seg.hours, 0.0
            ),
        }
    )
    out = frame.groupby("day_obs").sum()
    return out[(out > 0.005).any(axis=1)]


def precedence_checks(seg):
    """Where the precedence moved time away from what the observing record shows; for review, not the ledger."""
    rows = []

    def add(name, mask, by="category"):
        sub = seg[mask]
        rows.append(
            {
                "check": name,
                "hours": float(sub.hours.sum()),
                "nights": int(sub.day_obs.nunique()),
                "where it went": ", ".join(
                    f"{k} {v:.2f} h"
                    for k, v in sub.groupby(by)
                    .hours.sum()
                    .sort_values(ascending=False)
                    .items()
                    if v > 0.005
                ),
            }
        )

    add(
        "Survey open-shutter time charged to another category",
        seg.a1_evidence & (seg.category != "A1"),
    )
    add(
        "WEATHER state set during survey activity (left in A)",
        seg.survey_active & (seg.status >= 0) & ((seg.status & 8) > 0),
    )
    add(
        "Idle time inside visit gaps below the D3 cap (D3)",
        seg.applies_E & (seg.category == "D3"),
    )
    add("Visit gaps beyond the D3 cap (E)", seg.detail == "visit gap beyond the D3 cap")
    add(
        "Measured weather inside override D4 / C5 windows (now C)",
        seg.on_override
        & seg.override_category.isin(["D4", "C5"])
        & seg.category.isin(["C1", "C2", "C3"]),
        by="override_category",
    )
    add(
        "Override C5 with EFD weather data within limits (now D4)",
        seg.detail == "override C5 with EFD weather data within limits",
        by="day_obs",
    )
    add(
        "T539 start-up outside the allowance (D1 overrun)",
        seg.detail == "start-up overrun (T539)",
        by="day_obs",
    )
    add("C3 weather standby", seg.category == "C3", by="detail")
    add(
        "No dome telemetry, treated as closed for C1",
        seg.applies_C1
        & seg.dome_open.isna()
        & (seg.category == "C1")
        & ~seg.on_override,
        by="day_obs",
    )
    c1_nights = seg.loc[seg.category == "C1", "day_obs"].unique()
    add(
        "D4 closed dome / good conditions on a night that also has C1 weather",
        (seg.category == "D4")
        & (seg.detail == "closed dome, good conditions, nothing running")
        & seg.day_obs.isin(c1_nights),
        by="day_obs",
    )
    return pd.DataFrame(rows).set_index("check")


# ---------------------------------------------------------------------------------------------------------------
# What-if corrections and pending-decision sensitivity (shown, never applied to the ledger above)
# ---------------------------------------------------------------------------------------------------------------

OVERRIDE_RAW_COLUMNS = [
    "day_obs",
    "start_utc",
    "end_utc",
    "states",
    "planned_downtime",
    "category",
    "group",
    "source",
    "notes",
]


def _raw_overrides(ov):
    raw = ov.copy()
    for c in OVERRIDE_RAW_COLUMNS:
        if c not in raw:
            raw[c] = ""
    return raw[OVERRIDE_RAW_COLUMNS]


def correction_rows(corrections):
    """Override rows (raw form) for the 'window' corrections."""
    rows = []
    for c in corrections:
        if c["kind"] != "window":
            continue
        rows.append(
            {
                "day_obs": int(c["day_obs"]),
                "start_utc": pd.Timestamp(
                    Time(c["start"], format="mjd", scale="tai").utc.datetime
                ),
                "end_utc": pd.Timestamp(
                    Time(c["end"], format="mjd", scale="tai").utc.datetime
                ),
                "states": c.get("states", ""),
                "planned_downtime": "FALSE",
                "category": c["category"],
                "group": c.get("group", ""),
                "source": "what-if: " + c["id"],
                "notes": c.get("evidence", ""),
            }
        )
    return pd.DataFrame(rows, columns=OVERRIDE_RAW_COLUMNS)


def ledger_with_corrections(inputs, corrections):
    """Rebuild the ledger with `corrections` applied, without touching the override file.

    Each correction is a dict with an `id` and a `kind`: "window" (an override row for [start, end) with a
    category and group, as the ROSP-58 rows would be), "override_group" (fill the `group` of override rows of
    `category` that have none), or "fault_group" (re-attribute a night's FAULT-state D1/D2 hours to `group`).
    `inputs` holds the build_ledger arguments by name.
    """
    ov = _raw_overrides(inputs["overrides"])
    for c in corrections:
        if c["kind"] == "override_group":
            m = (ov.category == c["category"]) & (ov.group.fillna("") == "")
            ov.loc[m, "group"] = c["group"]
    extra = correction_rows(corrections)
    if len(extra):
        ov = pd.concat([ov, extra], ignore_index=True)
    args = dict(inputs)
    args["overrides"] = prepare_overrides(ov)
    seg = build_ledger(**args)
    return apply_group_corrections(
        seg,
        {
            int(c["day_obs"]): c["group"]
            for c in corrections
            if c["kind"] == "fault_group"
        },
    )


def ledger_delta(base, seg):
    """Hours by category and by (category, fault group) that a corrected ledger moves relative to `base`."""
    by_cat = (
        seg.groupby("category")
        .hours.sum()
        .sub(base.groupby("category").hours.sum(), fill_value=0.0)
    )
    key = lambda s: s.category + np.where(
        s.category.isin(["D1", "D2", "D4"]), " / " + s.fault_group, ""
    )
    by_group = (
        seg.groupby(key(seg))
        .hours.sum()
        .sub(base.groupby(key(base)).hours.sum(), fill_value=0.0)
    )
    return by_cat[by_cat.abs() > 0.005], by_group[by_group.abs() > 0.005]


def what_if_table(inputs, base, corrections):
    """One row per correction: what it would move, applied alone; and a last row with all applied together."""
    rows = []
    for c in corrections:
        if c["kind"] == "note":
            rows.append(
                {
                    "correction": c["id"],
                    "status": c.get("status", ""),
                    "evidence": c.get("evidence", ""),
                    "moves": c.get("moves", "not applied"),
                }
            )
            continue
        seg = ledger_with_corrections(inputs, [c])
        by_cat, by_group = ledger_delta(base, seg)
        moves = ", ".join(f"{k} {v:+.2f} h" for k, v in by_group.sort_values().items())
        rows.append(
            {
                "correction": c["id"],
                "status": c.get("status", ""),
                "evidence": c.get("evidence", ""),
                "moves": moves or "no change",
            }
        )
    applicable = [
        c for c in corrections if c["kind"] != "note" and not c.get("conditional")
    ]
    seg_all = ledger_with_corrections(inputs, applicable)
    by_cat, _ = ledger_delta(base, seg_all)
    rows.append(
        {
            "correction": f"all {len(applicable)} non-conditional corrections together",
            "status": "",
            "evidence": "",
            "moves": ", ".join(
                f"{k} {v:+.2f} h" for k, v in by_cat.sort_values().items()
            )
            or "no change",
        }
    )
    return pd.DataFrame(rows).set_index("correction"), seg_all


def sensitivity(inputs, variants, windows):
    """Category hours for rule variants: variants = {label: (rules, d3_cap_s)}."""
    out = {}
    for label, (rules, cap) in variants.items():
        args = dict(inputs)
        args["rules"], args["d3_cap_s"] = rules, cap
        out[label] = per_night_table(build_ledger(**args), windows)[
            CATEGORY_ORDER
        ].sum()
    return pd.DataFrame(out)
