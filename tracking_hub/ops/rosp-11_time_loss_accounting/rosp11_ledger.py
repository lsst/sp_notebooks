"""Shared functions for the ROSP-11 time-loss ledger.

The ledger assigns every second of the observable night (Sun -12 deg to -12 deg, ROSP-22) to exactly one
category of the ROSP-24 loss-cause taxonomy, by the ROSP-24 precedence

    B1 -> C -> B2 -> D (D2, D1, D4, D3) -> E -> A -> F

It is used by ``rosp-11_time_loss_accounting.ipynb``. The state handling (seeding each night from the last
Scheduler ``observatoryStatus`` message before twilight, the 30 h freshness limit, the FAULT-clause parser and
the ROSP-42 override reader) is adapted from the supporting notebook ``fault_downtime.ipynb``.

Every rule that is still pending a decision is a field of :class:`Rules` (or a row of its Weather Constraints
regime table), not a literal in the code, so the notebook can print the full rule set it ran with.

Times are TAI MJD floats throughout. The EFD is UTC and is converted on read; ConsDB ``obs_start_mjd`` and
``obs_end_mjd`` are already TAI.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import os
import re
import urllib.request
from collections import OrderedDict

import numpy as np
import pandas as pd
from astropy.time import Time, TimeDelta

# Bumped whenever a function signature or a returned column changes, so the notebook can refuse a module
# that does not match it (on Times Square the module is fetched from GitHub, not read from the checkout).
MODULE_API_VERSION = 3

TAXONOMY_VERSION = (
    "v1.0 (pending ROSP-24 sign-off; rule updates of 2026-11-16/17, round 3)"
)

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
OPERATIONS_DECISION = "Operations decision"
FAULT_GROUPS = (
    "Telescope mount and structure",
    "Optics and active optics",
    "Camera",
    "Dome and enclosure",
    "Facility utilities",
    "Safety systems",
    "Observatory control software",
    "IT, networks and data services",
    OPERATIONS_DECISION,
    "Site and external",
    PENDING,
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
# CSCs whose summaryState is screened when a FAULT has no CSC clause (is any CSC actually in FAULT?).
SUMMARY_SCREEN_CSCS = (
    "MTMount",
    "MTRotator",
    "MTPtg",
    "MTM1M3",
    "MTM2",
    "MTHexapod:1",
    "MTHexapod:2",
    "MTAOS",
    "MTCamera",
    "MTDome",
    "MTDomeTrajectory",
    "Scheduler:1",
    "ScriptQueue:1",
)
# Cause keywords in narrative-log text -> attribution group (for a FAULT set by hand with no CSC in FAULT).
NARRATIVE_CAUSES = (
    (
        "Camera",
        r"\bcamera\b|mtcamera|\blsstcam\b|\bfes\b|filter (change|exchange|changer)|carousel|cryo|\bmcm\b|\bccs\b",
    ),
    (
        "Telescope mount and structure",
        r"\btma\b|\bmount\b|mtmount|\boss\b|rotator|azimuth|elevation|encoder|mtptg",
    ),
    (
        "Optics and active optics",
        r"m1m3|\bm2\b|hexapod|\baos\b|mirror|actuator|hardpoint",
    ),
    ("Dome and enclosure", r"\bdome\b|louver|aperture shutter"),
    (
        "Facility utilities",
        r"dynalene|glycol|chiller|hvac|electric|power (cut|outage|failure)|generator|compressor",
    ),
    ("Safety systems", r"\bgis\b|interlock|e-?stop"),
    ("Observatory control software", r"scheduler|script ?queue|watcher"),
    ("IT, networks and data services", r"network|internet|\bdns\b|\bvpn\b"),
)
NARRATIVE_IGNORE = r"auxtel|latiss|\bat(dome|dometrajectory|mcs|ptg|camera|aos|hexapod|pneumatics|spectrograph|headerservice|oods|whitelight)\b"

# Only the machine-generated clause, and only its current ("are in") form: "were in" is sticky and repeats long
# after recovery. Operator text before the clause must not be searched. (From fault_downtime.ipynb.)
FAULT_CSC_RE = re.compile(r"components are in <State\.FAULT: \d+> state:\s*([^.]*)\.")
FAULT_CSC_WERE_RE = re.compile(
    r"components were in <State\.FAULT: \d+> state:\s*([^.]*)\."
)

# Weather Constraints regimes (Confluence page 39696225 history, Brian 2026-11-17). Only criteria measurable from
# telemetry are implemented; the rest is listed in KNOWN_GAPS. Spread = T - Td at the weather tower (ESS:301).
WEATHER_REGIMES = (
    {
        "regime": "R1",
        "versions": "v27",
        "from_dayobs": 0,
        "spread_close": 2.0,  # close at once below this
        "spread_close_sustained": None,  # (limit, minutes): close if below limit continuously
        "inside_close": 2.0,  # any inside sensor ESS:111/112/113 within this of its dew point ...
        "inside_close_min": 20,  # ... for this many minutes
        "glass_close": None,  # M1M3 glass (ESS:115 item 3) within this of ESS:113 Td: close
        "rh_do_not_open": 70.0,
        "cloud_closes": True,  # "heavy and dark" only: the deep-index thick class
        "reopen": "stable",  # all closing conditions clear for `stable_min`
        "stable_min": 30,
        "initial": "limits",  # initial opening: below limits (and RH <= rh_do_not_open)
    },
    {
        "regime": "R2",
        "versions": "v28-v36",
        "from_dayobs": 20261014,
        "spread_close": 2.0,
        "spread_close_sustained": None,
        "inside_close": 2.0,
        "inside_close_min": 20,
        "glass_close": None,
        "rh_do_not_open": 70.0,
        "cloud_closes": True,
        "reopen": "stable",
        "stable_min": 30,
        "initial": "limits",
    },
    {
        "regime": "R3",
        "versions": "v37-v38",
        "from_dayobs": 20261102,
        "spread_close": 2.0,
        "spread_close_sustained": None,
        "inside_close": 2.0,
        "inside_close_min": 20,
        "glass_close": None,
        "rh_do_not_open": None,
        "cloud_closes": True,
        "reopen": "stable",
        "stable_min": 30,
        "initial": "limits",
    },
    {
        "regime": "R4",
        "versions": "v39+",
        "from_dayobs": 20261107,
        "spread_close": 1.0,
        "spread_close_sustained": (2.0, 20),
        "inside_close": None,
        "inside_close_min": 20,
        "glass_close": 2.0,
        "rh_do_not_open": None,
        "cloud_closes": False,
        "reopen": "spread",  # 30-min median spread > 5 and 30-min minimum > 2, counted from the close
        "stable_min": 30,
        "reopen_spread_median": 5.0,
        "reopen_spread_min": 2.0,
        "initial": "spread",  # spread > 2 and stable or rising over 30 min
        "initial_spread": 2.0,
    },
)
# Wind rules by date (SCC-20 switched them on 20261005, inside R1's dates).
WIND_REGIMES = (
    {
        "from_dayobs": 0,
        "gust_close": 18.0,
        "gust_points": 1,
        "avg5_close": None,
        "reopen_gust": 15.0,
        "reopen_min": 15,
    },
    {
        "from_dayobs": 20261005,
        "gust_close": 25.0,
        "gust_points": 3,
        "avg5_close": 20.0,
        "reopen_gust": 20.0,
        "reopen_min": 15,
    },
)
# Post-precipitation proposal (page 1241514042, a DRAFT): its hours are flagged provisional.
POST_PRECIP = {
    "rain_hold_min": 60,  # rain with T > 0 C: 30 min dry + dome drain
    "humid_rh": 75.0,  # RH above this (or fog) with T > 0 C ...
    "humid_hold_min": 20,  # ... 20 min safe + walk-around
    "icing_temp_c": 0.0,  # T <= this while structures are still wet: C3 to the end of the night
    "wet_memory_min": 120,  # how long structures count as wet after rain or a humid period
}
KNOWN_GAPS = (
    "Forecast-based rules: precipitation probability, expected night-minimum temperature (not in telemetry).",
    "Outside observations: fog, snow or ice seen on the dome or structures, 'heavy and dark' cloud judged by eye "
    "(the deep-index thick class stands in for the latter).",
    "Snow or sleet: ESS:301 logevent_precipitation never reports snowing, so the 'C3 until the daytime "
    "temperature goes above 0 C' rule cannot be triggered.",
    "R1-R3 'near limits at handover ... open only below limits with no upward trend': 'near limits' has no number; "
    "only 'below limits' is applied.",
    "Tier 3 judgment (operator discretion) and the walk-around itself: only the fixed hold times are applied.",
    "Gusts 'for several data points' (R2-R4) read as >= 3 of the last 5 one-minute maxima; R2 're-open after 15 min "
    "under 20 m/s' read as 1-min gust maxima under 20 m/s.",
    "Inside-sensor and glass dew-point criteria use 1-min means of ESS:111/112/113 and ESS:115 item 3; a sensor "
    "outage removes the criterion silently for that minute.",
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
    # BLOCK-T539 (initial alignment): start-up when it is the first of the night, before the first science visit;
    # otherwise recovery, charged with the interruption it follows (ROSP-58, 2026-11-17). Never B2a.
    alignment_programs: tuple = ("BLOCK-T539",)
    startup_allowance_min: float = 10.0  # A3 from -12 deg evening twilight (Q6, 10/26)
    startup_overrun_category: str = (
        "D1"  # D4 only where a delay was declared (DOWNTIME state or override D4)
    )
    # A recovery T539 after a weather closure: "C3" (default) or "D1".
    t539_after_weather: str = "C3"
    gap_scatter_s: float = 10.0
    gap_fallback_s: float = 60.0
    d3_cap_s: float | None = None
    d3_cap_quantile: float = 0.99
    activity_join_min: float = 10.0
    weather_regimes: tuple = WEATHER_REGIMES
    wind_regimes: tuple = WIND_REGIMES
    post_precip: tuple = tuple(POST_PRECIP.items())
    # Cloud: the deep allskycam index (ext_deep, mag). "Unusable" (C1 closed / C2 open) and R1-R3 "heavy and
    # dark" = the thick class, ext_deep above this. The standard index (> 0.5) is the fallback.
    cloud_source: str = "deep"
    cloud_ext_deep_max: float = 1.0
    cloud_index_max: float = 0.5
    morning_margin_h: float = 1.0  # no (re-)opening within this of morning -12 deg
    dome_open_pct: float = 5.0
    rain_stale_h: float = 3.0
    standby_lead_h: float = 6.0
    d2_tolerance_min: float = 30.0
    max_state_age_h: float = 30.0
    fault_onset_tolerance_s: float = 120.0
    fault_onset_lookback_h: float = 24.0
    # Narrative-log window for a hand-set FAULT: from this long before the FAULT episode began to the run's end.
    narrative_margin_min: float = 30.0

    def regime(self, day_obs):
        rows = sorted(self.weather_regimes, key=lambda r: r["from_dayobs"])
        out = rows[0]
        for r in rows:
            if day_obs >= r["from_dayobs"]:
                out = r
        return out

    def wind(self, day_obs):
        rows = sorted(self.wind_regimes, key=lambda r: r["from_dayobs"])
        out = rows[0]
        for r in rows:
            if day_obs >= r["from_dayobs"]:
                out = r
        return out

    def regimes_table(self):
        return pd.DataFrame(list(self.weather_regimes)).set_index("regime")

    def as_table(self):
        rows = []
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if f.name in ("weather_regimes", "wind_regimes"):
                value = f"{len(value)} rows (table below)"
            elif f.name == "post_precip":
                value = ", ".join(f"{k}={v}" for k, v in value)
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
        "ROSP-24 v1.0 + Decision Log v32/v33 rule updates (2026-11-16/17)",
    ),
    "twilight_alt": ("decided", "Q1, ROSP-22"),
    "survey_programs": ("parameter", "Early Operations science blocks, as ROSP-2"),
    "a1_img_types": ("parameter", "acq/cwfs inside a survey block count as A2"),
    "alignment_programs": (
        "decided",
        "T539 by context: start-up or recovery (ROSP-58, 2026-11-17)",
    ),
    "startup_allowance_min": ("decided", "Q6 / ROSP-50, from -12 deg evening twilight"),
    "startup_overrun_category": (
        "decided",
        "Q6 / ROSP-50; D4 where a delay was declared",
    ),
    "t539_after_weather": (
        "pending",
        "recovery alignment after a weather closure: C3 (default) or D1",
    ),
    "gap_scatter_s": ("pending", "'slew model plus normal scatter' has no number yet"),
    "gap_fallback_s": ("pending", "only where the slew model cannot be evaluated"),
    "d3_cap_s": (
        "pending",
        "ROSP-23: Erik to set; None = d3_cap_quantile of non-fault gaps",
    ),
    "d3_cap_quantile": ("pending", "ROSP-23 proposal: 99th percentile"),
    "activity_join_min": ("pending", "how B2 activity is bridged between exposures"),
    "weather_regimes": (
        "decided",
        "Weather Constraints R1-R4 (page 39696225 history, Brian 2026-11-17)",
    ),
    "wind_regimes": ("decided", "wind switch 20261005 (SCC-20)"),
    "post_precip": (
        "provisional",
        "page 1241514042 is a draft proposal: its C3 hours are provisional",
    ),
    "cloud_source": (
        "decided",
        "deep allskycam index (ext_deep), rubin_notebooks 5e8d76b",
    ),
    "cloud_ext_deep_max": (
        "pending",
        "thick class (> 1.0 mag) as 'unusable' / 'heavy and dark'",
    ),
    "cloud_index_max": (
        "decided",
        "ROSP-24 C1/C2 on the standard index (fallback only)",
    ),
    "morning_margin_h": (
        "decided",
        "Weather Constraints: no opening within 1 h of morning -12 deg",
    ),
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
    "narrative_margin_min": (
        "parameter",
        "narrative-log window before a hand-set FAULT episode",
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

    def index_of(self, t):
        """Index of the interval containing each t, -1 outside every interval."""
        t = np.asarray(t, dtype=float)
        if len(self.starts) == 0:
            return np.full(t.shape, -1)
        i = np.searchsorted(self.starts, t, side="right") - 1
        ok = (i >= 0) & (t < self.ends[np.clip(i, 0, None)])
        return np.where(ok, i, -1)

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
# Observatory state (EFD) and fault attribution
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
    or "later in run". A run with none of these has no CSC clause ("no CSC"): assign_primary_csc screens the CSC
    summary states and, for a FAULT set by hand, the narrative log. `episode_start` is when the unbroken FAULT
    stretch began; `suggested_cscs` keeps the last clause within `lookback_h` (or the sticky "were in" clause).
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
        episode_start = msg_t[first_msg] if first_msg >= 0 else g.start.min()
        if first_msg >= 0:
            j = first_msg - 1
            while j >= 0 and (msg_status[j] & STATUS_BITS["FAULT"]) > 0:
                episode_start = msg_t[j]
                j -= 1
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
        records.append(
            {
                "fault_run": int(run_id),
                "day_obs": int(g.day_obs.iloc[0]),
                "start": g.start.min(),
                "end": g.end.max(),
                "hours": g.hours.sum(),
                "episode_start": episode_start,
                "onset_time": (
                    onset_time if np.isfinite(onset_time) else g.msg_time.iloc[0]
                ),
                "onset_cscs": onset,
                "onset_source": source or "none",
                "suggested_cscs": suggested,
                "suggested_from": suggested_from,
                "all_cscs": sorted(
                    set().union(*(set(parse_fault_cscs(n)) for n in g.note))
                )
                or list(onset),
                "first_note": g.note.iloc[0],
            }
        )
    cols = [
        "fault_run",
        "day_obs",
        "start",
        "end",
        "hours",
        "episode_start",
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


def fetch_narrative(narrative_client, t_start, t_end):
    """Simonyi narrative-log entries (TAI MJD begin/end, text) over the range."""
    try:
        m = narrative_client.query_log(t_start, t_end, {"limit": 50000})
    except Exception as err:
        print(f"  narrative log: {type(err).__name__}: {err}")
        return pd.DataFrame(columns=["t0", "t1", "text"])
    if len(m) == 0:
        return pd.DataFrame(columns=["t0", "t1", "text"])
    comp = (
        m["component"].fillna("").astype(str)
        if "component" in m
        else pd.Series("", index=m.index)
    )
    m = m[~comp.str.contains("AuxTel", case=False)].copy()
    begin = pd.to_datetime(m.date_begin, utc=True, format="ISO8601")
    end = pd.to_datetime(m.date_end, utc=True, format="ISO8601")
    text = m.message_text.fillna("").astype(str)
    out = pd.DataFrame(
        {
            "t0": utc_index_to_tai_mjd(pd.DatetimeIndex(begin)),
            "t1": utc_index_to_tai_mjd(pd.DatetimeIndex(end)),
            "text": text.values,
        }
    )
    out["t1"] = np.maximum(out.t1, out.t0)
    return out.sort_values("t0").reset_index(drop=True)


def narrative_cause(narrative, t0, t1):
    """The attribution group named by cause keywords in the most narrative-log entries overlapping [t0, t1]
    (Watcher alarm posts, script names, paths and sensor names are ignored; ties go to the earlier group).
    """
    if narrative is None or len(narrative) == 0:
        return None, ""
    sub = narrative[(narrative.t0 <= t1) & (narrative.t1 >= t0)]
    names = [g for g, _ in NARRATIVE_CAUSES]
    counts, quote = {}, {}
    for text in sub.text:
        low = text.lower().strip()
        if low.startswith(
            "alarm:"
        ):  # Watcher alarm posts name sensor locations, not causes
            continue
        # Script names (take_image_lsstcam.py), file and module paths, SAL indices and ESS sensor names are not causes.
        low = re.sub(
            r"\S+\.py\b|/\S+|\b\w+(\.\w+){2,}\b|salindex:?\s*\d+|camera[- ]?(inlet|ess\d*)",
            " ",
            low,
        )
        if re.search(NARRATIVE_IGNORE, low) and not re.search(
            r"simonyi|lsstcam|mtmount|tma", low
        ):
            continue
        for group, pattern in NARRATIVE_CAUSES:
            if re.search(pattern, low):  # one vote per entry and group
                counts[group] = counts.get(group, 0) + 1
                quote.setdefault(group, " ".join(text.split())[:110])
    if not counts:
        return None, ""
    best = max(counts, key=lambda g: (counts[g], -names.index(g)))
    return best, quote[best]


def assign_primary_csc(runs, summary_states, rules, narrative=None):
    """The primary CSC and group of each fault run.

    A run with a CSC clause: the first CSC of the onset set to have entered FAULT (CSC summaryState decides when
    several are named). A run without one: the CSC summary states are screened, and a CSC actually in FAULT at
    the run's start is the primary. Otherwise the FAULT was set by hand: a narrative-log note in the same window
    (from `narrative_margin_min` before the FAULT episode began to the run's end) that names a cause sets the
    group; failing that the group is Operations decision (ROSP-58, 2026-11-17). The 24 h look-back stays only as
    `suggested_group`.
    """
    tol = rules.fault_onset_tolerance_s / SECONDS_PER_DAY
    lookback = rules.fault_onset_lookback_h / 24.0
    out = {
        k: []
        for k in (
            "primary_csc",
            "primary_rule",
            "fault_entered",
            "group",
            "suggested_group",
            "narrative_quote",
        )
    }
    for r in runs.itertuples():
        quote = ""
        suggested = csc_group(r.suggested_cscs[0]) if r.suggested_cscs else ""
        if r.onset_source == "no CSC":
            best, best_t = None, np.inf
            for csc, (times, states, _) in summary_states.items():
                if len(times) == 0:
                    continue
                k = np.searchsorted(times, r.start + tol, side="right") - 1
                if (
                    k >= 0
                    and states[k] == SUMMARY_STATE_FAULT
                    and times[k] >= r.start - lookback
                    and times[k] < best_t
                ):
                    best, best_t = csc, times[k]
            if best is not None:
                primary, rule, entered, group = (
                    best,
                    "no clause; CSC in FAULT by summaryState",
                    best_t,
                    csc_group(best),
                )
            else:
                cause, quote = narrative_cause(
                    narrative,
                    r.episode_start - rules.narrative_margin_min / 1440.0,
                    r.end,
                )
                if cause:
                    primary, rule, entered, group = (
                        None,
                        "hand-set FAULT; cause from the narrative log",
                        np.nan,
                        cause,
                    )
                else:
                    primary, rule, entered, group = (
                        None,
                        "hand-set FAULT; no cause recorded",
                        np.nan,
                        OPERATIONS_DECISION,
                    )
        else:
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
                if ok.any() and times[ok].max() < best_t:
                    best, best_t = csc, times[ok].max()
            where = f" ({r.onset_source})"
            if best is not None:
                primary, entered = best, best_t
                rule = (
                    "first to enter FAULT (summaryState)"
                    if len(r.onset_cscs) > 1
                    else "only CSC named"
                ) + where
            else:
                primary, entered = r.onset_cscs[0], np.nan
                rule = (
                    "first named, no summaryState match"
                    if len(r.onset_cscs) > 1
                    else "only CSC named"
                ) + where
            group = csc_group(primary)
        for key, v in zip(out, (primary, rule, entered, group, suggested, quote)):
            out[key].append(v)
    runs = runs.copy()
    for key, v in out.items():
        runs[key] = v
    return runs


# ---------------------------------------------------------------------------------------------------------------
# Environment (EFD): weather tower, inside sensors, dome shutter, precipitation
# ---------------------------------------------------------------------------------------------------------------


def _influx_minutes(efd_client, select, measurement, where, t0_iso, t1_iso):
    conditions = ([where] if where else []) + [
        f"time >= '{t0_iso}'",
        f"time < '{t1_iso}'",
    ]
    query = f'SELECT {select} FROM "efd"."autogen"."{measurement}" WHERE {" AND ".join(conditions)} GROUP BY time(1m)'
    df = efd_client.query(query)
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()


ENV_SPECS = (
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
    ('mean("dewPointItem") AS dew_point', "lsst.sal.ESS.dewPoint", '"salIndex" = 301'),
    (
        'mean("temperatureItem0") AS air_temp',
        "lsst.sal.ESS.temperature",
        '"salIndex" = 301',
    ),
    (
        'mean("temperatureItem0") AS t_111',
        "lsst.sal.ESS.temperature",
        '"salIndex" = 111',
    ),
    ('mean("dewPointItem") AS td_111', "lsst.sal.ESS.dewPoint", '"salIndex" = 111'),
    (
        'mean("temperatureItem0") AS t_112',
        "lsst.sal.ESS.temperature",
        '"salIndex" = 112',
    ),
    ('mean("dewPointItem") AS td_112', "lsst.sal.ESS.dewPoint", '"salIndex" = 112'),
    (
        'mean("temperatureItem0") AS t_113',
        "lsst.sal.ESS.temperature",
        '"salIndex" = 113',
    ),
    ('mean("dewPointItem") AS td_113', "lsst.sal.ESS.dewPoint", '"salIndex" = 113'),
    (
        'mean("temperatureItem3") AS t_glass',
        "lsst.sal.ESS.temperature",
        '"salIndex" = 115',
    ),
    (
        'max("positionActual0") AS shutter0, max("positionActual1") AS shutter1',
        "lsst.sal.MTDome.apertureShutter",
        None,
    ),
)
ENV_COLUMNS = [
    "wind",
    "gust",
    "rh",
    "dew_point",
    "air_temp",
    "t_111",
    "td_111",
    "t_112",
    "td_112",
    "t_113",
    "td_113",
    "t_glass",
    "shutter0",
    "shutter1",
]


def fetch_environment(efd_client, first_day_obs, last_day_obs, chunk_days=10):
    """1-minute weather-tower (ESS:301), inside-sensor (ESS:111/112/113), M1M3-glass (ESS:115 item 3) and
    dome-shutter bins over dayObs first..last (chunked: InfluxDB refuses more than 20 000 buckets per query).
    """
    import rubin_nights.dayobs_utils as rn_dayobs

    frames = []
    for lo, hi in day_obs_chunks(first_day_obs, last_day_obs, chunk_days):
        t0 = rn_dayobs.day_obs_to_time(lo).utc.isot + "Z"
        t1 = (rn_dayobs.day_obs_to_time(hi) + TimeDelta(1, format="jd")).utc.isot + "Z"
        parts = []
        for select, measurement, where in ENV_SPECS:
            try:
                parts.append(
                    _influx_minutes(efd_client, select, measurement, where, t0, t1)
                )
            except Exception as err:
                print(
                    f"  EFD {measurement} {where} {lo}-{hi}: {type(err).__name__}: {err}"
                )
        parts = [p for p in parts if len(p)]
        if parts:
            frames.append(pd.concat(parts, axis=1))
    if not frames:
        env = pd.DataFrame(columns=ENV_COLUMNS)
    else:
        env = pd.concat(frames).sort_index()
        env = env[~env.index.duplicated()].reindex(columns=ENV_COLUMNS)
    env["dpd"] = env.air_temp - env.dew_point
    env["shutter"] = env[["shutter0", "shutter1"]].max(axis=1, skipna=True)
    env["t"] = utc_index_to_tai_mjd(env.index)
    return env


def fetch_precipitation(efd_client, t_start, t_end, stale_h=3.0):
    """Rain intervals (TAI MJD) from ESS:301 logevent_precipitation: raining from a True report to the next
    report, or for at most `stale_h` without a further report."""
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


# ---------------------------------------------------------------------------------------------------------------
# Allskycam (optional; SDF only)
# ---------------------------------------------------------------------------------------------------------------


def load_allskycam_deep(allsky_out, first_day_obs, last_day_obs):
    """Deep allskycam index per frame (TAI MJD of mid-frame, ext_deep in mag, cloud_idx_deep) from rubin_allskycam's
    deep_cloud/<dayObs>.parquet joined on seq to cloud/<dayObs>.parquet, as read by rubin_notebooks
    dome_open_close_retrospective (5e8d76b). Unlike the retrospective, frames that fail `geometry_ok` are kept: they
    are the overcast frames (no stars, so no astrometric solution; ext_deep sits at its 3-mag cap and the standard
    index near 1), and dropping them would turn overcast time into "no cloud". Returns (frames, note, last night).
    """
    if not os.path.isdir(os.path.join(allsky_out, "deep_cloud")):
        return None, f"not available ({allsky_out}/deep_cloud not readable)", None
    frames, last = [], None
    for d in day_obs_range(first_day_obs, last_day_obs):
        p_std = os.path.join(allsky_out, "cloud", f"{d}.parquet")
        p_deep = os.path.join(allsky_out, "deep_cloud", f"{d}.parquet")
        if not (os.path.exists(p_std) and os.path.exists(p_deep)):
            continue
        s = pd.read_parquet(p_std, columns=["seq", "mjd", "geometry_ok", "cloud_idx"])
        dd = pd.read_parquet(p_deep, columns=["seq", "cloud_idx_deep", "ext_deep"])
        m = s.merge(dd, on="seq").dropna(subset=["ext_deep"])
        if len(m):
            m["day_obs"] = d
            frames.append(m)
            last = d
    if not frames:
        return None, f"{allsky_out}: no deep frames in range", None
    cam = pd.concat(frames, ignore_index=True)
    cam["geometry_ok"] = cam.geometry_ok.astype(bool)
    cam["t"] = Time(cam.mjd.values, format="mjd", scale="utc").tai.mjd
    cam = cam.sort_values("t").reset_index(drop=True)
    note = (
        f"deep index {allsky_out}/deep_cloud: {cam.day_obs.nunique()} nights in range, last {last}; "
        f"{int((~cam.geometry_ok).sum())} of {len(cam)} frames without an astrometric solution kept (overcast)"
    )
    return (
        cam[["day_obs", "t", "ext_deep", "cloud_idx_deep", "cloud_idx", "geometry_ok"]],
        note,
        last,
    )


def load_allskycam(path, first_day_obs, last_day_obs, rms_max=1.4):
    """Standard allskycam index frames (fallback): (frames with t, cloud_idx), note, last night in the file."""
    try:
        cam = pd.read_parquet(
            path, columns=["day_obs", "mjd", "cloud_idx", "ast_mode", "ast_rms_px"]
        )
    except Exception as err:
        return None, f"not available ({type(err).__name__}: {err})", None
    last_in_file = int(cam.day_obs.max()) if len(cam) else None
    cam = cam[(cam.day_obs >= first_day_obs) & (cam.day_obs <= last_day_obs)].copy()
    rms = cam[cam.ast_mode == "refine"].groupby("day_obs").ast_rms_px.median()
    cam = cam[~cam.day_obs.isin(rms.index[rms > rms_max])]
    cam["t"] = (
        Time(cam.mjd.values, format="mjd", scale="utc").tai.mjd if len(cam) else []
    )
    return (
        cam.sort_values("t")[["day_obs", "t", "cloud_idx"]].reset_index(drop=True),
        f"standard index {path}",
        last_in_file,
    )


def cloud_flags(cam, rules):
    """Per-frame cloudy flag for the ledger: deep index ext_deep > cloud_ext_deep_max, else standard index."""
    if cam is None or len(cam) == 0:
        return None
    cam = cam.copy()
    if rules.cloud_source == "deep" and "ext_deep" in cam:
        cam["cloud_value"] = cam.ext_deep
        cam["cloudy"] = cam.ext_deep > rules.cloud_ext_deep_max
    else:
        cam["cloud_value"] = cam.cloud_idx
        cam["cloudy"] = cam.cloud_idx > rules.cloud_index_max
    return cam


# ---------------------------------------------------------------------------------------------------------------
# Weather Constraints: closing, re-opening and standby (C3), per regime
# ---------------------------------------------------------------------------------------------------------------


def weather_minutes(env, cam, rain, windows, rules):
    """Per-minute weather evidence by the Weather Constraints regime in force each night.

    `close`: a closing criterion of the regime (tower spread, inside sensors or M1M3 glass near the dew point,
    wind, rain, and for R1-R3 heavy cloud). `bad` = close or allskycam-unusable cloud (ROSP-24 C1/C2).
    `standby` (C3) applies only with the dome closed: after a closure, until the regime's re-opening criteria are
    met; at the start of the night, until its initial-opening criteria are met; within `morning_margin_h` of
    morning -12 deg when the dome could not yet open; and the post-precipitation holds (provisional). An
    opening ends any standby. `reason` names the criterion; `provisional` marks the post-precipitation proposal.
    """
    cols = [
        "t",
        "day_obs",
        "known",
        "close",
        "bad",
        "close_reason",
        "standby",
        "reason",
        "provisional",
        "regime",
    ]
    if len(env) == 0:
        return pd.DataFrame(columns=cols)
    pp = dict(rules.post_precip)
    lead = rules.standby_lead_h / 24.0
    t_env = env.t.values
    out = []
    for day_obs, w in windows.iterrows():
        sel = (t_env >= w.start - lead) & (t_env < w.end)
        if not sel.any():
            continue
        e = env[sel]
        t = e.t.values
        R, W = rules.regime(day_obs), rules.wind(day_obs)
        spread = pd.Series(e.dpd.values)
        rh, temp = e.rh.values, e.air_temp.values
        wind, gust = pd.Series(e.wind.values), pd.Series(e.gust.values)
        none = np.zeros(len(t), dtype=bool)
        with np.errstate(invalid="ignore"):
            m_spread = (spread < R["spread_close"]).values
            m_sustained = none
            if R["spread_close_sustained"]:
                lim, n = R["spread_close_sustained"]
                m_sustained = (spread < lim).astype(float).rolling(
                    n, min_periods=n
                ).min().values == 1
            m_inside = none
            if R["inside_close"] is not None:
                m_inside = np.zeros(len(t), dtype=bool)
                for k in ("111", "112", "113"):
                    low = pd.Series(
                        e[f"t_{k}"].values - e[f"td_{k}"].values < R["inside_close"]
                    ).astype(float)
                    m_inside |= (
                        low.rolling(
                            R["inside_close_min"], min_periods=R["inside_close_min"]
                        )
                        .min()
                        .values
                        == 1
                    )
            m_glass = none
            if R["glass_close"] is not None:
                m_glass = (e.t_glass.values - e.td_113.values) < R["glass_close"]
            if W["avg5_close"]:
                wind_close = (
                    wind.rolling(5, min_periods=3).mean() >= W["avg5_close"]
                ).values
                wind_close |= (gust >= W["gust_close"]).astype(float).rolling(
                    5, min_periods=1
                ).sum().values >= W["gust_points"]
            else:
                wind_close = (gust >= W["gust_close"]).values
        wet = rain.contains(t + 0.5 * MINUTE)
        close = m_spread | m_sustained | m_inside | m_glass | wind_close | wet
        if cam is not None and len(cam):
            cloudy = (
                _lookup_bins(
                    cam.t.values - 0.5 * MINUTE,
                    MINUTE,
                    cam.cloudy.values.astype(float),
                    t + 0.5 * MINUTE,
                )
                == 1
            )
            cloud_known = np.isfinite(
                _lookup_bins(
                    cam.t.values - 0.5 * MINUTE,
                    MINUTE,
                    cam.cloud_value.values,
                    t + 0.5 * MINUTE,
                )
            )
        else:
            cloudy = np.zeros(len(t), dtype=bool)
            cloud_known = np.zeros(len(t), dtype=bool)
        if R["cloud_closes"]:
            close |= cloudy
        bad = close | cloudy
        close_reason = np.select(
            [wet, wind_close, m_spread, m_sustained, m_inside, m_glass, cloudy],
            [
                "rain",
                "wind",
                "tower spread",
                "tower spread, sustained",
                "inside sensor near dew point",
                "M1M3 glass near dew point",
                "thick cloud (allskycam)",
            ],
            "",
        )
        known = np.isfinite(rh) | np.isfinite(wind.values) | cloud_known
        with np.errstate(invalid="ignore"):
            dome_open = e.shutter.values > rules.dome_open_pct
            spread_med30 = spread.rolling(30, min_periods=20).median().values
            spread_min30 = spread.rolling(30, min_periods=20).min().values
            spread_prev30 = spread.shift(30).values
            gust_ok = (
                gust.rolling(W["reopen_min"], min_periods=W["reopen_min"]).max().values
                < W["reopen_gust"]
            )
            humid = (rh > pp["humid_rh"]) & (temp > pp["icing_temp_c"])
        last_morning = w.full_end - rules.morning_margin_h / 24.0
        standby = np.zeros(len(t), dtype=bool)
        reason = np.full(len(t), "", dtype=object)
        prov = np.zeros(len(t), dtype=bool)
        closed, wind_episode, last_close, stable, iced, opened_once = (
            False,
            False,
            -np.inf,
            0,
            False,
            False,
        )
        rain_hold, humid_hold, wet_until = -np.inf, -np.inf, -np.inf
        for i in range(len(t)):
            if wet[i]:
                if not temp[i] <= pp["icing_temp_c"]:
                    rain_hold = t[i] + pp["rain_hold_min"] * MINUTE
                wet_until = t[i] + pp["wet_memory_min"] * MINUTE
            if humid[i]:
                humid_hold = t[i] + pp["humid_hold_min"] * MINUTE
                wet_until = t[i] + pp["wet_memory_min"] * MINUTE
            if temp[i] <= pp["icing_temp_c"] and t[i] < wet_until:
                iced = True
            if dome_open[i]:
                closed, wind_episode, iced, stable, opened_once = (
                    False,
                    False,
                    False,
                    0,
                    True,
                )
                rain_hold = humid_hold = -np.inf
                continue
            if close[i]:
                closed, last_close, stable = True, t[i], 0
                wind_episode |= bool(wind_close[i])
                continue
            stable += 1
            if t[i] < w.start:
                continue
            why, provisional = "", False
            if iced:
                why, provisional = "icing: structures wet below 0 C (provisional)", True
            elif closed:
                if R["reopen"] == "spread":
                    ok = (
                        (t[i] - last_close) * 1440 >= 30
                        and spread_med30[i] > R["reopen_spread_median"]
                        and spread_min30[i] > R["reopen_spread_min"]
                    )
                else:
                    ok = stable >= R["stable_min"] and (
                        R["rh_do_not_open"] is None or rh[i] <= R["rh_do_not_open"]
                    )
                if wind_episode:
                    ok = ok and bool(gust_ok[i])
                if ok and t[i] < last_morning:
                    closed, wind_episode = False, False
                else:
                    why = (
                        f"re-opening criteria not met ({R['regime']})"
                        if t[i] < last_morning
                        else "within 1 h of morning -12 deg"
                    )
            elif not opened_once:
                if R["initial"] == "spread":
                    prev = (
                        spread_prev30[i] if np.isfinite(spread_prev30[i]) else -np.inf
                    )
                    ok = (
                        spread.values[i] > R["initial_spread"]
                        and spread.values[i] >= prev - 0.1
                        and gust.values[i] < 20
                    )
                else:
                    ok = R["rh_do_not_open"] is None or rh[i] <= R["rh_do_not_open"]
                if ok and t[i] < last_morning:
                    opened_once = True  # the criteria allowed opening; a closed dome from here is not weather
                else:
                    why = (
                        f"initial-opening criteria not met ({R['regime']})"
                        if t[i] < last_morning
                        else "within 1 h of morning -12 deg"
                    )
            if not why and t[i] < rain_hold:
                why, provisional = "post-precipitation hold (provisional)", True
            if not why and t[i] < humid_hold:
                why, provisional = "post-humidity hold (provisional)", True
            if why:
                standby[i], reason[i], prov[i] = True, why, provisional
        night = t >= w.start
        out.append(
            pd.DataFrame(
                {
                    "t": t[night],
                    "day_obs": day_obs,
                    "known": known[night],
                    "close": close[night],
                    "bad": bad[night],
                    "close_reason": close_reason[night],
                    "standby": standby[night] & known[night],
                    "reason": reason[night],
                    "provisional": prov[night],
                    "regime": R["regime"],
                }
            )
        )
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=cols)


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
    gap between consecutive survey exposures with nothing in between), activity (non-survey, non-alignment
    exposures joined into B2 activity runs), alignment (T539 runs), the per-night first survey science start, and
    the gap table (whose excess the ledger splits into D3 and E with the cap).
    """
    exp = exposures.copy()
    exp["survey"] = exp.science_program.isin(rules.survey_programs)
    exp["alignment"] = exp.science_program.isin(rules.alignment_programs)
    exp["a1"] = exp.survey & exp.img_type.isin(rules.a1_img_types)
    survey = exp[exp.survey].sort_values("obs_start_mjd")
    other = exp[~exp.survey].sort_values("obs_start_mjd")
    eng = other[~other.alignment]
    align = other[other.alignment]
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
                continue
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
        "alignment": _runs(align.obs_start_mjd.values, align.obs_end_mjd.values, join),
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


# ---------------------------------------------------------------------------------------------------------------
# The ledger
# ---------------------------------------------------------------------------------------------------------------


def build_ledger(
    windows, intervals, runs, overrides, env, cam, wxm, seq, rules, d3_cap_s=None
):
    """Assign every part of every observable night to one category.

    `overrides` is the prepared override table; `cam` the allskycam frames with a `cloudy` flag (cloud_flags);
    `wxm` the per-minute Weather Constraints evidence (weather_minutes). Returns a segment table: one row per
    elementary segment with day_obs, start, end, hours, category, detail, the evidence used, and the fault group.
    """
    gaps = seq["gaps"]
    d3_iv, e_gap_iv = split_gap_excess(gaps, intervals, seq["activity"], d3_cap_s)

    # 1. Elementary segments: every evidence boundary inside a night.
    edges = [
        windows.start.values,
        windows.end.values,
        intervals.start.values,
        intervals.end.values,
    ]
    for key in ("a1", "a2_exp", "a2_gap", "activity", "alignment"):
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

    shutter = (
        _lookup_bins(env.t.values, MINUTE, env.shutter.values, mid)
        if len(env)
        else np.full(n, np.nan)
    )
    dome_known = np.isfinite(shutter)
    with np.errstate(invalid="ignore"):
        dome_open = dome_known & (shutter > rules.dome_open_pct)
    dome_closed = dome_known & ~dome_open
    if cam is not None and len(cam):
        cloud_v = _lookup_bins(
            cam.t.values - 0.5 * MINUTE, MINUTE, cam.cloud_value.values, mid
        )
        cloudy = (
            _lookup_bins(
                cam.t.values - 0.5 * MINUTE,
                MINUTE,
                cam.cloudy.values.astype(float),
                mid,
            )
            == 1
        )
    else:
        cloud_v = np.full(n, np.nan)
        cloudy = np.zeros(n, dtype=bool)
    cam_known = np.isfinite(cloud_v)
    if len(wxm):
        wt = wxm.t.values.astype(float)
        wx_known = _lookup_bins(wt, MINUTE, wxm.known.values.astype(float), mid) == 1
        wx_bad = (
            _lookup_bins(wt, MINUTE, wxm.bad.values.astype(float), mid) == 1
        ) & wx_known
        standby = _lookup_bins(wt, MINUTE, wxm.standby.values.astype(float), mid) == 1
        k = np.clip(np.searchsorted(wt, mid, side="right") - 1, 0, None)
        standby_reason = np.where(standby, wxm.reason.values[k], "")
        provisional = standby & wxm.provisional.values[k].astype(bool)
        regime = np.where(wx_known, wxm.regime.values[k], "")
        close_reason = np.where(wx_bad, wxm.close_reason.values[k], "")
    else:
        wx_known = wx_bad = standby = provisional = np.zeros(n, dtype=bool)
        standby_reason = regime = close_reason = np.full(n, "", dtype=object)
    wx_bad = wx_bad | cloudy
    weather_data = wx_known | cam_known

    a1 = seq["a1"].contains(mid)
    a2 = seq["a2_exp"].contains(mid) | seq["a2_gap"].contains(mid)
    d3 = d3_iv.contains(mid)
    e_gap = e_gap_iv.contains(mid)
    activity = seq["activity"].contains(mid)
    alignment_run = seq["alignment"].index_of(mid)
    alignment = alignment_run >= 0
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
        dome_closed & good & ~activity & ~alignment & ~survey_active & ~on_override
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
        (applies["C2"], "C2", "allskycam"),
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
    measured = np.where(close_reason != "", close_reason, "thick cloud (allskycam)")
    c_ev = np.where(
        ~st_W & wx_bad,
        np.char.add("measured: ", measured.astype(str)).astype(object),
        c_ev,
    )
    c_ev = np.where((c_ev == "") & standby, standby_reason, c_ev)
    sel = np.isin(category, ["C1", "C3"]) & (detail == "")
    detail[sel] = [
        f"{e or 'measured'} (inside override {o})" if on else e
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
            "cloud_value": cloud_v,
            "standby": standby,
            "provisional": provisional & (category == "C3"),
            "regime": regime,
            "survey_active": survey_active,
            "a1_evidence": a1,
            "activity": activity,
            "alignment_run": alignment_run,
            "on_override": on_override,
            "override_category": ov_cat,
            "override_group": ov_group,
        }
    )

    # 6. Start-up (A3 / overrun) and whole-night faults (D2).
    seg = _apply_startup(seg, windows, seq["first_science"], rules)
    seg = _apply_d2(seg, rules)

    # 7. Shadow evidence (kept for statistics, never added to the totals).
    seg["sh_weather_curated"] = ov_weather
    seg["sh_weather_measured"] = wx_bad | st_W | standby
    seg["sh_fault"] = st_F
    seg["sh_downtime"] = st_D | ov_downtime
    seg["sh_unavailable_activity"] = activity & ~available
    for key, val in applies.items():
        seg[f"applies_{key}"] = val

    # 8. Fault attribution, then recovery alignments (which take the group of the interruption they follow).
    seg = _attribute_faults(seg, intervals, runs)
    seg = _apply_alignment_recovery(seg, rules)
    seg["provisional"] = seg.provisional & (seg.category == "C3")
    seg.attrs["d3_cap_s"] = d3_cap_s
    return seg


def _apply_startup(seg, windows, first_science, rules):
    """Start-up from -12 deg evening twilight.

    The start-up alignment is the night's first T539 run if it begins before the first survey science visit with
    no weather (C) or fault (D1 / D2) time between -12 deg and its start; every other T539 run is a recovery
    alignment (_apply_alignment_recovery). Inside the 10-min allowance, start-up alignment time and (before the
    first survey visit) idle E / F time is A3. Start-up alignment time beyond the allowance, and the idle stretch
    from the end of the allowance to the first survey visit (stopping at any other category), is the start-up
    overrun: D1, or D4 with the declaration's group where a delay was declared before the first survey visit (a
    DOWNTIME state or an override D4 row, which keep their own time).
    """
    allowance = rules.startup_allowance_min / 1440.0
    seg = seg.copy()
    cat = seg.category.values.copy()
    det = seg.detail.values.copy()
    run = seg.alignment_run.values
    on_ov = seg.on_override.values
    mids = 0.5 * (seg.start.values + seg.end.values)
    role = np.full(len(seg), "", dtype=object)
    group = np.full(len(seg), "", dtype=object)
    for day_obs in np.unique(seg.day_obs.values):
        idx = np.where(seg.day_obs.values == day_obs)[0]
        w = windows.loc[day_obs]
        m = mids[idx]
        t_first = first_science.get(day_obs, np.nan)
        has_first = bool(np.isfinite(t_first) and w.start <= t_first < w.end)
        startup_run = -1
        runs_here = [r for r in dict.fromkeys(run[idx]) if r >= 0]
        if runs_here:
            first_run = runs_here[0]
            r_start = seg.start.values[idx[run[idx] == first_run]].min()
            before = idx[m < r_start]
            interrupted = any(c[0] == "C" or c in ("D1", "D2") for c in cat[before])
            if (not has_first or r_start < t_first) and not interrupted:
                startup_run = first_run
        # A declared delay (DOWNTIME state or an override D4 row) before the first survey visit makes the overrun D4.
        horizon = t_first if has_first else w.end
        declared = idx[
            (m < horizon)
            & (cat[idx] == "D4")
            & np.isin(det[idx], ["override", "DOWNTIME state"])
        ]
        overrun_cat = "D4" if len(declared) else rules.startup_overrun_category
        overrun_group = (
            next((g for g in seg.override_group.values[declared] if g), "")
            if len(declared)
            else ""
        )
        is_start = (run[idx] == startup_run) & (startup_run >= 0) & ~on_ov[idx]
        role[idx[is_start]] = "start-up"
        role[idx[(run[idx] >= 0) & ~is_start]] = "recovery"
        in_allow = (m >= w.start) & (m < w.start + allowance)
        free = np.isin(cat[idx], ["E", "F"]) & ~on_ov[idx]
        a3 = (is_start & in_allow & np.isin(cat[idx], ["E", "F"])) | (
            free & (run[idx] < 0) & in_allow & has_first & (m < t_first)
        )
        cat[idx[a3]] = "A3"
        det[idx[a3]] = "start-up allowance"
        over_align = is_start & ~in_allow & np.isin(cat[idx], ["E", "F"])
        cat[idx[over_align]] = overrun_cat
        det[idx[over_align]] = "start-up overrun (T539)"
        group[idx[over_align]] = overrun_group
        if has_first:
            declared_set = set(declared.tolist())
            for j in np.where((m >= w.start + allowance) & (m < t_first))[0]:
                i = idx[j]
                if role[i] == "start-up" or i in declared_set:
                    continue
                if cat[i] in ("E", "F") and not on_ov[i] and run[i] < 0:
                    cat[i] = overrun_cat
                    det[i] = "start-up overrun"
                    group[i] = overrun_group
                else:
                    break
    seg["category"] = cat
    seg["detail"] = det
    seg["alignment_role"] = role
    seg["startup_group"] = group
    return seg


def _apply_alignment_recovery(seg, rules):
    """Recovery alignments (T539 runs that are not the start-up): charged with the interruption they follow.

    Looking back from the run's start over the same night, past idle (E / F) time and other alignment time, the
    first other segment decides: a fault (D1 / D2) -> the same category and group; unplanned downtime (D4) -> D4,
    same group; weather (C) -> `t539_after_weather` (C3 by default, D1 the alternative); engineering (B) or survey
    (A, D3) -> E (no interruption). T539 is never B2a. Only E / F alignment time is re-categorised.
    """
    seg = seg.copy()
    cat, det, grp = (
        seg.category.values.copy(),
        seg.detail.values.copy(),
        seg.fault_group.values.copy(),
    )
    role, run = seg.alignment_role.values, seg.alignment_run.values
    follows = np.full(len(seg), "", dtype=object)
    for day_obs in np.unique(seg.day_obs.values[role == "recovery"]):
        idx = np.where(seg.day_obs.values == day_obs)[0]
        for r in dict.fromkeys(run[idx][role[idx] == "recovery"]):
            r_idx = idx[run[idx] == r]
            first = r_idx.min()
            kind, src = "none", None
            for i in range(first - 1, idx.min() - 1, -1):
                if run[i] >= 0 or cat[i] in ("E", "F"):
                    continue
                c = cat[i]
                if c in ("D1", "D2"):
                    kind, src = "fault", i
                elif c == "D4":
                    kind, src = "downtime", i
                elif c.startswith("C"):
                    kind, src = "weather", i
                elif c.startswith("B"):
                    kind = "engineering"
                else:
                    kind = "none"
                break
            take = r_idx[np.isin(cat[r_idx], ["E", "F"])]
            follows[r_idx] = kind
            if kind in ("fault", "downtime"):
                cat[take], grp[take] = cat[src], grp[src]
                det[take] = f"T539 recovery after {kind}"
            elif kind == "weather":
                cat[take] = rules.t539_after_weather
                det[take] = "T539 recovery after weather"
                if rules.t539_after_weather.startswith("D"):
                    grp[take] = PENDING
            else:
                cat[take] = "E"
                det[take] = f"T539 recovery, no interruption before it ({kind})"
    seg["category"], seg["detail"], seg["fault_group"] = cat, det, grp
    seg["alignment_follows"] = follows
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


def _attribute_faults(seg, intervals, runs):
    """Fault group of D1 / D2 / D4 segments.

    FAULT-state D1/D2: the run's group (assign_primary_csc: primary CSC, CSC summary state, narrative log, or
    Operations decision for a hand-set FAULT with no cause recorded). D2 without a run of its own: the night's
    largest run. Override D rows: the row's `group` (pending if empty). D4 closed dome with nothing running:
    Operations decision. Other D4 and the start-up overrun: pending. D3: not attributed.
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
        else pd.DataFrame(
            columns=["group", "primary_csc", "suggested_group", "primary_rule"]
        )
    )
    run_group, run_csc, run_sugg, run_rule = (
        rg.group.to_dict(),
        rg.primary_csc.to_dict(),
        rg.suggested_group.to_dict(),
        rg.primary_rule.to_dict(),
    )
    n = len(seg)
    group = np.full(n, "", dtype=object)
    csc = np.full(n, "", dtype=object)
    sugg = np.full(n, "", dtype=object)
    rule = np.full(n, "", dtype=object)
    cat, det = seg.category.values, seg.detail.values
    fr = seg.fault_run.values
    state_fault = np.isin(cat, ["D1", "D2"]) & np.isfinite(fr)
    for i in np.where(state_fault)[0]:
        r = int(fr[i])
        group[i], csc[i], sugg[i], rule[i] = (
            run_group.get(r, PENDING),
            run_csc.get(r) or "",
            run_sugg.get(r, ""),
            run_rule.get(r, ""),
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
                group[i], csc[i], sugg[i], rule[i] = (
                    run_group.get(r, PENDING),
                    run_csc.get(r) or "",
                    run_sugg.get(r, ""),
                    run_rule.get(r, ""),
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
        OPERATIONS_DECISION
    )
    su = np.char.startswith(det.astype(str), "start-up overrun") & (
        seg.startup_group.values != ""
    )
    group[su] = seg.startup_group.values[su]
    group[np.isin(cat, ["D1", "D2", "D4"]) & (group == "")] = PENDING
    seg["fault_group"] = group
    seg["fault_csc"] = csc
    seg["suggested_group"] = sugg
    seg["fault_rule"] = rule
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


def c3_by_regime(seg):
    """C3 weather-standby hours by Weather Constraints regime and reason (provisional rows flagged)."""
    c3 = seg[seg.category == "C3"].copy()
    if len(c3) == 0:
        return pd.DataFrame(columns=["hours", "nights"])
    c3["regime"] = c3.regime.replace("", "(no weather data)")
    c3["reason"] = np.where(c3.detail == "", "other", c3.detail)
    return c3.groupby(["regime", "reason"]).agg(
        hours=("hours", "sum"), nights=("day_obs", "nunique")
    )


def alignment_table(seg):
    """T539 hours by role (start-up / recovery), what a recovery follows, and the category it went to."""
    a = seg[seg.alignment_run >= 0]
    if len(a) == 0:
        return pd.DataFrame(columns=["hours", "runs", "nights"])
    a = a.assign(
        follows=np.where(a.alignment_role == "start-up", "-", a.alignment_follows)
    )
    return a.groupby(["alignment_role", "follows", "category"]).agg(
        hours=("hours", "sum"),
        runs=("alignment_run", "nunique"),
        nights=("day_obs", "nunique"),
    )


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
        "Start-up overrun (D1)",
        seg.detail.str.startswith("start-up overrun"),
        by="day_obs",
    )
    add(
        "C3 weather standby, provisional (post-precipitation proposal)",
        seg.provisional,
        by="detail",
    )
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
        _, by_group = ledger_delta(base, seg)
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
    """Category hours for rule variants: variants = {label: (rules, d3_cap_s)} or {label: (rules, d3_cap_s, cam, wxm)}."""
    out = {}
    for label, v in variants.items():
        args = dict(inputs)
        args["rules"], args["d3_cap_s"] = v[0], v[1]
        if len(v) > 2:
            args["cam"], args["wxm"] = v[2], v[3]
        out[label] = per_night_table(build_ledger(**args), windows)[
            CATEGORY_ORDER
        ].sum()
    return pd.DataFrame(out)
