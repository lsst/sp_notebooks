"""Shared functions for the ROSP-11 time-loss ledger.

The ledger assigns every second of the observable night (Sun -12 deg to -12 deg, ROSP-22) to exactly one
category of the ROSP-24 loss-cause taxonomy, by the ROSP-24 precedence

    B1 -> C -> B2 -> D (D2, D1, D4, D3) -> E -> A -> F

It is used by ``rosp-11_time_loss_accounting.ipynb``. The state handling (seeding each night from the last
Scheduler ``observatoryStatus`` message before twilight, the 30 h freshness limit, the FAULT-clause parser and
the ROSP-42 override reader) is adapted from the supporting notebook ``fault_downtime.ipynb``; that notebook is
not changed and does not import this module.

Every rule that is still pending ROSP-24 sign-off is a field of :class:`Rules`, not a literal in the code, so
the notebook can print the full rule set it ran with and a change of rule is a change of one field.

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
MODULE_API_VERSION = 1

TAXONOMY_VERSION = "v1.0 (pending ROSP-24 sign-off)"

GITHUB_REPO = "lsst/sp_notebooks"
OVERRIDES_PATH = (
    "tracking_hub/ops/observatory_state_overrides/observatory_state_overrides.csv"
)

SECONDS_PER_DAY = 86400.0

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
CATEGORY_GROUP = {c: c[0] for c in CATEGORY_ORDER}
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
OVERRIDE_CATEGORIES = ("B1", "C1", "C4", "C5", "D4")

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

# CSC summaryState enum.
SUMMARY_STATE_FAULT = 3

# ROSP-24 fault attribution groups (proposal in the ROSP-24 comments; Q5, mapping check by the subsystem leads,
# is still open). Keyed on the CSC base name, so SAL-indexed CSCs (MTHexapod:1, Scheduler:1) resolve.
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
    "Pending attribution",
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
PENDING = "Pending attribution"

# fault_downtime.ipynb's own subsystem map, kept only to reproduce its equal-split attribution for comparison.
FAULT_DOWNTIME_SUBSYSTEMS = {
    "MTMount": "TMA",
    "MTPtg": "TMA",
    "MTRotator": "TMA",
    "MTM1M3": "M1M3",
    "MTM2": "M2/hexapods",
    "MTHexapod": "M2/hexapods",
    "MTAOS": "AOS",
    "MTCamera": "Camera",
    "MTOODS": "Camera",
    "MTHeaderService": "Camera",
    "MTDome": "Dome",
    "Scheduler": "Scheduler/queue",
    "ScriptQueue": "Scheduler/queue",
}

# Only the machine-generated clause, and only its current ("are in") form: "were in" is sticky and repeats long
# after recovery. Operator text before the clause must not be searched. (From fault_downtime.ipynb.)
FAULT_CSC_RE = re.compile(r"components are in <State\.FAULT: \d+> state:\s*([^.]*)\.")
# The sticky past-tense clause, used only as the last resort for naming a run's CSC.
FAULT_CSC_WERE_RE = re.compile(
    r"components were in <State\.FAULT: \d+> state:\s*([^.]*)\."
)


@dataclasses.dataclass(frozen=True)
class Rules:
    """Every ledger rule that is a parameter rather than a literal. Defaults are taxonomy v1.0 as written in
    ROSP-24 on 2026-11-13 (Decision Log v28); fields marked pending in RULE_NOTES await sign-off or a decision.
    """

    taxonomy_version: str = TAXONOMY_VERSION
    # Observable night (ROSP-22).
    twilight_alt: float = -12.0
    # Survey vs engineering, from program metadata. Survey programs are the Early Operations science blocks.
    survey_programs: tuple = (
        "BLOCK-407",
        "BLOCK-408",
        "BLOCK-416",
        "BLOCK-417",
        "BLOCK-419",
        "BLOCK-421",
    )
    # Survey exposures of these img_types are A1 open-shutter time; other survey exposures count as A2.
    a1_img_types: tuple = ("science",)
    # A3: start-up allowance after evening -12 deg, measured to the first survey science visit (ROSP-50).
    startup_allowance_min: float = 10.0
    # An overrun of the start-up allowance is a short fault (ROSP-50, Q6).
    startup_overrun_category: str = "D1"
    # A2 vs D3: a visit gap is overhead up to the rubin_scheduler slew model plus this scatter; the rest is D3.
    gap_scatter_s: float = 10.0
    # Allowance used when the slew model is unavailable for a visit (no rubin_scheduler, or no pointing).
    gap_fallback_s: float = 60.0
    # Non-survey exposures closer together than this form one engineering / calibration activity (B2).
    activity_join_min: float = 10.0
    # C1 limits (ROSP-24 C1). Wind: (first dayObs, sustained m/s, gust m/s), applied from that dayObs on.
    rh_max: float = 80.0
    dpd_min: float = 2.0
    wind_limits: tuple = ((0, 15.0, 18.0), (20261001, 20.0, 25.0))
    # Cloud: allskycam cloud index above this is "unusable" (C1 when closed, C2 when open).
    cloud_index_max: float = 0.5
    # MTDome aperture shutter: open when either door is above this many percent.
    dome_open_pct: float = 5.0
    # D2: a night with no survey science visit whose FAULT state covers the observable night to within this
    # many minutes (ROSP-49 "D2 = whole-night fault"; Lynne's rule "within 30 minutes of the night length").
    d2_tolerance_min: float = 30.0
    # A status message is trusted for at most this long (from fault_downtime: the Scheduler emits at least daily).
    max_state_age_h: float = 30.0
    # Tolerance when matching a FAULT run's onset to the CSC summaryState transitions.
    fault_onset_tolerance_s: float = 120.0
    fault_onset_lookback_h: float = 24.0

    def wind_limit(self, day_obs):
        """(sustained, gust) m/s in force on day_obs (array-like)."""
        day_obs = np.asarray(day_obs)
        sustained = np.full(day_obs.shape, np.nan)
        gust = np.full(day_obs.shape, np.nan)
        for first, ws, wg in sorted(self.wind_limits):
            on = day_obs >= first
            sustained[on] = ws
            gust[on] = wg
        return sustained, gust

    def as_table(self):
        rows = []
        for f in dataclasses.fields(self):
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
        "ROSP-24 v1.0 text of 2026-11-13; sign-off requested for 2026-11-16",
    ),
    "twilight_alt": ("decided", "Q1, ROSP-22"),
    "survey_programs": (
        "parameter",
        "Early Operations science blocks, as ROSP-2; engineering = everything else",
    ),
    "a1_img_types": (
        "parameter",
        "open-shutter science; acq/cwfs inside a survey block count as A2",
    ),
    "startup_allowance_min": (
        "decided",
        "Q6 / ROSP-50 (10 min after -12 deg, to the first science visit)",
    ),
    "startup_overrun_category": (
        "decided",
        "Q6 / ROSP-50: an overrun is a short fault",
    ),
    "gap_scatter_s": (
        "pending",
        "'slew model plus normal scatter' (ROSP-24 A2) has no number yet",
    ),
    "gap_fallback_s": ("pending", "used only where the slew model cannot be evaluated"),
    "activity_join_min": (
        "pending",
        "how B2 activity is bridged between exposures is not in ROSP-24",
    ),
    "rh_max": ("decided", "ROSP-24 C1"),
    "dpd_min": ("decided", "ROSP-24 C1"),
    "wind_limits": (
        "pending",
        "ROSP-24 C1: 15/18 m/s 'before Oct2026', 20/25 after; switch date read as 20261001",
    ),
    "cloud_index_max": ("decided", "ROSP-24 C1/C2 (allskycam index > 0.5)"),
    "dome_open_pct": ("parameter", "MTDome.apertureShutter positionActual0/1"),
    "d2_tolerance_min": (
        "pending",
        "ROSP-49 'whole-night fault' read with Lynne's 30-minute tolerance",
    ),
    "max_state_age_h": (
        "parameter",
        "from fault_downtime (Scheduler emits at least one message a day)",
    ),
    "fault_onset_tolerance_s": (
        "parameter",
        "first CSC to enter FAULT, from CSC summaryState (ROSP-24 §5)",
    ),
    "fault_onset_lookback_h": (
        "parameter",
        "first CSC to enter FAULT, from CSC summaryState (ROSP-24 §5)",
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
    """One row per dayObs whose observable night has started: -12 deg evening to -12 deg morning (TAI MJD).

    A night in progress is clipped at `now`; `complete` says whether it has ended.
    """
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
    """Where to read the override file from, and the commit it is pinned to.

    With no `source`, ask GitHub for the last commit on `ref` that touched the file and read the file at that
    commit, so the content and the recorded SHA cannot disagree. A `source` (path or URL, e.g. from
    ROSP_STATE_OVERRIDES_CSV) is read as given and its commit is reported as "not from GitHub".
    """
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


def read_overrides(source=None, ref="main"):
    """The ROSP-42 override table indexed by day_obs, and a provenance dict (URL, commit SHA, blob SHA).

    Validates what ROSP-9 validates (12:00 UT to 12:00 UT windows, one row per night) and that every row has a
    ROSP-24 `category` consistent with its `states` / `planned_downtime`, as the override README specifies.
    """
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
        parse_dates=["start_utc", "end_utc"],
        keep_default_na=False,
    )
    ov["planned_downtime"] = ov.planned_downtime.str.upper() == "TRUE"
    expected = pd.to_datetime(ov.day_obs.astype(str), format="%Y%m%d") + pd.Timedelta(
        hours=12
    )
    bad = ov[
        (ov.start_utc != expected) | (ov.end_utc != expected + pd.Timedelta(days=1))
    ]
    if len(bad):
        raise ValueError(
            f"override windows must be 12:00 UT to 12:00 UT of dayObs: {bad.day_obs.tolist()}"
        )
    if ov.day_obs.duplicated().any():
        raise ValueError(
            f"duplicate override nights: {ov.day_obs[ov.day_obs.duplicated()].tolist()}"
        )
    unknown = ov[~ov.category.isin(OVERRIDE_CATEGORIES)]
    if len(unknown):
        raise ValueError(
            f"override category not in {OVERRIDE_CATEGORIES}: {unknown.day_obs.tolist()}"
        )
    labels = ov.states.str.split("|").apply(lambda s: {x.strip() for x in s})
    ov["has_weather"] = labels.apply(lambda s: "WEATHER" in s)
    ov["has_downtime"] = labels.apply(lambda s: "DOWNTIME" in s)
    # README mapping: planned DOWNTIME -> B1; otherwise any WEATHER -> C1/C4/C5; otherwise DOWNTIME -> D4.
    expected = np.where(
        ov.planned_downtime & ov.has_downtime,
        "B1",
        np.where(ov.has_weather, "C", np.where(ov.has_downtime, "D4", "?")),
    )
    consistent = [
        (e == "B1" and c == "B1")
        or (e == "C" and c in ("C1", "C4", "C5"))
        or (e == "D4" and c == "D4")
        for e, c in zip(expected, ov.category)
    ]
    prov["inconsistent_rows"] = ov.day_obs[~np.array(consistent)].tolist()
    prov["n_rows"] = len(ov)
    prov["first"] = int(ov.day_obs.min()) if len(ov) else None
    prov["last"] = int(ov.day_obs.max()) if len(ov) else None
    return ov.set_index("day_obs"), prov


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
    """Partition each night into state runs, each clipped to the night and carrying its status bitmask.

    The state at twilight is seeded from the last message at or before it; a message is trusted for at most
    `max_state_age_h` (fault_downtime's rule) and time past that, or before the first message, has
    status NO_STATE. Returns columns day_obs, start, end, status, note, msg_index, msg_time.
    """
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
    """Faulting CSC names (SAL index kept) from a status note's "are in <State.FAULT>" clause.

    Handles both payload formats seen in 2026 (Python-repr list up to 2026-06-08, bare list after), as in
    fault_downtime. With ordered=True the names keep their order in the note, otherwise they are sorted.
    """
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
    """Consecutive FAULT-labelled state runs within a night, one row per run.

    `onset_cscs` name the CSCs behind the run, from the first source that has them (`onset_source`):

    1. "run start": the run's first message carries the current ("are in") FAULT clause;
    2. "FAULT stretch": the run continues an unbroken stretch of FAULT-bit messages from before (a fault carried
       across twilight, whose night messages repeat only the sticky "were in" clause): the *earliest* "are in"
       clause of that stretch, i.e. the fault that started it;
    3. "lookback": the latest "are in" clause within `lookback_h` before the run (the FAULT bit was set by hand,
       e.g. "Leaving observatory in fault state due to camera issues", after a CSC fault);
    4. "later in run": the first "are in" clause further into the run;
    5. "were in": the CSCs of the sticky past-tense clause in the run's own messages.

    `all_cscs` is the union of the "are in" names over the run, which is what fault_downtime splits hours over.
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
        if not onset and first_msg >= 0:
            j = first_msg - 1
            while j >= 0 and msg_t[j] >= msg_t[first_msg] - lookback_h / 24.0:
                if msg_names[j]:
                    onset, onset_time, source = msg_names[j], msg_t[j], "lookback"
                    break
                j -= 1
        if not onset:
            for note, t in zip(g.note, g.msg_time):
                names = parse_fault_cscs(note, ordered=True)
                if names:
                    onset, onset_time, source = names, t, "later in run"
                    break
        if not onset:
            for note, t in zip(g.note, g.msg_time):
                names = parse_fault_cscs(note, ordered=True, regex=FAULT_CSC_WERE_RE)
                if names:
                    onset, onset_time, source = names, t, "were in"
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
        "all_cscs",
        "first_note",
    ]
    runs = pd.DataFrame(records, columns=cols)
    return iv, runs


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

    For each onset CSC, the latest summaryState -> FAULT transition at or before the run onset (plus a tolerance,
    since the Scheduler reports after the fact, and within a lookback) is found; the earliest of those wins.
    Without a usable transition the first CSC named in the note is used (`primary_rule` says which).
    """
    tol = rules.fault_onset_tolerance_s / SECONDS_PER_DAY
    lookback = rules.fault_onset_lookback_h / 24.0
    primary, rule, entered = [], [], []
    for r in runs.itertuples():
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
    runs["group"] = [csc_group(c) if c else PENDING for c in primary]
    return runs


# ---------------------------------------------------------------------------------------------------------------
# Environment (EFD): weather tower and dome shutter, 1-minute bins
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
    """1-minute weather-tower (ESS:301) and dome-shutter bins over dayObs first..last.

    Chunked at `chunk_days` because InfluxDB refuses more than 20 000 GROUP BY buckets per query.
    Columns: wind (mean speed), gust (max speed), rh, dew_point, air_temp, dpd, shutter (max of the two doors, %).
    """
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
        env = env[~env.index.duplicated()]
        env = env.reindex(columns=cols)
    env["dpd"] = env.air_temp - env.dew_point
    env["shutter"] = env[["shutter0", "shutter1"]].max(axis=1, skipna=True)
    env["t"] = utc_index_to_tai_mjd(env.index)
    return env


# ---------------------------------------------------------------------------------------------------------------
# Allskycam (optional; SDF only)
# ---------------------------------------------------------------------------------------------------------------


def load_allskycam(path, first_day_obs, last_day_obs, rms_max=1.4):
    """Allskycam frames (TAI MJD of mid-frame, cloud_idx) for the range, or None if the file is not readable.

    Applies the night-level astrometric gate of weather_downtime_analysis (night median ast_rms_px > rms_max on
    'refine' frames drops the night). The reduction is in bstalder/rubin_allskycam; the file is the one the
    supporting notebooks read.
    """
    try:
        cam = pd.read_parquet(
            path, columns=["day_obs", "mjd", "cloud_idx", "ast_mode", "ast_rms_px"]
        )
    except Exception as err:
        return None, f"not available ({type(err).__name__}: {err})"
    cam = cam[(cam.day_obs >= first_day_obs) & (cam.day_obs <= last_day_obs)].copy()
    rms = cam[cam.ast_mode == "refine"].groupby("day_obs").ast_rms_px.median()
    bad = rms.index[rms > rms_max]
    cam = cam[~cam.day_obs.isin(bad)]
    cam["t"] = (
        Time(cam.mjd.values, format="mjd", scale="utc").tai.mjd if len(cam) else []
    )
    cam = cam.sort_values("t")
    note = f"{path}; {cam.day_obs.nunique()} nights in range, last {cam.day_obs.max() if len(cam) else 'none'}"
    if len(bad):
        note += f"; astrometric gate dropped {len(bad)} night(s)"
    return cam[["day_obs", "t", "cloud_idx"]].reset_index(drop=True), note


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
        sql = (
            f"SELECT {', '.join(EXPOSURE_COLUMNS)} FROM cdb_lsstcam.exposure "
            f"WHERE day_obs >= {lo} AND day_obs <= {hi}"
        )
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
    """Model slew time *to* each survey exposure, from rubin_nights.add_model_slew_times (rubin_scheduler
    kinematic model with the TMA limits applied at the time). Returns (survey with `slew_model`, status text).
    """
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
    n_ok = int(out.slew_model.notna().sum())
    status = f"rubin_nights add_model_slew_times: {n_ok}/{len(out)} survey exposures modelled"
    if problems:
        status += f"; failed on {len(problems)} night(s): {', '.join(problems[:5])}"
    return out, status


def survey_sequence(exposures, rules):
    """Split exposures into the evidence sets the ledger needs.

    Returns a dict of Intervals: a1 (survey open shutter), a2_exp (other survey exposures), a2_gap / d3_gap (the
    allowed and excess parts of each gap between consecutive survey exposures with nothing in between), activity
    (non-survey exposures joined into B2 activity runs), and per night the first survey science start.
    """
    exp = exposures.copy()
    exp["survey"] = exp.science_program.isin(rules.survey_programs)
    exp["a1"] = exp.survey & exp.img_type.isin(rules.a1_img_types)
    survey = exp[exp.survey].sort_values("obs_start_mjd")
    other = exp[~exp.survey].sort_values("obs_start_mjd")

    gap_allowed, gap_excess, gap_rows = [], [], []
    other_starts = other.obs_start_mjd.values
    for day_obs, night in survey.groupby("day_obs"):
        night = night.sort_values("obs_start_mjd")
        ends = night.obs_end_mjd.values
        starts = night.obs_start_mjd.values
        model = (
            night.slew_model.values
            if "slew_model" in night
            else np.full(len(night), np.nan)
        )
        for i in range(1, len(night)):
            g0, g1 = ends[i - 1], starts[i]
            if g1 <= g0:
                continue
            # A non-survey exposure inside the gap means the survey sequence was interrupted.
            k = np.searchsorted(other_starts, g0, side="left")
            if k < len(other_starts) and other_starts[k] < g1:
                continue
            allowance_s = (
                model[i] if np.isfinite(model[i]) else rules.gap_fallback_s
            ) + (rules.gap_scatter_s if np.isfinite(model[i]) else 0.0)
            cut = min(g0 + allowance_s / SECONDS_PER_DAY, g1)
            gap_allowed.append((g0, cut))
            if g1 > cut:
                gap_excess.append((cut, g1))
            gap_rows.append(
                {
                    "day_obs": day_obs,
                    "gap_start": g0,
                    "gap_s": (g1 - g0) * SECONDS_PER_DAY,
                    "model_s": model[i],
                    "allowance_s": allowance_s,
                    "excess_s": max(g1 - cut, 0.0) * SECONDS_PER_DAY,
                }
            )

    join = rules.activity_join_min / 1440.0
    runs = []
    for s, e in zip(other.obs_start_mjd.values, other.obs_end_mjd.values):
        if runs and s - runs[-1][1] <= join:
            runs[-1][1] = max(runs[-1][1], e)
        else:
            runs.append([s, e])

    def iv(pairs):
        pairs = list(pairs)
        if not pairs:
            return Intervals([], [])
        a = np.array(pairs, dtype=float)
        return Intervals(a[:, 0], a[:, 1])

    a1 = survey[survey.a1]
    first_science = a1.groupby("day_obs").obs_start_mjd.min()
    return {
        "a1": Intervals(a1.obs_start_mjd, a1.obs_end_mjd),
        "a2_exp": Intervals(
            survey[~survey.a1].obs_start_mjd, survey[~survey.a1].obs_end_mjd
        ),
        "a2_gap": iv(gap_allowed),
        "d3_gap": iv(gap_excess),
        "activity": iv(runs),
        "first_science": first_science,
        "gaps": pd.DataFrame(gap_rows),
        "exposures": exp,
    }


# ---------------------------------------------------------------------------------------------------------------
# The ledger
# ---------------------------------------------------------------------------------------------------------------


def build_ledger(windows, intervals, runs, overrides, env, cam, seq, rules):
    """Assign every part of every observable night to one category.

    Returns a segment table: one row per elementary time segment (between consecutive evidence boundaries),
    with day_obs, start, end, hours, category, detail, the evidence flags used, and the fault attribution.
    """
    # 1. Elementary segments: every evidence boundary inside a night.
    edges = [
        windows.start.values,
        windows.end.values,
        intervals.start.values,
        intervals.end.values,
    ]
    for key in ("a1", "a2_exp", "a2_gap", "d3_gap", "activity"):
        edges.append(seq[key].edges())
    minute = 1.0 / 1440.0
    if len(env):
        edges += [env.t.values, env.t.values + minute]
    if cam is not None and len(cam):
        edges += [cam.t.values - 0.5 * minute, cam.t.values + 0.5 * minute]
    allowance = rules.startup_allowance_min / 1440.0
    edges += [windows.start.values + allowance]
    fs = seq["first_science"]
    edges.append(fs.values.astype(float))
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

    # 2. Evidence at each segment.
    iv_index = np.searchsorted(intervals.start.values, mid, side="right") - 1
    iv_ok = (iv_index >= 0) & (mid < intervals.end.values[np.clip(iv_index, 0, None)])
    status = np.where(
        iv_ok, intervals.status.values[np.clip(iv_index, 0, None)], NO_STATE
    )
    fresh = status >= 0

    def bit(name):
        return fresh & ((status & STATUS_BITS[name]) > 0)

    st_W, st_F, st_D, st_I, st_O = (
        bit(k) for k in ("WEATHER", "FAULT", "DOWNTIME", "IDLE", "OPERATIONAL")
    )
    st_U = fresh & (status == 0)
    st_day_only = fresh & (status == STATUS_BITS["DAYTIME"])

    if len(env):
        t_env = env.t.values
        rh = _lookup_bins(t_env, minute, env.rh.values, mid)
        dpd = _lookup_bins(t_env, minute, env.dpd.values, mid)
        wind = _lookup_bins(t_env, minute, env.wind.values, mid)
        gust = _lookup_bins(t_env, minute, env.gust.values, mid)
        shutter = _lookup_bins(t_env, minute, env.shutter.values, mid)
    else:
        rh = dpd = wind = gust = shutter = np.full(n, np.nan)
    ws_lim, wg_lim = rules.wind_limit(day_obs)
    with np.errstate(invalid="ignore"):
        lim_rh = rh > rules.rh_max
        lim_dpd = dpd < rules.dpd_min
        lim_wind = (wind > ws_lim) | (gust > wg_lim)
    wx_known = np.isfinite(rh) | np.isfinite(wind) | np.isfinite(dpd)
    wx_bad = lim_rh | lim_dpd | lim_wind
    dome_known = np.isfinite(shutter)
    with np.errstate(invalid="ignore"):
        dome_open = dome_known & (shutter > rules.dome_open_pct)
    dome_closed = dome_known & ~dome_open

    if cam is not None and len(cam):
        cloud = _lookup_bins(
            cam.t.values - 0.5 * minute, minute, cam.cloud_idx.values, mid
        )
    else:
        cloud = np.full(n, np.nan)
    cam_known = np.isfinite(cloud)
    with np.errstate(invalid="ignore"):
        cloudy = cam_known & (cloud > rules.cloud_index_max)

    a1 = seq["a1"].contains(mid)
    a2 = seq["a2_exp"].contains(mid) | seq["a2_gap"].contains(mid)
    d3 = seq["d3_gap"].contains(mid)
    activity = seq["activity"].contains(mid)
    survey_active = a1 | a2

    day_series = pd.Series(day_obs)
    if len(overrides):
        ov_cat = (
            day_series.map(overrides.category.to_dict())
            .fillna("")
            .values.astype(object)
        )
        ov_weather = day_series.map(overrides.has_weather.to_dict()).eq(True).values
        ov_downtime = day_series.map(overrides.has_downtime.to_dict()).eq(True).values
    else:
        ov_cat = np.full(n, "", dtype=object)
        ov_weather = ov_downtime = np.zeros(n, dtype=bool)
    on_override = ov_cat != ""

    # 3. Which categories apply (before precedence).
    good = wx_known & ~wx_bad & ~st_W & ~cloudy
    applies = OrderedDict()
    applies["B1"] = ov_cat == "B1"
    applies["C_override"] = np.isin(ov_cat, ["C1", "C4", "C5"])
    applies["C1"] = ~survey_active & ~dome_open & (st_W | wx_bad | cloudy)
    applies["C2"] = ~survey_active & dome_open & cloudy
    applies["C3"] = ~survey_active & dome_open & (st_W | wx_bad)
    applies["B2a"] = activity & ~dome_closed
    applies["B2b"] = activity & dome_closed
    applies["D1"] = st_F
    applies["D4_override"] = ov_cat == "D4"
    applies["D4_state"] = st_D
    applies["D4_closed"] = dome_closed & good & ~activity & ~survey_active
    applies["D3"] = d3
    applies["E"] = (st_I | st_O) & ~survey_active
    applies["A1"] = a1
    applies["A2"] = a2

    # 4. Precedence B1 -> C -> B2 -> D (D2, D1, D4, D3) -> E -> A -> F. Override nights take the override's
    #    category for the whole night (the README: an override replaces the EFD state of the whole window).
    choices = [
        (applies["B1"], "B1", "override B1"),
        (applies["C_override"], ov_cat, "override"),
        (on_override & (ov_cat == "D4"), "D4", "override D4"),
        (applies["C1"], "C1", ""),
        (applies["C2"], "C2", "allskycam index"),
        (applies["C3"], "C3", ""),
        (applies["B2a"], "B2a", ""),
        (applies["B2b"], "B2b", ""),
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
    # Evidence detail for C1 / C3 and F.
    c_ev = np.where(st_W, "WEATHER state", "")
    c_ev = np.where(~st_W & wx_bad, "limits (tower)", c_ev)
    c_ev = np.where(~st_W & ~wx_bad & cloudy, "allskycam index", c_ev)
    sel = np.isin(category, ["C1", "C3"]) & (detail == "")
    detail[sel] = c_ev[sel]
    f = category == "F"
    detail[f & st_U] = "F1 UNKNOWN state"
    detail[f & ~fresh] = "no fresh state"
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
            "interval": np.where(iv_ok, iv_index, -1),
            "dome_open": np.where(dome_known, dome_open, np.nan),
            "wx_bad": np.where(wx_known, wx_bad, np.nan),
            "cloudy": np.where(cam_known, cloudy, np.nan),
            "survey_active": survey_active,
            "a1_evidence": a1,
            "activity": activity,
            "on_override": on_override,
        }
    )

    # 5. Start-up (A3, and the overrun to the first survey science visit) and whole-night faults (D2).
    seg = _apply_startup(seg, windows, fs, rules)
    seg = _apply_d2(seg, rules)

    # 6. Shadow evidence (kept for statistics, never added to the totals).
    seg["sh_weather_curated"] = ov_weather
    seg["sh_weather_measured"] = (wx_known & wx_bad) | cloudy | st_W
    seg["sh_fault"] = st_F
    seg["sh_downtime"] = st_D | ov_downtime
    for key, val in applies.items():
        seg[f"applies_{key}"] = val

    # 7. Fault attribution.
    seg = _attribute_faults(seg, intervals, runs)
    return seg


def _apply_startup(seg, windows, first_science, rules):
    """A3 inside the allowance (taking E and F time), then the overrun up to the first survey science visit.

    The overrun is the stretch from the end of the allowance that would otherwise be E or F, up to the first
    survey science visit, stopping at the first segment of any other category: a weather closure or a fault at
    the start of the night ends the start-up rather than being part of it.
    """
    allowance = rules.startup_allowance_min / 1440.0
    seg = seg.copy()
    cat = seg.category.values.copy()
    det = seg.detail.values.copy()
    for day_obs, t_first in first_science.items():
        if day_obs not in windows.index:
            continue
        w = windows.loc[day_obs]
        if not (w.start <= t_first < w.end):
            continue
        idx = np.where(seg.day_obs.values == day_obs)[0]
        mids = 0.5 * (seg.start.values[idx] + seg.end.values[idx])
        in_allow = idx[(mids < min(w.start + allowance, t_first))]
        take = in_allow[np.isin(cat[in_allow], ["E", "F"])]
        cat[take] = "A3"
        det[take] = "start-up allowance"
        over = idx[(mids >= w.start + allowance) & (mids < t_first)]
        for i in over:
            if cat[i] in ("E", "F"):
                cat[i] = rules.startup_overrun_category
                det[i] = "start-up overrun"
            else:
                break
    seg["category"] = cat
    seg["detail"] = det
    return seg


def _apply_d2(seg, rules):
    """Whole-night faults: a night with no survey science visit whose FAULT state covers the night to within
    `d2_tolerance_min` (counting FAULT under every category, since the fault lasted regardless). Its time not
    claimed by B1, C or B2 becomes D2."""
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
    # An override night keeps its override category, so it is never a D2 night.
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
    """Fault group for D1 / D2 / D4 segments.

    D1 from the FAULT state: the primary CSC of the run. D2: the night's largest fault run. D4: "Operations
    decision" for a closed dome with nothing running, otherwise pending (the prompting subsystem is not in the
    data). Start-up overrun: pending. D3 is unattributed by construction.
    """
    seg = seg.copy()
    run_of_interval = {}
    if len(runs):
        # Map each state interval to its fault run through the interval table's fault_run column.
        for i, r in intervals.dropna(subset=["fault_run"]).iterrows():
            run_of_interval[i] = int(r.fault_run)
    seg["fault_run"] = [
        run_of_interval.get(int(i), np.nan) if i >= 0 else np.nan
        for i in seg.interval.values
    ]
    run_group = runs.set_index("fault_run").group.to_dict() if len(runs) else {}
    run_csc = runs.set_index("fault_run").primary_csc.to_dict() if len(runs) else {}
    group = np.full(len(seg), "", dtype=object)
    csc = np.full(len(seg), "", dtype=object)
    is_d1_state = (seg.category == "D1") & (seg.detail == "FAULT state")
    for i in np.where(is_d1_state | (seg.category == "D2"))[0]:
        r = seg.fault_run.values[i]
        if np.isfinite(r):
            group[i] = run_group.get(int(r), PENDING)
            csc[i] = run_csc.get(int(r)) or ""
    # D2 segments without a fault run of their own take the night's largest run.
    d2 = seg.category == "D2"
    if d2.any():
        big = (
            seg[d2 & seg.fault_run.notna()]
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
                group[i], csc[i] = run_group.get(r, PENDING), run_csc.get(r) or ""
    group[(seg.category == "D1") & (seg.detail == "start-up overrun")] = PENDING
    group[
        (seg.category == "D4")
        & (seg.detail == "closed dome, good conditions, nothing running")
    ] = "Operations decision"
    group[(seg.category.isin(["D1", "D2", "D4"])) & (group == "")] = PENDING
    seg["fault_group"] = group
    seg["fault_csc"] = csc
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
    """Hours and fraction of the observable night by category, by group, and D1+E together."""
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
    d1e = hours["D1"] + hours["E"]
    combined = pd.DataFrame(
        {
            "description": ["D1 short fault + E idle (reported together, Q7)"],
            "hours": [d1e],
            "fraction": [d1e / total if total else np.nan],
        },
        index=["D1+E"],
    )
    return out, groups, combined, total


def fault_by_group(seg, runs):
    """Fault-category hours by attribution group (D1, D2, D4), plus fault_downtime's equal split for comparison."""
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
    order = [g for g in FAULT_GROUPS if g in table.index] + [
        g for g in table.index if g not in FAULT_GROUPS
    ]
    table = table.reindex(order)

    # Equal split over every CSC named in the run (fault_downtime), applied to the same FAULT-state hours.
    state_fault = seg[seg.category.isin(["D1", "D2"]) & seg.fault_run.notna()]
    by_run = state_fault.groupby("fault_run").hours.sum()
    split = {}
    run_cscs = runs.set_index("fault_run").all_cscs.to_dict() if len(runs) else {}
    for r, h in by_run.items():
        cscs = run_cscs.get(int(r)) or []
        groups = [csc_group(c) for c in cscs] or [PENDING]
        for g in groups:
            split[g] = split.get(g, 0.0) + h / len(groups)
    primary = state_fault.groupby("fault_group").hours.sum()
    for g in list(split) + list(primary.index):
        if g not in table.index:
            table.loc[g] = 0.0
    table["FAULT-state hours, primary CSC"] = primary.reindex(table.index).fillna(0.0)
    table["FAULT-state hours, equal split (fault_downtime)"] = (
        pd.Series(split).reindex(table.index).fillna(0.0)
    )
    return table


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
    add(
        "Weather during planned maintenance (B1), from the override WEATHER state",
        b1 & seg.sh_weather_curated,
    )
    add(
        "Weather during planned maintenance (B1), measured (tower limits, allskycam, WEATHER state)",
        b1 & seg.sh_weather_measured,
    )
    add("FAULT state under weather (C)", c & seg.sh_fault)
    add("DOWNTIME (state or override) under weather (C)", c & seg.sh_downtime)
    add("Engineering / calibration activity under weather (C)", c & seg.activity)
    return pd.DataFrame(rows).set_index("shadow record")


def shadow_by_night(seg):
    b1 = seg.category == "B1"
    c = seg.category.str.startswith("C")
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
        "IDLE/OPERATIONAL-no-visit time inside a visit gap (D3 by precedence, not E)",
        seg.applies_E & (seg.category == "D3"),
    )
    add(
        "Measured weather on non-C override nights",
        seg.on_override & ~seg.category.str.startswith("C") & seg.sh_weather_measured,
    )
    add("Survey visits on override nights", seg.on_override & seg.a1_evidence)
    add(
        "No dome telemetry, treated as closed for C1",
        seg.applies_C1
        & seg.dome_open.isna()
        & (seg.category == "C1")
        & ~seg.on_override,
        by="day_obs",
    )
    fault_or_down = (seg.status >= 0) & (
        (seg.status & (STATUS_BITS["FAULT"] | STATUS_BITS["DOWNTIME"])) > 0
    )
    add(
        "Engineering / calibration (B2) while the state was FAULT or DOWNTIME",
        seg.category.str.startswith("B2") & fault_or_down,
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
