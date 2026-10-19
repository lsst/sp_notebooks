# AI session summary: ROSP-9 observable night definition (ROSP-22)

Last updated: 2026-10-19 (notebook refactored; see Notebook structure). Purpose: recover this session quickly. Update this file as the work progresses.

## Goal

Ticket (ROSP-9, raised at the PTH steward kickoff, Oct 2026): choose the **denominator of system availability**, i.e. the
definition of the *observable night*. Changing it later is a series break.
Done when: definition chosen (incl. twilight limit and weather data source); recorded in the Decision Log and on the
System Availability page; agreed with the time-loss accounting steward (ROSP-11) so denominators do not double-count.

**Decision (agreed with the user, 2026-10-19): the observable night is the time between evening and morning twilight with the
Sun at altitude -12 deg (`TWILIGHT_ALT = -12`), from the Sun alone; no weather, state or dome input.** The notebook was
refactored to state this up front. The other definitions that were compared are kept only as a historical record and are
labelled as not valid:

1. Full night, -12 to -12 deg twilight (adopted).
2. Full night minus `DOWNTIME` (historical, not valid).
3. Full night minus `WEATHER` (historical, not valid; added 2026-10-19 and not yet run).
4. Full night minus `DOWNTIME` and `WEATHER` (historical, not valid; was definition 3 before 2026-10-19).

`UNKNOWN` is disregarded (never removed). A `DOWNTIME`+`UNKNOWN` definition was considered and dropped.

The argument behind keeping `WEATHER` time: observing is not limited to an open dome (closed-dome calibration observations
can be taken during weather closures). The historical part measures how much `OPERATIONAL` time falls inside `WEATHER`
periods as supporting evidence. Dome-open telemetry is deliberately not compared.

## Files here

- `observable_night_definition_ROSP-22.ipynb`: the analysis notebook (edited with the Jupyter MCP tools; do not write the .ipynb directly).
- `AI_SUMMARY.md`: this file.

## Notebook structure

Run locally at USDF (`lsst` kernel) only. No Times Square code (no parameter-cell convention, no `pip install`).

**Part 1: the definition** (needs only `rubin_nights.dayobs_utils`; no EFD, no states). Edited by the user on 2026-10-19
after the refactor; the notebook is the reference, this is a description of it.
1. Markdown: title, decision, layout (says the definition is in "the next cell").
2. A scratch cell (`observable_night(20260629)[0].isot`) sits here, before `observable_night` is defined. It is not part of the
   design and raises `NameError` on a top-to-bottom run (see Known issues).
3. The definition cell: imports (`numpy`, `pandas`, `rubin_nights.dayobs_utils`), `TWILIGHT_ALT = -12.0`,
   `observable_night(day_obs)` (start/end Times from `rn_dayobs.day_obs_sunset_sunrise(day_obs, TWILIGHT_ALT)`), and a printed
   example for `example_dayObs = 20260629`.
4. Example plot cell: night length over the year. It also holds `start_observable_night = 20260629`,
   `end_observable_night = 20270628` (dayObs, inclusive; one full year), `day_obs_range`, `observable_night_table` and
   `observable_nights` (`start`, `end` TAI MJD, `night_hours`; 365 nights, 7.87 to 11.93 h, mean 10.06 h). Part 2 reuses
   `day_obs_range` and `observable_night_table`, so this cell must run before part 2.
5. Example plot: calendar of night length (weeks as rows, Monday to Sunday, one decimal per cell). Also defines
   `calendar_grid`, `draw_calendar`, `BLUES`, reused by part 2.

**Part 2: historical comparison (NOT VALID definitions; needs the EFD).** Banner markdown, and every figure carries the red
`HISTORICAL_TITLE`.
5. Setup cell: `history_start_dayobs = 20260501`, `history_end_dayobs = 20261016` (independent of the part 1 year), `day_obs_list`,
   `t_start`, `t_end` (Times for the EFD queries), `connections.get_clients()`.
6. `dome_open` (only an input to `count_observatory_states`) and `night_times = observable_night_table(day_obs_list)`.
7. EFD states: `get_observatory_state_times` + `count_observatory_states` give per-night `operational_hours`, `fault_down`,
   `idle_down`, `weather_down`, `downtime_down`, `unknown_down`. These overlap and are NOT a partition. Two manual
   adjustments happen before the per-night sums (see below).
8. Definitions (`EXCLUDE`, keys 1 to 4): hours per night are computed from the raw periods (`start`, `end`, `type`) as the
   **union** of excluded intervals clipped to the night window, so overlapping `WEATHER`/`DOWNTIME` are not double-subtracted.
   Also reports `operational_excluded`, the `OPERATIONAL` hours inside the excluded intervals.
9. Plots, in order: EFD state per night as a stacked bar (defines `STACK`, `LABEL`, `state_codes`, `night_len`, `part`);
   night-by-time raster (diagnostic); calendar of observable hours per night for each definition (uses `draw_calendar`);
   observable hours per night as lines. The raster, calendar and lines cells depend on the stacked-bar cell and on part 1.
   - Each night is split into exclusive pieces (1440 equal time bins). Priority, first match wins: `DOWNTIME` > `WEATHER`
     (split into "recorded OPERATIONAL" and the rest) > `OPERATIONAL` > `FAULT` > `UNKNOWN` > `IDLE`; time with none of these is
     "no state recorded". `DOWNTIME` overriding the rest matches `count_observatory_states`.
   - Colours are the reference palette in a stack order validated for adjacent-pair CVD separation. A grey strip below
     the bars marks carried-forward nights.
10. Notes markdown.

## Handling of missing and wrong states

- **Nights with no status messages.** `observatoryStatus` is logged only on a change of state and the library
  (`_obs_status_state_changes`) builds periods only for nights that have a message, so quiet nights silently vanished.
  `fill_missing_nights` carries the state in force at sunrise of the last night with periods forward over those nights
  (clipped to now; a night that has not started is skipped; nights before the first message cannot be filled).
- **`NIGHT_STATE_OVERRIDES`** (top of the states cell) replaces the logged state of named nights where it is known to be
  wrong, as `{dayObs: [types]}`. Applied only to the nights listed. Currently `{20260925: ["DOWNTIME", "IDLE"]}`: that night
  is logged `UNKNOWN` but the observatory was in `DOWNTIME`. The overridden state is also what gets carried forward over
  20260926-20260928 (no messages until `DOWNTIME` is logged on 20260929). A bare `DOWNTIME` appears in the library as
  `IDLE | DOWNTIME`, hence both types.
- Tonight (not yet started) has no periods and is dropped rather than shown as zeros.

## Historical results (table below predates definition 3; rerun to fill it in) (default window, 109 nights, 2026-06-29 to 2026-10-15; run 2026-10-16)

Note: these figures are for the default window (`start_dayobs=20260629`, `end_dayobs=0`, as in the YAML). On 2026-10-16
the parameters cell in the working notebook had been set by the user to `start_dayobs=20260601`, `end_dayobs=20261016`;
re-run the notebook for figures over that window (the first night, 20260601, starts mid-state, so it has some
"no state recorded" time).

| Definition | Pooled hours | % of full night |
|---|---|---|
| 1. Full night | 1187.8 | 100% |
| 2. Minus `DOWNTIME` | 185.5 | 15.6% |
| 4. Minus `DOWNTIME` and `WEATHER` | 143.7 | 12.1% |

- 26.2 h of `OPERATIONAL` state fall inside `WEATHER` periods, so definition 4 discards 26.2 operational hours (definition 3
  not yet run); definitions 1 and 2 discard none. This is the main support for the argument above.
- 93 of 109 nights have zero observable hours under definitions 2 and 4 because the observatory was in `DOWNTIME`
  (mostly carried-forward nights from 2026-07-20). Pooled totals therefore depend heavily on the window chosen.
- `UNKNOWN` is negligible in the first weeks but is about 40 h over the whole window, so disregarding it is not neutral
  over long windows.

## Known issues and caveats (read first)

- **Top of the notebook (user edits, 2026-10-19), not yet fixed:** the scratch cell before the definition cell runs before
  `observable_night` exists (`NameError` on a fresh run); the definition cell prints `example_observable_night[0].isot` for both
  the start and the end of the night (the end should be `[1]`). The "TAI" label in that print is correct:
  `day_obs_sunset_sunrise` returns `(sunset, sunrise)` as Times built with `scale="tai"` (checked in
  `rubin_nights/dayobs_utils.py`; its docstring wrongly says civil / -6 deg, but it uses the `sun_alt` argument).
- **`WEATHER` is carried forward as well as `DOWNTIME`.** The last logged state before the quiet period was
  `WEATHER | DOWNTIME`, so `WEATHER` is carried over 67 nights (about 735 h, 2026-07-20 to 2026-09-24). This does not
  change the table (`DOWNTIME` covers the same nights for definitions 2 and 4) but it inflates the `WEATHER` line in the state
  plot and any `WEATHER` share quoted for this window. A possible fix (carry only `DOWNTIME`/`IDLE`) was suggested and
  **explicitly declined by the user**; do not apply it unless asked.
- **Carry-forward is an assumption.** A long run of nights with no events could also be a gap in the EFD record rather than a
  persistent state.
- Weather source so far is only the EFD `WEATHER` state. Not compared with narrative-log weather time lost or a weather
  station criterion.
- The case for keeping `WEATHER` time rests on the state record only. Counting calibration exposures taken during
  `WEATHER` periods (Consolidated Database or exposure log) would show directly that observations happened; not done.
- Overlap treatment (union of states) only matters for the historical comparison now; the adopted definition does not use states.
- Plot cells embed large images; `run_cell` through the MCP tool errors on oversized output even though the cell ran.
  Check the output file with `jq`.

## Location note

The user asked for this folder under `tracking_hub/data_production/`. The repo README places ROSP-9 under
`tracking_hub/ops/rosp-09_system_availability` (official stub, steward Erik Dennihy) and says never to move metric folders
and that diagnostic notebooks must not carry the `rosp-NN` tag. The notebook follows the user's path but uses the
`diagnostic` tag. Decide whether this should move to `ops/` or `dev/` before opening a PR. Branch: `tickets/ROSP-9`.
Nothing has been committed.

## Suggested next steps

1. Re-run part 2 to fill in definition 3 and refresh the saved outputs (the saved outputs predate the refactor).
2. Record the decision (Sun -12 to -12 deg) in the Decision Log and on the System Availability page; tell the ROSP-11 steward.
3. Decide whether the notebook should move to `ops/` or `dev/` before opening a PR (see Location note).
