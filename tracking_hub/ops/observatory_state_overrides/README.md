# Observatory state overrides

`observatory_state_overrides.csv` holds manual corrections to the observatory state recorded in the EFD, so that any
notebook under `tracking_hub/ops/` can read the same file and apply the same overrides. Notebooks must not import from
each other, so each notebook reads the CSV itself.

## What this file asks of the observatory states, and what it does not

The Scheduler `ObservatoryState` enums (`OPERATIONAL`, `WEATHER`, `DOWNTIME`, `FAULT`, ...) record how the time was
used. They are read first and are never edited: this file does not change the EFD, and `states` keeps the EFD
vocabulary so anything that reads it like the Scheduler topic keeps working.

The ROSP-24 ledger (ROSP-9, 11, 12) is a layer on top of the states. Everything the states do not say — engineering vs
survey, calibration blocks, idle time, which subsystem a fault belongs to — comes from program and visit metadata
(scheduler blocks, `observation_reason`, ConsDB visits, CSC summary states), not from the state enums. The one
thing the ledger asks of the state system is a planned/unplanned split of `DOWNTIME` (ROSP-48); until that exists, the
`planned_downtime` column here carries it.

This file is the one place where the ledger deliberately departs from a recorded state: nights where the EFD record is
missing (`UNKNOWN` from mid-July 2026) or known to be wrong. Each row names its evidence (`source`) so the departure is
auditable, and the file version is recorded with every report that uses it. A corrective path for the EFD record itself
is being discussed in OBS-2062; if that lands, rows here become redundant rather than contradictory. Reviewers of the
state semantics (ROSP-24) only need to check this boundary: does any row here contradict a state that was recorded
correctly?

One row per override night:

| Column | Meaning |
|---|---|
| `day_obs` | UT date (YYYYMMDD) being overridden |
| `start_utc`, `end_utc` | Override window: 12:00 UT on `day_obs` to 12:00 UT on the following date. This covers the whole -12 to -12 deg night. |
| `states` | EFD observatory states in force for the whole window, `\|`-separated (e.g. `DOWNTIME\|WEATHER`) |
| `planned_downtime` | `TRUE` if a `DOWNTIME` in this row is planned (scheduled) downtime; `FALSE` is unplanned |
| `category` | ROSP-24 category attributed to the night: `B1`, `C1`, `C4`, `C5` or `D4` (see below) |
| `source` | Where the override came from |
| `notes` | Free text |

Where an override exists it replaces the EFD state for the whole window. Nights without a row are not overridden.

## ROSP-24 categories

`states` stays in EFD vocabulary, so anything that reads it like the Scheduler topic keeps working. `category` holds the
ROSP-24 attribution that ROSP-9, 11 and 12 report by. ROSP-24 precedence is B1 -> C -> B2 -> D -> E -> A -> F, which is
*not* the Scheduler-style priority (`DOWNTIME` > `WEATHER`). A row maps to a category as follows:

| Row | Category |
|---|---|
| `DOWNTIME` or `DOWNTIME\|WEATHER`, `planned_downtime = TRUE` | **B1** planned downtime |
| Any other row containing `WEATHER` | **C**: C1, C4 or C5 (below) |
| `DOWNTIME`, `planned_downtime = FALSE` | **D4** unplanned downtime |

Because B1 comes first, weather during planned maintenance does not count against allocations. It is kept as
`DOWNTIME|WEATHER` with `planned_downtime = TRUE` so the weather stays in the shadow record for year-over-year statistics.
Do not read `DOWNTIME|WEATHER` with `planned_downtime = FALSE` as downtime: the storm block 20260715 to 20260831 is **C1**
(catastrophe/weather downtime, as in the ROSP-42 table).

The `WEATHER` categories are different evidence behind the same state:

- **C1**: catastrophe/weather downtime, from the ROSP-42 table.
- **C4**: CFOps summit-access road closure.
- **C5**: weather loss at the neighbouring observatories. Rubin is marked lost when all three observatories (Gemini,
  SOAR, Blanco) lost the night, or when Gemini and SOAR were both closed with Blanco open. **Missing data:** when Gemini
  has no data, SOAR decides, i.e. a night where SOAR reports >50% weather loss is C5 whether or not Blanco was open
  (20260908, 20260909, 20260911).

## Current contents (103 nights, 20260715 to 20261025)

| dayObs | States | Source |
|---|---|---|
| 20260715 to 20260831 | `DOWNTIME\|WEATHER` | Erik Dennihy (mid-July to 1 Sep). Extreme weather event: the observatory was down, and this period is deliberately not counted against availability. |
| 20260901 to 20260913 | `WEATHER` or `DOWNTIME` | `ROSP-53_Aggregated_Weather_Loss.csv`, column "Rubin Nightly Weather Loss Projection" (TRUE = `WEATHER`). Bus access resumed 7 Sep when weather permitted. |
| 20260914 to 20261004 | `DOWNTIME`, `planned_downtime = TRUE` (B1). `DOWNTIME\|WEATHER` on 20260918 to 20260921 and 20261004, from the ROSP-53 summit-access closures (AC10, AC11, AC12) | Erik Dennihy (planned downtime, 14 Sep to 5 Oct noon UTC to noon UTC, as in ROSP-42) |
| 20261005 to 20261025 | `WEATHER` or `DOWNTIME` | `ROSP-53_Aggregated_Weather_Loss.csv`, same column |

From 26 October 2026 on the state was recorded correctly on the summit and no override is needed.
