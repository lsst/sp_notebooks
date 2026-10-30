# Observatory state overrides

`observatory_state_overrides.csv` holds manual corrections to the observatory state recorded in the EFD, so that any
notebook under `tracking_hub/ops/` can read the same file and apply the same overrides. Notebooks must not import from
each other, so each notebook reads the CSV itself.

One row per override night:

| Column | Meaning |
|---|---|
| `day_obs` | UT date (YYYYMMDD) being overridden |
| `start_utc`, `end_utc` | Override window: 12:00 UT on `day_obs` to 12:00 UT on the following date. This covers the whole -12 to -12 deg night. |
| `states` | EFD observatory states in force for the whole window, `\|`-separated (e.g. `DOWNTIME\|WEATHER`) |
| `planned_downtime` | `TRUE` if a `DOWNTIME` in this row is planned (scheduled) downtime; `FALSE` is unplanned |
| `source` | Where the override came from |
| `notes` | Free text |

Where an override exists it replaces the EFD state for the whole window. Nights without a row are not overridden.

## Current contents (103 nights, 20260715 to 20261025)

| dayObs | States | Source |
|---|---|---|
| 20260715 to 20260831 | `DOWNTIME\|WEATHER` | Erik Dennihy (mid-July to 1 Sep). Extreme weather event: the observatory was down, and this period is deliberately not counted against availability. |
| 20260901 to 20260914 | `WEATHER` or `DOWNTIME` | `ROSP-53_Aggregated_Weather_Loss.csv`, column "Rubin Nightly Weather Loss Projection" (TRUE = `WEATHER`) |
| 20260915 to 20261004 | `DOWNTIME`, `planned_downtime = TRUE` | Erik Dennihy (planned downtime, 14 Sep to 5 Oct) |
| 20261005 to 20261025 | `WEATHER` or `DOWNTIME` | `ROSP-53_Aggregated_Weather_Loss.csv`, same column |

From 26 October 2026 on the state was recorded correctly on the summit and no override is needed.
