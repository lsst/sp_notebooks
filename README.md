# sp_notebooks
Performance of the integrated Rubin System for use with the [Times Square](https://sqr-062.lsst.io/) service.

See the [Times Square documentation](https://rsp.lsst.io/v/usdfdev/guides/times-square/index.html) for more information or the [deployed version](https://usdf-rsp-dev.slac.stanford.edu/times-square/).

These notebooks track how well the integrated Rubin system is performing on-sky —
visit acquisition rate, delivered image quality, observing efficiency, and on-sky
time utilization — by querying the Consolidated Database (ConsDB), the EFD, and the
exposure / narrative logs and comparing against the `baseline_v5.1.0_10yrs` survey
simulation. Data access is handled through
[`rubin_nights`](https://github.com/lsst-sims/rubin_nights), whose
`connections.get_clients()` resolves the correct endpoints and tokens for whatever
environment the notebook runs in (Times Square / Nublado, the USDF RSP, or an SDF
login node).

## Notebooks

| Notebook | What it measures | Key parameters |
|----------|------------------|----------------|
| [`diq_vs_visit_rate.ipynb`](notebooks/diq_vs_visit_rate.ipynb) | Science-visit acquisition rate and delivered image quality (DIQ) over the last `n_nights`, placed on a visit-rate vs. image-quality plane against the "LSST = 1" goal. | `day_obs_max`, `n_nights` |
| [`efficiency.ipynb`](notebooks/efficiency.ipynb) | On-sky observing efficiency: modeled slew/settle overheads vs. actual visit gaps, plus dome-open hours, narrative-log fault/weather time, and EFD-recorded observatory states, extrapolated to a per-night system availability × `fO`. The notebook's own markdown explains how its three fault/idle accountings (inferred, narrative-log, EFD) differ and why they must not be summed. | `day_obs`, `n_days` |
| [`image_quality_trending.ipynb`](notebooks/image_quality_trending.ipynb) | PSF / delivered image-quality trends across a night range: per-detector and per-visit FWHM, ellipticity and moment-score distributions, decomposing DIQ into atmosphere / optics+camera / across-FoV variation. | `day_obs_min`, `day_obs_max` |
| [`on-sky_utilization.ipynb`](notebooks/on-sky_utilization.ipynb) | On-sky time utilization: visit timeline vs. twilight, acquired-vs-ideal visit rate, and inter-visit gap-time trending. | `day_obs_min`, `day_obs_max` |
| [`weekly_metrics.ipynb`](notebooks/weekly_metrics.ipynb) | Weekly community-forum report: trailing-week and season-to-date open-shutter time, visits by band, delivered IQ vs. design, fault/weather downtime, open-shutter efficiency, DDF/main-survey allocation, sky coverage and depth vs. the baseline, and prompt-processing throughput. Ends with an auto-filled plain-text headline summary for the community post. Scheduled to run weekly on Times Square (Mondays, 14:00). | `day_obs_max`, `n_days`, `season_start_day_obs` |

Each notebook has a sidecar `.yaml` (e.g. [`notebooks/efficiency.yaml`](notebooks/efficiency.yaml))
of the same name that registers it as a Times Square page.

## Performance Tracking Hub (`tracking_hub/`)

[`tracking_hub/`](tracking_hub/) holds one Times Square notebook per Tier 1 system-performance
metric tracked by the
[Performance Tracking Hub (PTH)](https://rubinobs.atlassian.net/wiki/spaces/ROSP/pages/1906802696).
Each metric is a Jira epic, `ROSP-1` through `ROSP-12`
(<https://rubinobs.atlassian.net/browse/ROSP-N>), and lives in its own folder
`tracking_hub/<category>/rosp-NN_<short_name>/` with an official notebook and sidecar of
the same name. Folder paths define the Times Square page URLs linked from Confluence and
Jira, so metric folders must never be renamed or moved. All 12 notebooks are currently
**stubs**: they document the metric and validate parameters but compute nothing yet.
Every notebook takes the same `start_dayobs` / `end_dayobs` parameters, defaulting to the
PTH baseline epoch (2026-06-29 to 2026-12-31). See
[`tracking_hub/README.md`](tracking_hub/README.md) for layout rules, tagging, versioning,
and the Method Change Process.

| Epic | Metric | Category | Owner |
|------|--------|----------|-------|
| [ROSP-1](tracking_hub/data_production/rosp-01_depth/) | Single-visit and coadded depth by band | Data Production | Colin Slater |
| [ROSP-2](tracking_hub/data_production/rosp-02_delivered_iq/) | Delivered image quality (including atmosphere) | Data Production | Elana Urbach |
| [ROSP-3](tracking_hub/data_delivery/rosp-03_alert_latency/) | Alert generation latency and completeness | Data Delivery & User Experience | Colin Slater |
| [ROSP-4](tracking_hub/data_production/rosp-04_survey_progress/) | Survey progress vs. 10-year projection | Data Production | Lynne Jones |
| [ROSP-5](tracking_hub/data_delivery/rosp-05_dr_query_latency/) | Data release query and image retrieval latency | Data Delivery & User Experience | Colin Slater |
| [ROSP-6](tracking_hub/data_production/rosp-06_effective_survey_speed/) | Effective survey speed (normalized etendue, fE) | Data Production | Lynne Jones |
| [ROSP-7](tracking_hub/data_production/rosp-07_sdiq/) | System delivered image quality (sDIQ) | Data Production | Elana Urbach |
| [ROSP-8](tracking_hub/data_production/rosp-08_field_star_ellipticity/) | Field star ellipticity | Data Production | Elana Urbach |
| [ROSP-9](tracking_hub/ops/rosp-09_system_availability/) | System availability / uptime | Observatory Operations | Erik Dennihy |
| [ROSP-10](tracking_hub/ops/rosp-10_open_shutter_slew_settle/) | Open-shutter fraction and slew/settle time | Observatory Operations | Erik Dennihy |
| [ROSP-11](tracking_hub/ops/rosp-11_time_loss_accounting/) | Time-loss accounting (weather / fault / scheduled) | Observatory Operations | Brian Stalder |
| [ROSP-12](tracking_hub/ops/rosp-12_fault_rate_mttr/) | Fault rate, MTTR and top fault categories | Observatory Operations | Brian Stalder |

`tracking_hub/dev/` is scratch space excluded from Times Square by the `ignore` glob in
the root [`times-square.yaml`](times-square.yaml); keep it flat, since the glob is not
recursive.

## Times Square parameters

The sidecar `.yaml` files declare each page's title, description, authors, and
**parameters** (with types and defaults), and optionally a run `schedule`. When Times
Square executes a notebook it replaces the first code cell with the sidecar parameter
values, so the notebook's own first cell is only a set of local defaults for
interactive use.

To let the same notebook run both under Times Square and standalone, the parameters
cell sets a `not_times_square = True` flag. Times Square drops that flag when it
substitutes parameters, so a following bootstrap cell detects the Times Square runtime
via the resulting `NameError` and pip-installs / upgrades `rubin_nights` in the
Nublado pod; elsewhere (e.g. an SDF login node) that cell is a no-op.

## Development

This repository uses pre-commit hooks to keep notebooks consistent (including
stripping notebook outputs). Install the pre-commit by running:

```bash
pip install pre-commit
pre-commit install
```

### Running the notebooks outside Times Square

The notebooks are expected to run both under Times Square and standalone (e.g. on an
SDF login node), which imposes two constraints worth knowing when editing them:

- **Don't gate environment setup on `EXTERNAL_INSTANCE_URL`.** That variable is only
  set on an RSP, so a check like `if "usdf" in current_location` silently does nothing
  on a login node. `efficiency.ipynb` instead probes for a readable
  `rubin_sim_data` directory and sets `RUBIN_SIM_DATA_DIR` only if it is not already
  set — needed for the site models (sunset/sunrise, seeing).
- **Coerce sidecar parameters, and don't rely on the parameters cell for imports.**
  Times Square substitutes parameters as **strings**, and a `format: dayobs` parameter
  arrives *dashed* (`"2026-08-21"`), so a bare `int()` raises `ValueError` — use
  `rn_dayobs.day_obs_str_to_int()`, which accepts both that and `"20260821"`. Note also
  that Times Square replaces the whole first code cell, discarding any imports it held,
  so the bootstrap cell must import what it uses rather than inheriting from above.

Keep imports limited to what a notebook actually uses; an unused import of a package
that is absent from the target environment breaks the whole notebook for no benefit.
