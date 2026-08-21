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
| [`efficiency.ipynb`](notebooks/efficiency.ipynb) | On-sky observing efficiency: modeled slew/settle overheads vs. actual visit gaps, plus dome-open hours, narrative-log fault/weather time, and EFD-recorded observatory states, extrapolated to a per-night system availability × `fO`. | `day_obs`, `n_days` |
| [`image_quality_trending.ipynb`](notebooks/image_quality_trending.ipynb) | PSF / delivered image-quality trends across a night range: per-detector and per-visit FWHM, ellipticity and moment-score distributions, decomposing DIQ into atmosphere / optics+camera / across-FoV variation. | `day_obs_min`, `day_obs_max` |
| [`on-sky_utilization.ipynb`](notebooks/on-sky_utilization.ipynb) | On-sky time utilization: visit timeline vs. twilight, acquired-vs-ideal visit rate, and inter-visit gap-time trending. | `day_obs_min`, `day_obs_max` |

Each notebook has a sidecar `.yaml` (e.g. [`notebooks/efficiency.yaml`](notebooks/efficiency.yaml))
of the same name that registers it as a Times Square page.

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

## Fault / idle accounting in `efficiency.ipynb`

`efficiency.ipynb` carries three independent accountings of lost time, which do not
agree and are not interchangeable:

| Column(s) | Source | Nature |
|-----------|--------|--------|
| `total_fault_idle`, `total_fault_idle_gap` | inferred from `visit_gap` minus the modeled slew | derived |
| `log_fault`, `log_weather` | narrative log | human-reported |
| `fault_down`, `idle_down`, `weather_down`, `downtime_down`, `state_fault_idle` | EFD observatory states, via `rubin_nights.observatory_status` | recorded |

Only the inferred `total_fault_idle_gap` feeds `ratio_active`, and therefore `eff_all`.
Because it is measured *between consecutive visits*, it cannot see time on a night when
observing stopped outright — a full-night fault, or a weather-shuttered night with no
visits, yields no gaps and so contributes no inferred fault time. Such nights are `NaN`
(not zero) in the inferred columns and drop out of the `np.nanmean` availability
figures. Read `eff_all` / `eff_fbs` as availability **conditional on observing having
happened**, not as a fraction of all calendar nights; the notebook prints a per-night
comparison table and flags nights where the inferred and recorded values diverge
sharply.

Note the EFD state periods **overlap** and must not be summed: a `WEATHER` period
typically spans the whole night while `OPERATIONAL` / `IDLE` / `FAULT` run inside it, so
the per-night columns can total well over `night_hours`. `count_observatory_states`
de-overlaps only `FAULT` occurring during `DOWNTIME` (via its `contributed_hours`
column). Treat each column as "hours during which this was true", not as a partition of
the night.

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
- **Coerce sidecar parameters.** Times Square substitutes parameters as **strings**,
  while the notebook's own defaults cell may produce ints. Notebooks that do arithmetic
  on a `day_obs` cast it explicitly (`day_obs = int(day_obs)`) in the bootstrap cell.

Keep imports limited to what a notebook actually uses; an unused import of a package
that is absent from the target environment breaks the whole notebook for no benefit.
