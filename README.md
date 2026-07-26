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
| [`efficiency.ipynb`](notebooks/efficiency.ipynb) | On-sky observing efficiency: modeled slew/settle overheads vs. actual visit gaps, plus dome-open hours and narrative-log fault/weather time, extrapolated to a per-night system availability × `fO`. | `day_obs`, `n_days` |
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

## Development

This repository uses pre-commit hooks to keep notebooks consistent (including
stripping notebook outputs). Install the pre-commit by running:

```bash
pip install pre-commit
pre-commit install
```
