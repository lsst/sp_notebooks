# Performance Tracking Hub notebooks

Times Square notebooks for the 12 Tier 1 system-performance metrics tracked by the
[Performance Tracking Hub (PTH)](https://rubinobs.atlassian.net/wiki/spaces/ROSP/pages/1906802696).
Each metric is a Jira epic, `ROSP-1` through `ROSP-12`
(<https://rubinobs.atlassian.net/browse/ROSP-N>). All notebooks here are currently
**stubs**: they document the metric and validate parameters but compute nothing yet.

## Layout

```
tracking_hub/
  ops/               Category 1 - Observatory Operations      (ROSP-9, 10, 11, 12)
  data_production/   Category 2 - Data Production             (ROSP-1, 2, 4, 6, 7, 8)
  data_delivery/     Category 3 - Data Delivery & User Experience (ROSP-3, 5)
  dev/               scratch space, not rendered by Times Square (flat; no subfolders)
```

Inside a category folder there is one folder per ROSP epic, named
`rosp-NN_<short_name>` (two-digit zero-padded key, lowercase, underscores). Category
and Layer are recorded as sidecar **tags** (`category-N`, `layer-N`), not as folders.

**Never rename or move a metric folder.** Times Square page URLs are derived from the
file path, and those URLs are linked from Confluence and Jira.

## One official notebook per epic

Each metric folder holds exactly one official notebook, `<folder>.ipynb`, with its
sidecar `<folder>.yaml`. Diagnostic notebooks may sit beside it, but they must carry a
`diagnostic` tag and must **not** carry the `rosp-NN` tag, so the official page stays
unambiguous.

## Versioning

Method versions live in git, not in file names. One PTH Method Change ticket maps to
one PR, and that PR's Times Square preview is the review artifact. When a baseline is
frozen, tag the commit `rosp-NN-vX.Y` and record the SHA on the metric's Confluence
page. Changes to a method go through the
[PTH Method Change Process](https://rubinobs.atlassian.net/wiki/spaces/ROSP/pages/2032959501).

## Standard parameters

Every notebook takes the same two parameters, declared in its sidecar and set in the
first code cell (which Times Square replaces):

| Parameter      | Meaning                    | Default    |
|----------------|----------------------------|------------|
| `start_dayobs` | First dayObs (inclusive)   | `20260629` |
| `end_dayobs`   | Last dayObs (inclusive)    | `20261231` |

The defaults are the PTH baseline epoch, 2026-06-29 to 2026-12-31.

## Scratch space

`dev/` is excluded from Times Square by the `ignore` glob in the root `times-square.yaml`.
**Keep `dev/` flat.** Times Square matches ignore globs with `PurePosixPath.match`, which
has no recursive wildcard, so a sidecar in a subfolder like `dev/sub/x.yaml` would be published.

## Shared helper code

Where shared helper code should live is still to be decided. Until then, do not use
relative imports between notebooks or folders here.
