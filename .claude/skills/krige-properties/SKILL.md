---
name: krige-properties
description: Property suite for pyvista-krige. Seeded random sample sets check promises that must hold for every input, such as exact kriging at the samples, grid and point paths agreeing, chunking and where masks not changing values, bounded exceedance, and round trips for the planned features (transform, cells, suggest_samples, decluster, save/load, ZMAP/.asc IO, simulate). Use when a change touches predict, _targets.py, _exceedance.py, or implements or edits one of issues #2 to #9.
---

# krige-properties

`tests/properties/` checks promises that hold for every sample set, not just the fixtures in `tests/conftest.py`. Each test runs once per seed. A seed builds a random sample set: 8 to 30 points on a jittered lattice, a random scale per axis, a random field, and a random zero-nugget variogram (spherical, exponential or linear). The suite runs in about 2 seconds at 8 seeds and is part of the normal `pytest` run and CI.

There is no formal verifier for numpy and PyKrige code, so this is property testing of the real code with generated inputs. A pass covers the seeds that ran, not every input.

## When to use it

Use it for any change to these files and functions, and for any PR that implements issues #2 to #9:

| Module | Promise (and the harm if broken) | Code it covers |
| --- | --- | --- |
| `test_prop_predict.py` | With no nugget, the estimate at a sample is the sample and the variance is 0. If broken, results disagree with the data the user gave. | `KrigingModel.predict`, `_targets.sample_arrays`, `_model._sample_cloud` |
| `test_prop_predict.py` | An `ImageData` target gives the same values as its points as a `PolyData`. If broken, grid results are scrambled. | `_targets.execute_on`, `_targets._execute_grid`, `_targets.grid_axes` |
| `test_prop_predict.py` | `max_memory` chunking does not change any value, on grids and point sets. If broken, large targets get wrong or missing values. | `execute_on` (point runs), `_execute_grid` (slabs) |
| `test_prop_predict.py` | `where` keeps selected values unchanged and writes `null_value` elsewhere. If broken, masks drop or leak values. | `resolve_where`, `_execute_grid` masked path, `predict` |
| `test_prop_exceedance.py` | P(v > t) equals the normal tail of the estimate and its kriging std, is in [0, 1], never rises with t, P(>) + P(<) = 1, and std input equals variance input. If broken, risk maps are wrong. | `_exceedance.exceedance`, `kriging_std`, `uncertainty_name` |
| `test_prop_transform.py` (#2) | Back-transformed estimate at a sample is the sample; log estimates are positive; `transform=None` equals no transform. | `fit(transform=...)`, `_transform.py` |
| `test_prop_cells.py` (#3) | `preference='cell'` equals kriging at the cell centers, cell for cell. | `predict(preference='cell')` |
| `test_prop_suggest.py` (#4) | Each pick has the largest std once the earlier picks are added; picks are distinct target points. | `KrigingModel.suggest_samples` |
| `test_prop_decluster.py` (#6) | Weights are positive, sum to n, follow the samples when shuffled, and are all 1 on a lattice with cell size equal to the spacing. | `KrigeAccessor.decluster` |
| `test_prop_save_load.py` (#7) | A loaded model predicts bit for bit like the saved one. | `KrigingModel.save`, `KrigingModel.load` |
| `test_prop_grid_io.py` (#8) | Writing then reading a grid returns the same geometry, NaN pattern and values. | `write_esri_ascii`, `read_esri_ascii`, `write_zmap`, `read_zmap` |
| `test_prop_simulate.py` (#9) | Every realization equals the data at the samples; the same seed repeats exactly. | `KrigingModel.simulate` |

A module for a planned feature skips with the issue number until the feature exists. It finds the feature by name: `fit(transform=...)`, the `preference` parameter of `predict`, `KrigingModel.suggest_samples`, `KrigeAccessor.decluster`, `KrigingModel.load`, `pyvista_krige.read_esri_ascii` and `read_zmap`, and `KrigingModel.simulate`. The tests use the API written in each issue. If you implement it under a different name or array name, update the module in the same PR. Do not leave it skipping.

## Run it

From the repo root:

```bash
PYTHONPATH=src python -m pytest -q tests/properties -rs
```

On the maintainer's machine, `python` is `/Users/kevin/VisualStudioProjects/pyvista-project/pyvista/.venv/bin/python`, and `PYTHONPATH=src` makes a worktree test its own `src`. `-rs` lists what skipped, so you can see whether your feature's module ran.

Before a PR, run a deeper pass with 100 seeds (about 15 seconds):

```bash
KRIGE_PROPERTY_CASES=100 PYTHONPATH=src python -m pytest -q tests/properties
```

## Read a failure

Every assertion message names the seed. Rebuild the case to look at it:

```python
import sys

sys.path.insert(0, 'tests/properties')
from conftest import random_samples, fit_exact, grid_over

samples = random_samples(seed=5, dim=3)
model = fit_exact(samples, 5)
grid = grid_over(samples, 5, 3)
```

Tell a real bug from a test error:

- The failure is a bug in the code if the property is a promise the docstring or issue makes and the case is valid: distinct points, finite values, a zero-nugget variogram. Fix the code, not the test. Do not loosen a tolerance until it passes.
- The failure is a test error if the case breaks the test's own assumptions. Examples are two points close enough to make the kriging matrix singular, or an API name that differs from the issue. Fix the generator or the module and say so in the PR.
- Tolerances are 1e-6 to 1e-9 relative to the data scale. They allow float round-off, not modeling error. A miss of 1e-3 is a bug.

## Prove a new check can fail

A check that has never failed proves nothing.

**When you add or change a property**, break the code it guards with one line, see the check fail, then restore the line. For the existing promises, run the mutant script:

```bash
PYTHON=/Users/kevin/VisualStudioProjects/pyvista-project/pyvista/.venv/bin/python .claude/skills/krige-properties/mutants.sh
```

Every line must report failures. A line with only "passed" means a check stopped catching that bug. A line that says SKIPPED means the source changed under the mutant's anchor text: update the anchor in `mutants.sh` to the new code.

**When your PR implements a planned feature:**
1. Remove nothing from its module.
2. Confirm it now runs (no longer listed by `-rs`).
3. Add one mutant for your code to `mutants.sh`, such as a dropped back-transform term, a reversed row order in the reader, or a missing conditioning step.
4. Confirm that the mutant fails.

## Keep it in step with the code

Last confirmed against commit `d3bbda7` (`main`, v0.1.0). At that commit:
- All 9 mutants in `mutants.sh` failed the suite.
- The unchanged code passed 128 checks.
- 100 seeds passed with no findings.
- The planned-feature modules for #3, #4, #6 and #7 were checked against throwaway reference implementations, and one deliberate bug per check made them fail.
- The modules for #2, #8 and #9 have not yet been seen to fail. The PR that implements each one must do that, as described in the previous section.

When a change edits the covered code above:
1. Rerun the suite and `mutants.sh`.
2. Check that each promise in the table still describes the code.
3. Update this commit only after both hold at the new commit.

If the commit here is older than the last change to a covered file (check `git log -1 --format=%h -- src/pyvista_krige/_targets.py src/pyvista_krige/_exceedance.py src/pyvista_krige/_model.py`), the suite needs another look before you trust a pass.
