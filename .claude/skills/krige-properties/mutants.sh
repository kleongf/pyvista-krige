#!/usr/bin/env zsh
# Prove the property suite still catches real bugs: apply one-line breaks to the
# source one at a time, run tests/properties, then restore the file with git.
# Every line must report failures; "passed" with no "failed" means a check went blind.
# Usage: PYTHON=/path/to/python .claude/skills/krige-properties/mutants.sh
set -u
cd "$(git rev-parse --show-toplevel)"
PY=${PYTHON:-python}
if [[ -n "$(git status --porcelain src)" ]]; then
  echo 'src/ has uncommitted changes; commit or set them aside first.' >&2
  exit 2
fi

mutate() {  # name file old new
  "$PY" - "$2" "$3" "$4" <<'PY'
import pathlib, sys
path, old, new = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
text = path.read_text()
if text.count(old) != 1:
    sys.exit(f'mutant anchor not found once in {path}: {old!r}')
path.write_text(text.replace(old, new))
PY
  if [[ $? -ne 0 ]]; then printf '%-26s SKIPPED (source changed, update the anchor)\n' "$1"; return; fi
  result=$(PYTHONPATH=src "$PY" -m pytest -q -p no:cacheprovider tests/properties 2>&1 | tail -1)
  git checkout -q -- "$2"
  printf '%-26s %s\n' "$1" "$result"
}

T=src/pyvista_krige/_targets.py
E=src/pyvista_krige/_exceedance.py
mutate samples-misaligned  $T 'return Samples(points[keep], values[keep], data, keep)' 'return Samples(points[keep], values[keep][::-1], data, keep)'
mutate grid-ravel-order    $T 'return np.concatenate(estimates).ravel(), np.concatenate(variances).ravel()' "return np.concatenate(estimates).ravel(order='F'), np.concatenate(variances).ravel(order='F')"
mutate chunk-off-by-one    $T 'chunk = selected[start : start + step]' 'chunk = selected[start : start + step - 1]'
mutate grid-slab-overlap   $T 'part = slice(start, start + rows)' 'part = slice(max(start - 1, 0), max(start - 1, 0) + rows)'
mutate cell-corner-not-center $T 'return pv.RectilinearGrid(*(0.5 * (a[:-1] + a[1:]) if a.size > 1 else a for a in axes))' 'return pv.RectilinearGrid(*(a[:-1] if a.size > 1 else a for a in axes))'
mutate where-mask-inverted $T 'skip = None if where is None else ~where.reshape(shape)' 'skip = None if where is None else where.reshape(shape)'
mutate exceedance-tail     $E 'probability = norm.sf(' 'probability = norm.cdf('
mutate exceedance-below    $E 'probability = 1.0 - probability' 'probability = probability'
mutate std-as-variance     $E "std = name.endswith(' std')" 'std = False'
mutate exceedance-scaled   $E '    probability[~np.isfinite(mean)] = np.nan' $'    probability = np.clip(probability * 1.01, 0, 1)\n    probability[~np.isfinite(mean)] = np.nan'
