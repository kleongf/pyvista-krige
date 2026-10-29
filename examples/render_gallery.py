"""Render every example to ``docs/images/<name>.png`` for the README gallery."""

from __future__ import annotations

import runpy
from pathlib import Path

import pyvista as pv

EXAMPLES = Path(__file__).parent
IMAGES = EXAMPLES.parent / 'docs' / 'images'

pv.OFF_SCREEN = True
original_show = pv.Plotter.show

for script in sorted(EXAMPLES.glob('*.py')):
    if script.name == Path(__file__).name:
        continue
    target = IMAGES / f'{script.stem}.png'

    def show(self, *args, _target=target, **kwargs):
        kwargs['screenshot'] = _target
        return original_show(self, *args, **kwargs)

    pv.Plotter.show = show
    runpy.run_path(str(script), run_name='__main__')
    print(f'wrote {target.relative_to(EXAMPLES.parent)}')

pv.Plotter.show = original_show
