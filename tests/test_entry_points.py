"""Run in fresh interpreters so no test has imported ``pyvista_krige`` beforehand."""

from __future__ import annotations

import subprocess
import sys


def run_python(code: str) -> str:
    result = subprocess.run(
        [sys.executable, '-c', code], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def test_import_pyvista_does_not_import_the_plugin():
    out = run_python('import sys, pyvista; print("pyvista_krige" in sys.modules)')
    assert out == 'False'


def test_accessor_and_component_resolve_without_importing_the_plugin():
    out = run_python(
        'import pyvista as pv\n'
        'pv.OFF_SCREEN = True\n'
        'print(type(pv.Sphere().krige).__name__, type(pv.Plotter().krige).__name__)'
    )
    assert out == 'KrigeAccessor KrigeComponent'
