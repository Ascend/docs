"""CPU-only equivalence checks for the NPU override module.

The quick-start doc runs the same script (tests/torchtitan/
check_npu_overrides.py) before the training steps; this test wires it into
the repo's pytest suite so a local run catches a drift between the
override module and the upstream components it stands in for.

The script runs in a subprocess with its own directory as cwd, exactly like
the doc does: the repo root holds a ``torchtitan/`` clone during a
quick-start run, and that directory would otherwise shadow the installed
package as a namespace package.
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, 'check_npu_overrides.py')

EXPECTED = 'npu override equivalence: rope OK, mask OK'


class NpuOverrideEquivalenceTests(unittest.TestCase):
    """The override components must match the upstream ones numerically."""

    def test_check_script_passes(self) -> None:
        proc = subprocess.run(
            [sys.executable, SCRIPT],
            cwd=HERE,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0 and 'ModuleNotFoundError' in proc.stderr:
            self.skipTest(f'torchtitan is not installed: {proc.stderr.strip()}')
        self.assertEqual(
            proc.returncode, 0, f'script failed:\n{proc.stdout}\n{proc.stderr}'
        )
        self.assertEqual(proc.stdout.strip(), EXPECTED)
        self.assertIn('max_abs_diff', proc.stderr)


if __name__ == '__main__':
    unittest.main()
