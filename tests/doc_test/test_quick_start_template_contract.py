"""Static contract for fixed upstream refs in the shared Quick Start guard."""

from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / ".github" / "workflows" / "quick-start-template.yml"


class TestQuickStartTemplateContract(unittest.TestCase):
    def test_fixed_ref_is_optional_and_propagated_to_the_test_job(self) -> None:
        text = TEMPLATE.read_text(encoding="utf-8")

        self.assertIn("fixed_ref:", text)
        self.assertIn("default: ''", text)
        self.assertIn("FIXED_REF: ${{ inputs.fixed_ref }}", text)

    def test_fixed_ref_commit_sha_drives_scheduled_change_detection(self) -> None:
        text = TEMPLATE.read_text(encoding="utf-8")

        self.assertIn('commits/$FIXED_REF', text)
        self.assertIn("caller-fixed ref", text)
        self.assertIn("ref=\"$FIXED_REF\"", text)


if __name__ == "__main__":
    unittest.main()
