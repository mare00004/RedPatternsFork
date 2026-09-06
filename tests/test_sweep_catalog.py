from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from red_patterns.models import ConvRun, TaylorRun
from red_patterns.sweep_catalog import SweepCatalogError, load_sweep_catalog
from red_patterns.sweep_jobs import runs_to_jsonl

from test_sweep_pipeline import build_sample_runs


class SweepCatalogTests(unittest.TestCase):
    def test_catalog_resolves_every_configured_run_and_reports_missing_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runs = build_sample_runs()
            (root / "runs.jsonl").write_text(runs_to_jsonl(runs), encoding="utf-8")
            present = root / "results" / runs[0].run_id / "run.h5"
            present.parent.mkdir(parents=True)
            present.touch()

            catalog = load_sweep_catalog(root)

            self.assertEqual(catalog.root, root)
            self.assertEqual([entry.run_id for entry in catalog.entries], ["r000001", "r000002"])
            self.assertTrue(catalog.entries[0].h5_exists)
            self.assertFalse(catalog.entries[1].h5_exists)
            self.assertEqual(catalog.entries[1].run_h5, root / "results" / "r000002" / "run.h5")
            self.assertIsInstance(catalog.entries[0].run, TaylorRun)
            self.assertIsInstance(catalog.entries[1].run, ConvRun)

    def test_catalog_rejects_a_directory_without_runs_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(SweepCatalogError, "does not contain"):
                load_sweep_catalog(tmpdir)

    def test_catalog_wraps_invalid_jsonl_as_a_catalog_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "runs.jsonl").write_text('{"not": "a run"}\n', encoding="utf-8")

            with self.assertRaisesRegex(SweepCatalogError, "Could not read"):
                load_sweep_catalog(root)
