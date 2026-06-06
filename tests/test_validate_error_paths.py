"""Error-path tests for benchmarks.validate.validate_artifact_dir."""

import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.validate import validate_artifact_dir


class ValidateArtifactDirErrorTests(unittest.TestCase):
    def test_rejects_missing_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            missing = Path(tempdir) / "does-not-exist"
            with self.assertRaises(ValueError) as ctx:
                validate_artifact_dir(missing)
            self.assertIn("does not exist", str(ctx.exception))

    def test_rejects_file_passed_as_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            target = Path(tempdir) / "file.txt"
            target.write_text("not a directory", encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                validate_artifact_dir(target)
            self.assertIn("not a directory", str(ctx.exception))

    def test_reports_missing_metadata_json(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            artifact_dir = Path(tempdir)
            with self.assertRaises(ValueError) as ctx:
                validate_artifact_dir(artifact_dir)
            self.assertIn("metadata.json", str(ctx.exception))

    def test_reports_invalid_json_with_location(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            artifact_dir = Path(tempdir)
            (artifact_dir / "metadata.json").write_text("{ not valid json", encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                validate_artifact_dir(artifact_dir)
            message = str(ctx.exception)
            self.assertIn("invalid JSON", message)
            self.assertIn("line", message)

    def test_reports_non_object_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            artifact_dir = Path(tempdir)
            (artifact_dir / "metadata.json").write_text(json.dumps([1, 2, 3]), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                validate_artifact_dir(artifact_dir)
            self.assertIn("JSON object", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
