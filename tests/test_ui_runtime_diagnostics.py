import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class UiRuntimeDiagnosticsTests(unittest.TestCase):
    def test_pyside_native_probe_uses_accelerator_runtime_labels(self) -> None:
        source = (REPO_ROOT / "pyside_sram_app_advanced.py").read_text(encoding="utf-8")

        self.assertNotIn("Torch CUDA backend available to wrapper", source)
        self.assertNotIn("torch.cuda.is_available()", source)
        self.assertIn("Torch accelerator backend available to wrapper", source)
        self.assertIn("Torch accelerator backend detail", source)


if __name__ == "__main__":
    unittest.main()
