import csv
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _format_throughput_range(values: list[float]) -> str:
    low = min(values)
    high = max(values)
    return f"{int(low // 1000.0)}k-{high / 1_000_000.0:.2f}M"


def _format_speedup_range(values: list[float]) -> str:
    return f"{min(values):.2f}x-{max(values):.2f}x"


class PortfolioPublicEvidenceTests(unittest.TestCase):
    def test_readme_representative_numbers_match_cuda_full_results(self) -> None:
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        results_path = REPO_ROOT / "reports" / "portability" / "cuda_full_results.csv"
        with results_path.open("r", encoding="utf-8", newline="") as fp:
            rows = list(csv.DictReader(fp))

        cpu_existing = {
            row["case_id"]: float(row["throughput_samples_per_sec"])
            for row in rows
            if row["lane"] == "cpu_existing" and row["status"] == "pass"
        }
        torch_rows = [row for row in rows if row["lane"] == "torch_accelerated" and row["status"] == "pass"]
        throughputs = [float(row["throughput_samples_per_sec"]) for row in torch_rows]
        speedups = [
            float(row["throughput_samples_per_sec"]) / cpu_existing[row["case_id"]]
            for row in torch_rows
        ]

        self.assertIn(_format_throughput_range(throughputs), readme)
        self.assertIn(_format_speedup_range(speedups), readme)
        self.assertIn("Speedup vs CPU Existing", (REPO_ROOT / "reports" / "portability" / "cuda_full_report.md").read_text(encoding="utf-8"))

    def test_public_docs_do_not_make_banned_claims(self) -> None:
        banned_claims = [
            "fully portable to AMD GPUs",
            "ROCm-ready and validated",
            "HIP support complete",
            "CUDA-optimized SRAM simulator",
            "GPU-accelerated SRAM physical simulator",
            "PDK signoff validated",
            "silicon accurate",
        ]
        public_docs = [
            REPO_ROOT / "README.md",
            REPO_ROOT / "PORTFOLIO_REVIEW.md",
            REPO_ROOT / "docs" / "benchmark_methodology.md",
            REPO_ROOT / "docs" / "reproduce_cuda_4060ti.md",
            REPO_ROOT / "reports" / "portability" / "README.md",
            REPO_ROOT / "reports" / "portability" / "cuda_full_report.md",
            REPO_ROOT / "reports" / "portability" / "cuda_full_fidelity.md",
            REPO_ROOT / "reports" / "portability" / "cuda_full_environment.txt",
            REPO_ROOT / "reports" / "portability" / "dashboard.md",
        ]
        for path in public_docs:
            text = path.read_text(encoding="utf-8")
            for claim in banned_claims:
                self.assertNotIn(claim, text, msg=f"{claim!r} found in {path}")


if __name__ == "__main__":
    unittest.main()
