import unittest

from scripts.run_chunk_size_sweep import build_report


class ChunkSizeSweepReportTests(unittest.TestCase):
    def test_build_report_documents_scope_and_results(self) -> None:
        text = build_report(
            case_id="20000x512",
            chunk_sizes=[512, 1024],
            rows=[
                {
                    "chunk_size": 512,
                    "median_wall_clock_sec": 0.020,
                    "mean_wall_clock_sec": 0.021,
                    "p95_wall_clock_sec": 0.022,
                    "throughput_samples_per_sec": 1_000_000.0,
                    "speedup_vs_1024": 0.90,
                    "mean_prediction": 0.060,
                },
                {
                    "chunk_size": 1024,
                    "median_wall_clock_sec": 0.018,
                    "mean_wall_clock_sec": 0.019,
                    "p95_wall_clock_sec": 0.020,
                    "throughput_samples_per_sec": 1_111_111.0,
                    "speedup_vs_1024": 1.00,
                    "mean_prediction": 0.061,
                },
            ],
            env={
                "python_version": "3.11.9",
                "platform": "Windows",
                "torch_version": "2.12.0+cu126",
                "cuda_version": "12.6",
                "runtime_kind": "cuda",
                "device_display_name": "NVIDIA GeForce RTX 4060 Ti",
            },
            warmup_runs=1,
            repeat_runs=3,
        )

        self.assertIn("RTX 4060 Ti Chunk-Size Sweep", text)
        self.assertIn("Speedup vs 1024 Chunk", text)
        self.assertIn("not a hand-written CUDA kernel microbenchmark", text)
        self.assertIn("docs/benchmark_methodology.md", text)
        self.assertIn("| 1024 |", text)


if __name__ == "__main__":
    unittest.main()
