import unittest
from unittest import mock

from backends.registry import (
    get_backend_capabilities,
    get_gpu_backend_capability,
    get_runtime_backend_capabilities,
)
from backends.torch_portable import TorchRuntimeInfo


class BackendRegistryTests(unittest.TestCase):
    def test_cpu_lanes_are_always_available(self) -> None:
        capabilities = {cap.name: cap for cap in get_backend_capabilities(device_mode="auto")}
        self.assertTrue(capabilities["cpu_existing"].available)
        self.assertTrue(capabilities["cpu_numpy"].available)
        self.assertEqual(capabilities["cpu_existing"].device, "cpu")
        self.assertEqual(capabilities["cpu_numpy"].device, "cpu")

    def test_gpu_lane_can_be_forced_off_by_device_mode(self) -> None:
        gpu_capability = get_gpu_backend_capability(device_mode="cpu")
        self.assertFalse(gpu_capability.available)
        self.assertEqual(gpu_capability.reason, "device_mode_cpu")
        self.assertTrue(gpu_capability.fallback_allowed)

    def test_gpu_lane_reports_availability_or_known_skip_reason(self) -> None:
        gpu_capability = get_gpu_backend_capability(device_mode="auto")
        self.assertIn(
            gpu_capability.reason,
            {"cuda-ready", "hip-ready", "torch-unavailable", "accelerator-unavailable"},
        )

    def test_gpu_lane_reports_hip_device_when_runtime_is_rocm(self) -> None:
        runtime = TorchRuntimeInfo(
            accelerator_available=True,
            torch_device="cuda",
            backend_kind="hip",
            runtime_kind="rocm",
            device_display_name="AMD Instinct MI300",
            torch_version="2.12.0+rocm",
            torch_build_tag="rocm",
            cuda_version=None,
            hip_version="6.4",
            reason="hip-ready",
        )
        with mock.patch("backends.accelerator_lane.get_torch_runtime_metadata", return_value=runtime):
            gpu_capability = get_gpu_backend_capability(device_mode="auto")

        self.assertTrue(gpu_capability.available)
        self.assertEqual(gpu_capability.device, "hip")
        self.assertEqual(gpu_capability.backend_kind, "hip")
        self.assertEqual(gpu_capability.runtime_kind, "rocm")
        self.assertEqual(gpu_capability.device_display_name, "AMD Instinct MI300")

    def test_runtime_capabilities_include_python_fallback(self) -> None:
        capabilities = {cap.name: cap for cap in get_runtime_backend_capabilities("simulate", native_module=None)}
        self.assertIn("simulate_python_fallback", capabilities)
        self.assertTrue(capabilities["simulate_python_fallback"].available)
        self.assertIn("simulate_torch_accelerated", capabilities)
