"""Real GPU telemetry through NVML when a GPU is present; otherwise None (callers fall back to the simulator)."""
from __future__ import annotations

from typing import Optional


class GpuSampler:
    def __init__(self) -> None:
        self._nvml = None
        self._handle = None
        try:
            import pynvml  # provided by the `nvidia-ml-py` package

            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._nvml = pynvml
        except Exception:
            self._nvml = None

    @property
    def available(self) -> bool:
        return self._nvml is not None

    def read(self) -> Optional[dict]:
        if not self._nvml:
            return None
        try:
            util = self._nvml.nvmlDeviceGetUtilizationRates(self._handle)
            mem = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
            return {"util": float(util.gpu), "mem_used": float(mem.used), "mem_total": float(mem.total)}
        except Exception:
            return None
