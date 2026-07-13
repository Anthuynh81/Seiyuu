"""GPU resource management: one heavy model resident at a time (TTS XOR the local LLM)."""

from seiyuu.gpu.manager import GpuBusyError, GpuConsumer, GpuResourceManager, get_gpu_manager

__all__ = ["GpuBusyError", "GpuConsumer", "GpuResourceManager", "get_gpu_manager"]
