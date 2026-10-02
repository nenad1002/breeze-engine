__all__ = ["InferenceSession", "Qwen35CpuModel", "Phi35CpuModel", "Qwen35GpuModel", "Qwen35ReferenceModel"]


def __getattr__(name):
	# Load each backend only when its public API is requested.
	if name == "Qwen35CpuModel":
		from .qwen35_cpu import Qwen35CpuModel
		return Qwen35CpuModel
	if name == "Phi35CpuModel":
		from .phi35_cpu import Phi35CpuModel
		return Phi35CpuModel
	if name == "Qwen35GpuModel":
		from .qwen35_gpu import Qwen35GpuModel
		return Qwen35GpuModel
	if name == "Qwen35ReferenceModel":
		from .qwen35_reference import Qwen35ReferenceModel
		return Qwen35ReferenceModel
	if name == "InferenceSession":
		from .graph_session import InferenceSession
		return InferenceSession
	raise AttributeError(name)
