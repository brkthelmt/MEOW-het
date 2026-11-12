"""MEOW package initialization and explicit accelerator selection."""
import os

# Configure optional CUDA extensions before importing model modules or torch hub.
if os.environ.get("MEOW_DEVICE", "").split(":")[0] == "npu":
    os.environ["XFORMERS_DISABLED"] = "1"
    import torch_npu  # noqa: F401
