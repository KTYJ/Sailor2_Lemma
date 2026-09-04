import torch

print("PyTorch:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA Version:", torch.version.cuda)
    print("GPU Device:", torch.cuda.get_device_name(0))
    print("Capability:", torch.cuda.get_device_capability(0))
    x = torch.ones((2, 2), device="cuda")
    print("Tensor on GPU test successful:", x.device)
else:
    print("WARNING: CUDA is not available.")