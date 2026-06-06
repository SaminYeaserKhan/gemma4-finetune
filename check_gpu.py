import torch


print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print("VRAM:", round(total_gb, 2), "GB")
else:
    print("No CUDA GPU detected.")

