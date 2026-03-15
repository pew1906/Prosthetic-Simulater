import torch
print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")
print(f"CUDA version    : {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU             : {torch.cuda.get_device_name(0)}")
    print(f"VRAM            : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("No CUDA — torch was installed without GPU support.")
    print("Fix: pip install torch --index-url https://download.pytorch.org/whl/cu121")
