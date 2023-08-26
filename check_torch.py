"""
If you get the version numbers and True for torch.cuda.is_available(),
it means PyTorch and Torchvision are installed and working correctly.
"""
import torch
import torchvision

print(torch.__version__)
print(torch.cuda.is_available())
print(torchvision.__version__)