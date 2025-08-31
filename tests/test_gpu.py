import torch

def test_cuda_available():
    assert torch.cuda.is_available(), "CUDA should be available on this machine"
    assert torch.cuda.device_count() >= 1
