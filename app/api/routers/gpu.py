from fastapi import APIRouter
import torch

router = APIRouter(prefix="/gpu", tags=["gpu"])

@router.get("/status")
def gpu_status():
    ok = torch.cuda.is_available()
    name = torch.cuda.get_device_name(0) if ok else "CPU"
    return {
        "cuda": ok,
        "device_count": torch.cuda.device_count() if ok else 0,
        "name": name,
        "torch": torch.__version__,
    }
