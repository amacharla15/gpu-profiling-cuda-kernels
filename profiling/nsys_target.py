
import torch
import torchvision.models as models

device = "cuda"
dtype = torch.float32
batch_size = 64

model = models.resnet50(weights=None).to(device=device, dtype=dtype).eval()
images = torch.randn(batch_size, 3, 224, 224, device=device, dtype=dtype)

with torch.no_grad():
    for _ in range(3):
        _ = model(images)
torch.cuda.synchronize()

torch.cuda.nvtx.range_push("resnet50_inference")
with torch.no_grad():
    for _ in range(5):
        _ = model(images)
        torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()

print("nsys target script complete.")
