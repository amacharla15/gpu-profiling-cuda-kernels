
import torch
import torchvision.models as models

device = "cuda"
batch_size = 64

model = models.resnet50(weights=None).to(device=device).eval()
images = torch.randn(batch_size, 3, 224, 224, device=device)

with torch.no_grad():
    for _ in range(3):
        _ = model(images)
torch.cuda.synchronize()

with torch.no_grad():
    _ = model(images)
torch.cuda.synchronize()

print("ncu target script complete.")
