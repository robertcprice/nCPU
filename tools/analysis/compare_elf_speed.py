import torch
import time
import sys
sys.path.insert(0, "/root/kvrm-spnc")

device = torch.device("cuda")

# Load Byte Classification model
from train_elf_curriculum import CurriculumELFLoader as ByteModel
byte_model = ByteModel().to(device)
checkpoint = torch.load("models/final/neural_elf_loader_best.pt", map_location=device)
if "model_state_dict" in checkpoint:
    byte_model.load_state_dict(checkpoint["model_state_dict"])
else:
    byte_model.load_state_dict(checkpoint)
byte_model.eval()

# Load Gumbel model  
from train_elf_gumbel import GumbelELFLoader as GumbelModel
gumbel_model = GumbelModel(hidden_dim=512).to(device)
gumbel_model.load_state_dict(torch.load("models/final/gumbel_elf_loader_best.pt", map_location=device))
gumbel_model.eval()

# Count parameters
byte_params = sum(p.numel() for p in byte_model.parameters())
gumbel_params = sum(p.numel() for p in gumbel_model.parameters())

print(f"Byte Classification: {byte_params:,} params")
print(f"Gumbel-Softmax:     {gumbel_params:,} params")
print()

# Benchmark
batch_size = 1024
x = torch.rand(batch_size, 64, 8, device=device)

# Warmup
for _ in range(10):
    with torch.no_grad():
        _ = byte_model(x)
        _ = gumbel_model(x, temperature=0.5)

torch.cuda.synchronize()

# Time Byte Classification
t0 = time.time()
for _ in range(100):
    with torch.no_grad():
        _ = byte_model(x)
torch.cuda.synchronize()
byte_time = (time.time() - t0) / 100

# Time Gumbel
t0 = time.time()
for _ in range(100):
    with torch.no_grad():
        _ = gumbel_model(x, temperature=0.5)
torch.cuda.synchronize()
gumbel_time = (time.time() - t0) / 100

print(f"Byte Classification: {byte_time*1000:.2f}ms per batch")
print(f"Gumbel-Softmax:     {gumbel_time*1000:.2f}ms per batch")
print()
print(f"Speed ratio: {byte_time/gumbel_time:.2f}x (>1 = Gumbel faster)")
