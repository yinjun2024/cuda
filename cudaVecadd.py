import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device : {device}")

size = 1 << 25
a = torch.randn(size, device=device)
b = torch.randn(size, device=device)

for _ in range(3):
    a + b

torch.cuda.synchronize() if device.type == "cuda" else None
start = time.time()

c = a + b

torch.cuda.synchronize() if device.type == "cuda" else None
end = time.time()

print(f"result(sliced) : \n{c[:5]}")

print(f"timeused : {(end - start) * 1000:.2f} ms")