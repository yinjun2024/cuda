import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device : {device}")

size = 1 << 13
a = torch.randn(size, size, device=device)
b = torch.randn(size, size, device=device)

for _ in range(15):
    torch.mm(a, b)
    torch.cuda.synchronize() if device.type == "cuda" else None

start = time.time()

c = torch.matmul(a, b)

torch.cuda.synchronize() if device.type == "cuda" else None
end = time.time()

print(f"result(sliced) : \n{c[:5, :5]}")

print(f"timeused : {(end - start) * 1000:.6f} ms")