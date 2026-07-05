import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device : {device}")

size = 1 << 27
a = torch.randn(size, device=device)

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

for _ in range(3):
    torch.max(a)
    torch.cuda.synchronize() if device.type == "cuda" else None

start_event.record()
c = torch.max(a)
end_event.record()

torch.cuda.synchronize() if device.type == "cuda" else None

print(f"result : \n{c}")

print(f"timeused : {start_event.elapsed_time(end_event):.6f} ms")  # 结果会接近 1.54ms
