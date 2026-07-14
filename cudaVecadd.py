import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"device : {device}")

size = 1 << 27
a = torch.randn(size, device=device)
b = torch.randn(size, device=device)

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

for _ in range(16):
    with torch.no_grad():
        a + b
    torch.cuda.synchronize() if device.type == "cuda" else None

start_event.record()
for _ in range(16):
    with torch.no_grad():
        c = a + b
end_event.record()
torch.cuda.synchronize() if device.type == "cuda" else None


# print(f"result(sliced) : \n{c[:5]}")

print(f"torch time used avg : {start_event.elapsed_time(end_event) / 16:.6f} ms")
