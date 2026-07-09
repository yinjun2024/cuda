import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"device : {device}")

size = 1 << 13
a = torch.randn(size, size, device=device)
b = torch.randn(size, size, device=device)

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

for _ in range(15):
    with torch.no_grad():
        torch.mm(a, b)
    torch.cuda.synchronize() if device.type == "cuda" else None

start_event.record()
with torch.no_grad():
    c = torch.matmul(a, b)
end_event.record()
torch.cuda.synchronize() if device.type == "cuda" else None


# print(f"result(sliced) : \n{c[:5, :5]}")

print(f"timeused : {start_event.elapsed_time(end_event):.6f} ms")
