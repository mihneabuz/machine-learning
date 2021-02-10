import torch
x = torch.rand(5, 3)
print(x)
print(torch.cuda.is_available())
if torch.cuda.is_available():
    x = torch.cuda.current_device()
    print(torch.cuda.get_device_name(x))
