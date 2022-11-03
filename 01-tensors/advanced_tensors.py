import torch
import math 
import matplotlib.pyplot as plt

dtype = torch.float
device = torch.device('cpu')

x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
print(x)
y = torch.sin(x)

# hidden layer
a = torch.randn((), device=device, dtype=dtype)
b = torch.randn((), device=device, dtype=dtype)
c = torch.randn((), device=device, dtype=dtype)
d = torch.randn((), device=device, dtype=dtype)

learning_rate = 1e-6 # 0.0000001

plt.figure()
for t in range(2000):
    y_pred = a + b * x + c * x ** 2 + d * x ** 3
    
    loss = (y_pred- y).pow(2).sum().item()
    
    grad_y_pred = 2.0 * (y_pred - y)
    
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()
    
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d
    
    plt.plot(x.numpy(), y_pred.detach().numpy())
    
plt.plot(x.numpy(), y.numpy(), 'r')
plt.show()