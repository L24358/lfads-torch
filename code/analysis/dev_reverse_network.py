"""
Develop "ReverseNetwork" to perform pseudo-inverse of a matrix with an ill-conditioned inverse.

The matrix inverse is ill-conditioned because the matrix is of dimension (r, f), r > f.
This means that the inverse will be projecting a high dimensional vector to lower dimensions, of which
there are infinite ways to do.

https://ieeexplore.ieee.org/document/5726567
"""

import numpy as np
import torch
import torch.nn as nn

class ReverseNetwork(nn.Module):
    def __init__(self, r_dim, f_dim):
        super().__init__()
        self.linear = nn.Linear(r_dim, f_dim)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, f_dim)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        
    def forward(self, r):
        x = self.linear(r)
        return x
    
batch_size = 1024
r_dim = 100
f_dim = 10
A = torch.randn(r_dim, f_dim)
net = ReverseNetwork(r_dim, f_dim)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

for epoch in range(1000):
    f = torch.randn(batch_size, f_dim)
    r = torch.matmul(f, A.t())
    f_pred = net(r)
    loss = criterion(f, f_pred)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
        
    if epoch % 100 == 0:
        norm = torch.norm(f, dim=1).mean() * torch.norm(A)
        print(f"EPOCH {epoch}: LOSS {loss.item()}, NORM {norm.item()}")
        
        with torch.no_grad():
            A_pinv = torch.linalg.pinv(A)
            loss_inv = criterion(f, torch.matmul(r, A_pinv.t()))
            print(f"LOSS with inverse {loss_inv.item()}")

print(f[0])
print(f_pred[0])