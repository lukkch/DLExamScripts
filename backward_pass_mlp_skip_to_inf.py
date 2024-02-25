import torch
import torch.nn as nn
import torch.optim as optim

x = [[2]]
y = [[1]]

w = [[0.5]]
w_skip = [[0.5]]


class MLP(nn.Module):

    def __init__(self, w, w_skip):
        super().__init__()

        self.fc0 = nn.Linear(1, 1, bias=False)
        self.g0 = nn.ReLU()
        self.fc1 = nn.Linear(1, 1, bias=False)
        self.g1 = nn.ReLU()
        self.fc2 = nn.Linear(1, 1, bias=False)
        self.g2 = nn.ReLU()
        self.fc3 = nn.Linear(1, 1, bias=False)
        self.g3 = nn.ReLU()
        self.fc4 = nn.Linear(1, 1, bias=False)
        self.g4 = nn.ReLU()
        self.fc5 = nn.Linear(1, 1, bias=False)
        self.g5 = nn.ReLU()
        self.fc6 = nn.Linear(1, 1, bias=False)
        self.g6 = nn.ReLU()
        self.fc7 = nn.Linear(1, 1, bias=False)
        self.g7 = nn.ReLU()
        self.fc8 = nn.Linear(1, 1, bias=False)
        self.g8 = nn.ReLU()
        self.fc9 = nn.Linear(1, 1, bias=False)
        self.g9 = nn.ReLU()
        self.fc10 = nn.Linear(1, 1, bias=False)
        self.g10 = nn.ReLU()
        self.fc11 = nn.Linear(1, 1, bias=False)
        self.g11 = nn.ReLU()
        self.fc12 = nn.Linear(1, 1, bias=False)
        self.g12 = nn.ReLU()
        self.fc13 = nn.Linear(1, 1, bias=False)
        self.g13 = nn.ReLU()
        self.fc14 = nn.Linear(1, 1, bias=False)
        self.g14 = nn.ReLU()
        
        self.skip = nn.Linear(1, 1, bias=False)

        self.fc0.weight.data = torch.tensor(w).float().reshape_as(self.fc0.weight.data)
        self.fc1.weight.data = torch.tensor(w).float().reshape_as(self.fc1.weight.data)
        self.fc2.weight.data = torch.tensor(w).float().reshape_as(self.fc2.weight.data)
        self.fc3.weight.data = torch.tensor(w).float().reshape_as(self.fc3.weight.data)
        self.fc4.weight.data = torch.tensor(w).float().reshape_as(self.fc4.weight.data)
        self.fc5.weight.data = torch.tensor(w).float().reshape_as(self.fc5.weight.data)
        self.fc6.weight.data = torch.tensor(w).float().reshape_as(self.fc6.weight.data)
        self.fc7.weight.data = torch.tensor(w).float().reshape_as(self.fc7.weight.data)
        self.fc8.weight.data = torch.tensor(w).float().reshape_as(self.fc8.weight.data)
        self.fc9.weight.data = torch.tensor(w).float().reshape_as(self.fc9.weight.data)
        self.fc10.weight.data = torch.tensor(w).float().reshape_as(self.fc10.weight.data)
        self.fc11.weight.data = torch.tensor(w).float().reshape_as(self.fc11.weight.data)
        self.fc12.weight.data = torch.tensor(w).float().reshape_as(self.fc12.weight.data)
        self.fc13.weight.data = torch.tensor(w).float().reshape_as(self.fc13.weight.data)
        self.fc14.weight.data = torch.tensor(w).float().reshape_as(self.fc14.weight.data)
        self.skip.weight.data = torch.tensor(w_skip).float().reshape_as(self.skip.weight.data)

    def forward(self, x):
        z0 = self.fc0(x)
        g0 = self.g0(z0)

        z1 = self.fc1(g0)
        g1 = self.g1(z1)
        
        z2 = self.fc2(g1)
        g2 = self.g2(z2)
        
        z3 = self.fc3(g2)
        g3 = self.g3(z3)
        
        z4 = self.fc4(g3)
        g4 = self.g4(z4)
        
        z5 = self.fc5(g4)
        g5 = self.g5(z5)
        
        z6 = self.fc6(g5)
        g6 = self.g6(z6)
        
        z7 = self.fc7(g6)
        g7 = self.g7(z7)
        
        z8 = self.fc8(g7)
        g8 = self.g8(z8)
        
        z9 = self.fc9(g8)
        g9 = self.g9(z9)
        
        z10 = self.fc10(g9)
        g10 = self.g10(z10)
        
        z11 = self.fc11(g10)
        g11 = self.g11(z11)
        
        z12 = self.fc12(g11)
        g12 = self.g12(z12)
        
        z13 = self.fc13(g12)
        g13 = self.g13(z13)
        
        z14 = self.fc14(g13)
        g14 = self.g14(z14)

        y_hat = self.fc2(g14) + self.skip(g0)

        print(f" z0         = {z0}")
        print(f" g0         = {g0}")
        print(f" z1         = {z1}")
        print(f" g1         = {g1}")
        print(f" y_hat      = {y_hat}")

        z0.register_hook(lambda grad: print(f" ∂L/∂z0: {grad}"))
        g0.register_hook(lambda grad: print(f" ∂L/∂g0: {grad}"))
        z1.register_hook(lambda grad: print(f" ∂L/∂z1: {grad}"))
        g1.register_hook(lambda grad: print(f" ∂L/∂g1: {grad}"))
        y_hat.register_hook(lambda grad: print(f" ∂L/∂y_hat: {grad}"))

        return y_hat


mlp = MLP(w, w_skip)

optimizer = optim.SGD(mlp.parameters(), lr=1.0, momentum=0.0, dampening=0.0, weight_decay=0.0)

# initial weights
print("\Initial weights:")
print(f" w      = {mlp.fc0.weight.data}")

# forward pass
optimizer.zero_grad()
x = torch.tensor(x).float()
y = torch.tensor(y).float()
print("\nForward pass:")
print(f" x          = {x}")
print(f" y          = {y}")
pred = mlp(x)

# backward pass
loss = torch.sum(torch.abs(pred - y))
print(f" loss       = {loss}") # slightly under 0.5, because predicted y = 0.5*h0 + (term slightly > 0)
print("\nBackward Pass:")
loss.backward()

print("\nGradients:")
print(f" ∇w0        = {mlp.fc0.weight.grad}")
print(f" ∇w1        = {mlp.fc1.weight.grad}")
print(f" ∇w2        = {mlp.fc2.weight.grad}")
print(f" ∇w13       = {mlp.fc13.weight.grad}")
print(f" ∇w14       = {mlp.fc14.weight.grad}")
print(f" ∇w_skip    = {mlp.skip.weight.grad}")

optimizer.step()
print("\nUpdated weights:")
print(f" w0     = {mlp.fc0.weight.data}")
print(f" w1     = {mlp.fc1.weight.data}")
print(f" w2     = {mlp.fc2.weight.data}")
print(f" w13    = {mlp.fc13.weight.data}")
print(f" w14    = {mlp.fc14.weight.data}")
print(f" w_skip = {mlp.skip.weight.data}")
