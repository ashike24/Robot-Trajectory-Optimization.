import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

X = np.load("X.npy")
Y = np.load("Y.npy")

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.float32)
X_test  = torch.tensor(X_test,  dtype=torch.float32)
Y_test  = torch.tensor(Y_test,  dtype=torch.float32)


class TrajectoryNet(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 128),  nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, out_dim)
        )
    def forward(self, x):
        return self.net(x)


out_dim   = Y.shape[1]
model     = TrajectoryNet(out_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
loss_fn   = nn.MSELoss()

EPOCHS = 500
train_losses, test_losses = [], []

for epoch in range(EPOCHS):
    model.train()
    pred = model(X_train)
    loss = loss_fn(pred, Y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    model.eval()
    with torch.no_grad():
        test_loss = loss_fn(model(X_test), Y_test).item()

    train_losses.append(loss.item())
    test_losses.append(test_loss)

    if epoch % 50 == 0:
        print(f"Epoch {epoch:4d} | Train: {loss.item():.6f} | Test: {test_loss:.6f}")

torch.save(model.state_dict(), "trajectory_net.pth")
print("Model saved to trajectory_net.pth")

plt.figure(figsize=(8, 4))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses,  label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.yscale('log')
plt.title('Training and Test Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("training_curve.png", dpi=150)
plt.show()
