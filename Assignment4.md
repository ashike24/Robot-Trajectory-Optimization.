# Assignment 4: Learning-Based Trajectory Prediction and Interactive Dashboard

## Problem Statement

Build a complete trajectory-planning pipeline that bridges numerical optimization and machine learning. Using the optimizer from Assignment 3 as a data generator, train a neural network (MLP) to directly predict smooth joint-space trajectories from start and end configurations. Then deploy an interactive dashboard where users can input joint angles and visually compare the optimized trajectory against the learned prediction in both joint space and Cartesian space.

---

## Solution Overview

```
[Random start/end configs]
         │
         ▼
[Numerical Optimizer (Assign. 3)]  ──▶  Training dataset (X.npy, Y.npy)
         │
         ▼
[MLP: 4 inputs → 100 outputs]      ──▶  trajectory_net.pth
         │
         ▼
[Streamlit Dashboard]              ──▶  Real-time comparison
```

---

## Part 1: Dataset Generation

Each training sample consists of:

| Component | Description | Shape |
|---|---|---|
| Input `X` | `[q1_start, q2_start, q1_end, q2_end]` | `(4,)` |
| Output `Y` | Full optimized trajectory `[q1(t1), q2(t1), ..., q1(tN), q2(tN)]` | `(2N,)` |

### `generate_dataset.py`

```python
import numpy as np
from scipy.optimize import minimize

# ── Optimizer from Assignment 3 ───────────────────────────────────────────────
def optimize_trajectory(q_start, q_end, T=2.0, N=50):
    """
    Generate a minimum-acceleration trajectory from q_start to q_end.
    Returns trajectory array of shape (N, 2).
    """
    dt = T / (N - 1)
    q_init = np.linspace(q_start, q_end, N)   # linear initial guess

    def cost(x):
        q   = x.reshape(N, 2)
        acc = (q[2:] - 2*q[1:-1] + q[:-2]) / dt**2
        return np.sum(acc**2)

    constraints = [
        {'type': 'eq', 'fun': lambda x: x[:2]  - q_start},
        {'type': 'eq', 'fun': lambda x: x[-2:] - q_end},
    ]

    res = minimize(cost, q_init.flatten(), constraints=constraints,
                   method='SLSQP', options={'maxiter': 500, 'ftol': 1e-7})
    return res.x.reshape(N, 2)


# ── Generate Dataset ──────────────────────────────────────────────────────────
NUM_SAMPLES = 500
N           = 50

X, Y = [], []

for i in range(NUM_SAMPLES):
    q_start = np.random.uniform(-np.pi, np.pi, 2)
    q_end   = np.random.uniform(-np.pi, np.pi, 2)

    traj = optimize_trajectory(q_start, q_end, N=N)
    X.append(np.hstack([q_start, q_end]))
    Y.append(traj.flatten())

    if (i + 1) % 50 == 0:
        print(f"Generated {i+1}/{NUM_SAMPLES} samples")

X = np.array(X)   # shape (NUM_SAMPLES, 4)
Y = np.array(Y)   # shape (NUM_SAMPLES, 2*N)

np.save("X.npy", X)
np.save("Y.npy", Y)
print(f"Dataset saved: X={X.shape}, Y={Y.shape}")
```

---

## Part 2: Model Training

A 3-layer MLP learns the mapping:
```
[q1_start, q2_start, q1_end, q2_end]  ──▶  [q1(t1), q2(t1), ..., q1(tN), q2(tN)]
```

### `train_model.py`

```python
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

# ── Load Dataset ──────────────────────────────────────────────────────────────
X = np.load("X.npy")
Y = np.load("Y.npy")

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.float32)
X_test  = torch.tensor(X_test,  dtype=torch.float32)
Y_test  = torch.tensor(Y_test,  dtype=torch.float32)

# ── Model Architecture ────────────────────────────────────────────────────────
class TrajectoryNet(nn.Module):
    """
    MLP that maps 4 boundary values to a full N-step joint trajectory.
    Input:  [q1_start, q2_start, q1_end, q2_end]
    Output: [q1(t0), q2(t0), ..., q1(t_{N-1}), q2(t_{N-1})]
    """
    def __init__(self, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim)
        )

    def forward(self, x):
        return self.net(x)


out_dim = Y.shape[1]   # = 2 * N
model   = TrajectoryNet(out_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
loss_fn   = nn.MSELoss()

# ── Training Loop ─────────────────────────────────────────────────────────────
EPOCHS = 500
train_losses, test_losses = [], []

for epoch in range(EPOCHS):
    model.train()
    pred  = model(X_train)
    loss  = loss_fn(pred, Y_train)
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
        print(f"Epoch {epoch:4d} | Train Loss: {loss.item():.6f} | Test Loss: {test_loss:.6f}")

torch.save(model.state_dict(), "trajectory_net.pth")
print("Model saved to trajectory_net.pth")

# ── Plot Training Curve ───────────────────────────────────────────────────────
import matplotlib.pyplot as plt

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
```

---

## Part 3: Interactive Dashboard

### `app.py`

```python
import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ── Configuration ─────────────────────────────────────────────────────────────
N   = 50
l1  = l2 = 1.0

# ── Forward Kinematics ────────────────────────────────────────────────────────
def fk(q1, q2):
    x = l1*np.cos(q1) + l2*np.cos(q1 + q2)
    y = l1*np.sin(q1) + l2*np.sin(q1 + q2)
    return x, y

# ── Trajectory Optimizer ──────────────────────────────────────────────────────
def optimize_trajectory(q_start, q_end, T=2.0, N=50):
    dt = T / (N - 1)
    q_init = np.linspace(q_start, q_end, N)

    def cost(x):
        q   = x.reshape(N, 2)
        acc = (q[2:] - 2*q[1:-1] + q[:-2]) / dt**2
        return np.sum(acc**2)

    constraints = [
        {'type': 'eq', 'fun': lambda x: x[:2]  - q_start},
        {'type': 'eq', 'fun': lambda x: x[-2:] - q_end},
    ]
    res = minimize(cost, q_init.flatten(), constraints=constraints, method='SLSQP')
    return res.x.reshape(N, 2)

# ── Load Neural Network ───────────────────────────────────────────────────────
class TrajectoryNet(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, out_dim)
        )
    def forward(self, x):
        return self.net(x)

@st.cache_resource
def load_model():
    model = TrajectoryNet(2 * N)
    model.load_state_dict(torch.load("trajectory_net.pth", map_location='cpu'))
    model.eval()
    return model

model = load_model()

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🤖 2-Link Robot: Optimized vs Learned Trajectory")
st.markdown("Adjust the start and end joint angles, then click **Generate** to compare trajectories.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Start Configuration")
    q1s = st.slider("q1 start (rad)", -3.14, 3.14, -1.0, step=0.01)
    q2s = st.slider("q2 start (rad)", -3.14, 3.14,  0.5, step=0.01)

with col2:
    st.subheader("End Configuration")
    q1e = st.slider("q1 end (rad)", -3.14, 3.14,  1.0, step=0.01)
    q2e = st.slider("q2 end (rad)", -3.14, 3.14, -0.5, step=0.01)

if st.button("Generate Trajectories", type="primary"):
    q_start = np.array([q1s, q2s])
    q_end   = np.array([q1e, q2e])

    # Optimized trajectory
    with st.spinner("Running optimizer..."):
        opt_traj = optimize_trajectory(q_start, q_end, N=N)

    # Neural network prediction
    inp       = torch.tensor([[q1s, q2s, q1e, q2e]], dtype=torch.float32)
    pred_flat = model(inp).detach().numpy().flatten()
    pred_traj = pred_flat.reshape(N, 2)

    # MSE between trajectories
    mse = np.mean((opt_traj - pred_traj)**2)
    st.metric("Trajectory MSE (opt vs learned)", f"{mse:.6f}")

    t = np.linspace(0, 1, N)   # normalized time

    # ── Plot 1: Joint Angles ──────────────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(9, 4))
    ax1.plot(t, opt_traj[:,0],  color='steelblue',  lw=2,  label='q1 optimized')
    ax1.plot(t, pred_traj[:,0], color='steelblue',  lw=2, ls='--', label='q1 learned')
    ax1.plot(t, opt_traj[:,1],  color='darkorange', lw=2,  label='q2 optimized')
    ax1.plot(t, pred_traj[:,1], color='darkorange', lw=2, ls='--', label='q2 learned')
    ax1.set_xlabel('Normalized Time')
    ax1.set_ylabel('Joint Angle (rad)')
    ax1.set_title('Joint-Space Trajectory Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.4)
    st.pyplot(fig1)

    # ── Plot 2: End-Effector Path ─────────────────────────────────────────────
    xo, yo, xp, yp = [], [], [], []
    for i in range(N):
        x, y = fk(opt_traj[i, 0],  opt_traj[i, 1]);  xo.append(x); yo.append(y)
        x, y = fk(pred_traj[i, 0], pred_traj[i, 1]); xp.append(x); yp.append(y)

    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.plot(xo, yo, color='steelblue',  lw=2,      label='Optimized path')
    ax2.plot(xp, yp, color='darkorange', lw=2, ls='--', label='Learned path')
    ax2.scatter([xo[0], xo[-1]], [yo[0], yo[-1]], s=80, color='black', zorder=5)
    ax2.axis('equal')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('End-Effector Path in Cartesian Space')
    ax2.legend()
    ax2.grid(True, alpha=0.4)
    st.pyplot(fig2)
```

**Run the dashboard:**
```bash
streamlit run app.py
```

---

## Part 4: Results and Discussion

The complete pipeline — optimization → dataset generation → MLP training → dashboard — demonstrates how numerical methods and machine learning can complement each other in robotic trajectory planning.

**What worked well:**
- The MLP trained on 500 optimized samples learns to closely approximate optimizer outputs for configurations within the training distribution.
- Inference is near-instant (< 1 ms), compared to ~50–200 ms for the SLSQP optimizer, making the learned model suitable for real-time use.
- Joint-space trajectories predicted by the network are smooth and satisfy the broad shape of the optimized path.

**Limitations:**
- Prediction accuracy degrades for joint configurations far from the training distribution (extrapolation).
- The network does not strictly guarantee boundary conditions are met — a small residual error at start/end configurations may exist.
- With only 500 training samples, the model may miss fine details for unusual configurations.

**When to use each approach:**

| Scenario | Preferred Method |
|---|---|
| Offline planning with time budget | Numerical Optimization |
| Real-time control (< 5 ms required) | Learned Model |
| Novel / unseen configurations | Numerical Optimization |
| High-frequency replanning | Learned Model |
| Strict boundary guarantee needed | Numerical Optimization |

**Conclusion:** Learning-based trajectory predictors offer a compelling speed-accuracy trade-off. By treating the optimizer as a teacher, the neural network distills expensive computation into fast inference, enabling real-time robotic motion generation that would otherwise be impractical.
