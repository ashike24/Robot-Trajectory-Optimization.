import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.optimize import minimize

N  = 50
l1 = l2 = 1.0

def fk(q1, q2):
    x = l1*np.cos(q1) + l2*np.cos(q1 + q2)
    y = l1*np.sin(q1) + l2*np.sin(q1 + q2)
    return x, y

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

st.title("🤖 2-Link Robot: Optimized vs Learned Trajectory")
st.markdown("Adjust joint angles and click **Generate** to compare trajectories.")

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

    with st.spinner("Running optimizer..."):
        opt_traj = optimize_trajectory(q_start, q_end, N=N)

    inp       = torch.tensor([[q1s, q2s, q1e, q2e]], dtype=torch.float32)
    pred_traj = model(inp).detach().numpy().reshape(N, 2)

    mse = np.mean((opt_traj - pred_traj)**2)
    st.metric("Trajectory MSE (optimized vs learned)", f"{mse:.6f}")

    t = np.linspace(0, 1, N)

    fig1, ax1 = plt.subplots(figsize=(9, 4))
    ax1.plot(t, opt_traj[:,0],  color='steelblue',  lw=2,       label='q1 optimized')
    ax1.plot(t, pred_traj[:,0], color='steelblue',  lw=2, ls='--', label='q1 learned')
    ax1.plot(t, opt_traj[:,1],  color='darkorange', lw=2,       label='q2 optimized')
    ax1.plot(t, pred_traj[:,1], color='darkorange', lw=2, ls='--', label='q2 learned')
    ax1.set_xlabel('Normalized Time')
    ax1.set_ylabel('Joint Angle (rad)')
    ax1.set_title('Joint-Space Trajectory Comparison')
    ax1.legend(); ax1.grid(True, alpha=0.4)
    st.pyplot(fig1)

    xo, yo, xp, yp = [], [], [], []
    for i in range(N):
        x, y = fk(opt_traj[i,0],  opt_traj[i,1]);  xo.append(x); yo.append(y)
        x, y = fk(pred_traj[i,0], pred_traj[i,1]); xp.append(x); yp.append(y)

    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.plot(xo, yo, color='steelblue',  lw=2,       label='Optimized path')
    ax2.plot(xp, yp, color='darkorange', lw=2, ls='--', label='Learned path')
    ax2.scatter([xo[0], xo[-1]], [yo[0], yo[-1]], s=80, color='black', zorder=5)
    ax2.axis('equal')
    ax2.set_xlabel('X (m)'); ax2.set_ylabel('Y (m)')
    ax2.set_title('End-Effector Path in Cartesian Space')
    ax2.legend(); ax2.grid(True, alpha=0.4)
    st.pyplot(fig2)
