import numpy as np
from scipy.optimize import minimize


def optimize_trajectory(q_start, q_end, T=2.0, N=50):
    """
    Generate a minimum-acceleration trajectory from q_start to q_end.
    Returns trajectory array of shape (N, 2).
    """
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
    res = minimize(cost, q_init.flatten(), constraints=constraints,
                   method='SLSQP', options={'maxiter': 500, 'ftol': 1e-7})
    return res.x.reshape(N, 2)


NUM_SAMPLES = 500
N = 50
X, Y = [], []

for i in range(NUM_SAMPLES):
    q_start = np.random.uniform(-np.pi, np.pi, 2)
    q_end   = np.random.uniform(-np.pi, np.pi, 2)
    traj    = optimize_trajectory(q_start, q_end, N=N)
    X.append(np.hstack([q_start, q_end]))
    Y.append(traj.flatten())
    if (i + 1) % 50 == 0:
        print(f"Generated {i+1}/{NUM_SAMPLES} samples")

X = np.array(X)
Y = np.array(Y)
np.save("X.npy", X)
np.save("Y.npy", Y)
print(f"Dataset saved: X={X.shape}, Y={Y.shape}")
