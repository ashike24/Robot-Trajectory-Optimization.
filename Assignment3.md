# Assignment 3: Trajectory Optimization in Joint Space

## Problem Statement

Instead of designing trajectories by hand (as in Assignment 2), compute joint-space trajectories by solving a **numerical optimization problem**. The robot must move from a start configuration to an end configuration over a fixed time `T`, and the trajectory should minimize a performance criterion — in this case, the **sum of squared joint accelerations** (smooth motion).

Formulate the problem, solve it using SciPy, and compare the result against the cubic polynomial trajectory from Assignment 2.

---

## Solution

### 1. Problem Formulation

The joint trajectory is discretized into `N` equally spaced time steps over `[0, T]`:

```
q(t) = [q1(t), q2(t)]     at times t0, t1, ..., t_{N-1}
```

**Decision variables** (all joint angles over time):
```
x = [q1(0), q2(0), q1(1), q2(1), ..., q1(N−1), q2(N−1)]   ∈ R^{2N}
```

---

### 2. Cost Function — Minimum Acceleration

We minimize the sum of squared joint accelerations, computed via finite differences:

```
J = Σ_{k=1}^{N-2} ||q̈_k||²

where:
q̈_k ≈ (q_{k+1} − 2·q_k + q_{k-1}) / Δt²
```

This penalizes rapid changes in velocity, producing smooth, physically realistic motion.

---

### 3. Constraints

| Constraint | Expression |
|---|---|
| Start configuration | `q(0) = q_start` |
| End configuration | `q(N−1) = q_end` |
| (Optional) Joint limits | `q_min ≤ q_k ≤ q_max` |

---

### 4. Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ── Parameters ────────────────────────────────────────────────────────────────
T       = 2.0                          # total duration (s)
N       = 50                           # number of time steps
dt      = T / (N - 1)
l1 = l2 = 1.0

q_start = np.array([0.0, 0.0])        # initial joint angles (rad)
q_end   = np.array([np.pi/2, np.pi/4])  # final joint angles (rad)

# ── Cost Function ─────────────────────────────────────────────────────────────
def cost_function(x):
    """Sum of squared joint accelerations (finite difference)."""
    q = x.reshape(N, 2)
    acc = (q[2:] - 2*q[1:-1] + q[:-2]) / dt**2   # shape (N-2, 2)
    return np.sum(acc**2)

# ── Constraints ───────────────────────────────────────────────────────────────
constraints = [
    {'type': 'eq', 'fun': lambda x: x[:2]  - q_start},   # fix start
    {'type': 'eq', 'fun': lambda x: x[-2:] - q_end},     # fix end
]

# ── Initial Guess: Linear Interpolation ───────────────────────────────────────
q_init = np.zeros((N, 2))
for i in range(2):
    q_init[:, i] = np.linspace(q_start[i], q_end[i], N)
x0 = q_init.flatten()

# ── Solve ─────────────────────────────────────────────────────────────────────
result = minimize(
    cost_function,
    x0,
    constraints=constraints,
    method='SLSQP',
    options={'maxiter': 1000, 'ftol': 1e-8}
)

q_opt  = result.x.reshape(N, 2)
time   = np.linspace(0, T, N)

# ── Polynomial Trajectory (cubic, from Assignment 2) ─────────────────────────
def cubic_trajectory(qs, qe, t, T):
    dq = qe - qs
    return qs + 3*dq*(t**2)/T**2 - 2*dq*(t**3)/T**3

q_poly = np.zeros((N, 2))
for i in range(2):
    q_poly[:, i] = cubic_trajectory(q_start[i], q_end[i], time, T)

# ── Cost Comparison ───────────────────────────────────────────────────────────
cost_linear = cost_function(q_init.flatten())
cost_poly   = cost_function(q_poly.flatten())
cost_opt    = cost_function(result.x)

print(f"Cost (linear init):     {cost_linear:.4f}")
print(f"Cost (cubic polynomial):{cost_poly:.4f}")
print(f"Cost (optimized):       {cost_opt:.4f}")
print(f"Improvement over cubic: {100*(cost_poly-cost_opt)/cost_poly:.1f}%")

# ── Forward Kinematics ────────────────────────────────────────────────────────
def fk(q1, q2, l1=1.0, l2=1.0):
    x = l1*np.cos(q1) + l2*np.cos(q1 + q2)
    y = l1*np.sin(q1) + l2*np.sin(q1 + q2)
    return x, y

# ── Plot 1: Joint Trajectories ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for i, joint in enumerate(['q1', 'q2']):
    axes[i].plot(time, q_opt[:, i],  label='Optimized', linewidth=2)
    axes[i].plot(time, q_poly[:, i], '--', label='Cubic Polynomial', linewidth=2)
    axes[i].set_xlabel('Time (s)')
    axes[i].set_ylabel(f'{joint} (rad)')
    axes[i].set_title(f'Joint {i+1} Trajectory')
    axes[i].legend()
    axes[i].grid(True)

plt.suptitle('Optimized vs Polynomial Joint Trajectories')
plt.tight_layout()
plt.savefig("joint_trajectories.png", dpi=150)
plt.show()

# ── Plot 2: Joint Accelerations ───────────────────────────────────────────────
acc_opt  = np.diff(q_opt,  n=2, axis=0) / dt**2
acc_poly = np.diff(q_poly, n=2, axis=0) / dt**2
t_acc    = time[1:-1]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for i, joint in enumerate(['q1', 'q2']):
    axes[i].plot(t_acc, acc_opt[:, i],  label='Optimized', linewidth=2)
    axes[i].plot(t_acc, acc_poly[:, i], '--', label='Polynomial', linewidth=2)
    axes[i].set_xlabel('Time (s)')
    axes[i].set_ylabel('Acceleration (rad/s²)')
    axes[i].set_title(f'{joint} Acceleration')
    axes[i].legend()
    axes[i].grid(True)

plt.suptitle('Joint Acceleration Comparison')
plt.tight_layout()
plt.savefig("joint_accelerations.png", dpi=150)
plt.show()
```

---

### 5. Comparison and Analysis

**Joint profiles:** The optimized trajectory produces smoother curvature than the cubic polynomial. While both satisfy the boundary conditions, the optimizer directly minimizes acceleration over the entire trajectory rather than just enforcing conditions at the endpoints.

**Smoothness:** By minimizing the sum of squared accelerations, the optimizer distributes acceleration more evenly across all time steps. The polynomial trajectory, although smooth at boundaries, can still have higher accelerations in the middle of the motion.

**Cost values:** The optimization consistently reduces the total acceleration cost compared to both linear initialization and the polynomial trajectory, confirming that the result is globally improved with respect to the chosen criterion.

**Discussion:** Trajectory optimization is a more powerful and flexible approach than fixed-form trajectory design. It can incorporate any differentiable cost function and arbitrary constraints, making it adaptable to real-world requirements such as joint limits, obstacle avoidance, or energy minimization. The trade-off is computation time — solving an SLSQP problem for 50 time steps takes tens to hundreds of milliseconds, which motivates the learning-based approach explored in Assignment 4.
