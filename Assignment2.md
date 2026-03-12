# Assignment 2: Joint-Space Trajectory Generation and Smoothness Analysis

## Problem Statement

Generate continuous joint-space trajectories for the 2-link planar arm moving between two configurations over a fixed time duration `T`. Implement and compare two trajectory types:

1. **Linear interpolation** — joint angles vary linearly with time.
2. **Cubic polynomial** — a smooth trajectory with zero velocity at the start and end.

Analyze how each approach affects motion smoothness, and discuss which is more suitable for real robotic systems.

---

## Solution

### 1. Forward Kinematics (for end-effector path)

For any joint trajectory `q1(t), q2(t)`, the end-effector Cartesian position is:

```
x(t) = l1·cos(q1(t)) + l2·cos(q1(t) + q2(t))
y(t) = l1·sin(q1(t)) + l2·sin(q1(t) + q2(t))
```

---

### 2. Linear Joint-Space Trajectory

Each joint angle changes at a constant rate from start to end:

```
q1(t) = q1_start + (t / T) · (q1_end − q1_start)
q2(t) = q2_start + (t / T) · (q2_end − q2_start)
```

**Joint velocities** are constant throughout:
```
dq1/dt = (q1_end − q1_start) / T
dq2/dt = (q2_end − q2_start) / T
```

**Limitation:** Velocities are non-zero at both the start and end, causing abrupt jumps — this is mechanically undesirable.

---

### 3. Smooth Cubic Polynomial Trajectory

A cubic polynomial per joint is used to enforce zero velocity at the boundaries:

```
qi(t) = a0 + a1·t + a2·t² + a3·t³
```

**Boundary conditions:**
```
qi(0)  = qi_start    (position at start)
qi(T)  = qi_end      (position at end)
dqi/dt |t=0 = 0      (zero velocity at start)
dqi/dt |t=T = 0      (zero velocity at end)
```

**Solving the 4 conditions gives:**
```
a0 = qi_start
a1 = 0
a2 =  3·(qi_end − qi_start) / T²
a3 = −2·(qi_end − qi_start) / T³
```

**Final expression:**
```
qi(t) = qi_start + 3·(Δq/T²)·t² − 2·(Δq/T³)·t³
```

where `Δq = qi_end − qi_start`.

**Velocity profile:**
```
dqi/dt = 6·(Δq/T²)·t − 6·(Δq/T³)·t²
```

This is zero at `t=0` and `t=T` — smooth start and stop guaranteed.

---

### 4. Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

# Robot parameters
l1, l2 = 1.5, 1.0

# Motion parameters
q1_start, q2_start = 0.0, 0.0
q1_end,   q2_end   = np.pi / 2, np.pi / 4
T = 5.0
t = np.linspace(0, T, 500)

# --- Linear Trajectory ---
q1_lin = q1_start + (q1_end - q1_start) * t / T
q2_lin = q2_start + (q2_end - q2_start) * t / T

# --- Cubic Polynomial Trajectory ---
def cubic_trajectory(qs, qe, t, T):
    """Cubic polynomial with zero velocity at start and end."""
    dq = qe - qs
    return qs + 3 * dq * (t**2) / T**2 - 2 * dq * (t**3) / T**3

def cubic_velocity(qs, qe, t, T):
    """Velocity of cubic polynomial trajectory."""
    dq = qe - qs
    return 6 * dq * t / T**2 - 6 * dq * t**2 / T**3

q1_cubic = cubic_trajectory(q1_start, q1_end, t, T)
q2_cubic = cubic_trajectory(q2_start, q2_end, t, T)

q1_vel_lin   = np.gradient(q1_lin, t)
q1_vel_cubic = cubic_velocity(q1_start, q1_end, t, T)

# --- Forward Kinematics ---
def end_effector(q1, q2, l1, l2):
    x = l1 * np.cos(q1) + l2 * np.cos(q1 + q2)
    y = l1 * np.sin(q1) + l2 * np.sin(q1 + q2)
    return x, y

x_lin,   y_lin   = end_effector(q1_lin,   q2_lin,   l1, l2)
x_cubic, y_cubic = end_effector(q1_cubic, q2_cubic, l1, l2)

# --- Plot 1: Joint Angles ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(t, q1_lin,   label='q1 Linear')
axes[0].plot(t, q1_cubic, '--', label='q1 Cubic')
axes[0].plot(t, q2_lin,   label='q2 Linear')
axes[0].plot(t, q2_cubic, '--', label='q2 Cubic')
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Joint Angle (rad)')
axes[0].set_title('Joint Angles vs Time')
axes[0].legend()
axes[0].grid(True)

# --- Plot 2: Joint Velocities ---
axes[1].plot(t, q1_vel_lin,   label='q1 velocity (linear)')
axes[1].plot(t, q1_vel_cubic, '--', label='q1 velocity (cubic)')
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Joint Velocity (rad/s)')
axes[1].set_title('Joint Velocity Comparison')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig("trajectory_comparison.png", dpi=150)
plt.show()

# --- Plot 3: End-Effector Path ---
plt.figure(figsize=(6, 6))
plt.plot(x_lin,   y_lin,   label='Linear path')
plt.plot(x_cubic, y_cubic, '--', label='Cubic path')
plt.scatter([x_lin[0], x_lin[-1]], [y_lin[0], y_lin[-1]],
            color='black', zorder=5, label='Start / End')
plt.axis('equal')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('End-Effector Path in Cartesian Space')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("end_effector_path.png", dpi=150)
plt.show()
```

---

### 5. Comparison and Discussion

Linear trajectories are straightforward to compute but produce discontinuous velocity profiles — the joint starts and stops abruptly. This creates impulsive forces at the actuators, increasing mechanical wear and reducing positioning accuracy. Cubic polynomial trajectories enforce zero velocity at both endpoints, producing a smooth bell-shaped velocity profile that starts and ends at rest. This significantly reduces mechanical stress, actuator effort, and vibration in real systems. Polynomial trajectories are therefore much better suited for physical robots, especially at higher speeds or when handling sensitive payloads. The trade-off is a slightly more complex computation, which is negligible in practice. For offline planning or simulation, either method may work, but smooth trajectories are the clear choice whenever hardware performance matters.
