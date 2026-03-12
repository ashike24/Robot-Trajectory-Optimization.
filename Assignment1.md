# Assignment 1: Forward Kinematics of a 2-Link Planar Arm

## Problem Statement

Derive and implement the **forward kinematics** of a 2-link planar robotic arm. Given joint angles `q1` and `q2` and link lengths `l1` and `l2`, compute the positions of the elbow joint and the end-effector. Visualize the arm in at least three different configurations and explain the effect of each joint angle on the robot's pose.

---

## Solution

### 1. Kinematic Model

For a 2-link planar arm fixed at the origin:

**Elbow position:**
```
x1 = l1 · cos(q1)
y1 = l1 · sin(q1)
```

**End-effector position:**
```
x2 = x1 + l2 · cos(q1 + q2)
   = l1·cos(q1) + l2·cos(q1 + q2)

y2 = y1 + l2 · sin(q1 + q2)
   = l1·sin(q1) + l2·sin(q1 + q2)
```

These equations hold for any choice of `l1` and `l2`.

---

### 2. Effect of Joint Angles

**Effect of q1:**
- Rotates the entire arm about the base joint at the origin.
- Changes the global orientation of both links simultaneously.
- Does not alter the relative angle between the two links.

**Effect of q2:**
- Controls the bend at the elbow joint.
- Determines how extended or folded the arm is.
- Large values (positive or negative) fold the arm back toward the base, reducing reach.

**Workspace:**
- Maximum reach (fully extended): `R_max = l1 + l2`
- Minimum reach (fully folded): `R_min = |l1 - l2|`
- With `l1 = l2 = 1`: reachable workspace is a disk of radius 2 centered at the origin.

---

### 3. Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

def forward_kinematics(q1, q2, l1=1.0, l2=1.0):
    """
    Compute elbow and end-effector positions for a 2-link planar arm.
    
    Args:
        q1: Shoulder joint angle (radians)
        q2: Elbow joint angle (radians)
        l1: Length of link 1
        l2: Length of link 2
    
    Returns:
        elbow: (x1, y1) position of elbow joint
        end_effector: (x2, y2) position of end-effector
    """
    x1 = l1 * np.cos(q1)
    y1 = l1 * np.sin(q1)

    x2 = x1 + l2 * np.cos(q1 + q2)
    y2 = y1 + l2 * np.sin(q1 + q2)

    return (x1, y1), (x2, y2)


# Three configurations to visualize
configs = {
    "Straight arm":  (0.0,       0.0),
    "Bent elbow":    (np.pi/4,   np.pi/2),
    "Folded back":   (np.pi/3,  -2*np.pi/3),
}

plt.figure(figsize=(7, 6))

for name, (q1, q2) in configs.items():
    elbow, ee = forward_kinematics(q1, q2)
    x_vals = [0, elbow[0], ee[0]]
    y_vals = [0, elbow[1], ee[1]]
    plt.plot(x_vals, y_vals, marker='o', linewidth=2, label=name)

plt.axhline(0, color='gray', linewidth=0.5)
plt.axvline(0, color='gray', linewidth=0.5)
plt.axis("equal")
plt.xlim(-2.5, 2.5)
plt.ylim(-2.5, 2.5)
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("2-Link Planar Robotic Arm — Three Configurations")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("arm_configurations.png", dpi=150)
plt.show()
```

---

### 4. Configuration Summary

| Configuration | q1 (rad) | q2 (rad) | End-effector (x, y) |
|---|---|---|---|
| Straight arm | 0.0 | 0.0 | (2.0, 0.0) |
| Bent elbow | π/4 | π/2 | (0.707, 1.707) |
| Folded back | π/3 | −2π/3 | (0.5, 0.866) |

---

### 5. Discussion

Forward kinematics provides a closed-form mapping from joint space `(q1, q2)` to Cartesian space `(x, y)`. Joint `q1` controls the global orientation of the entire arm — rotating it sweeps the end-effector around the base in a large arc. Joint `q2` controls the elbow bend, directly affecting how close or far the end-effector is from the base without changing the global arm direction. Together, these two degrees of freedom allow the end-effector to reach any point within an annular workspace bounded by `|l1 − l2|` and `l1 + l2`. When `l1 = l2`, the minimum reach is zero, meaning the arm can fold completely back on itself. Understanding this kinematic structure is the foundation for all subsequent trajectory planning tasks.
