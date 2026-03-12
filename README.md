# Robot Trajectory Optimization — 2-Link Planar Arm

A progressive 4-assignment project covering forward kinematics, trajectory generation, numerical optimization, and learning-based trajectory prediction for a 2-link planar robotic arm.

---

## Project Structure

```
robot-trajectory-optimization/
│
├── assignment1/
│   └── assignment1.md          # Forward Kinematics
│
├── assignment2/
│   └── assignment2.md          # Joint-Space Trajectory Generation
│
├── assignment3/
│   └── assignment3.md          # Trajectory Optimization in Joint Space
│
└── assignment4/
    ├── assignment4.md           # Learning-Based Trajectory Prediction
    ├── generate_dataset.py      # Dataset generation via optimization
    ├── train_model.py           # MLP training script
    └── app.py                   # Streamlit interactive dashboard
```

---

## Project Progression

| Assignment | Topic | Key Concept |
|---|---|---|
| 1 | Forward Kinematics | Static pose computation |
| 2 | Trajectory Generation | Linear vs. polynomial interpolation |
| 3 | Trajectory Optimization | Numerical optimization (min acceleration) |
| 4 | Learning-Based Prediction | MLP trained on optimized trajectories |

---

## Robot Model

A **2-link planar robotic arm** with joint angles `q1`, `q2` and link lengths `l1`, `l2`.

**Forward Kinematics:**
```
x = l1·cos(q1) + l2·cos(q1 + q2)
y = l1·sin(q1) + l2·sin(q1 + q2)
```

**Workspace:**
- Maximum reach: `R_max = l1 + l2`
- Minimum reach: `R_min = |l1 - l2|`

---

## Dependencies

```bash
pip install numpy scipy matplotlib torch scikit-learn streamlit
```

---

## How to Run

**Assignment 3 — Trajectory Optimization:**
```bash
python assignment3/trajectory_optimization.py
```

**Assignment 4 — Generate Dataset:**
```bash
python assignment4/generate_dataset.py
```

**Assignment 4 — Train Model:**
```bash
python assignment4/train_model.py
```

**Assignment 4 — Launch Dashboard:**
```bash
streamlit run assignment4/app.py
```

---

## Key Results

- Optimized trajectories achieve significantly lower acceleration cost than polynomial trajectories
- The trained MLP approximates optimized trajectories with near-instant inference
- The interactive dashboard enables real-time comparison of optimized vs. learned trajectories in both joint space and Cartesian space
