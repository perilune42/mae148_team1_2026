# Opponent State Estimation

This repository contains the implementation of state estimation filters for an autonomous racing opponent. The module transforms raw, noisy XYZ tracklet data from an onboard camera (via YOLO spatial detection) into a filtered state estimate in the global/inertial frame.

## Project Overview
The primary goal is to provide a robust estimate of an opponent's position and velocity to inform overtaking and collision avoidance logic.

### Estimation Approaches
We provide two primary filtering strategies, each with distinct trade-offs in complexity and reliability:

| Feature | **Extended Kalman Filter (EKF)** | **Linear Kalman Filter (KF)** |
| :--- | :--- | :--- |
| **Motion Model** | Kinematic Bicycle Model | Linear Constant Velocity |
| **State Vector** | $[x, y, \psi, v]^T$ (Pos, Yaw, Vel) | $[x, y, \dot{x}, \dot{y}]^T$ (Pos, Vel Components) |
| **Stability** | **Low.** Prone to instability due to finite difference Jacobians. | **High.** Decoupled X/Y dimensions; mathematically robust. |
| **Use Case** | Complex maneuvering/heading tracking. | General-purpose tracking and baseline stability. |

---

## Repository Structure

* `src/opp_ekf.py`: Implementation of the Extended Kalman Filter using a Kinematic Bicycle Model.
* `src/opp_kf.py`: Implementation of the Linear Kalman Filter using a Constant Velocity Model.
* `src/main_ekf.py`: The main entry point for the EKF implementation.
* `src/test_opp_ekf.py`: Unit tests for validating the EKF logic and convergence.
* `depthai_live_tracker.py`: (Reference) Pipeline script for integration with OAK-D/DepthAI hardware.
