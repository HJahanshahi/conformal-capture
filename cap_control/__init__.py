"""
cap_control - Uncertainty-aware MPC for space debris capture.

Phase 4+ of the debris-capture pipeline. Builds on:
    capture_lib     - Phases 0-3 (target sim, UPN, feasibility, grasp trigger)
    upn             - Uncertainty Propagation Networks
    space-robot-dq  - free-floating 7-DOF manipulator kinematics/dynamics

Submodules:
    dynamics    - free-floating chaser dynamics wrapper
    prediction  - UPN predictor wrapper for receding-horizon use
    controller  - MPC and cost-term library
    simulation  - closed-loop simulator (chaser + target + sensors + MPC)
    utils       - SE(3) helpers, logging
"""
__version__ = "0.1.0"
